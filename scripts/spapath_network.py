import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_st(data_type):
    return str(data_type).strip().upper() in {"ST", "ST_WITH_HE"}


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_features, out_features, dropout, alpha, concat=False, temp=1.0
    ):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.temp = temp
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, attention=True, tied_attention=None):
        Wh = torch.mm(
            h, self.W
        )  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        if tied_attention is not None:
            self.attention = tied_attention
        else:
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            self.attention = attention

        h_prime = torch.matmul(self.attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.xavier_uniform_(self.linear.weight.data, gain=1.414)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x, adj=None):

        x = self.linear(x)
        return x


class SingleNet(nn.Module):
    def __init__(self, feature_dim, latent_dim, data_type):
        super().__init__()
        self.use_gnn = _is_st(data_type)

        if self.use_gnn:
            self.encoder_layer1 = GraphAttentionLayer(
                feature_dim, latent_dim[0], dropout=0, alpha=0.2
            )
            self.encoder_layer2 = DenseLayer(latent_dim[0], latent_dim[1])
            self.decoder_layer1 = GraphAttentionLayer(
                latent_dim[1], latent_dim[0], dropout=0, alpha=0.2
            )
            self.decoder_layer2 = DenseLayer(latent_dim[0], feature_dim)
        else:
            self.encoder_layer1 = DenseLayer(feature_dim, latent_dim[0])
            self.encoder_layer2 = DenseLayer(latent_dim[0], latent_dim[1])
            self.decoder_layer1 = DenseLayer(latent_dim[1], latent_dim[0])
            self.decoder_layer2 = DenseLayer(latent_dim[0], feature_dim)

    def forward(self, node_feats, adj_matrix=None):
        if self.use_gnn:
            S = self.encoder(node_feats, adj_matrix)
            X_recon = self.decoder(S, adj_matrix)
        else:
            S = self.encoder(node_feats)
            X_recon = self.decoder(S)
        return S, X_recon

    def encoder(self, node_feats, adj_matrix=None):
        if self.use_gnn:
            H = F.elu(self.encoder_layer1(node_feats, adj_matrix))
        else:
            H = F.elu(self.encoder_layer1(node_feats))
        S = self.encoder_layer2(H)
        return S

    def decoder(self, S, adj_matrix=None):
        if self.use_gnn:
            H = F.elu(
                self.decoder_layer1(
                    S, adj_matrix, tied_attention=self.encoder_layer1.attention
                )
            )
        else:
            H = F.elu(self.decoder_layer1(S))
        X_recon = self.decoder_layer2(H)
        return X_recon


class IndEmbedNet(nn.Module):
    def __init__(self, feature_dim, latent_dim, n_slices, data_type):
        super().__init__()

        self.use_gnn = [_is_st(dt) for dt in data_type]
        self.n_slices = n_slices

        self.encoder_layers1 = nn.ModuleList(
            [
                self._make_layer1(feature_dim, latent_dim[0], self.use_gnn[i])
                for i in range(n_slices)
            ]
        )
        self.encoder_layers2 = nn.ModuleList(
            [DenseLayer(latent_dim[0], latent_dim[1]) for _ in range(n_slices)]
        )

        self.decoder_layers1 = nn.ModuleList(
            [
                self._make_layer1(latent_dim[1], latent_dim[0], self.use_gnn[i])
                for i in range(n_slices)
            ]
        )
        self.decoder_layers2 = nn.ModuleList(
            [DenseLayer(latent_dim[0], feature_dim) for _ in range(n_slices)]
        )

    def forward(self, node_feats_dict, adj_matrix_dict):
        S_dict, X_recon_dict = {}, {}
        for i in range(self.n_slices):
            S_dict[i] = self.encoder(i, node_feats_dict[i], adj_matrix_dict[i])
            X_recon_dict[i] = self.decoder(i, S_dict[i], adj_matrix_dict[i])
        return S_dict, X_recon_dict

    def _make_layer1(self, in_dim, out_dim, use_gnn):
        if use_gnn:
            return GraphAttentionLayer(in_dim, out_dim, dropout=0, alpha=0.2)
        else:
            return DenseLayer(in_dim, out_dim)

    def encoder(self, i, node_feats, adj_matrix=None):
        H = F.elu(self.encoder_layers1[i](node_feats, adj_matrix))
        S = self.encoder_layers2[i](H)
        return S

    def decoder(self, i, S, adj_matrix=None):
        if self.use_gnn[i]:
            H = F.elu(
                self.decoder_layers1[i](
                    S, adj_matrix, tied_attention=self.encoder_layers1[i].attention
                )
            )
        else:
            H = F.elu(self.decoder_layers1[i](S))
        X_recon = self.decoder_layers2[i](H)
        return X_recon
