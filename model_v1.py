import scipy.sparse as sp
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
import scanpy as sc
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
import importlib
import network_v1
import tools_v1
importlib.reload(network_v1)
importlib.reload(tools_v1)

MIN_TEMPERATURE = 1e-4

def _assert_finite_tensor(name, value):
    if torch.is_tensor(value) and not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} became non-finite.")
    return value

def _clip_grad_norm(parameters, max_norm):
    params = list(parameters)
    if len(params) == 0:
        return torch.tensor(0.0)
    try:
        return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    except TypeError:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if not torch.isfinite(total_norm):
            raise FloatingPointError("Gradient norm became non-finite.")
        return total_norm

def q_distribution(S, alpha, mu, eps=1e-8):
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    dist = torch.sum((S.unsqueeze(1) - mu) ** 2, dim=2)

    q = 1.0 / (1.0 + dist / alpha + eps)
    q = q ** ((alpha + 1.0) / 2.0)

    q = q / (torch.sum(q, dim=1, keepdim=True) + eps)
    return q

def target_distribution(q, eps=1e-8):
    p = q**2 / (torch.sum(q, dim=0) + eps)
    p = p  / (torch.sum(p, dim=1, keepdim=True) + eps)
    return p

def kld(target, pred, eps=1e-8):
    target = torch.clamp(target, min=eps)
    pred = torch.clamp(pred, min=eps)
    return torch.mean(torch.sum(target * torch.log(target / pred), dim=1))


def spv_contrastive_loss(features, labels, temperature=0.2):
    if temperature < MIN_TEMPERATURE:
        raise ValueError(f"temperature must be at least {MIN_TEMPERATURE}.")
    features = F.normalize(features, dim=1, eps=1e-8)
    similarity_matrix = torch.matmul(features, features.T)
    sim = similarity_matrix / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    self_mask = torch.eye(len(features), device=features.device, dtype=torch.bool)
    mask = mask * (~self_mask).float()

    logits = sim.masked_fill(self_mask, -torch.inf)
    log_den = torch.logsumexp(logits, dim=1, keepdim=True)
    log_den = torch.where(torch.isfinite(log_den), log_den, torch.zeros_like(log_den))
    log_prob = sim - log_den

    masked_log_prob = torch.where(mask.bool(), log_prob, torch.zeros_like(log_prob))
    loss = - masked_log_prob.sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    return loss.mean()

def mmd_loss_fun(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if kernel_mul <= 0:
        raise ValueError("kernel_mul must be positive.")
    if kernel_num <= 0:
        raise ValueError("kernel_num must be positive.")
    n = int(source.size(0))
    m = int(target.size(0))
    if n == 0 or m == 0:
        raise ValueError("source and target must both contain at least one sample.")

    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)

    loss = (XX + XY).sum() + (YX + YY).sum()
    return torch.sqrt(torch.clamp(loss, min=1e-12))

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        if temperature < MIN_TEMPERATURE:
            raise ValueError(f"temperature must be at least {MIN_TEMPERATURE}.")
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        device = z_i.device

        z_i = F.normalize(z_i, dim=1, eps=1e-8)
        z_j = F.normalize(z_j, dim=1, eps=1e-8)

        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.matmul(z, z.T)
        sim = sim / self.temperature

        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])

        mask = ~torch.eye(2 * batch_size, device=device).bool()
        sim = sim.masked_select(mask).view(2 * batch_size, -1)

        positives = torch.cat([
            torch.arange(batch_size, device=device) + batch_size,
            torch.arange(batch_size, device=device)
        ])

        positive_indices = positives.clone()
        for i in range(2 * batch_size):
            if positives[i] > i:
                positive_indices[i] -= 1

        loss = F.cross_entropy(sim, positive_indices)
        return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the Gram kernel matrix.
    - source: A data matrix with dimensions (sample_size_1 x feature_size), representing the first sample set.
    - target: A data matrix with dimensions (sample_size_2 x feature_size), representing the second sample set.
    - kernel_mul: This concept is somewhat unclear but appears to be related to calculating the bandwidth for each kernel.
    - kernel_num: The number of kernels, indicating the use of multiple kernels.
    - fix_sigma: Indicates whether to use a fixed standard deviation.
    
    Returns: A matrix of size ((sample_size_1 + sample_size_2) x (sample_size_1 + sample_size_2)), 
             structured as:
                            [   K_ss K_st
                                K_ts K_tt ]
             where K_ss and K_tt are the kernel matrices within the same sample sets, and K_st and K_ts are the 
             kernel matrices between the two different sample sets.
    """
    device = source.device
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # merge
    L2_distance = torch.cdist(total, total, p=2).pow(2)

    # Calculate the bandwidth for each kernel in a multi-kernel setup.
    if kernel_mul <= 0:
        raise ValueError("kernel_mul must be positive.")
    if kernel_num <= 0:
        raise ValueError("kernel_num must be positive.")
    if fix_sigma is not None and fix_sigma <= 0:
        raise ValueError("fix_sigma must be positive when provided.")

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, dtype=L2_distance.dtype, device=device)
    else:
        bandwidth = torch.sum(L2_distance.detach()) / max(n_samples**2-n_samples, 1)
        
    bandwidth = torch.clamp(bandwidth, min=torch.finfo(L2_distance.dtype).eps)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # The formula for the Gaussian kernel: exp(-|x-y|/bandwidth)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for 
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # Combine multiple kernels together.

def _is_st(data_type):
    return str(data_type).strip().upper() in {"ST", "ST_WITH_HE"}

def _int_label(label):
    return int(label)

def _cluster_size_records(labels, n_clusters=None):
    labels = np.asarray(labels).astype(int)
    if n_clusters is None:
        label_order = np.sort(np.unique(labels))
    else:
        label_order = np.arange(int(n_clusters))
    total = max(int(labels.shape[0]), 1)
    rows = []
    for label in label_order:
        count = int(np.sum(labels == label))
        rows.append({
            "cluster": int(label),
            "count": count,
            "fraction": float(count / total),
        })
    return rows

def _print_cluster_size_records(prefix, rows):
    size_text = ", ".join(
        f"{row['cluster']}:{row['count']}({row['fraction']:.3f})"
        for row in rows
    )
    print(f"{prefix} {size_text}")

class Model():
    def __init__(self,
                 adata_full,
                 adata_type_map,
                 batch_key = 'batch',
                 pre_cellembed_key = 'pre_embed',
                 cellembed_key = 'cell_embed',
                 geneembed_key = 'gene_embed',
                 cluster_key = 'cluster',
                 cell_gene_dist_key = 'cell_gene_dist',
                 n_pre_training_steps=500,
                 n_training_steps=200,
                 step_interval=100,
                 hidden_dims=[512,30],
                 coef_recon=1.0,
                 coef_inter=1.0,
                 coef_intra=0.1,
                 coef_geom=0.01,
                 coef_gene=1.0,
                 coef_kid = 1.0,
                 lr_pre= 1e-4,
                 lr_dk = 1e-4,
                 lr = 1e-4,
                 gradient_clipping =5.,
                 weight_decay = 5e-4,
                 seed= 123,
                 device='cuda:0',
                 update_interval=3,
                 tol = 5e-4,
                 alpha= 0.1,
                 warmup_steps=100,
                 mu_lr_scale=1.0,
                 min_ref_anchor_frac=0.05,
                 use_dec_finetune=False,
                 verbose=True,
                ):
        self.adata_full = adata_full
        self.batch_key = batch_key
        self.pre_cellembed_key = pre_cellembed_key
        self.cellembed_key = cellembed_key
        self.geneembed_key = geneembed_key
        self.cluster_key = cluster_key
        self.cell_gene_dist_key = cell_gene_dist_key
        data_type = list(adata_type_map.values())
        self.data_type = data_type
        self.device = torch.device(device)
        self.seed = seed
        self.lr_pre = lr_pre
        self.lr_dk = lr_dk
        self.lr = lr
        self.n_pre_training_steps = n_pre_training_steps
        self.step_interval = step_interval
        self.n_training_steps = n_training_steps
        if len(hidden_dims) < 2:
            raise ValueError("hidden_dims must contain at least two dimensions.")
        if n_pre_training_steps < 0:
            raise ValueError("n_pre_training_steps must be non-negative.")
        if n_training_steps <= 0:
            raise ValueError("n_training_steps must be positive.")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive.")
        if step_interval <= 0:
            raise ValueError("step_interval must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if mu_lr_scale <= 0:
            raise ValueError("mu_lr_scale must be positive.")
        if min_ref_anchor_frac < 0 or min_ref_anchor_frac >= 1:
            raise ValueError("min_ref_anchor_frac must be in [0, 1).")
        
        self.hvgs_shared = self.adata_full.uns['hvgs_shared']
        self.feature_dim = len(self.hvgs_shared)
        
        self.hk = self.adata_full.uns['hk']
        self.hidden_dims = hidden_dims
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.mu_lr_scale = mu_lr_scale
        self.min_ref_anchor_frac = min_ref_anchor_frac
        self.use_dec_finetune = bool(use_dec_finetune)
        self.verbose = verbose
        
        self.n_ST = sum(_is_st(t) for t in self.data_type)
        
        self.update_interval = update_interval
        self.coef_kid = coef_kid
        self.coef_recon=coef_recon
        self.coef_geom= coef_geom
        self.coef_inter=coef_inter
        self.coef_intra=coef_intra
        self.coef_gene=coef_gene
        self.genename = self.adata_full.var_names
        
        self.section_ids = list(adata_type_map.keys())
        # self.section_ids = section_ids
        self.n_slices = len(self.section_ids)
        if self.n_slices < 2:
            raise ValueError("integrate() requires at least 2 slices.")
        if len(self.data_type) != len(self.section_ids):
            raise ValueError(
                f"len(data_type)={len(self.data_type)} must equal len(section_ids)={len(self.section_ids)}"
            )
        
        self.tol = tol
        self.alpha = alpha
        
        tools_v1.seed_everything(self.seed)
        
        self.node_feats_dict = {}
        self.adj_matrix_dict = {}
        self.X_dict = {}
        self.graph_cos_dict = {}
        
        self.batch_list = []
        for i, bid in enumerate(self.section_ids):
            adata_batch = self.adata_full[self.adata_full.obs[self.batch_key] == bid].copy()
            if _is_st(self.data_type[i]):
                graph_key = f"graph_{bid}"
                graph_cos_key = f"graph_cos_{bid}"
                if graph_key not in self.adata_full.uns:
                    raise KeyError(f"{graph_key} not found in adata_full.uns")
                if graph_cos_key not in self.adata_full.uns:
                    raise KeyError(f"{graph_cos_key} not found in adata_full.uns")
                adata_batch.obsp["graph"] = self.adata_full.uns[graph_key].copy()
                adata_batch.obsp["graph_cos"] = self.adata_full.uns[graph_cos_key].copy()
            adata_batch.uns = {}
            self.batch_list.append(adata_batch)
        
        for i in range(self.n_slices):
            
            X_all = self.batch_list[i].X
            if sp.issparse(X_all):
                X_all = X_all.toarray()
            else:
                X_all = np.asarray(X_all)

            X_hvg = self.batch_list[i][:, self.hvgs_shared].X
            if sp.issparse(X_hvg):
                X_hvg = X_hvg.toarray()
            else:
                X_hvg = np.asarray(X_hvg)
            
            self.X_dict[i] = torch.from_numpy(X_all.copy()).float().to(self.device)
            self.node_feats_dict[i] = torch.from_numpy(X_hvg.copy()).float().to(self.device)
            
            
            
            # if sp.issparse(self.batch_list[i].X):
            #     self.batch_list[i].X = self.batch_list[i].X.toarray()
            #     self.X_dict[i] = torch.from_numpy(self.batch_list[i].X.copy()).float().to(self.device)
            #     self.node_feats_dict[i] = torch.from_numpy(self.batch_list[i][:,self.hvgs_shared].X).float().to(self.device)
            # else:
            #     self.X_dict[i] = torch.from_numpy(self.batch_list[i].X).float().to(self.device)
            #     self.node_feats_dict[i] = torch.from_numpy(self.batch_list[i][:,self.hvgs_shared].X).float().to(self.device)
            
            if _is_st(self.data_type[i]):
                
                graph = self.batch_list[i].obsp['graph']
                if sp.issparse(graph):
                    graph = graph.toarray()
                else:
                    graph = np.asarray(graph)
                self.adj_matrix_dict[i] = torch.from_numpy(graph).float().to(self.device)
                
                graph_cos = self.batch_list[i].obsp['graph_cos']
                if sp.issparse(graph_cos):
                    graph_cos = graph_cos.toarray()
                else:
                    graph_cos = np.asarray(graph_cos)
                self.graph_cos_dict[i] = torch.from_numpy(graph_cos).float().to(self.device)
                
            else:   
                self.adj_matrix_dict[i] = None
                self.graph_cos_dict[i] = None

    def initial_embedding(self):
        self.netmodel = {}
        self.optimizer_net = {}
        self.S_dict = {}
        
        for i in range(self.n_slices):
            self.netmodel[i] = network_v1.SingleNet(feature_dim=self.feature_dim,latent_dim=self.hidden_dims,data_type = self.data_type[i]).to(self.device)
            self.optimizer_net[i] = torch.optim.Adamax(self.netmodel[i].parameters(), weight_decay=self.weight_decay, lr=self.lr_pre)
            self.netmodel[i].train()
            
            for step in tqdm(range(self.n_pre_training_steps), desc=f"Pretrain slice {i}", disable=not self.verbose):
                Ss, X_recons = self.netmodel[i](self.node_feats_dict[i],self.adj_matrix_dict[i])
                
                diff = self.node_feats_dict[i] - X_recons
                recon_loss = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8))
                
                # recon_loss = F.mse_loss(X_recons, self.node_feats_dict[i])
                loss_total = self.coef_recon * recon_loss
                _assert_finite_tensor(f"pretrain slice {i} loss_total", loss_total)
                self.optimizer_net[i].zero_grad(set_to_none=True)
                loss_total.backward()
                _clip_grad_norm(self.netmodel[i].parameters(), self.gradient_clipping)
                self.optimizer_net[i].step()
                
            self.netmodel[i].eval()
            
            with torch.no_grad():
                Ss, X_recons = self.netmodel[i](self.node_feats_dict[i], self.adj_matrix_dict[i])
            self.S_dict[i] = Ss.detach()
            self.batch_list[i].obsm[self.pre_cellembed_key] = Ss.detach().cpu().numpy()
        
        self.adata_full.obsm[self.pre_cellembed_key] = np.zeros((self.adata_full.n_obs, self.hidden_dims[-1]), dtype=np.float32)
        for i, bid in enumerate(self.section_ids):
            mask = self.adata_full.obs[self.batch_key] == bid
            self.adata_full.obsm[self.pre_cellembed_key][mask.values, :] = self.S_dict[i].detach().cpu().numpy()
        return self.adata_full

    def clustering(self,ifmerge=True,init_res=1.5,intopk=30,n_clusters=None):
        if not hasattr(self, "S_dict"):
            raise RuntimeError("Call initial_embedding() before clustering().")
        if n_clusters is not None and len(n_clusters) != self.n_slices:
            raise ValueError("n_clusters must have the same length as section_ids.")
        
        y_pred_dict = {}
        
        if n_clusters is None:
            n_clusters = []
            for i in range(self.n_slices):
                adata=sc.AnnData(self.S_dict[i].detach().cpu().numpy())
                sc.pp.neighbors(adata, n_neighbors=15)
                sc.tl.leiden(adata, 
                             resolution=init_res,
                             random_state=self.seed)
                y_pred_dict[i] = adata.obs['leiden'].astype(int).to_numpy()
                n_clusters.append(len(np.unique(y_pred_dict[i])))
                del adata
                if self.verbose:
                    print(f'There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}')
        else:
            for i in range(self.n_slices):
                if self.pre_cellembed_key not in self.batch_list[i].obsm:
                    raise KeyError(f"{self.pre_cellembed_key} not found in batch_list[{i}].obsm")
                y_pred_dict[i] = tools_v1.search_res(adata = self.batch_list[i], 
                                                  embed_key=self.pre_cellembed_key, 
                                                  target_n=n_clusters[i], 
                                                  seed=self.seed)
                if self.verbose:
                    print(f'There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}')
        
        y_combined_dict = {}
        
        if ifmerge:
            for i in range(self.n_slices):
                self.batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()

                X_all = self.batch_list[i].X
                if sp.issparse(X_all):
                    X_all = X_all.toarray()
                else:
                    X_all = np.asarray(X_all)

                if _is_st(self.data_type[i]):
                    graph = self.batch_list[i].obsp['graph']
                    if sp.issparse(graph):
                        graph = graph.toarray()
                    else:
                        graph = np.asarray(graph)

                    s_gene_embed = tools_v1.gene_embed_weight(
                        X_all, self.batch_list[i].obsm[self.pre_cellembed_key], graph
                    )
                    del graph, X_all
                else:
                    s_gene_embed = tools_v1.gene_embed_weight(
                        X_all, self.batch_list[i].obsm[self.pre_cellembed_key]
                    )
                    del X_all

                s_gene2cell_mat = tools_v1.cell_to_gene_pdistance(
                    self.batch_list[i].obsm[self.pre_cellembed_key], s_gene_embed
                )
                self.batch_list[i].layers['dist'] = s_gene2cell_mat

                while True:
                    current_labels = self.batch_list[i].obs[self.cluster_key].astype(int).values
                    unique_labels = np.unique(current_labels)
                    n_current_clusters = len(unique_labels)

                    if n_current_clusters <= 1:
                        break


                    s_gene_list = tools_v1.select_sig_genes(self.batch_list[i], label_key=self.cluster_key, topk=intopk)
                    label_order = np.array([_int_label(label) for label in s_gene_list.keys()])

                    _, score_mat = tools_v1.self_compute_gene_pvalue(s_gene_list, self.genename.values)

                    tri_indx = np.triu_indices_from(score_mat, k=1)
                    upper_tri_values = score_mat[tri_indx]

                    if np.any(upper_tri_values <= 0.05):
                        min_idx = np.argmin(upper_tri_values)
                        cluster_idx_i = tri_indx[0][min_idx]
                        cluster_idx_j = tri_indx[1][min_idx]

                        actual_label_i = label_order[cluster_idx_i]
                        actual_label_j = label_order[cluster_idx_j]

                        mask = current_labels == actual_label_j
                        self.batch_list[i].obs.loc[mask, self.cluster_key] = actual_label_i

                        le = LabelEncoder()
                        self.batch_list[i].obs[self.cluster_key] = le.fit_transform(
                            self.batch_list[i].obs[self.cluster_key]
                        )

                    else:
                        break

                y_combined_dict[i] = self.batch_list[i].obs[self.cluster_key].copy()
                if self.verbose:
                    print(f'There are {len(np.unique(y_combined_dict[i]))} clusters for Slice - {i} after merging.')
        else:
            for i in range(self.n_slices):
                self.batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()
                y_combined_dict[i] = self.batch_list[i].obs[self.cluster_key]

        ##### 使用DEC进行fine tune ####
        mu_dict = {}
        y_combined_last = {}
        cluster_size_diagnostics = {}
        
        for i in range(self.n_slices):
            cluster_labels = np.asarray(y_combined_dict[i]).astype(int)
            label_order = np.sort(np.unique(cluster_labels))
            label_to_index = {label: idx for idx, label in enumerate(label_order)}
            y_combined_last[i] = np.array([label_to_index[label] for label in cluster_labels])
            after_merge_counts = _cluster_size_records(y_combined_last[i])
            cluster_size_diagnostics[str(i)] = {
                "section_id": self.section_ids[i],
                "data_type": str(self.data_type[i]),
                "after_merge": after_merge_counts,
            }
            if self.verbose:
                _print_cluster_size_records(
                    f"[Cluster size] Slice {i} after merge:",
                    after_merge_counts,
                )
            features = pd.DataFrame(self.S_dict[i].detach().cpu().numpy(),
                                   index=np.arange(0,self.S_dict[i].shape[0]))
            Group = pd.Series(cluster_labels,name='Group',index=features.index)
            Mergefeature = pd.concat([features, Group], axis=1)
            cluster_centers = Mergefeature.groupby("Group", sort=True).mean().to_numpy()
            mu_dict[i] = nn.Parameter(torch.tensor(cluster_centers, dtype=torch.float32, device=self.device))

        if not self.use_dec_finetune:
            if self.verbose:
                print("Skip Deep Embedded Clustering (DEC) fine-tuning.")
            for i in range(self.n_slices):
                cluster_size_diagnostics[str(i)]["after_dec"] = cluster_size_diagnostics[str(i)]["after_merge"]
                cluster_size_diagnostics[str(i)]["dec_skipped"] = True
                self.batch_list[i].obsm[self.pre_cellembed_key] = self.S_dict[i].detach().cpu().numpy()
                self.batch_list[i].obs[self.cluster_key] = y_combined_last[i].astype(str)

            self.adata_full.obsm[self.pre_cellembed_key] = np.zeros(
                (self.adata_full.n_obs, self.hidden_dims[-1]), dtype=np.float32
            )
            self.adata_full.obs[self.cluster_key] = ""

            for i, bid in enumerate(self.section_ids):
                mask = (self.adata_full.obs[self.batch_key] == bid).values
                self.adata_full.obsm[self.pre_cellembed_key][mask, :] = self.batch_list[i].obsm[self.pre_cellembed_key]
                self.adata_full.obs.loc[mask, self.cluster_key] = self.batch_list[i].obs[self.cluster_key].values.astype(str)

            self.adata_full.uns['cluster_size_diagnostics'] = cluster_size_diagnostics
            if self.verbose:
                print("Add cluster size diagnostics into adata_full.uns['cluster_size_diagnostics'].")
            return self.adata_full
        
        for i in range(self.n_slices):
            self.optimizer_net[i] = torch.optim.Adamax(
                [
                    {"params": self.netmodel[i].parameters(), "lr": self.lr_dk},
                    {"params": [mu_dict[i]], "lr": self.lr_dk * self.mu_lr_scale},
                ],
                weight_decay=self.weight_decay,
                lr=self.lr_dk
            )
            self.netmodel[i].train()
        
        if self.verbose:
            print('Apply Deep Embedded Clustering (DEC) to fine-tune the clustering results.')
        for i in range(self.n_slices):
            for step in range(self.n_training_steps):
                Ss, X_recons = self.netmodel[i](self.node_feats_dict[i],self.adj_matrix_dict[i])

                q = q_distribution(Ss, self.alpha, mu_dict[i])
                if step % self.update_interval == 0:
                    p = target_distribution(q.detach())
                
                diff = self.node_feats_dict[i].to(self.device) - X_recons
                recon_loss = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8))
                kid_loss = kld(p.detach(), q)
                loss_total = self.coef_kid * kid_loss + self.coef_recon * recon_loss
                _assert_finite_tensor(f"DEC slice {i} recon_loss", recon_loss)
                _assert_finite_tensor(f"DEC slice {i} kid_loss", kid_loss)
                _assert_finite_tensor(f"DEC slice {i} loss_total", loss_total)
                self.optimizer_net[i].zero_grad(set_to_none=True)
                loss_total.backward()
                _clip_grad_norm(
                    list(self.netmodel[i].parameters()) + [mu_dict[i]],
                    self.gradient_clipping
                )
                self.optimizer_net[i].step()
        
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_combined_last[i]).astype(np.float32) / Ss.shape[0]
                y_combined_last[i] = y_pred
                if step > 0 and step % self.update_interval == 0 and delta_label < self.tol:
                    break
        
            self.netmodel[i].eval()
            with torch.no_grad():
                Ss, X_recons = self.netmodel[i](self.node_feats_dict[i], self.adj_matrix_dict[i])
                q = q_distribution(Ss, self.alpha, mu_dict[i])

            le = LabelEncoder()
            self.S_dict[i] = Ss
            cell_embed = pd.DataFrame(Ss.detach().cpu().numpy())
            cell_embed.index = [f"slice-{i}-cell-{j}" for j in range(cell_embed.shape[0])]
            y_pred = torch.argmax(q, dim=1).data.detach().cpu().numpy()
            dec_counts = _cluster_size_records(y_pred, n_clusters=q.shape[1])
            cluster_size_diagnostics[str(i)]["after_dec"] = dec_counts
            if self.verbose:
                _print_cluster_size_records(
                    f"[Cluster size] Slice {i} after DEC:",
                    dec_counts,
                )
            new_pred = le.fit_transform(y_pred)
            cell_pred = pd.DataFrame(new_pred)
            cell_pred.index = [f"slice-{i}-cell-{j}" for j in range(cell_pred.shape[0])]
            if i == 0:
                self.batch_list[i].obsm[self.pre_cellembed_key] = cell_embed.values
                self.batch_list[i].obs[self.cluster_key] = cell_pred.values.ravel().astype(str)

            else:
                self.batch_list[i].obsm[self.pre_cellembed_key] = cell_embed.values
                self.batch_list[i].obs[self.cluster_key] = cell_pred.values.ravel().astype(str)


        self.adata_full.obsm[self.pre_cellembed_key] = np.zeros(
            (self.adata_full.n_obs, self.hidden_dims[-1]), dtype=np.float32
        )
        self.adata_full.obs[self.cluster_key] = ""

        for i, bid in enumerate(self.section_ids):
            mask = (self.adata_full.obs[self.batch_key] == bid).values
            self.adata_full.obsm[self.pre_cellembed_key][mask, :] = self.batch_list[i].obsm[self.pre_cellembed_key]
            self.adata_full.obs.loc[mask, self.cluster_key] = self.batch_list[i].obs[self.cluster_key].values.astype(str)

        self.adata_full.uns['cluster_size_diagnostics'] = cluster_size_diagnostics
        if self.verbose:
            print("Add cluster size diagnostics into adata_full.uns['cluster_size_diagnostics'].")

        # for i, bid in enumerate(self.section_ids):
        #     mask = (self.adata_full.obs[self.batch_key] == bid).values
        #     self.adata_full.obs.loc[mask, self.cluster_key] = y_combined_dict[i].values.astype(str)
            
        return self.adata_full

    def integrate(self, topk = 30, temp=0.1):
        if temp < MIN_TEMPERATURE:
            raise ValueError(f"temp must be at least {MIN_TEMPERATURE}.")
        if not hasattr(self, "netmodel"):
            raise RuntimeError("Call initial_embedding() and clustering() before integrate().")
        missing_cluster = [
            i for i in range(self.n_slices)
            if self.cluster_key not in self.batch_list[i].obs
        ]
        if missing_cluster:
            raise RuntimeError("Call clustering() before integrate().")

        tar_index = self.n_slices - 1
        
        self.net = network_v1.IndEmbedNet(feature_dim=self.feature_dim,
                    latent_dim=self.hidden_dims,
                    n_slices=self.n_slices,
                    data_type=self.data_type).to(self.device)
        
        for i in range(self.n_slices):
            state_dict = self.netmodel[i].state_dict()
            self.net.encoder_layers1[i].load_state_dict(
                {k.replace("encoder_layer1.", ""): v for k, v in state_dict.items() if k.startswith("encoder_layer1.")}
            )
            self.net.encoder_layers2[i].load_state_dict(
                {k.replace("encoder_layer2.", ""): v for k, v in state_dict.items() if k.startswith("encoder_layer2.")}
            )
        
            self.net.decoder_layers1[i].load_state_dict(
                {k.replace("decoder_layer1.", ""): v for k, v in state_dict.items() if k.startswith("decoder_layer1.")}
            )
            self.net.decoder_layers2[i].load_state_dict(
                {k.replace("decoder_layer2.", ""): v for k, v in state_dict.items() if k.startswith("decoder_layer2.")}
            )
        
            if self.verbose:
                print(f"Loaded weights for slice {i}")
            
        self.optimizer_net = torch.optim.Adamax(self.net.parameters(),weight_decay=self.weight_decay,lr=self.lr)
        self.net.train()
        early_stop_prev_state = None
        early_stop_stable_count = 0
        for step in tqdm(range(self.n_training_steps), disable=not self.verbose):
            self.optimizer_net.zero_grad(set_to_none=True)
            Ss, X_recons = self.net(self.node_feats_dict, self.adj_matrix_dict)   
            
            recon_loss = torch.tensor(0.0, device=self.device)
            geom_loss = torch.tensor(0.0, device=self.device)
            inter_loss = torch.tensor(0.0, device=self.device)
            intra_loss = torch.tensor(0.0, device=self.device)
            # gene_loss = torch.tensor(0.0, device=self.device)
            
            Ss_norm_dict = {}
            all_gene_embed_dict={}
            for i in range(self.n_slices):
                if _is_st(self.data_type[i]):
                    s_gene_embed = tools_v1.gene_embed_weight_torch(self.X_dict[i], Ss[i], self.adj_matrix_dict[i])
                else:
                    s_gene_embed = tools_v1.gene_embed_weight_torch(self.X_dict[i], Ss[i])
                all_gene_embed_dict[i] = {}
                all_gene_embed_dict[i][self.geneembed_key] = s_gene_embed
                if step % 20 == 0:
                    s_gene2cell_mat = tools_v1.cell_to_gene_pdistance_torch(Ss[i], s_gene_embed)
                    self.batch_list[i].layers['dist'] = s_gene2cell_mat.detach().cpu().numpy()
                    s_gene_list = tools_v1.select_sig_genes(self.batch_list[i],label_key=self.cluster_key,topk=topk)
                    all_gene_embed_dict[i]['gene_list'] = s_gene_list

            if step % 20 == 0:
                # =========================================================
                # initialize persistent pools at first update
                # =========================================================
                if step == 0:
                    ref_tar_dict = {i: pd.DataFrame(columns=['Ref', 'Tar', 'Score'])
                                    for i in range(tar_index)}
                    global_far_pool = []
                    
                # containers rebuilt at every update round
                inter_in_dict = {}
                inter_out_list = []
                score_mat_dict = {}
                tar_label_to_idx_dict = {}
                eligible_ref_row_idx_dict = {}
                far_candidate_sets = []   
                ref_anchor_eligibility = {}
                
                # target cluster labels
                tar_clusters = self.batch_list[tar_index].obs[self.cluster_key].astype(int).values
                
                # =========================================================
                # 1. update pair pool for each ref
                # =========================================================
                
                for i in range(tar_index):
                    # ---------- compute score matrix ----------
                    _, score_mat = tools_v1.compute_gene_pvalue(all_gene_embed_dict[i]['gene_list'],all_gene_embed_dict[tar_index]['gene_list'],self.genename.values)
                    
                    ref_label_order = [_int_label(label) for label in all_gene_embed_dict[i]['gene_list'].keys()]
                    tar_label_order = [_int_label(label) for label in all_gene_embed_dict[tar_index]['gene_list'].keys()]
                    ref_label_to_idx = {label: idx for idx, label in enumerate(ref_label_order)}
                    tar_label_to_idx = {label: idx for idx, label in enumerate(tar_label_order)}
                    ref_clusters_for_anchor = self.batch_list[i].obs[self.cluster_key].astype(int).values
                    ref_anchor_records = _cluster_size_records(ref_clusters_for_anchor)
                    eligible_ref_labels = {
                        row["cluster"]
                        for row in ref_anchor_records
                        if row["fraction"] >= self.min_ref_anchor_frac
                    }
                    eligible_ref_row_idx = [
                        idx
                        for idx, ref_label in enumerate(ref_label_order)
                        if ref_label in eligible_ref_labels
                    ]
                    eligible_ref_row_idx_dict[i] = eligible_ref_row_idx
                    ref_anchor_eligibility[str(i)] = {
                        "section_id": self.section_ids[i],
                        "min_ref_anchor_frac": float(self.min_ref_anchor_frac),
                        "eligible_clusters": sorted(int(label) for label in eligible_ref_labels),
                        "filtered_clusters": [
                            int(row["cluster"])
                            for row in ref_anchor_records
                            if row["fraction"] < self.min_ref_anchor_frac
                        ],
                        "cluster_sizes": ref_anchor_records,
                    }
                    score_mat_dict[i] = score_mat
                    tar_label_to_idx_dict[i] = tar_label_to_idx

                    # ---------- (optional) print score_mat ----------
                    # print(f"Ref {i} score_mat:")
                    # print(np.array2string(score_mat, formatter={'float_kind': lambda x: f"{x:.2f}"}))

                    # =====================================================
                    # 1a. clean old pair pool: keep only p < 0.05
                    # =====================================================
                    if len(ref_tar_dict[i]) > 0:
                        def _pair_still_valid(row):
                            ref_label = _int_label(row['Ref'])
                            tar_label = _int_label(row['Tar'])
                            if ref_label not in eligible_ref_labels:
                                return False
                            if ref_label not in ref_label_to_idx or tar_label not in tar_label_to_idx:
                                return False
                            return score_mat[ref_label_to_idx[ref_label], tar_label_to_idx[tar_label]] < 0.05

                        keep_mask = ref_tar_dict[i].apply(
                            _pair_still_valid,
                            axis=1
                        )
                        ref_tar_dict[i] = ref_tar_dict[i][keep_mask].reset_index(drop=True)

                    # =====================================================
                    # 1b. add one new pair:
                    # among pairs not already in pair_pool[i], pick the
                    # smallest p-value; if < 0.05, add it
                    # =====================================================
                    existing_pairs = set(
                        zip(
                            ref_tar_dict[i]['Ref'].astype(int).tolist(),
                            ref_tar_dict[i]['Tar'].astype(int).tolist()
                        )
                    ) if len(ref_tar_dict[i]) > 0 else set()

                    candidate_pairs = []
                    for r_idx, ref_label in enumerate(ref_label_order):
                        if ref_label not in eligible_ref_labels:
                            continue
                        for t_idx, tar_label in enumerate(tar_label_order):
                            if (ref_label, tar_label) in existing_pairs:
                                continue
                            candidate_pairs.append((ref_label, tar_label, score_mat[r_idx, t_idx]))

                    if len(candidate_pairs) > 0:
                        candidate_pairs.sort(key=lambda x: x[2])  # ascending by p-value
                        best_ref_label, best_tar_label, best_p = candidate_pairs[0]
                        if best_p < 0.05:
                            new_entry = pd.DataFrame({
                                'Ref': [int(best_ref_label)],
                                'Tar': [int(best_tar_label)],
                                'Score': [float(best_p)]
                            })
                            ref_tar_dict[i] = pd.concat([ref_tar_dict[i], new_entry], ignore_index=True)

                    # =====================================================
                    # 1c. build far candidate set for this ref:
                    # tar cluster t is candidate iff all p-values in col t > 0.05
                    # and it is not already matched in this ref pair pool
                    # =====================================================
                    if len(eligible_ref_row_idx) > 0:
                        eligible_score_mat = score_mat[eligible_ref_row_idx, :]
                        far_candidate_idx = np.where(np.min(eligible_score_mat, axis=0) > 0.05)[0]
                        far_candidates_i = set(tar_label_order[t_idx] for t_idx in far_candidate_idx)
                    else:
                        far_candidates_i = set()
                    far_candidate_sets.append(far_candidates_i)

                self.adata_full.uns['ref_anchor_eligibility'] = ref_anchor_eligibility

                # =========================================================
                # 2. update global far pool
                # =========================================================
                matched_target_clusters = set()
                for i in range(tar_index):
                    if len(ref_tar_dict[i]) > 0:
                        matched_target_clusters.update(
                            ref_tar_dict[i]['Tar'].astype(int).tolist()
                        )
                
                if len(far_candidate_sets) > 0:
                    # global_far_candidates = set.union(*far_candidate_sets)
                    # To use the stricter intersection strategy instead, replace
                    # the line above with the line below.
                    global_far_candidates = set.intersection(*far_candidate_sets)
                else:
                    global_far_candidates = set()

                # global_far_candidates -= matched_target_clusters

                # ---------- clean old far pool ----------
                # keep only those still in global far candidate set
                global_far_pool = [t for t in global_far_pool if t in global_far_candidates]

                # ---------- add one new far ----------
                # choose the one with largest mean p-value across all refs / all ref clusters
                remaining_far_candidates = [t for t in global_far_candidates if t not in global_far_pool]

                if len(remaining_far_candidates) > 0:
                    far_scores = []
                    for t in remaining_far_candidates:
                        all_vals = []
                        for i in range(tar_index):
                            t_idx = tar_label_to_idx_dict[i][t]
                            ref_rows = eligible_ref_row_idx_dict.get(i, [])
                            if len(ref_rows) == 0:
                                continue
                            all_vals.extend(score_mat_dict[i][ref_rows, t_idx].tolist())
                        if len(all_vals) == 0:
                            continue
                        # far_score_t = float(np.mean(all_vals)) #找均值最大的
                        far_score_t = float(np.min(all_vals)) #找最小值最大的
                        far_scores.append((int(t), far_score_t))

                    if len(far_scores) > 0:
                        far_scores.sort(key=lambda x: x[1], reverse=True)  # largest mean first
                        best_far_t = far_scores[0][0]
                        global_far_pool.append(best_far_t)
                
                # =========================================================
                # 3. rebuild inter_in_dict from pair pools
                # =========================================================
                for i in range(tar_index):
                    inter_in_dict[i] = []
                    ref_list = []
                    tar_list = []

                    ref_clusters = self.batch_list[i].obs[self.cluster_key].astype(int).values

                    for _, row in ref_tar_dict[i].iterrows():
                        ref_cluster = int(row['Ref'])
                        tar_cluster = int(row['Tar'])

                        ref_indices = np.where(ref_clusters == ref_cluster)[0]
                        tar_indices = np.where(tar_clusters == tar_cluster)[0]

                        ref_list.append(ref_indices.tolist())
                        tar_list.append(tar_indices.tolist())

                    inter_in_dict[i].append(ref_list)
                    inter_in_dict[i].append(tar_list)
                
                # =========================================================
                # 4. rebuild inter_out_list from global far pool
                # =========================================================
                global_far_pool = sorted(list(set(global_far_pool)))

                far_cell_indices = np.where(np.isin(tar_clusters, global_far_pool))[0]
                non_far_cell_indices = np.where(~np.isin(tar_clusters, global_far_pool))[0]

                inter_out_list.append(far_cell_indices)
                inter_out_list.append(non_far_cell_indices)
                self.adata_full.uns['global_far_pool'] = global_far_pool
                current_state = (
                    tuple(sorted(matched_target_clusters)),
                    tuple(sorted(global_far_pool)),
                )

                # if current_state == early_stop_prev_state:
                #     early_stop_stable_count += 1
                # else:
                #     early_stop_prev_state = current_state
                #     early_stop_stable_count = 1

                # if early_stop_stable_count >= 4:
                #     if self.verbose:
                #         print(
                #             f"Early stopping at step {step}: "
                #             "Matched target clusters and Global far pool stayed unchanged "
                #             "for 3 consecutive update rounds."
                #         )
                #     break
                # =========================================================
                # 5. debug prints
                # =========================================================
                if self.verbose and step % self.step_interval == 0:
                    print(f"Matched target clusters: {sorted(list(matched_target_clusters))}")
                    print(far_candidate_sets)
                    print(f"Global far candidates: {sorted(list(global_far_candidates))}")
                    print(f"Global far pool: {global_far_pool}")

                    for i in range(tar_index):
                        eligibility = ref_anchor_eligibility[str(i)]
                        print(
                            f"Ref {i} eligible anchors "
                            f"(min_frac={self.min_ref_anchor_frac}): "
                            f"{eligibility['eligible_clusters']}; "
                            f"filtered: {eligibility['filtered_clusters']}"
                        )
                        print(f"Ref {i} pair pool:")
                        print(ref_tar_dict[i])

            for i in range(self.n_slices):
                diff = self.node_feats_dict[i] - X_recons[i]
                recon_loss += torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8))
                # if i < tar_index:
                #     hk_ref = all_gene_embed_dict[i][self.geneembed_key][self.hk, :]
                #     hk_tar = all_gene_embed_dict[tar_index][self.geneembed_key][self.hk, :]
                #     gene_loss += (1 - F.cosine_similarity(hk_ref, hk_tar, dim=1)).mean()
                
                if _is_st(self.data_type[i]):
                    Ss_norm_dict[i] = F.normalize(Ss[i], p=2, dim=1, eps=1e-8)
                    graph_cos = self.graph_cos_dict[i].to(self.device)
                    mat_diff = torch.matmul(Ss_norm_dict[i], Ss_norm_dict[i].T) - graph_cos
                    geom_loss += torch.mean(torch.sum(mat_diff ** 2, dim=1))
            
            # gene_loss = gene_loss / (self.n_slices - 1)

            recon_loss = recon_loss / self.n_slices
            if self.n_ST > 0:
                geom_loss = geom_loss / self.n_ST
            else:
                geom_loss = torch.tensor(0.0, device=self.device)
            
            n_out = Ss[tar_index][inter_out_list[0], :].shape[0]
            if n_out > 0:
                k = min(256, n_out)
                idx_out = torch.randperm(n_out, device=Ss[tar_index].device)[:k]
                Ss_out_sub = Ss[tar_index][inter_out_list[0], :][idx_out,:]
                
                nozero_ref = 0
                for i in range(tar_index):
                    ref_list = inter_in_dict[i][0]
                    tar_list = inter_in_dict[i][1]
                    n_pairs = len(ref_list)
                    if n_pairs == 0:
                        continue
                    else:
                        nozero_ref += 1
                        tmp_pair_loss = 0.0
                        for j in range(len(ref_list)):
                            n_0 = Ss[i][ref_list[j], :].shape[0]
                            n_1 = Ss[tar_index][tar_list[j], :].shape[0]
                            n_2 = Ss[i].shape[0]
                            k = min(256, n_0, n_1, n_2)
                            idx_0 = torch.randperm(n_0, device=Ss[i].device)[:k]
                            idx_1 = torch.randperm(n_1, device=Ss[tar_index].device)[:k]
                            idx_2 = torch.randperm(n_2, device=Ss[i].device)[:k]
                            Ss0_sub = Ss[i][ref_list[j],:][idx_0,:]
                            Ss1_sub = Ss[tar_index][tar_list[j],:][idx_1,:]
                            Ss2_sub = Ss[i][idx_2,:]
                            mmd_pos = mmd_loss_fun(Ss0_sub, Ss1_sub, kernel_num=5)
                            mmd_neg_tar = mmd_loss_fun(Ss1_sub, Ss_out_sub, kernel_num=5)
                            mmd_neg_ref = mmd_loss_fun(Ss0_sub, Ss_out_sub, kernel_num=5)
                            mmd_neg = 0.5 * (mmd_neg_tar + mmd_neg_ref)
                            
                            margin = 2.0
                            tmp_pair_loss += torch.relu(mmd_pos - mmd_neg + margin)
                            
                            # tmp_pair_loss += mmd_pos - mmd_neg

                        inter_loss += tmp_pair_loss / n_pairs
                if nozero_ref > 0:
                    inter_loss = inter_loss / nozero_ref
                # else:
                #     inter_loss = torch.tensor(0.0, device=Ss[0].device)
            else:
                nozero_ref = 0
                for i in range(tar_index):
                    ref_list = inter_in_dict[i][0]
                    tar_list = inter_in_dict[i][1]
                    n_pairs = len(ref_list)
                    if n_pairs == 0:
                        continue
                    else:
                        nozero_ref += 1
                        tmp_pair_loss = 0.0
                        for j in range(len(ref_list)):
                            n_0 = Ss[i][ref_list[j], :].shape[0]
                            n_1 = Ss[tar_index][tar_list[j], :].shape[0]
                            k = min(256, n_0, n_1)
                            idx_0 = torch.randperm(n_0, device=Ss[i].device)[:k]
                            idx_1 = torch.randperm(n_1, device=Ss[tar_index].device)[:k]
                            Ss0_sub = Ss[i][ref_list[j],:][idx_0,:]
                            Ss1_sub = Ss[tar_index][tar_list[j],:][idx_1,:]
                            mmd_pos = mmd_loss_fun(Ss0_sub, Ss1_sub, kernel_num=5)

                            tmp_pair_loss += mmd_pos

                        inter_loss += tmp_pair_loss / n_pairs
                
                if nozero_ref > 0:
                    inter_loss = inter_loss / nozero_ref
                # else:
                #     inter_loss = torch.tensor(0.0, device=Ss[0].device)
            for i in range(self.n_slices):
                labels = self.batch_list[i].obs[self.cluster_key].astype(int).values
                labels = torch.from_numpy(labels).to(self.device)
                intra_loss += spv_contrastive_loss(Ss[i], labels, temperature=temp)
            
            intra_loss = intra_loss / self.n_slices
            # this_coef_gene = self.coef_gene
            # if step <= self.warmup_steps:
            #     this_coef_gene = self.coef_gene
            # else:
            #     this_coef_gene = self.coef_gene * 0.001
            
            # loss_total = self.coef_inter * inter_loss + this_coef_gene * gene_loss + self.coef_recon * recon_loss + self.coef_intra * intra_loss +self.coef_geom * geom_loss
            loss_total = self.coef_inter * inter_loss + self.coef_intra * intra_loss + self.coef_recon * recon_loss + self.coef_geom * geom_loss     
            for loss_name, loss_value in {
                "recon_loss": recon_loss,
                "inter_loss": inter_loss,
                "intra_loss": intra_loss,
                "geom_loss": geom_loss,
                "loss_total": loss_total,
            }.items():
                _assert_finite_tensor(f"integrate step {step} {loss_name}", loss_value)
            loss_total.backward()
            _clip_grad_norm(self.net.parameters(), self.gradient_clipping)
            self.optimizer_net.step()
            if self.verbose and step % self.step_interval == 0:
                print(f"[Train] Step: {step},recon_loss: {recon_loss.item():.4f}, inter_loss: {inter_loss.item():.4f},intra_loss: {intra_loss.item():.4f},geom_loss: {geom_loss.item():.4f}")

        
        self.net.eval()
        with torch.no_grad():
            Ss, X_recons = self.net(self.node_feats_dict, self.adj_matrix_dict)
        
        for i in range(self.n_slices):
            S = Ss[i].detach().cpu().numpy()
            cell_reps = pd.DataFrame(S)
            cell_reps.index = [f"slice-{i}-cell-{j}" for j in range(cell_reps.shape[0])]
            if i == 0:
                self.batch_list[i].obsm[self.cellembed_key] = cell_reps.values
            else:
                self.batch_list[i].obsm[self.cellembed_key] = cell_reps.values
        
        self.adata_full.obsm[self.cellembed_key] = np.zeros(
            (self.adata_full.n_obs, self.hidden_dims[-1]), dtype=np.float32
        )

        for i, bid in enumerate(self.section_ids):
            mask = (self.adata_full.obs[self.batch_key] == bid).values
            self.adata_full.obsm[self.cellembed_key][mask, :] = self.batch_list[i].obsm[self.cellembed_key]
        
        self.adata_full.uns['pair_dict'] = {}
        for k, v in ref_tar_dict.items():
            self.adata_full.uns['pair_dict'][str(k)] = v 
        if self.verbose:
            print(f"Add cell/spot embeddings into adata_full.obsm[{self.cellembed_key}]...")
        

        dist_list = []
        for i in range(self.n_slices):
            # adata_batch = self.adata_full[self.adata_full.obs[self.batch_key]==self.section_ids[i],].copy()
            
            adata_batch = self.batch_list[i].copy()
            
            if sp.issparse(adata_batch.X):
                adata_batch.X = adata_batch.X.toarray()
            else:
                adata_batch.X = adata_batch.X
            if self.verbose:
                print(f'Add gene embedding in adata_full.uns.....')
            
            X_all = adata_batch.X
            if sp.issparse(X_all):
                X_all = X_all.toarray()
            else:
                X_all = np.asarray(X_all)
            
            if _is_st(self.data_type[i]):
                adata_batch.obsp['graph'] = self.batch_list[i].obsp['graph']
                
                graph = adata_batch.obsp['graph']
                if sp.issparse(graph):
                    graph = graph.toarray()
                else:
                    graph = np.asarray(graph)

                s_gene_embed = tools_v1.gene_embed_weight(X_all,adata_batch.obsm[self.cellembed_key],graph)
                self.adata_full.uns[f'Slice_{i}_{self.geneembed_key}'] = s_gene_embed
                del X_all, graph
            else:
                s_gene_embed = tools_v1.gene_embed_weight(X_all,adata_batch.obsm[self.cellembed_key])
                self.adata_full.uns[f'Slice_{i}_{self.geneembed_key}'] = s_gene_embed
                del X_all
            dist_list.append(tools_v1.cell_to_gene_pdistance(adata_batch.obsm[self.cellembed_key], s_gene_embed))

        if self.verbose:
            print(f"Add gene cell_gene distance in adata_full.layers[{self.cell_gene_dist_key}]......")
        full_dist = np.zeros((self.adata_full.n_obs, self.adata_full.n_vars), dtype=np.float32)
        for i, bid in enumerate(self.section_ids):
            mask = (self.adata_full.obs[self.batch_key] == bid).values
            full_dist[mask, :] = dist_list[i]
        self.adata_full.layers[self.cell_gene_dist_key] = full_dist
    
        return self.adata_full
