import random
import os
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy import stats
from functools import reduce
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import liana
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import ranksums
from joblib import Parallel, delayed
from tqdm import tqdm
import umap

def seed_everything(seed: int = 123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _label_sort_key(label):
    try:
        return (0, int(label))
    except (TypeError, ValueError):
        return (1, str(label))
    
def create_dir(dir, exist_ok=True):
    os.makedirs(dir, exist_ok=exist_ok)
    print(f'{dir} has been created or already exists.')

def preprocess(adata_list,
               adata_type_map,
               full_num_hvgs=3000,
               min_genes_qc=10,
               min_cells_qc=10,
               ifco_expressed_genes=False
              ):
    adata_type = list(adata_type_map.values())
    assert len(adata_type) == len(adata_list)
    
    # 对每个adata进行质控，去细胞和去基因
    ads = []
    for i, adata_st in enumerate(adata_list):
        if not sp.issparse(adata_st.X):
            adata_st.X = sp.csr_matrix(adata_st.X)
        adata_st.var_names_make_unique()
        adata_st = adata_st[:, np.array(~adata_st.var.index.isna())
                    & np.array(~adata_st.var_names.str.startswith("mt-"))
                    & np.array(~adata_st.var_names.str.startswith("MT-"))]
        print(f"shape of adata {i} before quality control: {adata_st.shape}")
        sc.pp.filter_cells(adata_st, min_genes=min_genes_qc)
        sc.pp.filter_genes(adata_st, min_cells=min_cells_qc)
        print(f"shape of adata {i} after quality control: {adata_st.shape}")
        adata_st.obs.index = adata_st.obs.index + "-" + str(i)
        adata_list[i] = adata_st
        ads.append(adata_st)
    
    # 保存tar_adata（最后一个数据集的原始副本，包含所有基因）
    adata_tar = adata_list[-1].copy()
    
    # 保留共有基因，concat
    adata_full = ad.concat(ads, join="inner")
    del ads
    shared_gene = adata_full.var_names
    print(f"Find {len(shared_gene)} shared genes among datasets.")
    
    # 过滤共表达基因（可选），直接在 adata_full 上按 batch 分组
    if ifco_expressed_genes:
        expr_ratio_per_batch = []
        for bid in adata_full.obs["batch"].unique():
            sub = adata_full[adata_full.obs["batch"] == bid]
            expr_ratio = np.array((sub.X > 0).mean(axis=0)).ravel()
            expr_ratio_per_batch.append(set(np.where(expr_ratio > 0.1)[0]))

        shared_filtered_indices = set.intersection(*expr_ratio_per_batch)
        mask = np.zeros(len(shared_gene), dtype=bool)
        mask[list(shared_filtered_indices)] = True
        coexp_gene = shared_gene[mask]

        if len(coexp_gene) == 0:
            print("No shared genes after co-expression filtering. Use shared genes.")
            coexp_gene = shared_gene
        else:
            print(f"Find {len(coexp_gene)} shared co-expressed genes among {len(adata_list)} datasets.")
    else:
        coexp_gene = shared_gene
    
    # 将 adata_full 统一裁剪到 coexp_gene 或者 shared_gene
    adata_full = adata_full[:, coexp_gene].copy()
    
    target_sum = 1e3 if len(coexp_gene) < 1e3 else 1e4
    
    # 标准化 adata_full（只做一次）
    print("Normalize data...")
    adata_full.layers["counts"] = adata_full.X.copy()
    sc.pp.normalize_total(adata_full, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata_full)
    
    # 不对ATAC进行同样操作的标准化
    # normed_ads = []
    # batch_ids = sorted(adata_full.obs.index.str.extract(r'-(\d+)$')[0].unique(), key=int)
    # for i, bid in enumerate(batch_ids):
    #     mask = adata_full.obs.index.str.endswith(f"-{bid}")
    #     adata_batch = adata_full[mask].copy()
    #     if adata_type[i] != 'scATAC':
    #         sc.pp.normalize_total(adata_batch, target_sum=target_sum, inplace=True)
    #         sc.pp.log1p(adata_batch)
    #     normed_ads.append(adata_batch)
    # adata_full = ad.concat(normed_ads)
    # del normed_ads
    
    # 计算HVG（使用batch_key避免batch effect影响）
    sc.pp.highly_variable_genes(adata_full, n_top_genes=full_num_hvgs, batch_key="batch")
    hvgs_shared = sorted(adata_full.var_names[adata_full.var.highly_variable].tolist())
    # hvgs_shared = sorted(adata_full.var_names[adata_full.var.highly_variable_intersection].tolist())
    adata_full.uns['hvgs_shared'] = hvgs_shared
    print(f"Find {len(hvgs_shared)} shared highly variable genes among datasets.")
    
    # hvg_dict = {}
    # for i, b in enumerate(adata_full.obs["batch"].unique()):
    #     adata_b = adata_full[adata_full.obs["batch"] == b].copy()
    #     sc.pp.highly_variable_genes(
    #         adata_b,
    #         n_top_genes=full_num_hvgs
    #     )
    #     hvg_dict[i] = sorted(
    #         adata_b.var_names[adata_b.var["highly_variable"]].tolist()
    #     )
    # adata_full.uns["hvg_dict"] = hvg_dict
    
    # 处理 tar_adata：裁剪到 coexp_gene 后 normalize（只做一次）
    tar_target_sum = 1e3 if adata_tar.shape[1] < 1e3 else 1e4
    adata_tar.layers["counts"] = adata_tar.X.copy()
    sc.pp.normalize_total(adata_tar, target_sum=tar_target_sum, inplace=True)
    sc.pp.log1p(adata_tar)
    

#     hvgs_shared = None
#     for i in range(len(adata_list)):
#         adata_list[i] = adata_list[i][:, coexp_gene].copy()

#         if adata_type[i] != 'scATAC':
#             sc.pp.highly_variable_genes(
#                 adata_list[i],
#                 flavor='seurat_v3',
#                 n_top_genes=full_num_hvgs
#             )

#             hvgs = adata_list[i].var_names[
#                 adata_list[i].var.highly_variable
#             ]

#             if hvgs_shared is None:
#                 hvgs_shared = hvgs
#             else:
#                 hvgs_shared = hvgs_shared.intersection(hvgs)

#     if hvgs_shared is None:
#         raise ValueError("No scRNA or ST batch found for HVG calculation.")

#     hvgs_shared = sorted(hvgs_shared)
    
#     adata_full.uns['hvgs_shared'] = hvgs_shared
#     print("Find", str(len(hvgs_shared)), "shared highly variable genes among datasets.")
    
#     print("Normalize data...")
#     if adata_list[0].shape[1] < 1e3:
#         target_sum = 1e3
#     else:
#         target_sum = 1e4
        
#     for i in range(len(adata_list)):
#         if adata_type[i] == 'scATAC':
#             continue
#         else:
#             sc.pp.normalize_total(adata_list[i], target_sum=target_sum, inplace=True)
#             sc.pp.log1p(adata_list[i])
    
#     if tar_adata.shape[1] < 1e3:
#         target_sum = 1e3
#     else:
#         target_sum = 1e4
#     sc.pp.normalize_total(tar_adata, target_sum=target_sum, inplace=True)
#     sc.pp.log1p(tar_adata)
    
    return adata_full, adata_tar

def build_graph_GAT_plus(adata_full,
                    adata_type_map: dict,
                    K=8,
                    img_threshold=0.0):
    print("Start building graphs...")

    batch_ids = adata_full.obs["batch"].unique()

    for bid in batch_ids:
        current_type = adata_type_map[bid]
        mask = adata_full.obs["batch"] == bid
        adata_st = adata_full[mask]
        n_cells = adata_st.shape[0]
        

        if current_type in ['sc', 'scATAC']:
            adata_full.uns[f"graph_{bid}"] = sp.eye(n_cells, dtype=int, format="csr")
            adata_full.uns[f"graph_cos_{bid}"] = sp.eye(n_cells, dtype=float, format="csr")
            print(f"Skipping spatial graph for batch {bid} ({current_type})")
            continue

        print(f"Building sparse graph for batch {bid} ({current_type}) using NearestNeighbors...")

        if 'spatial' not in adata_st.obsm:
            raise KeyError(f"Batch {bid} is missing .obsm['spatial'].")

        coords = adata_st.obsm['spatial']
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', metric='euclidean').fit(coords)
        _, knn_indices = nbrs.kneighbors(coords)  # shape: (n_cells, K+1)，含自身

        row_indices_list = []
        col_indices_list = []

        if current_type == 'ST':
            row_indices_list = np.repeat(np.arange(n_cells), K+1)
            col_indices_list = knn_indices.flatten()

        elif current_type == 'ST_with_HE':
            if 'image_embedding' not in adata_st.obsm:
                raise ValueError(f"Batch {bid} is 'ST_with_HE' but .obsm['image_embedding'] is missing.")

            img_emb = adata_st.obsm['image_embedding']
            norm_img_emb = normalize(img_emb, axis=1)
            cosim_values = []

            for row_idx, neighbors in enumerate(knn_indices):
                curr_vec = norm_img_emb[row_idx]
                neighbor_vecs = norm_img_emb[neighbors]
                sims = np.dot(neighbor_vecs, curr_vec)
                cosim_values.extend(sims)
                valid_mask = sims >= img_threshold
                valid_neighbors = neighbors[valid_mask]

                if len(valid_neighbors) > 0:
                    row_indices_list.extend([row_idx] * len(valid_neighbors))
                    col_indices_list.extend(valid_neighbors)

            if len(cosim_values) > 0:
                quantiles = np.percentile(cosim_values, [0, 25, 50, 75, 100])
                print(f"  Image similarity quantiles (neighbors only): {quantiles}")

        data = np.ones(len(row_indices_list), dtype=int)
        
        G_sparse = sp.csr_matrix(
            (data, (row_indices_list, col_indices_list)),
            shape=(n_cells, n_cells)
        )
        
        adata_full.uns[f"graph_{bid}"] = G_sparse

        avg_neighbors = G_sparse.sum(axis=1).mean()
        print(f"  Average neighbors for Slice {bid}: {avg_neighbors-1:.2f}")

        try:
            pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine")
            adata_full.uns[f"graph_cos_{bid}"] = 1 - pair_dist_cos
        except MemoryError:
            print(f"  Warning: MemoryError when computing cosine matrix for batch {bid}. Using identity.")
            adata_full.uns[f"graph_cos_{bid}"] = np.eye(n_cells, dtype=float)

    return adata_full


def build_graph_GAT(adata_full,
                    adata_type=['ST', 'ST'],
                    K=8,
                    img_threshold=0.0):

    print("Start building graphs...")
    
    for i, adata_st in enumerate(adata_list):
        current_type = adata_type[i]
        n_cells = adata_st.shape[0]
        
        if current_type in ['sc', 'scATAC']:
            adata_st.obsm["graph"] = sp.csr_matrix((n_cells, n_cells), dtype=int)
            print(f"Skipping spatial graph for batch {i} ({current_type})")
            continue

        print(f"Building sparse graph for batch {i} ({current_type}) using NearestNeighbors...")
        
        if 'spatial' not in adata_st.obsm:
             raise KeyError(f"Batch {i} is missing .obsm['spatial'].")
             
        coords = adata_st.obsm['spatial']
        
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', metric='euclidean').fit(coords)
        _, knn_indices = nbrs.kneighbors(coords)
        
        
        row_indices_list = []
        col_indices_list = []
        
        if current_type == 'ST':
            row_indices_list = np.repeat(np.arange(n_cells), K+1)
            col_indices_list = knn_indices.flatten()
                
        elif current_type == 'ST_with_HE':
            if 'image_embedding' not in adata_st.obsm:
                raise ValueError(f"Batch {i} is 'ST_with_HE' but .obsm['image_embedding'] is missing.")
            
            img_emb = adata_st.obsm['image_embedding']
            norm_img_emb = normalize(img_emb, axis=1)
            
            cosim_values = []
            
            for row_idx, neighbors in enumerate(knn_indices):
                curr_vec = norm_img_emb[row_idx]
                neighbor_vecs = norm_img_emb[neighbors]
                sims = np.dot(neighbor_vecs, curr_vec)
                cosim_values.extend(sims)
                valid_mask = sims >= img_threshold
                valid_neighbors = neighbors[valid_mask]
                
                if len(valid_neighbors) > 0:
                    row_indices_list.extend([row_idx] * len(valid_neighbors))
                    col_indices_list.extend(valid_neighbors)
                    
            if len(cosim_values) > 0:
                quantiles = np.percentile(cosim_values, [0, 25, 50, 75, 100])
                print(f"  Image similarity quantiles (neighbors only): {quantiles}")

        data = np.ones(len(row_indices_list), dtype=int)
        
        G_sparse = sp.csr_matrix(
            (data, (row_indices_list, col_indices_list)), 
            shape=(n_cells, n_cells)
        )
        
        adata_st.obsm["graph"] = G_sparse
        
        avg_neighbors = G_sparse.sum(axis=1).mean()
        print(f"  Average neighbors for Slice {i}: {avg_neighbors-1:.2f}")
        
        try:
            pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine")
            adata_st.obsm["graph_cos"] = 1 - pair_dist_cos
        except MemoryError:
            print("  Warning: MemoryError when computing full gene expression cosine matrix. Skipping graph_cos.")

    return adata_list


# def build_graph_GAT(adata_list,
#                     adata_type=['ST', 'ST'],
#                     K=10,
#                     img_threshold=0.0):

#     print("Start building graphs...")
    
#     for i, adata_st in enumerate(adata_list):
#         current_type = adata_type[i]
#         n_cells = adata_st.shape[0]
        
#         if current_type in ['sc', 'scATAC']:
#             adata_st.obsm["graph"] = sp.csr_matrix((n_cells, n_cells), dtype=int)
#             print(f"Skipping spatial graph for batch {i} ({current_type})")
#             continue

#         print(f"Building sparse graph for batch {i} ({current_type}) using NearestNeighbors...")
        
#         if 'spatial' not in adata_st.obsm:
#              raise KeyError(f"Batch {i} is missing .obsm['spatial'].")
             
#         coords = adata_st.obsm['spatial']
        
#         nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', metric='euclidean').fit(coords)
#         _, knn_indices = nbrs.kneighbors(coords)
        
        
#         row_indices_list = []
#         col_indices_list = []
        
#         if current_type == 'ST':
#             row_indices_list = np.repeat(np.arange(n_cells), K+1)
#             col_indices_list = knn_indices.flatten()
                
#         elif current_type == 'ST_with_HE':
#             if 'image_embedding' not in adata_st.obsm:
#                 raise ValueError(f"Batch {i} is 'ST_with_HE' but .obsm['image_embedding'] is missing.")
            
#             img_emb = adata_st.obsm['image_embedding']
#             norm_img_emb = normalize(img_emb, axis=1)
            
#             cosim_values = []
            
#             for row_idx, neighbors in enumerate(knn_indices):
#                 curr_vec = norm_img_emb[row_idx]
#                 neighbor_vecs = norm_img_emb[neighbors]
#                 sims = np.dot(neighbor_vecs, curr_vec)
#                 cosim_values.extend(sims)
#                 valid_mask = sims >= img_threshold
#                 valid_neighbors = neighbors[valid_mask]
                
#                 if len(valid_neighbors) > 0:
#                     row_indices_list.extend([row_idx] * len(valid_neighbors))
#                     col_indices_list.extend(valid_neighbors)
                    
#             if len(cosim_values) > 0:
#                 quantiles = np.percentile(cosim_values, [0, 25, 50, 75, 100])
#                 print(f"  Image similarity quantiles (neighbors only): {quantiles}")

#         data = np.ones(len(row_indices_list), dtype=int)
        
#         G_sparse = sp.csr_matrix(
#             (data, (row_indices_list, col_indices_list)), 
#             shape=(n_cells, n_cells)
#         )
        
#         adata_st.obsm["graph"] = G_sparse
        
#         avg_neighbors = G_sparse.sum(axis=1).mean()
#         print(f"  Average neighbors for Slice {i}: {avg_neighbors-1:.2f}")
        
#         try:
#             pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine")
#             adata_st.obsm["graph_cos"] = 1 - pair_dist_cos
#         except MemoryError:
#             print("  Warning: MemoryError when computing full gene expression cosine matrix. Skipping graph_cos.")

#     return adata_list

def search_res(adata, 
               embed_key, 
               target_n=2, 
               n_neighbors=15, 
               start_res=0.1, 
               step=0.05, 
               max_res=5.0, 
               seed=123):
    
    data = sc.AnnData(adata.obsm[embed_key])
    sc.pp.neighbors(data, n_neighbors=n_neighbors)
    
    res = start_res

    while res <= max_res:
        sc.tl.leiden(data, resolution=res, random_state=seed)
        y_pred = data.obs['leiden'].to_numpy()
        n_clusters = len(np.unique(y_pred))
        
        if n_clusters == target_n:
            print(f"Found resolution={res:.3f} with {n_clusters} clusters")
            return y_pred
        elif n_clusters > target_n:
            break
        res += step
    
    print("Did not find exact match, use GaussianMixture fallback")
    gm = GaussianMixture(
        n_components=target_n,
        covariance_type='tied',
        init_params='kmeans',
        random_state=seed,
        n_init=5
    )
    y_pred = gm.fit_predict(adata.obsm[embed_key])
    return y_pred.astype(str)
    
#     gm = GaussianMixture(n_components=target_n,
#                      covariance_type='tied',
#                      init_params='kmeans',
#                      random_state=seed,
#                      n_init=5)
#     y_pred = gm.fit_predict(adata.obsm[embed_key])
#     print(f"Did not find exact match, use gaussian clustering")
#     return y_pred.astype(str)

def gene_embed_weight(X, ce_cell, adj=None, c=1.0):
    X = X.T
    if adj is None:
        sumW = np.sum(X, axis=1, keepdims=True) + 1e-8
        weight = X / sumW
        return weight @ ce_cell
        # weight = calculate_tfidf_weights(X) 
        # return np.dot(weight, ce_cell)
    else:
        indicatorX = (X != 0).astype(float)
        n_express = adj.dot(indicatorX.T) + c
        n_neighbor = np.array(adj.sum(axis=1)).flatten() + c
        n_express = n_express / n_neighbor[:, None]
        weight = X * n_express.T
        sumW = np.sum(weight, axis=1, keepdims=True) + 1e-10
        weight = weight / sumW
        return weight @ ce_cell

def gene_embed_weight_torch(X, ce_cell, adj=None, c=1.0):
    X = X.T  # genes × cells

    if adj is None:
        sumW = torch.sum(X, dim=1, keepdim=True) + 1e-8
        weight = X / sumW
        return torch.matmul(weight, ce_cell)
    else:
        indicatorX = (X != 0).float()
        n_express = torch.matmul(adj, indicatorX.T) + c
        n_neighbor = torch.sum(adj, dim=1, keepdim=True) + c
        n_express = n_express / n_neighbor
        weight = X * n_express.T
        sumW = torch.sum(weight, dim=1, keepdim=True) + 1e-10
        weight = weight / sumW
        return torch.matmul(weight, ce_cell)
    
def cell_to_gene_pdistance(cell_embed, gene_embed, eta=1e-10):
    # cell_embed = normalize(cell_embed, norm='l2', axis=1)
    # gene_embed = normalize(gene_embed, norm='l2', axis=1)

    An = np.sum(cell_embed ** 2, axis=1, keepdims=True)
    Bn = np.sum(gene_embed ** 2, axis=1, keepdims=True)
    C = -2 * np.dot(cell_embed, gene_embed.T)
    C += An
    C += Bn.T
    return np.sqrt(np.maximum(C, 0.0) + eta)

def cell_to_gene_pdistance_torch(cell_embed, gene_embed, eta=1e-10):
    An = torch.sum(cell_embed ** 2, dim=1, keepdim=True)
    Bn = torch.sum(gene_embed ** 2, dim=1, keepdim=True)
    
    C = -2 * torch.matmul(cell_embed, gene_embed.T)
    C = C + An
    C = C + Bn.T
    
    return torch.sqrt(torch.clamp(C, min=0.0) + eta)

def select_sig_genes(
    adata, 
    dist_key="dist", 
    label_key=None, 
    topk=100, 
    genes_use=None,
    expr_prop_cutoff=0.1, 
    ntop_max=200, 
    overlap_max=1
    # rm_mito_ribo=False, 
    # species="ms"
):
    
    cell_label_vec = adata.obs[label_key].astype(str)
    cell_IDs = sorted(cell_label_vec.unique(), key=_label_sort_key)

    if genes_use is None:
        genes_use = adata.var_names.to_numpy()
    else:
        genes_use = np.array(genes_use)

    expr_data = adata[:, genes_use].X
    distce_data = adata[:, genes_use].layers[dist_key]

    if sp.issparse(expr_data):
        expr_data = expr_data.toarray()
    if sp.issparse(distce_data):
        distce_data = distce_data.toarray()

    n_cells = adata.n_obs
    expr_all = (expr_data > 0).sum(axis=0)

    ref_sig_list = {}

    for label in cell_IDs:
        idx = np.where(cell_label_vec == label)[0]
        n_idx = len(idx)

        expr_prop = (expr_data[idx, :] > 0).mean(axis=0)
        distce_vals = distce_data[idx, :].mean(axis=0)

        mask = expr_prop > expr_prop_cutoff
        filtered_indices = np.where(mask)[0]

        if filtered_indices.size == 0:
            ref_sig_list[label] = {'genes': [], 'gene_index': []}
            continue

        sorted_idx = np.argsort(distce_vals[filtered_indices])[:topk]
        final_idx = filtered_indices[sorted_idx]

        gene_names = genes_use[final_idx]

        # if rm_mito_ribo:
        #     mito_pattern = r"^(mt-|Mt-)" if species == "ms" else r"^MT-"
        #     ribo_pattern = r"^(Rps|Rpl)" if species == "ms" else r"^(RPS|RPL)"
        #     is_mito = np.char.startswith(gene_names, tuple(["mt-", "Mt-", "MT-"]))
        #     is_ribo = np.char.startswith(gene_names, tuple(["Rps", "Rpl", "RPS", "RPL"]))
        #     keep_mask = ~(is_mito | is_ribo)
        #     gene_names = gene_names[keep_mask]
        #     final_idx = final_idx[keep_mask]

        ref_sig_list[label] = {
            'genes': gene_names.tolist(),
            'gene_index': final_idx.tolist()
        }

    return ref_sig_list

def self_compute_gene_pvalue(A_dict, background_genes):
    B_dict = A_dict
    A_classes = list(A_dict.keys())
    B_classes = A_classes

    N = len(background_genes)
    num_classes = len(A_classes)
    p_values = np.zeros((num_classes, num_classes))

    raw_p_list = []
    index_list = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes): 
            A_class = A_classes[i]
            B_class = B_classes[j]

            A_genes = set(A_dict[A_class]['genes'])
            B_genes = set(B_dict[B_class]['genes'])
            K = len(A_genes)
            M = len(B_genes)
            x = len(A_genes & B_genes)

            p_value = stats.hypergeom.sf(x - 1, N, K, M)
            p_values[i, j] = p_value
            raw_p_list.append(p_value)
            index_list.append((i, j))

    fdr_corrected = multipletests(raw_p_list, method='fdr_bh')[1]
    fdr_matrix = np.zeros((num_classes, num_classes))
    for val, (i, j) in zip(fdr_corrected, index_list):
        fdr_matrix[i, j] = val

    p_values = p_values + p_values.T
    fdr_matrix = fdr_matrix + fdr_matrix.T
    return p_values, fdr_matrix

def compute_gene_pvalue(A_dict, B_dict, background_genes):
    A_classes = list(A_dict.keys())
    B_classes = list(B_dict.keys())
    N = len(background_genes)
    num_A = len(A_classes)
    num_B = len(B_classes)
    p_values = np.zeros((num_A, num_B))

    for i, A_class in enumerate(A_classes):
        for j, B_class in enumerate(B_classes):
            A_genes = set(A_dict[A_class]['genes'])
            B_genes = set(B_dict[B_class]['genes'])
            K = len(A_genes)
            M = len(B_genes)
            x = len(A_genes & B_genes)

            p_value = stats.hypergeom.sf(x - 1, N, K, M)
            p_values[i, j] = p_value

    p_values_flat = p_values.flatten()
    fdr_corrected = multipletests(p_values_flat, method='fdr_bh')[1]
    fdr_matrix = fdr_corrected.reshape(num_A, num_B)

    return p_values, fdr_matrix

# def calculate_tfidf_weights(X):
#     sum_per_gene = np.sum(X, axis=1, keepdims=True) + 1e-8
#     tf = X / sum_per_gene
#     n_cells = X.shape[1]
#     n_cells_expressing_gene = np.sum(X > 0, axis=1) + 1
#     idf = np.log(n_cells / n_cells_expressing_gene)
#     tfidf_weights = tf * idf[:, np.newaxis]
#     sum_tfidf = np.sum(tfidf_weights, axis=1, keepdims=True) + 1e-8
#     final_weights = tfidf_weights / sum_tfidf
#     return final_weights

def detect_core(adata,
               embed,
               section_ids,
               label_core = 'Disease Core',
               label_other = 'Surrounding Area',
               core_types = ['Tumor'],
               celltype_key='cell_type',
               batch_key = 'batch',
               seed=123,
               neighbors = 30,
               ifumap=True,
               threshold = 0.05,
               strategy = 'cluster'):
    
    cond_section = section_ids[-1]
    adata_all = adata.copy()
    adata_all.obs['pred_label'] = label_other
    adata_all.obs['truth_label'] = label_other
    adata_all.uns['result'] = {}
    normal_mask = adata_all.obs[batch_key].isin(section_ids[:-1])
    cond_mask = adata_all.obs[batch_key] == section_ids[-1]
    adata_normal = adata_all[normal_mask]
    adata_cond = adata_all[cond_mask]
    
    X_all = adata_all.obsm[embed]
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1).fit(X_all)
    _, indices_all = nbrs.kneighbors(X_all)
    is_normal_all = np.array(adata_all.obs[batch_key].isin(section_ids[:-1]))
    
    if strategy == 'cluster':
        adata_cond.obs['binary'] = search_res(adata_cond,embed,target_n=2,seed=seed).astype(str)
        
        # X_all = adata_all.obsm[embed]
        # nbrs = NearestNeighbors(n_neighbors=neighbors+1).fit(X_all)
        # _, indices_all = nbrs.kneighbors(X_all)
        # is_normal_all = np.array(adata_all.obs[batch_key].isin(section_ids[:-1]))

        binary_labels = ['0', '1']

        ratios = {}
        for binary_label in binary_labels:
            adata_cluster = adata_cond[adata_cond.obs['binary'] == binary_label]
            cluster_indices = np.where(adata_all.obs_names.isin(adata_cluster.obs_names))[0]
            hit_flags = []
            
            # neighbor_indices = indices_all[cluster_indices, 1:]
            # neighbor_is_normal = is_normal_all[neighbor_indices]
            # has_normal_per_cell = np.any(neighbor_is_normal, axis=1)
            # ratios[binary_label] = np.mean(has_normal_per_cell)
            
            for idx in cluster_indices:
                neighbor_indices = indices_all[idx][1:]
                has_normal = np.any(is_normal_all[neighbor_indices])
                hit_flags.append(int(has_normal))
            ratios[binary_label] = np.mean(hit_flags)
            if ratios[binary_label] <= threshold:
                cond_binary_cells = adata_cluster.obs_names
                adata_all.obs.loc[cond_binary_cells, 'pred_label'] = label_core
        print(ratios)
        print(f"Add predicted label into adata_full.obsm[pred_label]...")
        all_above = all(v > threshold for v in ratios.values())
    else:
        all_above = True
    
    if all_above or strategy == 'individual':
        # X_all = adata_all.obsm[embed]
        # nbrs = NearestNeighbors(n_neighbors=neighbors + 1).fit(X_all)
        # _, indices_all = nbrs.kneighbors(X_all)

        cond_indices = np.where(adata_all.obs[batch_key] == section_ids[-1])[0]
        pad_flags = []

        for idx in cond_indices:
            neighbor_indices = indices_all[idx][1:]
            has_normal = np.any(normal_mask[neighbor_indices])
            pad_flags.append(int(not has_normal))

        pad_cells = cond_indices[np.array(pad_flags) == 1]
        pad_cell_names = adata_all.obs_names[pad_cells].tolist()
        adata_all.obs.loc[pad_cell_names, 'pred_label'] = label_core
        print(f"Add predicted label into adata_full.obsm[pred_label]...")
    
    if celltype_key is not None:
        adata_all.obs.loc[adata_all.obs[celltype_key].isin(core_types), 'truth_label'] = label_core
        adata_batch = adata_all[adata_all.obs[batch_key] == cond_section].copy()
        truth = adata_batch.obs['truth_label'].map({label_other: 0, label_core: 1}).values
        pred = adata_batch.obs['pred_label'].map({label_other: 0, label_core: 1}).values
        f1 = f1_score(truth, pred, pos_label=1)
        acc = accuracy_score(truth, pred)
        bacc = balanced_accuracy_score(truth, pred)

        TP = np.sum((truth == 1) & (pred == 1))
        TN = np.sum((truth == 0) & (pred == 0))
        FP = np.sum((truth == 0) & (pred == 1))
        FN = np.sum((truth == 1) & (pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        DTS = np.mean([f1, bacc])
        metrics_df = pd.DataFrame({'DTS': [DTS],
                                   'F1': [f1],
                                   'Balanced Accuracy': [bacc],
                                   'Accuracy': [acc],
                                   'Precision': [precision],
                                   'Recall': [recall],
                                   'specificity': [specificity]})
        adata_all.uns['result'] = metrics_df
        
        if ifumap:
            labels = adata_batch.obs['pred_label'].astype('category')
            sc.pp.neighbors(adata_batch,use_rep=embed)
            sc.tl.umap(adata_batch,key_added=f'{embed}_umap',random_state=seed)
            if len(labels.cat.categories) == 1:
                single_color = ['#1f77b4']
                adata_batch.uns['pred_label_colors'] = single_color
                sc.pl.embedding(
                    adata_batch,
                    basis=f"{embed}_umap",
                    color=['pred_label',celltype_key],
                    palette=None
                )
            else:
                sc.pl.embedding(adata_batch,basis=f'{embed}_umap',color=['pred_label',celltype_key],palette=None)
    else:
        adata_batch = adata_all[adata_all.obs[batch_key] == cond_section].copy()
        if ifumap:
            labels = adata_batch.obs['pred_label'].astype('category')
            sc.pp.neighbors(adata_batch,use_rep=embed)
            sc.tl.umap(adata_batch,key_added=f'{embed}_umap',random_state=seed)
            if len(labels.cat.categories) == 1:
                single_color = ['#1f77b4']
                adata_batch.uns['pred_label_colors'] = single_color
                sc.pl.embedding(
                    adata_batch,
                    basis=f"{embed}_umap",
                    color=['pred_label'],
                    palette=None
                )
            else:
                sc.pl.embedding(adata_batch,basis=f'{embed}_umap',color=['pred_label'],palette=None)

            
            # adata_batch = adata_all[adata_all.obs[batch_key] == cond_section].copy()
            # labels = adata_batch.obs['pred_label'].astype('category')
            # adata_batch.obs['binary'] = '2'
            # common_cells = adata_batch.obs_names.intersection(adata_cond.obs_names)
            # adata_batch.obs.loc[common_cells, 'binary'] = adata_cond.obs.loc[common_cells, 'binary']
            
            # sc.pp.neighbors(adata_batch,use_rep=embed)
            # sc.tl.umap(adata_batch,key_added=f'{embed}_umap',random_state=seed)
            # if len(labels.cat.categories) == 1:
            #     single_color = ['#1f77b4']
            #     adata_batch.uns['pred_label_colors'] = single_color
            #     sc.pl.embedding(
            #         adata_batch,
            #         basis=f"{embed}_umap",
            #         color=['pred_label','binary'],
            #         palette=None
            #     )
            

            # else:
            #     sc.pl.embedding(adata_batch,basis=f'{embed}_umap',color=['pred_label','binary'],palette=None)
    
    return adata_all

def subclustering(adata_full,embed_key='cell_embed',label='pred_label',
                  ifmerge=True,init_res=1.5,intopk=30,n_clusters=None):

    y_pred_dict = {}
    n_slices = len(np.unique(adata_full.obs['batch'].values))

    if n_clusters is None:
        n_clusters = []
        for i in range(n_slices):
            adata=sc.AnnData(adata_full.obsm[embed_key])
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, 
                         resolution=init_res,
                         random_state=seed)
            y_pred_dict[i] = adata.obs['leiden'].astype(int).to_numpy()
            n_clusters.append(len(np.unique(y_pred_dict[i])))
            del adata
            print(f'There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}')
    else:
        for i in range(self.n_slices):
            y_pred_dict[i] = tools_v1.search_res(adata = self.Batch_list[i], 
                                              embed_key=self.pre_cellembed_key, 
                                              target_n=n_clusters[i], 
                                              seed=seed)
            print(f'There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}')
    
    y_combined_dict = {}
    genename = self.Batch_list[0].var_names

    if ifmerge == True:
        for i in range(self.n_slices):
            self.Batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()
            if 'ST' in self.data_type[i]:
                s_gene_embed = tools_v1.gene_embed_weight(self.Batch_list[i].X,self.Batch_list[i].obsm[self.pre_cellembed_key],self.Batch_list[i].obsm['graph'])
            else:
                s_gene_embed = tools_v1.gene_embed_weight(self.Batch_list[i].X,self.Batch_list[i].obsm[self.pre_cellembed_key])
            s_gene2cell_mat = tools_v1.cell_to_gene_pdistance(self.Batch_list[i].obsm[self.pre_cellembed_key], s_gene_embed)
            self.Batch_list[i].layers['dist'] = s_gene2cell_mat
            s_gene_list = tools_v1.select_sig_genes(self.Batch_list[i],label_key=self.cluster_key,topk=intopk)

            _,score_mat = tools_v1.self_compute_gene_pvalue(s_gene_list, genename.values)
            tri_indx = np.triu_indices_from(score_mat, k=1)
            upper_tri_values = score_mat[tri_indx]

            while any(upper_tri_values<=0.05):
                min_idx = np.argmin(upper_tri_values)
                x = tri_indx[0][min_idx]
                y = tri_indx[1][min_idx]
                self.Batch_list[i].obs[self.cluster_key][self.Batch_list[i].obs[self.cluster_key] == y] = x
                self.Batch_list[i].obs[self.cluster_key] = LabelEncoder().fit_transform(self.Batch_list[i].obs[self.cluster_key])
                s_gene_list = tools_v1.select_sig_genes(self.Batch_list[i],label_key=self.cluster_key,topk=intopk)
                _,score_mat = tools_v1.self_compute_gene_pvalue(s_gene_list, genename.values)
                tri_indx = np.triu_indices_from(score_mat, k=1)
                upper_tri_values = score_mat[tri_indx]
            y_combined_dict[i] = self.Batch_list[i].obs[self.cluster_key]
            print(f'There are {len(np.unique(y_combined_dict[i]))} clusters for Slice - {i} after merging.')
    else:
        for i in range(self.n_slices):
            self.Batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()
            y_combined_dict[i] = self.Batch_list[i].obs[self.cluster_key]
    print(y_combined_dict)




def _dist_to_potential(dist_matrix):
    if dist_matrix.size > 0:
        sigma = np.median(dist_matrix)
    else:
        sigma = 1.0

    if sigma == 0:
        sigma = 1e-6
    return np.exp(-(dist_matrix**2) / (2 * sigma**2))


def _run_single_permutation(seed, all_involved_indices, n_senders_count, 
                            lr_database, gene_map, potential_mat, spatial_weight_matrix):

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(all_involved_indices)

    pseudo_sender_idx = shuffled_indices[:n_senders_count]
    pseudo_receiver_idx = shuffled_indices[n_senders_count:]

    current_perm_scores = {}

    for ligand, receptor in lr_database:
        if ligand not in gene_map or receptor not in gene_map:
            continue

        l_idx = gene_map[ligand]
        r_idx = gene_map[receptor]

        vec_pseudo_sender = potential_mat[pseudo_sender_idx, l_idx]
        vec_pseudo_receiver = potential_mat[pseudo_receiver_idx, r_idx]

        null_score = np.dot(vec_pseudo_sender.T, np.dot(spatial_weight_matrix, vec_pseudo_receiver))
        current_perm_scores[f"{ligand}-{receptor}"] = null_score

    return current_perm_scores

def check_cluster_umap(adata_full, embed, celltype_key, section_ids, seed=123):
    results = []
    
    for section in section_ids:
        adata_batch = adata_full[adata_full.obs['batch'] == section].copy()
        
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=seed, verbose=False)
        adata_batch.obsm[f'{embed}_umap'] = umap_model.fit_transform(adata_batch.obsm[embed])
        
        sc.pl.embedding(
            adata_batch,
            basis=f'{embed}_umap',
            color=['cluster', celltype_key]
        )
        
        y_true = adata_batch.obs[celltype_key].values
        y_pred = adata_batch.obs['cluster'].values
        
#         ari = adjusted_rand_score(y_true, y_pred)
#         nmi = normalized_mutual_info_score(y_true, y_pred)
#         print(f"Section {section}: ARI={ari:.3f}, NMI={nmi:.3f}")
        
#         results.append({'section': section, 'ARI': ari, 'NMI': nmi})
    
    # return results

class CCC_Interact:
    def __init__(self, 
                 adata,
                 direction = 'core_to_other',
                 resource_name='consensus',
                 dist_layer='cell_gene_dist', 
                 spatial_key='spatial', 
                 label_col='pred_label',
                 core_label='Disease Core',
                 other_label='Surrounding Area'):

        try:
            plt.style.use('seaborn-v0_8-white')
        except OSError:
            plt.style.use('default')
            mpl.rcParams['axes.spines.top'] = False
            mpl.rcParams['axes.spines.right'] = False
    
        self.adata = adata
        self.spatial_key = spatial_key
        self.label_col = label_col
        self.core_label = core_label
        self.other_label = other_label
        self.direction = direction
        try:
            self.dist_mat = adata.layers[dist_layer]
            self.coords = adata.obsm[spatial_key]
            self.labels = adata.obs[label_col].values
            self.genes = adata.var_names.values
        except KeyError as e:
            raise KeyError(f"Key missing in adata: {e}")

        self.gene_map = {g: i for i, g in enumerate(self.genes)}
        self.other_idx = np.where(self.labels == self.other_label)[0]
        self.core_idx = np.where(self.labels == self.core_label)[0] 
        
        print(f"Initialized with {len(self.other_idx)} '{other_label}' cells and {len(self.core_idx)} '{core_label}' cells.")
        
        if len(self.core_idx) == 0:
            raise ValueError(f"No cells found for label '{pad_label}'.")
            
        self.gene_names = list(adata.var_names)
        lr_df = liana.resource.select_resource(resource_name=resource_name)
        self.lr_database = list(zip(lr_df['ligand'], lr_df['receptor']))
        available_genes_set = set(self.gene_names)

        print(f"Total genes in your dataset: {len(available_genes_set)}")
        print(f"Original L-R pairs: {len(self.lr_database)}")

        filtered_lr_database = [
            (l, r) for l, r in self.lr_database
            if l in available_genes_set and r in available_genes_set
        ]

        print("-" * 30)
        print(f"Filtered L-R pairs: {len(filtered_lr_database)}")
        self.lr_database = filtered_lr_database


    def run_analysis(self,   
                     normalization=True,      
                     kernel_bandwidth=500,
                     n_perms=1000,
                     calc_stats=True,
                     p_adj_threshold=0.05,
                     seed=123,
                     n_jobs=8,
                     result_key_suffix=''):

        if self.direction == 'both':
            target_directions = ['other_to_core', 'core_to_other']
        elif self.direction in ['other_to_core', 'core_to_other']:
            target_directions = [self.direction]
        else:
            raise ValueError("direction must be 'other_to_core', 'core_to_other', or 'both'")

        print(f"Start Analyzing {len(self.lr_database)} L-R pairs...")
        # print(f"Config: Mode={self.direction}, Bandwidth={kernel_bandwidth}")

        nc_coords = self.coords[self.other_idx]   
        pad_coords = self.coords[self.core_idx] 
        
        cached_spatial_weights = {}
        self.potential_mat = _dist_to_potential(self.dist_mat)
        
        for current_dir in target_directions:
            print(f"\n=== Processing Direction: {current_dir} ===")
            
            if current_dir == 'core_to_other' and 'other_to_core' in cached_spatial_weights:
                print("Transposing existing spatial weight matrix from other_to_core...")
                spatial_weight_matrix = cached_spatial_weights['other_to_core'].T
            else:
                print(f"Calculating full distance matrix for {current_dir}...")
                if current_dir == 'other_to_core':
                    dist_matrix = cdist(nc_coords, pad_coords, metric='euclidean')
                else:
                    dist_matrix = cdist(pad_coords, nc_coords, metric='euclidean')
                
                spatial_weight_matrix = np.exp(-(dist_matrix**2) / (2 * kernel_bandwidth**2))
                
                cached_spatial_weights[current_dir] = spatial_weight_matrix
                del dist_matrix

            if current_dir == 'other_to_core':
                sender_idx = self.other_idx
                receiver_idx = self.core_idx
            else:
                sender_idx = self.core_idx
                receiver_idx = self.other_idx

            pathway_contributions = {}
            
            real_total_scores = {}
            real_pathway_sums = {}

            for ligand, receptor in self.lr_database:
                if ligand not in self.gene_map or receptor not in self.gene_map:
                    continue
                
                l_idx = self.gene_map[ligand]
                r_idx = self.gene_map[receptor]
                pair_name = f"{ligand}-{receptor}"

                vec_sender = self.potential_mat[sender_idx, l_idx]
                vec_receiver = self.potential_mat[receiver_idx, r_idx]
                
                receiver_context_sum = np.dot(spatial_weight_matrix, vec_receiver)

                current_pathway_score = vec_sender * receiver_context_sum

                full_length_score = np.zeros(self.adata.shape[0])
                full_length_score[sender_idx] = current_pathway_score
                pathway_contributions[pair_name] = full_length_score

                real_total_scores[pair_name] = current_pathway_score
                real_pathway_sums[pair_name] = np.sum(current_pathway_score)

            pathway_stats = pd.DataFrame(index=real_pathway_sums.keys())
            pathway_stats['Real_Sum'] = list(real_pathway_sums.values())

            if calc_stats and n_perms > 0:
                print(f"Running permutation test ({n_perms} perms) for {current_dir}...")
                
                all_involved_indices = np.concatenate([self.other_idx, self.core_idx])
                n_senders_count = len(sender_idx)
                
                null_scores = {k: np.zeros(n_perms) for k in real_pathway_sums.keys()}
                
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_run_single_permutation)(
                        p,
                        all_involved_indices,
                        n_senders_count, 
                        self.lr_database, 
                        self.gene_map, 
                        self.potential_mat,                  
                        spatial_weight_matrix
                    ) 
                    for p in tqdm(range(n_perms), desc="Parallel Permutation")
                )
                
                for p, perm_result in enumerate(results):
                    for pair_name, score in perm_result.items():
                        null_scores[pair_name][p] = score
                     
                p_values = []
                for pair_name, real_score in real_pathway_sums.items():
                    null_dist = null_scores[pair_name]
                    p_val = (np.sum(null_dist >= real_score) + 1) / (n_perms + 1)
                    p_values.append(p_val)
                
                pathway_stats['P_value'] = p_values
                reject, pvals_corrected, _, _ = multipletests(p_values, alpha=p_adj_threshold, method='fdr_bh')
                pathway_stats['P_adj'] = pvals_corrected
                pathway_stats = pathway_stats.sort_values('P_adj')

                stat_key = f'ccc_stats_{current_dir}{result_key_suffix}'
                
                significant_df = pathway_stats[pathway_stats['P_adj'] < p_adj_threshold].copy()
                significant_df = significant_df.sort_values(by='Real_Sum', ascending=False)
                
                self.adata.uns[stat_key] = significant_df
                
                if not significant_df.empty:
                    print(f"[{current_dir}] Found {len(significant_df)} significant pathways.")
                else:
                    print(f"[{current_dir}] No significant pathways found.")
            
            significant_pairs = significant_df.index.tolist()
            valid_vectors = [real_total_scores[pair] for pair in significant_pairs if pair in real_total_scores]
            if not valid_vectors:
                print("No valid pairing vector was found！")
                combined_score = None
                
            else:
                combined_score = np.sum(valid_vectors, axis=0)
                if normalization:
                    _min = np.min(combined_score)
                    _max = np.max(combined_score)
                    if _max - _min > 1e-9:
                        combined_score = (combined_score - _min) / (_max - _min)
                    else:
                        combined_score[:] = 0

                n_total = self.adata.shape[0]
                full_scores = np.zeros(n_total)
                full_scores[sender_idx] = combined_score
            
                score_key = f'ccc_score_{current_dir}{result_key_suffix}'
                uns_pathway_key = f'ccc_pathways_{current_dir}{result_key_suffix}'
            
                print(f"Saving results to obs['{score_key}'] and uns['{uns_pathway_key}']")
            
                self.adata.obs[score_key] = full_scores
                self.adata.uns[uns_pathway_key] = pathway_contributions

        print("\nAnalysis Completed.")      

    
    def get_sender_composition(self, 
                               base_score_key='ccc_score', 
                               result_key_suffix='',       
                               threshold=0.5,
                               cell_type_col='CellType',
                               save=None):

            if self.direction == 'both':
                target_directions = ['other_to_core', 'core_to_other']
            elif self.direction in ['other_to_core', 'core_to_other']:
                target_directions = [self.direction]
            else:
                raise ValueError(f"Unknown direction: {self.direction}")

            results_dict = {}

            for current_dir in target_directions:
                full_key = f"{base_score_key}_{current_dir}{result_key_suffix}"

                if full_key not in self.adata.obs:
                    print(f"Warning: Key '{full_key}' not found. Skipping {current_dir}.")
                    continue

                if current_dir == 'other_to_core':
                    target_label = self.other_label
                    role_name = "NC (Sender)"
                else:
                    target_label = self.core_label
                    role_name = "PAD (Sender)"

                print(f"\n=== Analyzing Sender Composition: {current_dir} ===")
                print(f"Target Role: {role_name} | Threshold > {threshold}")

                mask = (self.adata.obs[self.label_col] == target_label) & \
                       (self.adata.obs[full_key] > threshold)

                sig_cells = self.adata.obs[mask]
                n_sig = len(sig_cells)

                if n_sig == 0:
                    print(f"No cells passed the threshold for {current_dir}.")
                    results_dict[current_dir] = None
                    continue

                if cell_type_col not in sig_cells.columns:
                     raise KeyError(f"Column '{cell_type_col}' not found in adata.obs.")

                counts = sig_cells[cell_type_col].value_counts()
                percentages = sig_cells[cell_type_col].value_counts(normalize=True) * 100

                stats_df = pd.DataFrame({
                    'Count': counts,
                    'Percentage (%)': percentages.round(2)
                })

                print("-" * 40)
                print(stats_df)
                print("-" * 40)
                
                uns_cell_key = f'ccc_top_senders_{current_dir}{result_key_suffix}'
            
                print(f"Saving results uns['{uns_cell_key}']")
            
                self.adata.uns[uns_cell_key] = stats_df
                
                if save:
                    import os
                    root, ext = os.path.splitext(save)
                    final_save_name = f"{root}_{current_dir}{ext}"
                    stats_df.to_csv(final_save_name, index_label=cell_type_col)
                    print(f"Results saved to {final_save_name}")

                results_dict[current_dir] = stats_df

            return results_dict
        
class CCC_plot:
    def __init__(self, 
                 adata,
                 direction = 'other_to_core',
                 resource_name='consensus',
                 dist_layer='cell_gene_dist', 
                 spatial_key='spatial', 
                 label_col='pred_label',
                 core_label='Disease Core',
                 other_label='Surrounding Area'):
        try:
            plt.style.use('seaborn-v0_8-white')
        except OSError:
            plt.style.use('default')
            mpl.rcParams['axes.spines.top'] = False
            mpl.rcParams['axes.spines.right'] = False
    
        self.adata = adata
        self.spatial_key = spatial_key
        self.label_col = label_col
        self.core_label = core_label
        self.other_label = other_label
        self.direction = direction

        try:
            self.dist_mat = adata.layers[dist_layer]
            self.coords = adata.obsm[spatial_key]
            self.labels = adata.obs[label_col].values
            self.genes = adata.var_names.values
        except KeyError as e:
            raise KeyError(f"Key missing in adata: {e}")

        self.gene_map = {g: i for i, g in enumerate(self.genes)}
        self.other_idx = np.where(self.labels == self.other_label)[0]
        self.core_idx = np.where(self.labels == self.core_label)[0] 
        
        print(f"Initialized with {len(self.other_idx)} '{self.other_label}' cells and {len(self.core_idx)} '{self.core_label}' cells.")
        
        if len(self.core_idx) == 0:
            raise ValueError(f"No cells found for label '{pad_label}'.")
    
    
    def plot_interaction(self, 
                     base_score_key='ccc_score',
                     color_bg = '#b8b0b0',
                     cmap = 'YlOrRd',
                     result_key_suffix='',       
                     point_size=5, 
                     title=None,
                     base_size=4, 
                     dpi=120,
                     save=None):

        if self.direction == 'both':
            target_directions = ['other_to_core', 'core_to_other']
        elif self.direction in ['other_to_core', 'core_to_other']:
            target_directions = [self.direction]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        for current_dir in target_directions:
            full_key = f"{base_score_key}_{current_dir}{result_key_suffix}"

            if full_key not in self.adata.obs:
                print(f"Warning: Key '{full_key}' not found. Skipping.")
                continue

            print(f"Plotting interaction: {current_dir} ...")

            coords = self.coords
            labels = self.labels
            scores = self.adata.obs[full_key].values

            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            delta_x = x_max - x_min
            delta_y = y_max - y_min
            aspect_ratio = delta_x / delta_y if delta_y > 0 else 1.0
            figsize = (base_size * aspect_ratio + 1.2, base_size)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if current_dir == 'other_to_core':
                mask_bg = (labels == self.core_label)
                mask_fg = (labels == self.other_label)

                bg_color = color_bg
                label_bg = f'{self.core_label} (Receiver)'
                label_fg = f'{self.other_label} (Sender)'

            else:

                mask_bg = (labels == self.other_label)
                mask_fg = (labels == self.core_label)

                bg_color = color_bg
                label_bg = f'{self.other_label} (Receiver)'
                label_fg = f'{self.core_label} (Sender)'

            if np.sum(mask_bg) > 0:
                ax.scatter(coords[mask_bg, 0], coords[mask_bg, 1],
                           c=bg_color, 
                           s=point_size*1.1, 
                           alpha=0.8,
                           edgecolors='none', rasterized=True,
                           label=label_bg)

            if np.sum(mask_fg) > 0:
                fg_coords = coords[mask_fg]
                fg_scores = scores[mask_fg]

                sort_idx = np.argsort(fg_scores)

                sc = ax.scatter(fg_coords[sort_idx, 0], fg_coords[sort_idx, 1],
                                c=fg_scores[sort_idx], 
                                cmap=cmap, 
                                s=point_size, 
                                alpha=1.0, 
                                edgecolors='none',
                                rasterized=True,
                                label=label_fg)

                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.005, aspect=30, shrink=0.8)
                cbar.set_label('Interaction Score for Senders', rotation=270, labelpad=20, fontsize=10, color='#555555')
                cbar.ax.tick_params(labelsize=8, color='#888888')
                cbar.outline.set_linewidth(0.5)
                cbar.outline.set_edgecolor('#AAAAAA')

            if current_dir == 'other_to_core':
                plot_title = 'Sender (Surrounding Area) - Receiver (Disease Core)'
            else:
                plot_title = 'Sender (Disease Core) - Receiver (Surrounding Area)'
            ax.set_title(plot_title, fontsize=14)

            ax.axis('off')
            ax.set_aspect('equal', 'datalim')
            ax.invert_yaxis()

            if save:
                import os
                root, ext = os.path.splitext(save)
                final_save_name = f"{root}_{current_dir}{ext}"

                plt.savefig(final_save_name, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                print(f"Saved figure to {final_save_name}")

            plt.show()
            plt.close()
    
    def plot_sender_distribution(self,
                                 color_map = None,
                                 celltype_key = 'CellType',
                                 result_key_suffix='', 
                                 figsize=(6, 5.5),
                                 save=None):

            if self.direction == 'both':
                target_directions = ['other_to_core', 'core_to_other']
            elif self.direction in ['other_to_core', 'core_to_other']:
                target_directions = [self.direction]
            else:
                raise ValueError(f"Unknown direction: {self.direction}")

            for current_dir in target_directions:

                uns_cell_key = f'ccc_top_senders_{current_dir}{result_key_suffix}'

                if uns_cell_key not in self.adata.uns:
                    print(f"Warning: Key '{uns_cell_key}' not found in adata.uns.")
                    continue

                celltype_composition = self.adata.uns[uns_cell_key]

                if celltype_composition is None or celltype_composition.empty:
                    print(f"No sender data available for {current_dir}.")
                    continue

                df_plot = celltype_composition.copy()

                df_plot = df_plot.reset_index() 

                df_plot.columns.values[0] = celltype_key
                if 'Count' not in df_plot.columns:
                     for col in df_plot.columns:
                         if pd.api.types.is_numeric_dtype(df_plot[col]) and col != celltype_key:
                             df_plot.rename(columns={col: 'Count'}, inplace=True)
                             break

                df_plot[celltype_key] = df_plot[celltype_key].astype(str)

                df_plot = df_plot[df_plot['Count'] > 0].sort_values(by='Count', ascending=False)

                if df_plot.empty:
                    continue

                unique_types = df_plot[celltype_key].unique()
                safe_palette = {}
                for ctype in unique_types:
                    safe_palette[ctype] = color_map.get(ctype, '#808080')

                plt.figure(figsize=figsize)

                ax = sns.barplot(data=df_plot, 
                                 x=celltype_key, 
                                 y='Count',
                                 order=df_plot[celltype_key],
                                 palette=safe_palette, 
                                 hue=celltype_key, 
                                 legend=False,
                                 edgecolor='black', 
                                 linewidth=1, 
                                 zorder=3)

                sender_role = "Surrounding Area" if current_dir == 'other_to_core' else "Disease Core"
                plt.title(f'{sender_role} Sender Distribution', fontsize=14, pad=15)

                plt.xlabel('Cell Type', fontsize=12)
                plt.ylabel('Count', fontsize=12)

                plt.xticks(rotation=45, ha='right')
                plt.tick_params(axis='both', which='major', direction='out', length=5, width=1)

                y_max = df_plot['Count'].max()
                for i, v in enumerate(df_plot['Count']):
                    plt.text(i, v + (y_max * 0.005), str(v), 
                             ha='center', va='bottom', fontsize=10, color='black')

                plt.tight_layout()

                if save:
                    root, ext = os.path.splitext(save)
                    final_save = f"{root}_{current_dir}_distribution{ext}"
                    plt.savefig(final_save, dpi=300, bbox_inches='tight')
                    print(f"Saved distribution plot to {final_save}")

                plt.show()
    
    def plot_top_pathways(self, 
                              result_key_suffix='', 
                              top_n=6,
                              n_cols=3,
                              point_size=5, 
                              percentile=99, 
                              alpha=0.8,
                              base_size=4,
                              dpi=120,
                              save=None):

            if self.direction == 'both':
                target_directions = ['other_to_core', 'core_to_other']
            elif self.direction in ['other_to_core', 'core_to_other']:
                target_directions = [self.direction]
            else:
                raise ValueError(f"Unknown direction: {self.direction}")

            for current_dir in target_directions:
                stats_key = f'ccc_stats_{current_dir}{result_key_suffix}'

                if stats_key not in self.adata.uns:
                    print(f"Warning: Stats key '{stats_key}' not found. Run analysis first.")
                    continue

                stats_df = self.adata.uns[stats_key]
                if stats_df.empty:
                    print(f"No significant pathways found for {current_dir}.")
                    continue

                top_pairs = stats_df.index[:top_n].tolist()
                n_pairs = len(top_pairs)
                print(f"\nPlotting top {n_pairs} pathways for {current_dir}...")

                if current_dir == 'other_to_core':
                    mask_sender = (self.labels == self.other_label)
                    mask_receiver = (self.labels == self.core_label)
                    sender_name, receiver_name = 'Surrounding Area', 'Disease Core'
                else: 
                    mask_sender = (self.labels == self.core_label)
                    mask_receiver = (self.labels == self.other_label)
                    sender_name, receiver_name = 'Disease Core', 'Surrounding Area'

                n_cols = n_cols
                n_rows = (n_pairs + n_cols - 1) // n_cols

                coords = self.coords
                x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
                aspect = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 1

                fig_w = base_size * aspect * n_cols
                fig_h = base_size * n_rows * 1.3 

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
                axes = axes.flatten() if n_pairs > 1 else [axes]

                def get_expr(gene, mask):
                    use_raw = self.adata.raw is not None and gene in self.adata.raw.var_names
                    if use_raw:
                        expr = self.adata.raw[mask, gene].X
                    elif gene in self.adata.var_names:
                        expr = self.adata[mask, gene].X
                    else:
                        return np.zeros(np.sum(mask))
                    if not isinstance(expr, np.ndarray):
                        expr = expr.toarray()
                    return expr.flatten()

                for i, pair_name in enumerate(top_pairs):
                    ax = axes[i]
                    ligand, receptor = pair_name.split('-')

                    sender_expr = get_expr(ligand, mask_sender)
                    sender_coords = coords[mask_sender]

                    receiver_expr = get_expr(receptor, mask_receiver)
                    receiver_coords = coords[mask_receiver]

                    vmax_s = np.percentile(sender_expr, percentile) if np.max(sender_expr) > 0 else 1
                    vmax_r = np.percentile(receiver_expr, percentile) if np.max(receiver_expr) > 0 else 1

                    norm_sender = Normalize(vmin=0, vmax=vmax_s)
                    norm_receiver = Normalize(vmin=0, vmax=vmax_r)

                    sc_r = ax.scatter(receiver_coords[:, 0], receiver_coords[:, 1],
                                      c=receiver_expr, cmap=cm.Blues, norm=norm_receiver,
                                      s=point_size, alpha=alpha, edgecolors='none', rasterized=True)

                    sc_s = ax.scatter(sender_coords[:, 0], sender_coords[:, 1],
                                      c=sender_expr, cmap=cm.Reds, norm=norm_sender,
                                      s=point_size, alpha=alpha, edgecolors='none', rasterized=True)

                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.invert_yaxis()

                    axins_top = inset_axes(ax, width="80%", height="5%", loc='upper center', 
                                           bbox_to_anchor=(0, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

                    cbar_top = plt.colorbar(sc_s, cax=axins_top, orientation='horizontal')

                    cbar_top.ax.xaxis.set_ticks_position('top')
                    cbar_top.ax.xaxis.set_label_position('top')
                    cbar_top.set_label(f'{ligand} (Sender: {sender_name})', size=9, labelpad=5, color='#8B0000', fontweight='bold')
                    cbar_top.ax.tick_params(labelsize=7, colors='#8B0000')
                    cbar_top.outline.set_edgecolor('#FFCCCC')

                    axins_bottom = inset_axes(ax, width="80%", height="5%", loc='lower center', 
                                              bbox_to_anchor=(0, -0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

                    cbar_bot = plt.colorbar(sc_r, cax=axins_bottom, orientation='horizontal')

                    cbar_bot.ax.xaxis.set_ticks_position('bottom')
                    cbar_bot.ax.xaxis.set_label_position('bottom')
                    cbar_bot.set_label(f'{receptor} (Receiver: {receiver_name})', size=9, labelpad=5, color='#00008B', fontweight='bold')
                    cbar_bot.ax.tick_params(labelsize=7, colors='#00008B')
                    cbar_bot.outline.set_edgecolor('#CCCCFF')

                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                plt.tight_layout(h_pad=3.0, w_pad=1.0)

                plt.subplots_adjust(hspace=0.4) 

                if save:
                    import os
                    root, ext = os.path.splitext(save)
                    final_save_name = f"{root}_{current_dir}_top{top_n}{ext}"
                    plt.savefig(final_save_name, dpi=dpi, bbox_inches='tight')
                    print(f"Saved figure to {final_save_name}")

                plt.show()
                plt.close()
                
    def plot_bubble_pathways(self, 
                                 top_n=10, 
                                 result_key_suffix='', 
                                 figsize=(6, 5), 
                                 bubble_color='#B0C4DE',
                                 save=None):

            FONT_TITLE  = 14
            FONT_LABEL  = 12
            FONT_TICK   = 11
            FONT_LEGEND = 10

            if self.direction == 'both':
                target_directions = ['other_to_core', 'core_to_other']
            elif self.direction in ['other_to_core', 'core_to_other']:
                target_directions = [self.direction]
            else:
                raise ValueError(f"Unknown direction: {self.direction}")

            for current_dir in target_directions:

                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

                stats_key = f'ccc_stats_{current_dir}{result_key_suffix}'

                if stats_key not in self.adata.uns:
                    ax.text(0.5, 0.5, "No stats found", ha='center', va='center')
                else:
                    df = self.adata.uns[stats_key]

                    df_plot = df.head(top_n).sort_values('Real_Sum', ascending=True).copy()

                    if df_plot.empty:
                        ax.text(0.5, 0.5, "No significant pathways", ha='center', va='center')
                    else:
                        y_pos = range(len(df_plot))
                        x_vals = df_plot['Real_Sum'].values
                        p_vals = df_plot['P_adj'].values

                        x_min, x_max = x_vals.min(), x_vals.max()
                        x_range = x_max - x_min
                        padding = x_range * 0.15 if x_range > 0 else (x_max * 0.1 if x_max > 0 else 1.0)
                        ax.set_xlim(x_min - padding, x_max + padding)

                        SIZE_LEVELS = {
                            'p_0.001': 300, # ***
                            'p_0.01':  120, # **
                            'p_0.05':  50,  # *
                            'ns':      20   # ns
                        }

                        sizes = []
                        for p in p_vals:
                            if p < 0.001:
                                sizes.append(SIZE_LEVELS['p_0.001'])
                            elif p < 0.01:
                                sizes.append(SIZE_LEVELS['p_0.01'])
                            elif p < 0.05:
                                sizes.append(SIZE_LEVELS['p_0.05'])
                            else:
                                sizes.append(SIZE_LEVELS['ns'])

                        ax.scatter(x_vals, y_pos, 
                                   s=sizes, 
                                   c=bubble_color, 
                                   edgecolors='black', 
                                   linewidth=0.8,
                                   alpha=0.9, 
                                   zorder=2)

                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(df_plot.index, fontsize=FONT_TICK)

                        ax.set_xlabel('Interaction Strength (Sender)', fontsize=FONT_LABEL, labelpad=8)

                        title_txt = "Surrounding Area -> Disease Core" if current_dir == 'other_to_core' else "Disease Core -> Surrounding Area"
                        ax.set_title(f"{title_txt} (Top {top_n})", fontsize=FONT_TITLE, fontweight='bold', pad=15)

                        for spine_name in ['top', 'bottom', 'left', 'right']:
                            ax.spines[spine_name].set_visible(True)
                            ax.spines[spine_name].set_color('black')
                            ax.spines[spine_name].set_linewidth(1.0)

                        ax.grid(False)

                        ax.tick_params(axis='both', which='major', labelsize=FONT_TICK, width=1, length=5, direction='out')
                        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)
                        ax.tick_params(axis='y', left=True, right=False)

                        legend_elements = [
                            plt.scatter([], [], s=SIZE_LEVELS['p_0.001'], c=bubble_color, edgecolors='black', alpha=0.9, label='P < 0.001'),
                            plt.scatter([], [], s=SIZE_LEVELS['p_0.01'],  c=bubble_color, edgecolors='black', alpha=0.9, label='P < 0.01'),
                            plt.scatter([], [], s=SIZE_LEVELS['p_0.05'],  c=bubble_color, edgecolors='black', alpha=0.9, label='P < 0.05')
                        ]

                        ax.legend(handles=legend_elements, 
                                  title="Significance",
                                  title_fontsize=FONT_LEGEND,
                                  loc='lower right', 
                                  frameon=True, 
                                  edgecolor='black',
                                  fontsize=FONT_LEGEND,
                                  labelspacing=1.0, 
                                  borderpad=0.8)

                if save:
                    import os
                    root, ext = os.path.splitext(save)
                    final_save = f"{root}_{current_dir}_bubble{ext}"
                    plt.savefig(final_save, dpi=300, bbox_inches='tight')
                    print(f"Saved: {final_save}")

                plt.show()
