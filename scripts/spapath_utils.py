import random
import os
import re
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    print(f"{dir} has been created or already exists.")


def attach_image_embedding(adata, embedding_path):
    image_info = pd.read_csv(embedding_path, index_col=0)
    embedding_columns = [
        column for column in image_info.columns if str(column).startswith("dim_")
    ]
    if not embedding_columns:
        raise ValueError(f"No image-embedding columns were found in {embedding_path}.")

    missing_observations = adata.obs_names.difference(image_info.index)
    if len(missing_observations) > 0:
        raise ValueError(
            f"{len(missing_observations)} observations are missing from {embedding_path}."
        )

    adata.obsm["image_embedding"] = image_info.loc[
        adata.obs_names, embedding_columns
    ].to_numpy()
    return adata


def preprocess(
    adata_list,
    adata_type_map,
    full_num_hvgs=3000,
    min_genes_qc=10,
    min_cells_qc=10,
    ifco_expressed_genes=False,
):
    adata_type = list(adata_type_map.values())
    assert len(adata_type) == len(adata_list)

    ads = []
    for i, adata_st in enumerate(adata_list):
        if not sp.issparse(adata_st.X):
            adata_st.X = sp.csr_matrix(adata_st.X)
        adata_st.var_names_make_unique()
        adata_st = adata_st[
            :,
            np.array(~adata_st.var.index.isna())
            & np.array(~adata_st.var_names.str.startswith("mt-"))
            & np.array(~adata_st.var_names.str.startswith("MT-")),
        ]
        print(f"shape of adata {i} before quality control: {adata_st.shape}")
        sc.pp.filter_cells(adata_st, min_genes=min_genes_qc)
        sc.pp.filter_genes(adata_st, min_cells=min_cells_qc)
        print(f"shape of adata {i} after quality control: {adata_st.shape}")
        adata_st.obs.index = adata_st.obs.index + "-" + str(i)
        adata_list[i] = adata_st
        ads.append(adata_st)

    adata_tar = adata_list[-1].copy()

    adata_full = ad.concat(ads, join="inner")
    del ads
    shared_gene = adata_full.var_names
    print(f"Find {len(shared_gene)} shared genes among datasets.")

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
            print(
                f"Find {len(coexp_gene)} shared co-expressed genes among {len(adata_list)} datasets."
            )
    else:
        coexp_gene = shared_gene

    adata_full = adata_full[:, coexp_gene].copy()

    target_sum = 1e3 if len(coexp_gene) < 1e3 else 1e4

    print("Normalize data...")
    adata_full.layers["counts"] = adata_full.X.copy()
    sc.pp.normalize_total(adata_full, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata_full)

    sc.pp.highly_variable_genes(
        adata_full, n_top_genes=full_num_hvgs, batch_key="batch"
    )
    hvgs_shared = sorted(adata_full.var_names[adata_full.var.highly_variable].tolist())
    adata_full.uns["hvgs_shared"] = hvgs_shared
    print(f"Find {len(hvgs_shared)} shared highly variable genes among datasets.")

    tar_target_sum = 1e3 if adata_tar.shape[1] < 1e3 else 1e4
    adata_tar.layers["counts"] = adata_tar.X.copy()
    sc.pp.normalize_total(adata_tar, target_sum=tar_target_sum, inplace=True)
    sc.pp.log1p(adata_tar)

    return adata_full, adata_tar


def build_graph_GAT_plus(adata_full, adata_type_map: dict, K=8, img_threshold=0.0):
    print("Start building graphs...")

    batch_ids = adata_full.obs["batch"].unique()

    for bid in batch_ids:
        current_type = adata_type_map[bid]
        mask = adata_full.obs["batch"] == bid
        adata_st = adata_full[mask]
        n_cells = adata_st.shape[0]

        if current_type in ["sc", "scATAC"]:
            adata_full.uns[f"graph_{bid}"] = sp.eye(n_cells, dtype=int, format="csr")
            adata_full.uns[f"graph_cos_{bid}"] = sp.eye(
                n_cells, dtype=float, format="csr"
            )
            print(f"Skipping spatial graph for batch {bid} ({current_type})")
            continue

        print(
            f"Building sparse graph for batch {bid} ({current_type}) using NearestNeighbors..."
        )

        if "spatial" not in adata_st.obsm:
            raise KeyError(f"Batch {bid} is missing .obsm['spatial'].")

        coords = adata_st.obsm["spatial"]
        nbrs = NearestNeighbors(
            n_neighbors=K + 1, algorithm="auto", metric="euclidean"
        ).fit(coords)
        _, knn_indices = nbrs.kneighbors(coords)

        row_indices_list = []
        col_indices_list = []

        if current_type == "ST":
            row_indices_list = np.repeat(np.arange(n_cells), K + 1)
            col_indices_list = knn_indices.flatten()

        elif current_type == "ST_with_HE":
            if "image_embedding" not in adata_st.obsm:
                raise ValueError(
                    f"Batch {bid} is 'ST_with_HE' but .obsm['image_embedding'] is missing."
                )

            img_emb = adata_st.obsm["image_embedding"]
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
            (data, (row_indices_list, col_indices_list)), shape=(n_cells, n_cells)
        )

        adata_full.uns[f"graph_{bid}"] = G_sparse

        avg_neighbors = G_sparse.sum(axis=1).mean()
        print(f"  Average neighbors for Slice {bid}: {avg_neighbors-1:.2f}")

        try:
            pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine")
            adata_full.uns[f"graph_cos_{bid}"] = 1 - pair_dist_cos
        except MemoryError:
            print(
                f"  Warning: MemoryError when computing cosine matrix for batch {bid}. Using identity."
            )
            adata_full.uns[f"graph_cos_{bid}"] = np.eye(n_cells, dtype=float)

    return adata_full


def build_graph_GAT(adata_full, adata_type=["ST", "ST"], K=8, img_threshold=0.0):

    print("Start building graphs...")

    for i, adata_st in enumerate(adata_list):
        current_type = adata_type[i]
        n_cells = adata_st.shape[0]

        if current_type in ["sc", "scATAC"]:
            adata_st.obsm["graph"] = sp.csr_matrix((n_cells, n_cells), dtype=int)
            print(f"Skipping spatial graph for batch {i} ({current_type})")
            continue

        print(
            f"Building sparse graph for batch {i} ({current_type}) using NearestNeighbors..."
        )

        if "spatial" not in adata_st.obsm:
            raise KeyError(f"Batch {i} is missing .obsm['spatial'].")

        coords = adata_st.obsm["spatial"]

        nbrs = NearestNeighbors(
            n_neighbors=K + 1, algorithm="auto", metric="euclidean"
        ).fit(coords)
        _, knn_indices = nbrs.kneighbors(coords)

        row_indices_list = []
        col_indices_list = []

        if current_type == "ST":
            row_indices_list = np.repeat(np.arange(n_cells), K + 1)
            col_indices_list = knn_indices.flatten()

        elif current_type == "ST_with_HE":
            if "image_embedding" not in adata_st.obsm:
                raise ValueError(
                    f"Batch {i} is 'ST_with_HE' but .obsm['image_embedding'] is missing."
                )

            img_emb = adata_st.obsm["image_embedding"]
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
            (data, (row_indices_list, col_indices_list)), shape=(n_cells, n_cells)
        )

        adata_st.obsm["graph"] = G_sparse

        avg_neighbors = G_sparse.sum(axis=1).mean()
        print(f"  Average neighbors for Slice {i}: {avg_neighbors-1:.2f}")

        try:
            pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine")
            adata_st.obsm["graph_cos"] = 1 - pair_dist_cos
        except MemoryError:
            print(
                "  Warning: MemoryError when computing full gene expression cosine matrix. Skipping graph_cos."
            )

    return adata_list


def search_res(
    adata,
    embed_key,
    target_n=2,
    n_neighbors=15,
    start_res=0.1,
    step=0.05,
    max_res=5.0,
    seed=123,
):

    data = sc.AnnData(adata.obsm[embed_key])
    sc.pp.neighbors(data, n_neighbors=n_neighbors)

    res = start_res

    while res <= max_res:
        sc.tl.leiden(data, resolution=res, random_state=seed)
        y_pred = data.obs["leiden"].to_numpy()
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
        covariance_type="tied",
        init_params="kmeans",
        random_state=seed,
        n_init=5,
    )
    y_pred = gm.fit_predict(adata.obsm[embed_key])
    return y_pred.astype(str)


def gene_embed_weight(X, ce_cell, adj=None, c=1.0):
    X = X.T
    if adj is None:
        if sp.issparse(X):
            sumW = np.asarray(X.sum(axis=1)) + 1e-8
            return np.asarray(X @ ce_cell) / sumW
        sumW = np.sum(X, axis=1, keepdims=True) + 1e-8
        weight = X / sumW
        return weight @ ce_cell
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
    X = X.T

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
    An = np.sum(cell_embed**2, axis=1, keepdims=True)
    Bn = np.sum(gene_embed**2, axis=1, keepdims=True)
    C = -2 * np.dot(cell_embed, gene_embed.T)
    C += An
    C += Bn.T
    return np.sqrt(np.maximum(C, 0.0) + eta)


def cell_to_gene_pdistance_torch(cell_embed, gene_embed, eta=1e-10):
    An = torch.sum(cell_embed**2, dim=1, keepdim=True)
    Bn = torch.sum(gene_embed**2, dim=1, keepdim=True)

    C = -2 * torch.matmul(cell_embed, gene_embed.T)
    C = C + An
    C = C + Bn.T

    return torch.sqrt(torch.clamp(C, min=0.0) + eta)


def build_disease_data(
    adata_full,
    disease_adata_all_genes,
    disease_section,
    embed_key="pre_embed",
    batch_key="batch",
    graph_key="graph",
    gene_embed_key="gene_embed",
    dist_key="cell_gene_dist",
):
    """Build an all-gene disease dataset and recalculate its co-embedding."""
    if batch_key not in adata_full.obs:
        raise KeyError(f"Batch column '{batch_key}' was not found in adata_full.obs.")
    disease_mask = adata_full.obs[batch_key] == disease_section
    if not np.any(disease_mask):
        raise ValueError(f"No observations were found for section '{disease_section}'.")
    disease_result = adata_full[disease_mask].copy()

    if not disease_adata_all_genes.obs_names.is_unique:
        raise ValueError("disease_adata_all_genes must have unique observation names.")
    if not disease_result.obs_names.is_unique:
        raise ValueError("disease_result must have unique observation names.")

    missing_observations = disease_result.obs_names.difference(
        disease_adata_all_genes.obs_names
    )
    if len(missing_observations) > 0:
        raise ValueError(
            f"{len(missing_observations)} observations in disease_result are missing "
            "from disease_adata_all_genes."
        )

    disease_data = disease_adata_all_genes[disease_result.obs_names].copy()
    aligned_result = disease_result[disease_data.obs_names]
    disease_data.obs = aligned_result.obs.copy()

    for key in list(disease_data.obsm.keys()):
        del disease_data.obsm[key]
    for key, value in aligned_result.obsm.items():
        disease_data.obsm[key] = value.copy()

    if embed_key not in disease_data.obsm:
        raise KeyError(f"Embedding '{embed_key}' was not found in disease_result.obsm.")

    if graph_key in aligned_result.obsp:
        graph = aligned_result.obsp[graph_key].copy()
    else:
        if batch_key not in aligned_result.obs:
            raise KeyError(
                f"Batch column '{batch_key}' was not found in disease_result.obs."
            )
        section_ids = aligned_result.obs[batch_key].unique()
        if len(section_ids) != 1:
            raise ValueError("disease_result must contain exactly one section.")
        stored_graph_key = f"{graph_key}_{section_ids[0]}"
        if stored_graph_key not in aligned_result.uns:
            raise KeyError(
                f"Graph '{graph_key}' was not found in disease_result.obsp, and "
                f"'{stored_graph_key}' was not found in disease_result.uns."
            )
        graph = aligned_result.uns[stored_graph_key].copy()

    if graph.shape != (disease_data.n_obs, disease_data.n_obs):
        raise ValueError(
            f"Graph shape {graph.shape} does not match the {disease_data.n_obs} "
            "observations in disease_data."
        )
    disease_data.obsp[graph_key] = graph

    expression = disease_data.X
    if sp.issparse(expression):
        expression = expression.toarray()
    else:
        expression = np.asarray(expression)
    cell_embedding = np.asarray(disease_data.obsm[embed_key])
    dense_graph = graph.toarray() if sp.issparse(graph) else np.asarray(graph)
    gene_embedding = gene_embed_weight(expression, cell_embedding, dense_graph)
    disease_data.uns[gene_embed_key] = gene_embedding
    disease_data.layers[dist_key] = cell_to_gene_pdistance(
        cell_embedding, disease_data.uns[gene_embed_key]
    ).astype(np.float32)

    return disease_data


def select_sig_genes(
    adata,
    dist_key="dist",
    label_key=None,
    topk=100,
    genes_use=None,
    expr_prop_cutoff=0.1,
    ntop_max=200,
    overlap_max=1,
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

    ref_sig_list = {}

    for label in cell_IDs:
        idx = np.where(cell_label_vec == label)[0]
        expr_prop = (expr_data[idx, :] > 0).mean(axis=0)
        distce_vals = distce_data[idx, :].mean(axis=0)

        mask = expr_prop > expr_prop_cutoff
        filtered_indices = np.where(mask)[0]

        if filtered_indices.size == 0:
            ref_sig_list[label] = {"genes": [], "gene_index": []}
            continue

        sorted_idx = np.argsort(distce_vals[filtered_indices])[:topk]
        final_idx = filtered_indices[sorted_idx]

        gene_names = genes_use[final_idx]

        ref_sig_list[label] = {
            "genes": gene_names.tolist(),
            "gene_index": final_idx.tolist(),
        }

    return ref_sig_list


def calculate_signature_genes(
    adata,
    target_label="Pathological regions",
    dist_key="cell_gene_dist",
    label_key="pred_label",
    topk=100,
    n_print=10,
    **selection_kwargs,
):
    """Calculate signature genes for one label and print the leading genes."""
    if n_print < 0:
        raise ValueError("n_print must be non-negative.")

    signature_results = select_sig_genes(
        adata=adata,
        dist_key=dist_key,
        label_key=label_key,
        topk=topk,
        **selection_kwargs,
    )
    if target_label not in signature_results:
        available_labels = ", ".join(map(str, signature_results))
        raise KeyError(
            f"Label '{target_label}' was not found. Available labels: "
            f"{available_labels}."
        )

    signature_genes = signature_results[target_label]["genes"]
    n_display = min(n_print, len(signature_genes))
    signature_table = pd.DataFrame(
        {
            "rank": np.arange(1, n_display + 1),
            "gene": signature_genes[:n_display],
        }
    )
    print(f"Top {n_display} signature genes for {target_label}:")
    print(signature_table.to_string(index=False))
    return signature_genes


def run_signature_enrichment(
    genes,
    gene_sets=None,
    group_name="Pathological regions",
    organism="human",
    adjusted_p_cutoff=0.05,
    min_count=5,
):
    """Run Enrichr and retain significant terms with sufficient gene overlap."""
    import gseapy as gp

    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2025",
            "MSigDB_Hallmark_2020",
        ]

    gene_list = pd.Series(genes).dropna().astype(str).drop_duplicates().tolist()
    if not gene_list:
        raise ValueError("At least one gene is required for enrichment analysis.")

    enrichment = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        cutoff=adjusted_p_cutoff,
    )
    results = enrichment.results.copy()
    results["Adjusted P-value"] = pd.to_numeric(
        results["Adjusted P-value"], errors="coerce"
    )
    results = results.loc[
        results["Adjusted P-value"] < adjusted_p_cutoff
    ].copy()
    if results.empty:
        return results

    overlap = results["Overlap"].astype(str).str.split("/", expand=True)
    overlap_count = pd.to_numeric(overlap[0], errors="coerce")
    overlap_total = pd.to_numeric(overlap[1], errors="coerce")
    results = results.loc[overlap_count >= min_count].copy()
    if results.empty:
        return results

    results["Group"] = group_name
    results["Count"] = overlap_count.loc[results.index].astype(int)
    results["GeneRatio"] = results["Count"] / overlap_total.loc[results.index]
    results["Term_Name"] = results["Term"].map(_extract_enrichment_term_name)
    results["Term_Name_wrap"] = results["Term_Name"].map(
        lambda term: _wrap_enrichment_label(term, width=32)
    )
    results["neglog10_adjP"] = -np.log10(
        results["Adjusted P-value"].clip(lower=1e-300)
    )
    return results.reset_index(drop=True)


def _extract_enrichment_term_name(term):
    term = str(term)
    if "(" in term:
        return term[: term.rfind("(")].strip()
    return term


def _wrap_enrichment_label(text, width=42):
    import textwrap

    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _scale_enrichment_marker(value, value_min, value_max, size_min, size_max):
    if value_min == value_max:
        return (size_min + size_max) / 2
    return size_min + (value - value_min) / (value_max - value_min) * (
        size_max - size_min
    )


def plot_enrichment_bubble(
    enrichment_results,
    go_library="GO_Biological_Process_2025",
    hallmark_library="MSigDB_Hallmark_2020",
    save=None,
    dpi=300,
    show=True,
):
    """Plot significant GO and Hallmark enrichment terms as a bubble chart."""
    from matplotlib.patches import Patch

    required_columns = {
        "Gene_set",
        "Term",
        "GeneRatio",
        "Count",
        "Adjusted P-value",
    }
    missing_columns = required_columns.difference(enrichment_results.columns)
    if missing_columns:
        raise KeyError(
            "Missing enrichment columns: " + ", ".join(sorted(missing_columns))
        )
    if enrichment_results.empty:
        raise ValueError("No enrichment results are available for plotting.")

    plot_data = pd.concat(
        [
            enrichment_results.loc[enrichment_results["Gene_set"] == go_library],
            enrichment_results.loc[
                enrichment_results["Gene_set"] == hallmark_library
            ],
        ],
        ignore_index=True,
    )
    if plot_data.empty:
        raise ValueError("No GO or Hallmark enrichment results are available.")
    plot_data["GeneRatio"] = pd.to_numeric(plot_data["GeneRatio"], errors="coerce")
    plot_data["Count"] = pd.to_numeric(plot_data["Count"], errors="coerce")
    plot_data["Adjusted P-value"] = pd.to_numeric(
        plot_data["Adjusted P-value"], errors="coerce"
    )
    numeric_columns = ["GeneRatio", "Count", "Adjusted P-value"]
    if plot_data[numeric_columns].isna().any().any():
        raise ValueError("Enrichment plotting columns must contain numeric values.")

    if "Term_Name" in plot_data:
        display_terms = plot_data["Term_Name"].astype(str)
    else:
        display_terms = plot_data["Term"].map(_extract_enrichment_term_name)
    plot_data["Label"] = display_terms.map(
        lambda term: _wrap_enrichment_label(term, width=42)
    )
    plot_data["neglog10_adjP"] = -np.log10(
        plot_data["Adjusted P-value"].clip(lower=1e-300)
    )

    y_step = 1.15
    plot_data["y"] = np.arange(len(plot_data) - 1, -1, -1) * y_step + 1
    count_min = plot_data["Count"].min()
    count_max = plot_data["Count"].max()
    plot_data["Size"] = plot_data["Count"].map(
        lambda count: _scale_enrichment_marker(
            count, count_min, count_max, 110, 520
        )
    )

    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    figure.subplots_adjust(left=0.42, right=0.73, bottom=0.11, top=0.96)
    cmap = LinearSegmentedColormap.from_list(
        "single_purple_gradient", ["#F2EEF7", "#6A4C93"]
    )
    color_min = plot_data["neglog10_adjP"].min()
    color_max = plot_data["neglog10_adjP"].max()
    if np.isclose(color_min, color_max):
        color_min -= 0.5
        color_max += 0.5
    norm = Normalize(vmin=color_min, vmax=color_max)

    scatter = axis.scatter(
        plot_data["GeneRatio"],
        plot_data["y"],
        s=plot_data["Size"],
        c=plot_data["neglog10_adjP"],
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    axis.set_yticks(plot_data["y"])
    axis.set_yticklabels(
        plot_data["Label"], fontsize=10, rotation=0, ha="right", va="center"
    )
    axis.tick_params(axis="y", length=3, width=1.0, direction="out", pad=3)
    axis.set_ylim(
        plot_data["y"].min() - 0.55 * y_step,
        plot_data["y"].max() + 0.55 * y_step,
    )

    tick_max = np.ceil(plot_data["GeneRatio"].max() / 0.01) * 0.01
    tick_max = max(tick_max, 0.01)
    axis.set_xlim(-0.004, tick_max + 0.015)
    x_ticks = np.arange(0, tick_max + 0.02, 0.02)
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)
    axis.tick_params(axis="x", length=4, width=1.0, direction="out")

    go_mask = plot_data["Gene_set"] == go_library
    hallmark_mask = plot_data["Gene_set"] == hallmark_library
    if go_mask.any() and hallmark_mask.any():
        boundary = (
            plot_data.loc[go_mask, "y"].min()
            + plot_data.loc[hallmark_mask, "y"].max()
        ) / 2
        axis.axhline(
            boundary,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            zorder=2,
        )

    axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25, zorder=1)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("black")

    colorbar_axis = figure.add_axes([0.79, 0.72, 0.022, 0.18])
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=9, width=1.0, length=3)
    colorbar.outline.set_visible(True)
    colorbar.outline.set_linewidth(1.2)
    colorbar.outline.set_edgecolor("black")

    gene_set_colors = {go_library: "#8DAA91", hallmark_library: "#D9B382"}
    gene_set_handles = [
        Patch(
            facecolor=gene_set_colors[go_library],
            edgecolor="black",
            label="GOBP",
            alpha=0.75,
        ),
        Patch(
            facecolor=gene_set_colors[hallmark_library],
            edgecolor="black",
            label="Hallmark",
            alpha=0.75,
        ),
    ]
    figure.legend(
        handles=gene_set_handles,
        title="Gene set",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.61),
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=1.5,
        handletextpad=0.8,
    )

    legend_counts = [5, 7, 9]
    size_handles = [
        axis.scatter(
            [],
            [],
            s=_scale_enrichment_marker(count, count_min, count_max, 55, 200),
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
        )
        for count in legend_counts
    ]
    figure.legend(
        size_handles,
        [str(count) for count in legend_counts],
        title="Gene count",
        scatterpoints=1,
        frameon=False,
        fontsize=9.5,
        title_fontsize=10.5,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.34),
        borderaxespad=0.0,
        labelspacing=1.0,
        handletextpad=1.0,
    )

    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return figure


def plot_gene_expression(
    adata,
    genes,
    spatial_key="spatial",
    n_columns=3,
    point_size=25,
    cmap=None,
    flip_y=True,
    save=None,
    dpi=150,
    show=True,
):
    """Plot spatial expression for selected genes."""
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates '{spatial_key}' were not found in adata.obsm."
        )
    if not genes:
        raise ValueError("At least one gene must be provided.")
    if n_columns <= 0:
        raise ValueError("n_columns must be positive.")

    missing_genes = [gene for gene in genes if gene not in adata.var_names]
    if missing_genes:
        raise KeyError(f"Genes not found in adata.var_names: {', '.join(missing_genes)}")

    coordinates = np.asarray(adata.obsm[spatial_key]).copy()
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"'{spatial_key}' must contain at least two coordinate columns.")
    if flip_y:
        coordinates[:, 1] *= -1

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "signature_expression", ["#636fa4", "#e8cbc0", "#ff7e5f"]
        )

    n_columns = min(n_columns, len(genes))
    n_rows = int(np.ceil(len(genes) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(n_columns * 3, n_rows * 2.5),
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, gene in zip(axes, genes):
        expression = adata[:, gene].X
        if sp.issparse(expression):
            expression = expression.toarray()
        expression = np.asarray(expression).ravel()

        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=expression,
            cmap=cmap,
            s=point_size,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(gene, fontsize=12, fontstyle="italic")
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()
        figure.colorbar(scatter, ax=axis, shrink=0.8, pad=0.02)

    for axis in axes[len(genes) :]:
        axis.set_visible(False)

    figure.tight_layout()
    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None

    return figure


def load_tc_le_gene_sets(
    gene_table,
    gene_column="Gene",
    score_column="sample_2",
):
    """Load TC and LE gene sets from a signed gene-score table."""
    if isinstance(gene_table, (str, os.PathLike)):
        gene_table = pd.read_csv(gene_table)
    elif isinstance(gene_table, pd.DataFrame):
        gene_table = gene_table.copy()
    else:
        raise TypeError("gene_table must be a path or a pandas DataFrame.")

    required_columns = {gene_column, score_column}
    missing_columns = required_columns.difference(gene_table.columns)
    if missing_columns:
        raise KeyError(
            "Missing TC/LE gene-table columns: "
            + ", ".join(sorted(missing_columns))
        )

    gene_table[score_column] = pd.to_numeric(
        gene_table[score_column], errors="coerce"
    )
    gene_table = gene_table.dropna(subset=[gene_column, score_column]).copy()
    gene_table[gene_column] = gene_table[gene_column].astype(str)
    gene_table = gene_table.drop_duplicates(subset=gene_column)

    gene_sets = {
        "TC": gene_table.loc[
            gene_table[score_column] > 0, gene_column
        ].tolist(),
        "LE": gene_table.loc[
            gene_table[score_column] <= 0, gene_column
        ].tolist(),
    }
    if not gene_sets["TC"] or not gene_sets["LE"]:
        raise ValueError("Both TC and LE gene sets must contain at least one gene.")
    return gene_sets


def calculate_distance_enrichment(
    adata,
    gene_sets,
    dist_key="cell_gene_dist",
    n_permutations=5000,
    min_genes=5,
    seed=123,
    fdr_method="fdr_bh",
    alpha=0.05,
    verbose=True,
):
    """Calculate observation-level gene-set enrichment from cell-gene distances."""
    if dist_key not in adata.layers:
        raise KeyError(f"adata.layers['{dist_key}'] was not found.")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    if min_genes <= 0:
        raise ValueError("min_genes must be positive.")

    distance_matrix = adata.layers[dist_key]
    n_observations, n_genes = distance_matrix.shape
    gene_names = adata.var_names.astype(str).to_numpy()
    gene_index = {gene: index for index, gene in enumerate(gene_names)}

    matched_indices = {}
    matched_genes = {}
    for name, genes in gene_sets.items():
        unique_genes = list(dict.fromkeys(map(str, genes)))
        current_genes = [gene for gene in unique_genes if gene in gene_index]
        if len(current_genes) >= min_genes:
            matched_genes[str(name)] = current_genes
            matched_indices[str(name)] = np.array(
                [gene_index[gene] for gene in current_genes], dtype=int
            )

    if not matched_indices:
        raise ValueError(
            "No gene sets retained enough genes after matching adata.var_names."
        )

    def sum_selected_genes(indices):
        values = distance_matrix[:, indices].sum(axis=1)
        if sp.issparse(values):
            values = values.toarray()
        return np.asarray(values).ravel().astype(float)

    observed = pd.DataFrame(
        {
            name: sum_selected_genes(indices)
            for name, indices in matched_indices.items()
        },
        index=adata.obs_names,
    )
    pathway_names = list(matched_indices)
    p_values = pd.DataFrame(
        np.nan, index=adata.obs_names, columns=pathway_names
    )
    z_scores = pd.DataFrame(
        np.nan, index=adata.obs_names, columns=pathway_names
    )

    pathways_by_size = {}
    for name, indices in matched_indices.items():
        pathways_by_size.setdefault(len(indices), []).append(name)

    rng = np.random.default_rng(seed)
    size_iterator = pathways_by_size.items()
    if verbose:
        size_iterator = tqdm(
            size_iterator,
            total=len(pathways_by_size),
            desc="Calculating distance enrichment",
        )

    for gene_set_size, names in size_iterator:
        null_distribution = np.empty(
            (n_observations, n_permutations), dtype=np.float32
        )
        for permutation_index in range(n_permutations):
            random_indices = rng.choice(
                n_genes, size=gene_set_size, replace=False
            )
            null_distribution[:, permutation_index] = sum_selected_genes(
                random_indices
            )

        null_mean = null_distribution.mean(axis=1)
        null_std = null_distribution.std(axis=1, ddof=1)
        null_std[null_std == 0] = np.nan

        for name in names:
            observed_values = observed[name].to_numpy(dtype=float)
            p_values[name] = (
                (null_distribution <= observed_values[:, None]).sum(axis=1) + 1
            ) / (n_permutations + 1)
            z_scores[name] = (null_mean - observed_values) / null_std

    adjusted_p_values = pd.DataFrame(
        np.nan, index=p_values.index, columns=p_values.columns
    )
    for observation in p_values.index:
        current_p_values = p_values.loc[observation]
        valid = current_p_values.notna()
        if valid.any():
            adjusted_p_values.loc[observation, valid] = multipletests(
                current_p_values.loc[valid].to_numpy(dtype=float),
                method=fdr_method,
            )[1]

    return {
        "p_values": p_values,
        "adjusted_p_values": adjusted_p_values,
        "significant": adjusted_p_values < alpha,
        "z_scores": z_scores,
        "observed_distances": observed,
        "matched_genes": matched_genes,
    }


def calculate_tc_le_enrichment(
    adata,
    gene_table,
    label_key="pred_label",
    pathological_label="Pathological regions",
    dist_key="cell_gene_dist",
    gene_column="Gene",
    score_column="sample_2",
    n_permutations=5000,
    min_genes=5,
    seed=123,
    alpha=0.05,
    output_score_key="TC_to_LE_score",
    verbose=True,
):
    """Calculate TC, LE, and TC-to-LE enrichment scores in pathological regions."""
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs['{label_key}'] was not found.")

    pathological_mask = adata.obs[label_key].astype(str) == pathological_label
    if not pathological_mask.any():
        raise ValueError(
            f"No observations were found for label '{pathological_label}'."
        )

    gene_sets = load_tc_le_gene_sets(
        gene_table=gene_table,
        gene_column=gene_column,
        score_column=score_column,
    )
    pathological_data = adata[pathological_mask].copy()
    results = calculate_distance_enrichment(
        adata=pathological_data,
        gene_sets=gene_sets,
        dist_key=dist_key,
        n_permutations=n_permutations,
        min_genes=min_genes,
        seed=seed,
        alpha=alpha,
        verbose=verbose,
    )

    required_programs = {"TC", "LE"}
    missing_programs = required_programs.difference(results["z_scores"].columns)
    if missing_programs:
        raise ValueError(
            "The following gene programs did not retain enough matched genes: "
            + ", ".join(sorted(missing_programs))
        )

    score_index = results["z_scores"].index
    tc_scores = results["z_scores"]["TC"]
    le_scores = results["z_scores"]["LE"]
    transition_scores = le_scores - tc_scores

    for key in ("TC_score", "LE_score", output_score_key):
        adata.obs[key] = np.nan
    adata.obs.loc[score_index, "TC_score"] = tc_scores
    adata.obs.loc[score_index, "LE_score"] = le_scores
    adata.obs.loc[score_index, output_score_key] = transition_scores

    results["transition_scores"] = transition_scores.rename(output_score_key)
    return results


def _significance_label(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_tc_le_violin(
    adata,
    score_key="TC_to_LE_score",
    group_key="CellType",
    predicted_label_key="pred_label",
    truth_label_key="truth_label",
    pathological_label="Pathological regions",
    group_order=None,
    comparisons=None,
    palette=None,
    save=None,
    dpi=300,
    show=True,
):
    """Plot TC-to-LE scores by ground-truth pathological tissue class."""
    required_columns = {
        score_key,
        group_key,
        predicted_label_key,
        truth_label_key,
    }
    missing_columns = required_columns.difference(adata.obs.columns)
    if missing_columns:
        raise KeyError(
            "Missing observation columns: " + ", ".join(sorted(missing_columns))
        )

    if group_order is None:
        group_order = ["core", "transitory", "edge"]
    if comparisons is None:
        comparisons = [("core", "transitory"), ("transitory", "edge")]
    if palette is None:
        palette = {
            "core": "#E09F3E",
            "transitory": "#C8553D",
            "edge": "#8E3B46",
        }

    plot_data = adata.obs.loc[
        adata.obs[predicted_label_key].astype(str).eq(pathological_label)
        & adata.obs[truth_label_key].astype(str).eq(pathological_label),
        [group_key, score_key],
    ].dropna()
    plot_data = plot_data.loc[plot_data[group_key].astype(str).isin(group_order)].copy()
    plot_data[group_key] = pd.Categorical(
        plot_data[group_key].astype(str), categories=group_order, ordered=True
    )

    group_counts = plot_data[group_key].value_counts()
    empty_groups = [group for group in group_order if group_counts.get(group, 0) == 0]
    if empty_groups:
        raise ValueError(
            "No observations were available for groups: "
            + ", ".join(empty_groups)
        )

    with sns.axes_style("white"):
        figure, axis = plt.subplots(figsize=(3.2, 3.0))
        sns.violinplot(
            data=plot_data,
            x=group_key,
            y=score_key,
            hue=group_key,
            order=group_order,
            hue_order=group_order,
            palette=palette,
            legend=False,
            width=0.4,
            inner="box",
            linewidth=1.2,
            ax=axis,
        )

    score_min = plot_data[score_key].min()
    score_max = plot_data[score_key].max()
    score_range = score_max - score_min
    if not np.isfinite(score_range) or score_range == 0:
        score_range = 1.0

    comparison_results = []
    bracket_step = score_range * 0.08
    bracket_height = score_range * 0.025
    for comparison_index, (group_one, group_two) in enumerate(comparisons):
        if group_one not in group_order or group_two not in group_order:
            raise ValueError("All comparison groups must be present in group_order.")
        values_one = plot_data.loc[
            plot_data[group_key] == group_one, score_key
        ].astype(float)
        values_two = plot_data.loc[
            plot_data[group_key] == group_two, score_key
        ].astype(float)
        statistic, p_value = stats.mannwhitneyu(
            values_one, values_two, alternative="two-sided"
        )
        significance = _significance_label(p_value)
        comparison_results.append(
            {
                "Group_1": group_one,
                "Group_2": group_two,
                "Mann_Whitney_U": statistic,
                "P_value": p_value,
                "Significance": significance,
            }
        )

        x_one = group_order.index(group_one)
        x_two = group_order.index(group_two)
        y_position = score_max + (comparison_index + 1) * bracket_step
        axis.plot(
            [x_one, x_one, x_two, x_two],
            [
                y_position,
                y_position + bracket_height,
                y_position + bracket_height,
                y_position,
            ],
            linewidth=1.2,
            color="black",
        )
        axis.text(
            (x_one + x_two) / 2,
            y_position + bracket_height,
            significance,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    axis.set_ylim(score_min - score_range * 0.10, score_max + score_range * 0.30)
    axis.set_xlabel("")
    axis.set_ylabel("TC-to-LE enrichment score", fontsize=11)
    axis.set_xticks(
        range(len(group_order)),
        labels=[group.capitalize() for group in group_order],
    )
    axis.tick_params(
        axis="both", direction="out", length=5, width=0.8, labelsize=11
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    figure.tight_layout()
    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return figure, pd.DataFrame(comparison_results)


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

            A_genes = set(A_dict[A_class]["genes"])
            B_genes = set(B_dict[B_class]["genes"])
            K = len(A_genes)
            M = len(B_genes)
            x = len(A_genes & B_genes)

            p_value = stats.hypergeom.sf(x - 1, N, K, M)
            p_values[i, j] = p_value
            raw_p_list.append(p_value)
            index_list.append((i, j))

    fdr_corrected = multipletests(raw_p_list, method="fdr_bh")[1]
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
            A_genes = set(A_dict[A_class]["genes"])
            B_genes = set(B_dict[B_class]["genes"])
            K = len(A_genes)
            M = len(B_genes)
            x = len(A_genes & B_genes)

            p_value = stats.hypergeom.sf(x - 1, N, K, M)
            p_values[i, j] = p_value

    p_values_flat = p_values.flatten()
    fdr_corrected = multipletests(p_values_flat, method="fdr_bh")[1]
    fdr_matrix = fdr_corrected.reshape(num_A, num_B)

    return p_values, fdr_matrix


def detection(
    adata,
    embed,
    section_ids,
    label_core="Disease Core",
    label_other="Surrounding Area",
    core_types=["Tumor"],
    celltype_key="cell_type",
    batch_key="batch",
    seed=123,
    neighbors=30,
    threshold=0.05,
    strategy="cluster",
):

    cond_section = section_ids[-1]
    adata_all = adata.copy()
    adata_all.obs["pred_label"] = label_other
    adata_all.obs["truth_label"] = label_other
    adata_all.uns["result"] = {}
    normal_mask = adata_all.obs[batch_key].isin(section_ids[:-1])
    cond_mask = adata_all.obs[batch_key] == section_ids[-1]
    adata_normal = adata_all[normal_mask]
    adata_cond = adata_all[cond_mask]

    X_all = adata_all.obsm[embed]
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1).fit(X_all)
    _, indices_all = nbrs.kneighbors(X_all)
    is_normal_all = np.array(adata_all.obs[batch_key].isin(section_ids[:-1]))

    if strategy == "cluster":
        adata_cond.obs["binary"] = search_res(
            adata_cond, embed, target_n=2, seed=seed
        ).astype(str)

        binary_labels = ["0", "1"]

        ratios = {}
        for binary_label in binary_labels:
            adata_cluster = adata_cond[adata_cond.obs["binary"] == binary_label]
            cluster_indices = np.where(
                adata_all.obs_names.isin(adata_cluster.obs_names)
            )[0]
            hit_flags = []

            for idx in cluster_indices:
                neighbor_indices = indices_all[idx][1:]
                has_normal = np.any(is_normal_all[neighbor_indices])
                hit_flags.append(int(has_normal))
            ratios[binary_label] = np.mean(hit_flags)
            if ratios[binary_label] <= threshold:
                cond_binary_cells = adata_cluster.obs_names
                adata_all.obs.loc[cond_binary_cells, "pred_label"] = label_core
        print(f"Detection completed with cluster reference-overlap ratios: {ratios}.")
        all_above = all(v > threshold for v in ratios.values())
    else:
        all_above = True

    if all_above or strategy == "individual":
        cond_indices = np.where(adata_all.obs[batch_key] == section_ids[-1])[0]
        pad_flags = []

        for idx in cond_indices:
            neighbor_indices = indices_all[idx][1:]
            has_normal = np.any(normal_mask[neighbor_indices])
            pad_flags.append(int(not has_normal))

        pad_cells = cond_indices[np.array(pad_flags) == 1]
        pad_cell_names = adata_all.obs_names[pad_cells].tolist()
        adata_all.obs.loc[pad_cell_names, "pred_label"] = label_core
        print("Detection completed at the observation level.")

    if celltype_key is not None:
        adata_all.obs.loc[
            adata_all.obs[celltype_key].isin(core_types), "truth_label"
        ] = label_core
        adata_batch = adata_all[adata_all.obs[batch_key] == cond_section].copy()
        truth = (
            adata_batch.obs["truth_label"].map({label_other: 0, label_core: 1}).values
        )
        pred = adata_batch.obs["pred_label"].map({label_other: 0, label_core: 1}).values
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
        metrics_df = pd.DataFrame(
            {
                "DTS": [DTS],
                "F1": [f1],
                "Balanced Accuracy": [bacc],
                "Accuracy": [acc],
                "Precision": [precision],
                "Recall": [recall],
                "specificity": [specificity],
            }
        )
        adata_all.uns["result"] = metrics_df

    return adata_all


def plot_detection_umap(
    adata,
    embed,
    section_id=None,
    batch_key="batch",
    label_key="pred_label",
    celltype_key=None,
    seed=123,
    point_size=18,
    label_palette=None,
    celltype_palette=None,
    save=None,
    dpi=150,
    show=True,
):
    if embed not in adata.obsm:
        raise KeyError(f"Embedding '{embed}' was not found in adata.obsm.")
    if label_key not in adata.obs:
        raise KeyError(f"Label column '{label_key}' was not found in adata.obs.")

    if section_id is None:
        plot_adata = adata.copy()
    else:
        if batch_key not in adata.obs:
            raise KeyError(f"Batch column '{batch_key}' was not found in adata.obs.")
        section_mask = adata.obs[batch_key] == section_id
        if not np.any(section_mask):
            raise ValueError(f"No observations were found for section '{section_id}'.")
        plot_adata = adata[section_mask].copy()

    color_keys = [label_key]
    if celltype_key is not None:
        if celltype_key not in plot_adata.obs:
            raise KeyError(
                f"Cell-type column '{celltype_key}' was not found in adata.obs."
            )
        color_keys.append(celltype_key)

    sc.pp.neighbors(plot_adata, use_rep=embed)
    sc.tl.umap(plot_adata, random_state=seed)

    region_palette = {
        "Healthy-like regions": "#6DBBD1",
        "Pathological regions": "#B6473F",
    }
    if label_palette is not None:
        region_palette.update(label_palette)

    right_margin = 0.82 if len(color_keys) > 1 else 0.72
    figure, axes = plt.subplots(
        1,
        len(color_keys),
        figsize=(5.0 * len(color_keys), 4.2),
        squeeze=False,
    )
    axes = axes.ravel()
    coordinates = plot_adata.obsm["X_umap"]

    for axis, color_key in zip(axes, color_keys):
        labels = plot_adata.obs[color_key].astype(str)
        categories = sorted(labels.unique(), key=_label_sort_key)

        if color_key == label_key:
            palette = {
                category: region_palette.get(category, "#7F7F7F")
                for category in categories
            }
            panel_title = "Predicted regions"
        else:
            if celltype_palette is None:
                fallback_colors = sns.color_palette("husl", n_colors=len(categories))
                palette = dict(zip(categories, fallback_colors))
            else:
                palette = {
                    category: celltype_palette.get(category, "#7F7F7F")
                    for category in categories
                }
            panel_title = str(color_key)

        for category in categories:
            category_mask = (labels == category).to_numpy()
            axis.scatter(
                coordinates[category_mask, 0],
                coordinates[category_mask, 1],
                s=point_size,
                c=[palette[category]],
                edgecolors="none",
                rasterized=True,
                label=category,
            )

        axis.set_title(panel_title, fontsize=14, pad=10)
        axis.set_xlabel("UMAP 1", fontsize=10)
        axis.set_ylabel("UMAP 2", fontsize=10)
        axis.set_box_aspect(1)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
        axis.legend(
            title=panel_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=2 if len(categories) > 12 else 1,
            borderaxespad=0.0,
        )

    figure.subplots_adjust(right=right_margin, wspace=0.9)

    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return figure


def _prepare_he_overlay(
    adata,
    image_path,
    section_id,
    batch_key,
    spatial_key,
    coordinate_scale,
    crop_margin,
):
    """Load and crop an H&E image to the scaled spatial coordinates."""
    image_path = os.fspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"H&E image was not found: {image_path}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] was not found.")
    if crop_margin < 0:
        raise ValueError("crop_margin must be non-negative.")

    if section_id is None:
        plot_adata = adata
    else:
        if batch_key not in adata.obs:
            raise KeyError(f"adata.obs['{batch_key}'] was not found.")
        section_mask = adata.obs[batch_key].astype(str) == str(section_id)
        if not section_mask.any():
            raise ValueError(f"No observations were found for section '{section_id}'.")
        plot_adata = adata[section_mask]

    image = plt.imread(image_path)
    if image.ndim not in (2, 3):
        raise ValueError("The H&E image must be a two- or three-dimensional array.")

    coordinates = np.asarray(plot_adata.obsm[spatial_key], dtype=float).copy()
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must contain at least two columns."
        )

    scale = np.asarray(coordinate_scale, dtype=float)
    if scale.ndim == 0:
        scale = np.repeat(scale, 2)
    if scale.shape != (2,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("coordinate_scale must contain one or two positive values.")
    coordinates[:, :2] *= scale

    image_height, image_width = image.shape[:2]
    x_min = max(0, int(np.floor(coordinates[:, 0].min() - crop_margin)))
    x_max = min(
        image_width,
        int(np.ceil(coordinates[:, 0].max() + crop_margin)) + 1,
    )
    y_min = max(0, int(np.floor(coordinates[:, 1].min() - crop_margin)))
    y_max = min(
        image_height,
        int(np.ceil(coordinates[:, 1].max() + crop_margin)) + 1,
    )
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "The scaled spatial coordinates do not overlap the H&E image."
        )

    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_coordinates = coordinates[:, :2] - np.array([x_min, y_min])
    return plot_adata, cropped_image, cropped_coordinates


def _resolve_he_point_size(
    coordinates,
    cropped_image_shape,
    figsize,
    point_size,
    point_scale,
):
    """Resolve a marker area from the median spatial nearest-neighbor distance."""
    if point_size is not None:
        if point_size <= 0:
            raise ValueError("point_size must be positive.")
        return float(point_size)
    if point_scale <= 0:
        raise ValueError("point_scale must be positive.")
    if coordinates.shape[0] < 2:
        return 20.0

    neighbor_model = NearestNeighbors(n_neighbors=2).fit(coordinates)
    distances, _ = neighbor_model.kneighbors(coordinates)
    median_neighbor_distance = float(np.median(distances[:, 1]))
    image_height, image_width = cropped_image_shape[:2]
    width_scale = float(figsize[0]) * 72 / image_width
    height_scale = float(figsize[1]) * 72 / image_height
    diameter_points = (
        median_neighbor_distance
        * min(width_scale, height_scale)
        * point_scale
    )
    return max(diameter_points**2, 1.0)


def _format_he_axis(axis, cropped_image, fill_figure=True):
    axis.set_xlim(0, cropped_image.shape[1])
    axis.set_ylim(cropped_image.shape[0], 0)
    axis.set_axis_off()
    axis.margins(0)
    if fill_figure:
        axis.set_position([0, 0, 1, 1])


def plot_prediction_on_he(
    adata,
    image_path,
    section_id=None,
    batch_key="batch",
    label_key="pred_label",
    spatial_key="spatial",
    label_palette=None,
    coordinate_scale=1.0,
    crop_margin=20,
    image_alpha=0.8,
    point_size=None,
    point_scale=0.85,
    point_alpha=1.0,
    figsize=(3, 3),
    show_legend=False,
    save=None,
    dpi=300,
    show=True,
    ground_truth_key=None,
    ground_truth_palette=None,
):
    """Overlay predictions and optional ground truth on H&E; figsize is per panel."""
    if label_key not in adata.obs:
        raise KeyError(f"adata.obs['{label_key}'] was not found.")
    if ground_truth_key is not None and ground_truth_key not in adata.obs:
        raise KeyError(f"adata.obs['{ground_truth_key}'] was not found.")
    plot_adata, cropped_image, cropped_coordinates = _prepare_he_overlay(
        adata=adata,
        image_path=image_path,
        section_id=section_id,
        batch_key=batch_key,
        spatial_key=spatial_key,
        coordinate_scale=coordinate_scale,
        crop_margin=crop_margin,
    )
    default_palette = {
        "Pathological regions": "#B6473F",
        "Healthy-like regions": "#6DBBD1",
    }
    if label_palette is not None:
        default_palette.update(label_palette)
    panels = [(label_key, default_palette, "SpaPath")]
    if ground_truth_key is not None:
        panels.append((ground_truth_key, ground_truth_palette or {}, "Ground Truth"))
    resolved_point_size = _resolve_he_point_size(
        coordinates=cropped_coordinates,
        cropped_image_shape=cropped_image.shape,
        figsize=figsize,
        point_size=point_size,
        point_scale=point_scale,
    )

    figure, axes = plt.subplots(
        1, len(panels), figsize=(figsize[0] * len(panels), figsize[1]), squeeze=False
    )
    if len(panels) > 1:
        figure.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.12)

    for axis, (key, palette, title) in zip(axes.ravel(), panels):
        labels = plot_adata.obs[key].astype(str)
        categories = sorted(labels.unique(), key=_label_sort_key)
        colors = labels.map(palette).fillna("#7F7F7F").to_numpy()
        axis.imshow(cropped_image, alpha=image_alpha)
        axis.scatter(
            cropped_coordinates[:, 0],
            cropped_coordinates[:, 1],
            c=colors,
            s=resolved_point_size,
            alpha=point_alpha,
            edgecolors="none",
            rasterized=True,
        )
        _format_he_axis(axis, cropped_image, fill_figure=len(panels) == 1)
        if len(panels) > 1:
            axis.set_title(title, fontsize=12, pad=8)

        if show_legend:
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=palette.get(category, "#7F7F7F"),
                    markeredgecolor="none",
                    markersize=5,
                    label=category,
                )
                for category in categories
            ]
            legend_options = (
                {
                    "loc": "upper center",
                    "bbox_to_anchor": (0.5, -0.02),
                    "ncol": 2,
                    "frameon": False,
                }
                if len(panels) > 1
                else {"loc": "lower left", "frameon": True, "framealpha": 0.8}
            )
            axis.legend(
                handles=legend_handles,
                fontsize=8,
                **legend_options,
            )

    if save is not None:
        figure.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
    if show:
        plt.show()

    return figure


def plot_score_on_he(
    adata,
    image_path,
    score_key,
    section_id=None,
    batch_key="batch",
    label_key="pred_label",
    focus_label="Pathological regions",
    spatial_key="spatial",
    coordinate_scale=1.0,
    crop_margin=20,
    image_alpha=0.8,
    background_color="#6DBBD1",
    cmap="Reds",
    vmin=None,
    vmax=None,
    point_size=None,
    point_scale=0.85,
    point_alpha=1.0,
    figsize=(3, 3),
    colorbar=False,
    save=None,
    dpi=300,
    show=True,
):
    """Overlay a continuous score for one predicted region on a cropped H&E image."""
    required_columns = {score_key, label_key}
    missing_columns = required_columns.difference(adata.obs.columns)
    if missing_columns:
        raise KeyError(
            "Missing observation columns: " + ", ".join(sorted(missing_columns))
        )

    plot_adata, cropped_image, cropped_coordinates = _prepare_he_overlay(
        adata=adata,
        image_path=image_path,
        section_id=section_id,
        batch_key=batch_key,
        spatial_key=spatial_key,
        coordinate_scale=coordinate_scale,
        crop_margin=crop_margin,
    )
    labels = plot_adata.obs[label_key].astype(str).to_numpy()
    scores = pd.to_numeric(plot_adata.obs[score_key], errors="coerce").to_numpy()
    focus_mask = labels == str(focus_label)
    valid_focus_mask = focus_mask & np.isfinite(scores)
    if not valid_focus_mask.any():
        raise ValueError(
            f"No finite '{score_key}' values were found for label '{focus_label}'."
        )

    resolved_point_size = _resolve_he_point_size(
        coordinates=cropped_coordinates,
        cropped_image_shape=cropped_image.shape,
        figsize=figsize,
        point_size=point_size,
        point_scale=point_scale,
    )
    focus_scores = scores[valid_focus_mask]
    if vmin is None:
        vmin = float(np.min(focus_scores))
    if vmax is None:
        vmax = float(np.max(focus_scores))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    normalization = Normalize(vmin=vmin, vmax=vmax)

    figure, axis = plt.subplots(figsize=figsize)
    axis.imshow(cropped_image, alpha=image_alpha)
    background_mask = ~focus_mask
    if background_mask.any():
        axis.scatter(
            cropped_coordinates[background_mask, 0],
            cropped_coordinates[background_mask, 1],
            c=background_color,
            s=resolved_point_size,
            alpha=point_alpha,
            edgecolors="none",
            rasterized=True,
        )
    score_scatter = axis.scatter(
        cropped_coordinates[valid_focus_mask, 0],
        cropped_coordinates[valid_focus_mask, 1],
        c=focus_scores,
        cmap=cmap,
        norm=normalization,
        s=resolved_point_size,
        alpha=point_alpha,
        edgecolors="none",
        rasterized=True,
    )
    _format_he_axis(axis, cropped_image)

    if colorbar:
        colorbar_artist = figure.colorbar(
            score_scatter,
            ax=axis,
            fraction=0.035,
            pad=0.01,
        )
        colorbar_artist.ax.tick_params(labelsize=8)

    if save is not None:
        figure.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
    if show:
        plt.show()

    return figure


def subclustering(
    adata_full,
    embed_key="cell_embed",
    label="pred_label",
    ifmerge=True,
    init_res=1.5,
    intopk=30,
    n_clusters=None,
):

    y_pred_dict = {}
    n_slices = len(np.unique(adata_full.obs["batch"].values))

    if n_clusters is None:
        n_clusters = []
        for i in range(n_slices):
            adata = sc.AnnData(adata_full.obsm[embed_key])
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, resolution=init_res, random_state=seed)
            y_pred_dict[i] = adata.obs["leiden"].astype(int).to_numpy()
            n_clusters.append(len(np.unique(y_pred_dict[i])))
            del adata
            print(
                f"There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}"
            )
    else:
        for i in range(self.n_slices):
            y_pred_dict[i] = search_res(
                adata=self.Batch_list[i],
                embed_key=self.pre_cellembed_key,
                target_n=n_clusters[i],
                seed=seed,
            )
            print(
                f"There are {len(np.unique(y_pred_dict[i]))} initial clusters for Slice - {i}"
            )

    y_combined_dict = {}
    genename = self.Batch_list[0].var_names

    if ifmerge == True:
        for i in range(self.n_slices):
            self.Batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()
            if "ST" in self.data_type[i]:
                s_gene_embed = gene_embed_weight(
                    self.Batch_list[i].X,
                    self.Batch_list[i].obsm[self.pre_cellembed_key],
                    self.Batch_list[i].obsm["graph"],
                )
            else:
                s_gene_embed = gene_embed_weight(
                    self.Batch_list[i].X,
                    self.Batch_list[i].obsm[self.pre_cellembed_key],
                )
            s_gene2cell_mat = cell_to_gene_pdistance(
                self.Batch_list[i].obsm[self.pre_cellembed_key], s_gene_embed
            )
            self.Batch_list[i].layers["dist"] = s_gene2cell_mat
            s_gene_list = select_sig_genes(
                self.Batch_list[i], label_key=self.cluster_key, topk=intopk
            )

            _, score_mat = self_compute_gene_pvalue(s_gene_list, genename.values)
            tri_indx = np.triu_indices_from(score_mat, k=1)
            upper_tri_values = score_mat[tri_indx]

            while any(upper_tri_values <= 0.05):
                min_idx = np.argmin(upper_tri_values)
                x = tri_indx[0][min_idx]
                y = tri_indx[1][min_idx]
                self.Batch_list[i].obs[self.cluster_key][
                    self.Batch_list[i].obs[self.cluster_key] == y
                ] = x
                self.Batch_list[i].obs[self.cluster_key] = LabelEncoder().fit_transform(
                    self.Batch_list[i].obs[self.cluster_key]
                )
                s_gene_list = select_sig_genes(
                    self.Batch_list[i], label_key=self.cluster_key, topk=intopk
                )
                _, score_mat = self_compute_gene_pvalue(s_gene_list, genename.values)
                tri_indx = np.triu_indices_from(score_mat, k=1)
                upper_tri_values = score_mat[tri_indx]
            y_combined_dict[i] = self.Batch_list[i].obs[self.cluster_key]
            print(
                f"There are {len(np.unique(y_combined_dict[i]))} clusters for Slice - {i} after merging."
            )
    else:
        for i in range(self.n_slices):
            self.Batch_list[i].obs[self.cluster_key] = y_pred_dict[i].copy()
            y_combined_dict[i] = self.Batch_list[i].obs[self.cluster_key]
    print(y_combined_dict)


def _safe_ccc_key(value):
    """Convert a label into a key that is safe for AnnData mappings."""
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value))
    return re.sub(r"_+", "_", value).strip("_")


def _cell_gene_distance_to_potential(distance_matrix, sigma=4.0):
    """Convert cell-gene distances to potentials with a Gaussian kernel."""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("potential_sigma must be positive.")

    if sp.issparse(distance_matrix):
        distance_matrix = distance_matrix.toarray()
    else:
        distance_matrix = np.asarray(distance_matrix)

    return np.exp(-(distance_matrix**2) / (2 * sigma**2))


def _score_ccc_lr_block(
    sender_positions,
    receiver_positions,
    spatial_indicator,
    ligand_potentials,
    receptor_potentials,
    lr_pair_names,
    return_cell_scores=True,
    return_cellcell_matrices=False,
):
    """Calculate ligand-receptor scores for one directed region pair."""
    adjacency = spatial_indicator[np.ix_(sender_positions, receiver_positions)]
    sender_ligands = ligand_potentials[sender_positions, :]
    receiver_receptors = receptor_potentials[receiver_positions, :]

    receiver_context = adjacency @ receiver_receptors
    total_scores = np.sum(sender_ligands * receiver_context, axis=0)

    if not return_cell_scores and not return_cellcell_matrices:
        return total_scores

    sender_scores = {}
    receiver_scores = {}
    if return_cell_scores:
        sender_score_matrix = sender_ligands * receiver_context
        sender_context = adjacency.T @ sender_ligands
        receiver_score_matrix = receiver_receptors * sender_context
        sender_scores = {
            pair_name: sender_score_matrix[:, index].copy()
            for index, pair_name in enumerate(lr_pair_names)
        }
        receiver_scores = {
            pair_name: receiver_score_matrix[:, index].copy()
            for index, pair_name in enumerate(lr_pair_names)
        }

    if not return_cellcell_matrices:
        return total_scores, sender_scores, receiver_scores

    cellcell_matrices = {
        pair_name: (
            sender_ligands[:, index, None]
            * receiver_receptors[None, :, index]
            * adjacency
        )
        for index, pair_name in enumerate(lr_pair_names)
    }
    return total_scores, sender_scores, receiver_scores, cellcell_matrices


def _run_ccc_permutation(
    permutation_id,
    role_labels,
    spatial_indicator,
    ligand_potentials,
    receptor_potentials,
    lr_pair_names,
    seed,
):
    """Run one region-label permutation while preserving group sizes."""
    rng = np.random.default_rng(seed + permutation_id)
    permuted_roles = rng.permutation(role_labels)
    positions = np.arange(len(role_labels))
    sender_positions = positions[permuted_roles == "sender"]
    receiver_positions = positions[permuted_roles == "receiver"]

    return _score_ccc_lr_block(
        sender_positions=sender_positions,
        receiver_positions=receiver_positions,
        spatial_indicator=spatial_indicator,
        ligand_potentials=ligand_potentials,
        receptor_potentials=receptor_potentials,
        lr_pair_names=lr_pair_names,
        return_cell_scores=False,
        return_cellcell_matrices=False,
    )


def estimate_spatial_scale(coords, spot_center_distance_um=100):
    """Estimate micrometers per pixel from median nearest-neighbor distance."""
    coordinates = np.asarray(coords)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("At least two spatial coordinates are required.")

    neighbor_model = NearestNeighbors(n_neighbors=2).fit(coordinates)
    distances, _ = neighbor_model.kneighbors(coordinates)
    median_neighbor_distance = float(np.median(distances[:, 1]))
    if median_neighbor_distance <= 0:
        raise ValueError("The median nearest-neighbor distance must be positive.")

    micrometers_per_pixel = spot_center_distance_um / median_neighbor_distance
    return micrometers_per_pixel, median_neighbor_distance


def get_ccc_platform_params(platform, adata=None):
    """Return spatial and permutation defaults for a supported platform."""
    platform = str(platform).lower()
    if platform == "cosmx":
        return {
            "q_um": 200,
            "um_per_pixel": 0.12028,
            "potential_sigma": 4.0,
            "p_adj_threshold": 0.05,
            "n_perms": 5000,
            "n_jobs": 8,
        }

    if platform == "visium":
        if adata is None:
            raise ValueError("adata is required when platform='visium'.")
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm['spatial'] is required for Visium data.")

        micrometers_per_pixel, neighbor_distance = estimate_spatial_scale(
            adata.obsm["spatial"], spot_center_distance_um=100
        )
        return {
            "q_um": 200,
            "um_per_pixel": micrometers_per_pixel,
            "potential_sigma": 4.0,
            "p_adj_threshold": 0.05,
            "n_perms": 5000,
            "n_jobs": 8,
            "spot_center_distance_um": 100,
            "visium_neighbor_distance_pixel": neighbor_distance,
        }

    raise ValueError("platform must be either 'cosmx' or 'visium'.")


class CellTypeCCC:
    """Calculate directed spatial ligand-receptor communication scores."""

    def __init__(
        self,
        adata,
        resource_name="cellchatdb",
        dist_key="cell_gene_dist",
        spatial_key="spatial",
        label_key="pred_label",
        labels=None,
        lr_pairs=None,
    ):
        if dist_key not in adata.layers:
            raise KeyError(f"adata.layers['{dist_key}'] was not found.")
        if spatial_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{spatial_key}'] was not found.")
        if label_key not in adata.obs:
            raise KeyError(f"adata.obs['{label_key}'] was not found.")

        self.adata = adata
        self.distance_matrix = adata.layers[dist_key]
        self.coordinates = np.asarray(adata.obsm[spatial_key])
        self.observation_labels = adata.obs[label_key].astype(str).to_numpy()
        self.gene_names = adata.var_names.astype(str).to_numpy()
        self.gene_index = {
            gene: index for index, gene in enumerate(self.gene_names)
        }

        if labels is None:
            self.labels = list(pd.unique(self.observation_labels))
        else:
            self.labels = [str(label) for label in labels]

        self.label_indices = {}
        for label in self.labels:
            indices = np.flatnonzero(self.observation_labels == label)
            if indices.size == 0:
                raise ValueError(f"No observations were found for label '{label}'.")
            self.label_indices[label] = indices

        if lr_pairs is None:
            resource = liana.resource.select_resource(resource_name=resource_name)
            all_lr_pairs = list(zip(resource["ligand"], resource["receptor"]))
        else:
            all_lr_pairs = [(str(ligand), str(receptor)) for ligand, receptor in lr_pairs]

        available_genes = set(self.gene_names)
        self.lr_pairs = [
            (ligand, receptor)
            for ligand, receptor in all_lr_pairs
            if ligand in available_genes and receptor in available_genes
        ]
        if not self.lr_pairs:
            raise ValueError("No ligand-receptor pairs matched adata.var_names.")

    def _resolve_target_pairs(self, target_pairs, include_reverse):
        if target_pairs == "all":
            return [
                (sender, receiver)
                for sender in self.labels
                for receiver in self.labels
                if sender != receiver
            ]

        resolved_pairs = []
        for sender, receiver in target_pairs:
            sender = str(sender)
            receiver = str(receiver)
            if sender not in self.label_indices:
                raise ValueError(f"Unknown sender label: '{sender}'.")
            if receiver not in self.label_indices:
                raise ValueError(f"Unknown receiver label: '{receiver}'.")
            resolved_pairs.append((sender, receiver))
            if include_reverse and sender != receiver:
                resolved_pairs.append((receiver, sender))

        return list(dict.fromkeys(resolved_pairs))

    def run(
        self,
        target_pairs="all",
        include_reverse=True,
        normalization=True,
        q_um=200,
        um_per_pixel=0.12028,
        potential_sigma=4.0,
        n_perms=5000,
        p_adj_threshold=0.05,
        seed=123,
        n_jobs=8,
        result_key_suffix="",
        score_lr_source="all",
        top_n_lr=20,
        store_cellcell_matrices=False,
        verbose=True,
    ):
        """Run CCC analysis and store direction-level results in the AnnData object."""
        if q_um <= 0 or um_per_pixel <= 0:
            raise ValueError("q_um and um_per_pixel must be positive.")
        if n_perms < 0:
            raise ValueError("n_perms must be non-negative.")

        directed_pairs = self._resolve_target_pairs(target_pairs, include_reverse)
        spatial_cutoff = q_um / um_per_pixel
        potential_matrix = _cell_gene_distance_to_potential(
            self.distance_matrix, sigma=potential_sigma
        )
        lr_pair_names = [
            f"{ligand}-{receptor}" for ligand, receptor in self.lr_pairs
        ]
        ligand_indices = np.array(
            [self.gene_index[ligand] for ligand, _ in self.lr_pairs], dtype=int
        )
        receptor_indices = np.array(
            [self.gene_index[receptor] for _, receptor in self.lr_pairs], dtype=int
        )
        summaries = []

        if verbose:
            print(
                f"Running CCC analysis for {len(directed_pairs)} directed label pairs "
                f"and {len(self.lr_pairs)} ligand-receptor pairs."
            )

        for sender_label, receiver_label in directed_pairs:
            sender_indices = self.label_indices[sender_label]
            receiver_indices = self.label_indices[receiver_label]
            n_sender = len(sender_indices)
            n_receiver = len(receiver_indices)
            pair_indices = np.concatenate([sender_indices, receiver_indices])
            sender_positions = np.arange(n_sender)
            receiver_positions = np.arange(n_sender, n_sender + n_receiver)
            role_labels = np.array(
                ["sender"] * n_sender + ["receiver"] * n_receiver
            )

            pair_coordinates = self.coordinates[pair_indices]
            spatial_indicator = (
                cdist(pair_coordinates, pair_coordinates, metric="euclidean")
                <= spatial_cutoff
            )
            ligand_potentials = potential_matrix[np.ix_(pair_indices, ligand_indices)]
            receptor_potentials = potential_matrix[
                np.ix_(pair_indices, receptor_indices)
            ]

            observed = _score_ccc_lr_block(
                sender_positions=sender_positions,
                receiver_positions=receiver_positions,
                spatial_indicator=spatial_indicator,
                ligand_potentials=ligand_potentials,
                receptor_potentials=receptor_potentials,
                lr_pair_names=lr_pair_names,
                return_cell_scores=True,
                return_cellcell_matrices=store_cellcell_matrices,
            )
            if store_cellcell_matrices:
                (
                    observed_totals,
                    sender_scores,
                    receiver_scores,
                    cellcell_matrices,
                ) = observed
            else:
                observed_totals, sender_scores, receiver_scores = observed
                cellcell_matrices = None

            statistics = pd.DataFrame(
                {
                    "Ligand_Receptor": lr_pair_names,
                    "Sender": sender_label,
                    "Receiver": receiver_label,
                    "Real_Sum": observed_totals,
                },
                index=lr_pair_names,
            )

            if n_perms > 0:
                iterator = range(n_perms)
                if verbose:
                    iterator = tqdm(
                        iterator,
                        desc=(
                            f"{_safe_ccc_key(sender_label)}_to_"
                            f"{_safe_ccc_key(receiver_label)}"
                        ),
                    )
                permutation_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_run_ccc_permutation)(
                        permutation_id=permutation_id,
                        role_labels=role_labels,
                        spatial_indicator=spatial_indicator,
                        ligand_potentials=ligand_potentials,
                        receptor_potentials=receptor_potentials,
                        lr_pair_names=lr_pair_names,
                        seed=seed,
                    )
                    for permutation_id in iterator
                )
                null_scores = np.vstack(permutation_scores)
                p_values = (
                    (null_scores >= observed_totals[None, :]).sum(axis=0) + 1
                ) / (n_perms + 1)
                statistics["P_value"] = p_values
                statistics["P_value_min_possible"] = 1 / (n_perms + 1)
                statistics["Null_mean"] = null_scores.mean(axis=0)
                statistics["Null_std"] = null_scores.std(axis=0)
                statistics["Null_median"] = np.median(null_scores, axis=0)
                statistics["Empirical_rank"] = (
                    observed_totals[None, :] > null_scores
                ).mean(axis=0)
                statistics["Enrichment"] = statistics["Real_Sum"] / (
                    statistics["Null_mean"] + 1e-12
                )
                statistics["P_adj"] = multipletests(
                    p_values, alpha=p_adj_threshold, method="fdr_bh"
                )[1]
                statistics = statistics.sort_values(
                    ["P_adj", "P_value", "Real_Sum"],
                    ascending=[True, True, False],
                )
                significant = statistics.loc[
                    statistics["P_adj"] < p_adj_threshold
                ].sort_values("Real_Sum", ascending=False)
            else:
                for column in (
                    "P_value",
                    "P_value_min_possible",
                    "Null_mean",
                    "Null_std",
                    "Null_median",
                    "Empirical_rank",
                    "Enrichment",
                    "P_adj",
                ):
                    statistics[column] = np.nan
                statistics = statistics.sort_values("Real_Sum", ascending=False)
                significant = statistics.iloc[0:0].copy()

            direction_key = (
                f"{_safe_ccc_key(sender_label)}_to_"
                f"{_safe_ccc_key(receiver_label)}{result_key_suffix}"
            )
            all_statistics_key = f"ccc_stats_all_{direction_key}"
            significant_statistics_key = f"ccc_stats_sig_{direction_key}"
            self.adata.uns[all_statistics_key] = statistics
            self.adata.uns[significant_statistics_key] = significant

            if score_lr_source == "significant":
                selected_pairs = significant.index.tolist()
            elif score_lr_source == "all":
                selected_pairs = statistics.index.tolist()
            elif score_lr_source == "top":
                selected_pairs = statistics.nlargest(
                    top_n_lr, "Real_Sum"
                ).index.tolist()
            else:
                raise ValueError(
                    "score_lr_source must be 'significant', 'all', or 'top'."
                )
            selected_pairs = [
                pair
                for pair in selected_pairs
                if pair in sender_scores and pair in receiver_scores
            ]

            full_sender_scores = np.zeros(self.adata.n_obs)
            full_receiver_scores = np.zeros(self.adata.n_obs)
            if selected_pairs:
                combined_sender_scores = np.sum(
                    [sender_scores[pair] for pair in selected_pairs], axis=0
                )
                combined_receiver_scores = np.sum(
                    [receiver_scores[pair] for pair in selected_pairs], axis=0
                )
                if normalization:
                    combined_sender_scores = _minmax_scale(combined_sender_scores)
                    combined_receiver_scores = _minmax_scale(combined_receiver_scores)
                full_sender_scores[sender_indices] = combined_sender_scores
                full_receiver_scores[receiver_indices] = combined_receiver_scores

            sender_score_key = f"ccc_sender_score_{direction_key}"
            receiver_score_key = f"ccc_receiver_score_{direction_key}"
            self.adata.obs[sender_score_key] = full_sender_scores
            self.adata.obs[receiver_score_key] = full_receiver_scores
            self.adata.uns[f"ccc_lr_sender_scores_{direction_key}"] = sender_scores
            self.adata.uns[f"ccc_lr_receiver_scores_{direction_key}"] = receiver_scores
            if store_cellcell_matrices:
                self.adata.uns[
                    f"ccc_cellcell_matrices_{direction_key}"
                ] = cellcell_matrices

            summaries.append(
                {
                    "Direction": f"{sender_label} -> {receiver_label}",
                    "N_sender_cells": n_sender,
                    "N_receiver_cells": n_receiver,
                    "N_LR_pairs": len(statistics),
                    "N_significant_LR_pairs": len(significant),
                    "Min_P_adj": statistics["P_adj"].min(),
                }
            )

            if verbose:
                print(
                    f"Completed {sender_label} -> {receiver_label}: "
                    f"{len(significant)} significant ligand-receptor pairs."
                )

        summary = pd.DataFrame(summaries)
        self.adata.uns[f"ccc_celltype_pair_summary{result_key_suffix}"] = summary
        return summary


def _minmax_scale(values):
    values = np.asarray(values, dtype=float)
    value_min = values.min()
    value_range = values.max() - value_min
    if value_range <= 1e-9:
        return np.zeros_like(values)
    return (values - value_min) / value_range


def run_ccc_analysis(
    adata,
    platform="cosmx",
    sender_label="Pathological regions",
    receiver_label="Healthy-like regions",
    label_key="pred_label",
    dist_key="cell_gene_dist",
    spatial_key="spatial",
    resource_name="cellchatdb",
    lr_pairs=None,
    include_reverse=True,
    seed=123,
    q_um=None,
    um_per_pixel=None,
    potential_sigma=None,
    n_perms=None,
    p_adj_threshold=None,
    n_jobs=None,
    **run_kwargs,
):
    """Run bidirectional CCC analysis for pathological and healthy-like regions."""
    parameters = get_ccc_platform_params(platform=platform, adata=adata)
    overrides = {
        "q_um": q_um,
        "um_per_pixel": um_per_pixel,
        "potential_sigma": potential_sigma,
        "n_perms": n_perms,
        "p_adj_threshold": p_adj_threshold,
        "n_jobs": n_jobs,
    }
    parameters.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    run_parameter_names = {
        "q_um",
        "um_per_pixel",
        "potential_sigma",
        "n_perms",
        "p_adj_threshold",
        "n_jobs",
    }
    run_parameters = {
        key: value for key, value in parameters.items() if key in run_parameter_names
    }

    analysis = CellTypeCCC(
        adata=adata,
        resource_name=resource_name,
        dist_key=dist_key,
        spatial_key=spatial_key,
        label_key=label_key,
        labels=[receiver_label, sender_label],
        lr_pairs=lr_pairs,
    )
    summary = analysis.run(
        target_pairs=[(sender_label, receiver_label)],
        include_reverse=include_reverse,
        seed=seed,
        **run_parameters,
        **run_kwargs,
    )
    adata.uns["ccc_parameters"] = {
        "platform": str(platform),
        "sender_label": str(sender_label),
        "receiver_label": str(receiver_label),
        "label_key": str(label_key),
        "dist_key": str(dist_key),
        "spatial_key": str(spatial_key),
        "resource_name": str(resource_name),
        "include_reverse": bool(include_reverse),
        "seed": int(seed),
        **parameters,
    }
    return summary


def _dist_to_potential(dist_matrix):
    if dist_matrix.size > 0:
        sigma = np.median(dist_matrix)
    else:
        sigma = 1.0

    if sigma == 0:
        sigma = 1e-6
    return np.exp(-(dist_matrix**2) / (2 * sigma**2))


def _run_single_permutation(
    seed,
    all_involved_indices,
    n_senders_count,
    lr_database,
    gene_map,
    potential_mat,
    spatial_weight_matrix,
):

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

        null_score = np.dot(
            vec_pseudo_sender.T, np.dot(spatial_weight_matrix, vec_pseudo_receiver)
        )
        current_perm_scores[f"{ligand}-{receptor}"] = null_score

    return current_perm_scores


def check_cluster_umap(adata_full, embed, celltype_key, section_ids, seed=123):
    results = []

    for section in section_ids:
        adata_batch = adata_full[adata_full.obs["batch"] == section].copy()

        umap_model = umap.UMAP(
            n_neighbors=15, min_dist=0.5, random_state=seed, verbose=False
        )
        adata_batch.obsm[f"{embed}_umap"] = umap_model.fit_transform(
            adata_batch.obsm[embed]
        )

        sc.pl.embedding(
            adata_batch, basis=f"{embed}_umap", color=["cluster", celltype_key]
        )

        y_true = adata_batch.obs[celltype_key].values
        y_pred = adata_batch.obs["cluster"].values


class CCC_Interact:
    def __init__(
        self,
        adata,
        direction="core_to_other",
        resource_name="consensus",
        dist_layer="cell_gene_dist",
        spatial_key="spatial",
        label_col="pred_label",
        core_label="Disease Core",
        other_label="Surrounding Area",
    ):

        try:
            plt.style.use("seaborn-v0_8-white")
        except OSError:
            plt.style.use("default")
            mpl.rcParams["axes.spines.top"] = False
            mpl.rcParams["axes.spines.right"] = False

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

        print(
            f"Initialized with {len(self.other_idx)} '{other_label}' cells and {len(self.core_idx)} '{core_label}' cells."
        )

        if len(self.core_idx) == 0:
            raise ValueError(f"No cells found for label '{pad_label}'.")

        self.gene_names = list(adata.var_names)
        lr_df = liana.resource.select_resource(resource_name=resource_name)
        self.lr_database = list(zip(lr_df["ligand"], lr_df["receptor"]))
        available_genes_set = set(self.gene_names)

        print(f"Total genes in your dataset: {len(available_genes_set)}")
        print(f"Original L-R pairs: {len(self.lr_database)}")

        filtered_lr_database = [
            (l, r)
            for l, r in self.lr_database
            if l in available_genes_set and r in available_genes_set
        ]

        print("-" * 30)
        print(f"Filtered L-R pairs: {len(filtered_lr_database)}")
        self.lr_database = filtered_lr_database

    def run_analysis(
        self,
        normalization=True,
        kernel_bandwidth=500,
        n_perms=1000,
        calc_stats=True,
        p_adj_threshold=0.05,
        seed=123,
        n_jobs=8,
        result_key_suffix="",
    ):

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
            target_directions = [self.direction]
        else:
            raise ValueError(
                "direction must be 'other_to_core', 'core_to_other', or 'both'"
            )

        print(f"Start Analyzing {len(self.lr_database)} L-R pairs...")

        nc_coords = self.coords[self.other_idx]
        pad_coords = self.coords[self.core_idx]

        cached_spatial_weights = {}
        self.potential_mat = _dist_to_potential(self.dist_mat)

        for current_dir in target_directions:
            print(f"\n=== Processing Direction: {current_dir} ===")

            if (
                current_dir == "core_to_other"
                and "other_to_core" in cached_spatial_weights
            ):
                print(
                    "Transposing existing spatial weight matrix from other_to_core..."
                )
                spatial_weight_matrix = cached_spatial_weights["other_to_core"].T
            else:
                print(f"Calculating full distance matrix for {current_dir}...")
                if current_dir == "other_to_core":
                    dist_matrix = cdist(nc_coords, pad_coords, metric="euclidean")
                else:
                    dist_matrix = cdist(pad_coords, nc_coords, metric="euclidean")

                spatial_weight_matrix = np.exp(
                    -(dist_matrix**2) / (2 * kernel_bandwidth**2)
                )

                cached_spatial_weights[current_dir] = spatial_weight_matrix
                del dist_matrix

            if current_dir == "other_to_core":
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
            pathway_stats["Real_Sum"] = list(real_pathway_sums.values())

            if calc_stats and n_perms > 0:
                print(
                    f"Running permutation test ({n_perms} perms) for {current_dir}..."
                )

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
                        spatial_weight_matrix,
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

                pathway_stats["P_value"] = p_values
                reject, pvals_corrected, _, _ = multipletests(
                    p_values, alpha=p_adj_threshold, method="fdr_bh"
                )
                pathway_stats["P_adj"] = pvals_corrected
                pathway_stats = pathway_stats.sort_values("P_adj")

                stat_key = f"ccc_stats_{current_dir}{result_key_suffix}"

                significant_df = pathway_stats[
                    pathway_stats["P_adj"] < p_adj_threshold
                ].copy()
                significant_df = significant_df.sort_values(
                    by="Real_Sum", ascending=False
                )

                self.adata.uns[stat_key] = significant_df

                if not significant_df.empty:
                    print(
                        f"[{current_dir}] Found {len(significant_df)} significant pathways."
                    )
                else:
                    print(f"[{current_dir}] No significant pathways found.")

            significant_pairs = significant_df.index.tolist()
            valid_vectors = [
                real_total_scores[pair]
                for pair in significant_pairs
                if pair in real_total_scores
            ]
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

                score_key = f"ccc_score_{current_dir}{result_key_suffix}"
                uns_pathway_key = f"ccc_pathways_{current_dir}{result_key_suffix}"

                print(
                    f"Saving results to obs['{score_key}'] and uns['{uns_pathway_key}']"
                )

                self.adata.obs[score_key] = full_scores
                self.adata.uns[uns_pathway_key] = pathway_contributions

        print("\nAnalysis Completed.")

    def get_sender_composition(
        self,
        base_score_key="ccc_score",
        result_key_suffix="",
        threshold=0.5,
        cell_type_col="CellType",
        save=None,
    ):

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
            target_directions = [self.direction]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        results_dict = {}

        for current_dir in target_directions:
            full_key = f"{base_score_key}_{current_dir}{result_key_suffix}"

            if full_key not in self.adata.obs:
                print(f"Warning: Key '{full_key}' not found. Skipping {current_dir}.")
                continue

            if current_dir == "other_to_core":
                target_label = self.other_label
                role_name = "NC (Sender)"
            else:
                target_label = self.core_label
                role_name = "PAD (Sender)"

            print(f"\n=== Analyzing Sender Composition: {current_dir} ===")
            print(f"Target Role: {role_name} | Threshold > {threshold}")

            mask = (self.adata.obs[self.label_col] == target_label) & (
                self.adata.obs[full_key] > threshold
            )

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

            stats_df = pd.DataFrame(
                {"Count": counts, "Percentage (%)": percentages.round(2)}
            )

            print("-" * 40)
            print(stats_df)
            print("-" * 40)

            uns_cell_key = f"ccc_top_senders_{current_dir}{result_key_suffix}"

            print(f"Saving results uns['{uns_cell_key}']")

            self.adata.uns[uns_cell_key] = stats_df

            if save:
                import os

                root, ext = os.path.splitext(save)
                final_save_name = f"{root}_{current_dir}{ext}"
                stats_df.to_csv(final_save_name, index_label=cell_type_col)

            results_dict[current_dir] = stats_df

        return results_dict


class CCC_plot:
    def __init__(
        self,
        adata,
        direction="other_to_core",
        resource_name="consensus",
        dist_layer="cell_gene_dist",
        spatial_key="spatial",
        label_col="pred_label",
        core_label="Disease Core",
        other_label="Surrounding Area",
    ):
        try:
            plt.style.use("seaborn-v0_8-white")
        except OSError:
            plt.style.use("default")
            mpl.rcParams["axes.spines.top"] = False
            mpl.rcParams["axes.spines.right"] = False

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

        print(
            f"Initialized with {len(self.other_idx)} '{self.other_label}' cells and {len(self.core_idx)} '{self.core_label}' cells."
        )

        if len(self.core_idx) == 0:
            raise ValueError(f"No cells found for label '{pad_label}'.")

    def plot_interaction(
        self,
        base_score_key="ccc_score",
        color_bg="#b8b0b0",
        cmap="YlOrRd",
        result_key_suffix="",
        point_size=5,
        title=None,
        base_size=4,
        dpi=120,
        save=None,
    ):

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
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

            if current_dir == "other_to_core":
                mask_bg = labels == self.core_label
                mask_fg = labels == self.other_label

                bg_color = color_bg
                label_bg = f"{self.core_label} (Receiver)"
                label_fg = f"{self.other_label} (Sender)"

            else:

                mask_bg = labels == self.other_label
                mask_fg = labels == self.core_label

                bg_color = color_bg
                label_bg = f"{self.other_label} (Receiver)"
                label_fg = f"{self.core_label} (Sender)"

            if np.sum(mask_bg) > 0:
                ax.scatter(
                    coords[mask_bg, 0],
                    coords[mask_bg, 1],
                    c=bg_color,
                    s=point_size * 1.1,
                    alpha=0.8,
                    edgecolors="none",
                    rasterized=True,
                    label=label_bg,
                )

            if np.sum(mask_fg) > 0:
                fg_coords = coords[mask_fg]
                fg_scores = scores[mask_fg]

                sort_idx = np.argsort(fg_scores)

                sc = ax.scatter(
                    fg_coords[sort_idx, 0],
                    fg_coords[sort_idx, 1],
                    c=fg_scores[sort_idx],
                    cmap=cmap,
                    s=point_size,
                    alpha=1.0,
                    edgecolors="none",
                    rasterized=True,
                    label=label_fg,
                )

                cbar = plt.colorbar(
                    sc, ax=ax, fraction=0.03, pad=0.005, aspect=30, shrink=0.8
                )
                cbar.set_label(
                    "Interaction Score for Senders",
                    rotation=270,
                    labelpad=20,
                    fontsize=10,
                    color="#555555",
                )
                cbar.ax.tick_params(labelsize=8, color="#888888")
                cbar.outline.set_linewidth(0.5)
                cbar.outline.set_edgecolor("#AAAAAA")

            if current_dir == "other_to_core":
                plot_title = "Sender (Surrounding Area) - Receiver (Disease Core)"
            else:
                plot_title = "Sender (Disease Core) - Receiver (Surrounding Area)"
            ax.set_title(plot_title, fontsize=14)

            ax.axis("off")
            ax.set_aspect("equal", "datalim")
            ax.invert_yaxis()

            if save:
                import os

                root, ext = os.path.splitext(save)
                final_save_name = f"{root}_{current_dir}{ext}"

                plt.savefig(
                    final_save_name, dpi=dpi, bbox_inches="tight", pad_inches=0.1
                )

            plt.show()
            plt.close()

    def plot_sender_distribution(
        self,
        color_map=None,
        celltype_key="CellType",
        result_key_suffix="",
        figsize=(6, 5.5),
        save=None,
    ):

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
            target_directions = [self.direction]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        for current_dir in target_directions:

            uns_cell_key = f"ccc_top_senders_{current_dir}{result_key_suffix}"

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
            if "Count" not in df_plot.columns:
                for col in df_plot.columns:
                    if (
                        pd.api.types.is_numeric_dtype(df_plot[col])
                        and col != celltype_key
                    ):
                        df_plot.rename(columns={col: "Count"}, inplace=True)
                        break

            df_plot[celltype_key] = df_plot[celltype_key].astype(str)

            df_plot = df_plot[df_plot["Count"] > 0].sort_values(
                by="Count", ascending=False
            )

            if df_plot.empty:
                continue

            unique_types = df_plot[celltype_key].unique()
            safe_palette = {}
            for ctype in unique_types:
                safe_palette[ctype] = color_map.get(ctype, "#808080")

            plt.figure(figsize=figsize)

            ax = sns.barplot(
                data=df_plot,
                x=celltype_key,
                y="Count",
                order=df_plot[celltype_key],
                palette=safe_palette,
                hue=celltype_key,
                legend=False,
                edgecolor="black",
                linewidth=1,
                zorder=3,
            )

            sender_role = (
                "Surrounding Area" if current_dir == "other_to_core" else "Disease Core"
            )
            plt.title(f"{sender_role} Sender Distribution", fontsize=14, pad=15)

            plt.xlabel("Cell Type", fontsize=12)
            plt.ylabel("Count", fontsize=12)

            plt.xticks(rotation=45, ha="right")
            plt.tick_params(
                axis="both", which="major", direction="out", length=5, width=1
            )

            y_max = df_plot["Count"].max()
            for i, v in enumerate(df_plot["Count"]):
                plt.text(
                    i,
                    v + (y_max * 0.005),
                    str(v),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )

            plt.tight_layout()

            if save:
                root, ext = os.path.splitext(save)
                final_save = f"{root}_{current_dir}_distribution{ext}"
                plt.savefig(final_save, dpi=300, bbox_inches="tight")
                print(f"Saved distribution plot to {final_save}")

            plt.show()

    def plot_top_pathways(
        self,
        result_key_suffix="",
        top_n=6,
        n_cols=3,
        point_size=5,
        percentile=99,
        alpha=0.8,
        base_size=4,
        dpi=120,
        save=None,
    ):

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
            target_directions = [self.direction]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        for current_dir in target_directions:
            stats_key = f"ccc_stats_{current_dir}{result_key_suffix}"

            if stats_key not in self.adata.uns:
                print(
                    f"Warning: Stats key '{stats_key}' not found. Run analysis first."
                )
                continue

            stats_df = self.adata.uns[stats_key]
            if stats_df.empty:
                print(f"No significant pathways found for {current_dir}.")
                continue

            top_pairs = stats_df.index[:top_n].tolist()
            n_pairs = len(top_pairs)
            print(f"\nPlotting top {n_pairs} pathways for {current_dir}...")

            if current_dir == "other_to_core":
                mask_sender = self.labels == self.other_label
                mask_receiver = self.labels == self.core_label
                sender_name, receiver_name = "Surrounding Area", "Disease Core"
            else:
                mask_sender = self.labels == self.core_label
                mask_receiver = self.labels == self.other_label
                sender_name, receiver_name = "Disease Core", "Surrounding Area"

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
                use_raw = (
                    self.adata.raw is not None and gene in self.adata.raw.var_names
                )
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
                ligand, receptor = pair_name.split("-")

                sender_expr = get_expr(ligand, mask_sender)
                sender_coords = coords[mask_sender]

                receiver_expr = get_expr(receptor, mask_receiver)
                receiver_coords = coords[mask_receiver]

                vmax_s = (
                    np.percentile(sender_expr, percentile)
                    if np.max(sender_expr) > 0
                    else 1
                )
                vmax_r = (
                    np.percentile(receiver_expr, percentile)
                    if np.max(receiver_expr) > 0
                    else 1
                )

                norm_sender = Normalize(vmin=0, vmax=vmax_s)
                norm_receiver = Normalize(vmin=0, vmax=vmax_r)

                sc_r = ax.scatter(
                    receiver_coords[:, 0],
                    receiver_coords[:, 1],
                    c=receiver_expr,
                    cmap=cm.Blues,
                    norm=norm_receiver,
                    s=point_size,
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                )

                sc_s = ax.scatter(
                    sender_coords[:, 0],
                    sender_coords[:, 1],
                    c=sender_expr,
                    cmap=cm.Reds,
                    norm=norm_sender,
                    s=point_size,
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                )

                ax.set_aspect("equal")
                ax.axis("off")
                ax.invert_yaxis()

                axins_top = inset_axes(
                    ax,
                    width="80%",
                    height="5%",
                    loc="upper center",
                    bbox_to_anchor=(0, 0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )

                cbar_top = plt.colorbar(sc_s, cax=axins_top, orientation="horizontal")

                cbar_top.ax.xaxis.set_ticks_position("top")
                cbar_top.ax.xaxis.set_label_position("top")
                cbar_top.set_label(
                    f"{ligand} (Sender: {sender_name})",
                    size=9,
                    labelpad=5,
                    color="#8B0000",
                    fontweight="bold",
                )
                cbar_top.ax.tick_params(labelsize=7, colors="#8B0000")
                cbar_top.outline.set_edgecolor("#FFCCCC")

                axins_bottom = inset_axes(
                    ax,
                    width="80%",
                    height="5%",
                    loc="lower center",
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )

                cbar_bot = plt.colorbar(
                    sc_r, cax=axins_bottom, orientation="horizontal"
                )

                cbar_bot.ax.xaxis.set_ticks_position("bottom")
                cbar_bot.ax.xaxis.set_label_position("bottom")
                cbar_bot.set_label(
                    f"{receptor} (Receiver: {receiver_name})",
                    size=9,
                    labelpad=5,
                    color="#00008B",
                    fontweight="bold",
                )
                cbar_bot.ax.tick_params(labelsize=7, colors="#00008B")
                cbar_bot.outline.set_edgecolor("#CCCCFF")

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout(h_pad=3.0, w_pad=1.0)

            plt.subplots_adjust(hspace=0.4)

            if save:
                import os

                root, ext = os.path.splitext(save)
                final_save_name = f"{root}_{current_dir}_top{top_n}{ext}"
                plt.savefig(final_save_name, dpi=dpi, bbox_inches="tight")

            plt.show()
            plt.close()

    def plot_bubble_pathways(
        self,
        top_n=10,
        result_key_suffix="",
        figsize=(6, 5),
        bubble_color="#B0C4DE",
        save=None,
    ):

        FONT_TITLE = 14
        FONT_LABEL = 12
        FONT_TICK = 11
        FONT_LEGEND = 10

        if self.direction == "both":
            target_directions = ["other_to_core", "core_to_other"]
        elif self.direction in ["other_to_core", "core_to_other"]:
            target_directions = [self.direction]
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        for current_dir in target_directions:

            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

            stats_key = f"ccc_stats_{current_dir}{result_key_suffix}"

            if stats_key not in self.adata.uns:
                ax.text(0.5, 0.5, "No stats found", ha="center", va="center")
            else:
                df = self.adata.uns[stats_key]

                df_plot = df.head(top_n).sort_values("Real_Sum", ascending=True).copy()

                if df_plot.empty:
                    ax.text(
                        0.5, 0.5, "No significant pathways", ha="center", va="center"
                    )
                else:
                    y_pos = range(len(df_plot))
                    x_vals = df_plot["Real_Sum"].values
                    p_vals = df_plot["P_adj"].values

                    x_min, x_max = x_vals.min(), x_vals.max()
                    x_range = x_max - x_min
                    padding = (
                        x_range * 0.15
                        if x_range > 0
                        else (x_max * 0.1 if x_max > 0 else 1.0)
                    )
                    ax.set_xlim(x_min - padding, x_max + padding)

                    SIZE_LEVELS = {
                        "p_0.001": 300,
                        "p_0.01": 120,
                        "p_0.05": 50,
                        "ns": 20,
                    }

                    sizes = []
                    for p in p_vals:
                        if p < 0.001:
                            sizes.append(SIZE_LEVELS["p_0.001"])
                        elif p < 0.01:
                            sizes.append(SIZE_LEVELS["p_0.01"])
                        elif p < 0.05:
                            sizes.append(SIZE_LEVELS["p_0.05"])
                        else:
                            sizes.append(SIZE_LEVELS["ns"])

                    ax.scatter(
                        x_vals,
                        y_pos,
                        s=sizes,
                        c=bubble_color,
                        edgecolors="black",
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=2,
                    )

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(df_plot.index, fontsize=FONT_TICK)

                    ax.set_xlabel(
                        "Interaction Strength (Sender)", fontsize=FONT_LABEL, labelpad=8
                    )

                    title_txt = (
                        "Surrounding Area -> Disease Core"
                        if current_dir == "other_to_core"
                        else "Disease Core -> Surrounding Area"
                    )
                    ax.set_title(
                        f"{title_txt} (Top {top_n})",
                        fontsize=FONT_TITLE,
                        fontweight="bold",
                        pad=15,
                    )

                    for spine_name in ["top", "bottom", "left", "right"]:
                        ax.spines[spine_name].set_visible(True)
                        ax.spines[spine_name].set_color("black")
                        ax.spines[spine_name].set_linewidth(1.0)

                    ax.grid(False)

                    ax.tick_params(
                        axis="both",
                        which="major",
                        labelsize=FONT_TICK,
                        width=1,
                        length=5,
                        direction="out",
                    )
                    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
                    ax.tick_params(axis="y", left=True, right=False)

                    legend_elements = [
                        plt.scatter(
                            [],
                            [],
                            s=SIZE_LEVELS["p_0.001"],
                            c=bubble_color,
                            edgecolors="black",
                            alpha=0.9,
                            label="P < 0.001",
                        ),
                        plt.scatter(
                            [],
                            [],
                            s=SIZE_LEVELS["p_0.01"],
                            c=bubble_color,
                            edgecolors="black",
                            alpha=0.9,
                            label="P < 0.01",
                        ),
                        plt.scatter(
                            [],
                            [],
                            s=SIZE_LEVELS["p_0.05"],
                            c=bubble_color,
                            edgecolors="black",
                            alpha=0.9,
                            label="P < 0.05",
                        ),
                    ]

                    ax.legend(
                        handles=legend_elements,
                        title="Significance",
                        title_fontsize=FONT_LEGEND,
                        loc="lower right",
                        frameon=True,
                        edgecolor="black",
                        fontsize=FONT_LEGEND,
                        labelspacing=1.0,
                        borderpad=0.8,
                    )

            if save:
                import os

                root, ext = os.path.splitext(save)
                final_save = f"{root}_{current_dir}_bubble{ext}"
                plt.savefig(final_save, dpi=300, bbox_inches="tight")

            plt.show()
