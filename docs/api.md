# API reference

Import `spapath_model` and `spapath_utils` after adding the repository's `scripts/` directory to `sys.path`. This page describes the main workflow functions; full signatures and implementation are in the [source files](https://github.com/edawu11/SpaPath/tree/main/scripts).

## Preparation

| Function in `spapath_utils` | Purpose and result |
| --- | --- |
| `attach_image_embedding(adata, embedding_path)` | Align CSV rows to observation names and store `dim_*` columns in `.obsm["image_embedding"]`. Modifies and returns the supplied AnnData. |
| `preprocess(adata_list, adata_type_map, ...)` | Filter and normalize data, select shared highly variable genes, and return `(adata_full, disease_adata_all_genes)`. Keep the disease sample last. |
| `build_graph_GAT_plus(adata_full, adata_type_map, K=8, img_threshold=0.0)` | Build sample-specific graphs and return the combined AnnData. This is the graph builder used by the maintained notebooks. |

## Model

Create `spapath_model.Model(adata_full, adata_type_map, device="cuda", seed=123, ...)`, then call these methods in order:

| Method | Output |
| --- | --- |
| `initial_embedding()` | AnnData with the initial `pre_embed` embedding |
| `clustering(init_res=1.5, intopk=40)` | AnnData with cluster assignments, with merging enabled by default |
| `integrate(topk=40)` | AnnData with integrated `cell_embed`, gene embeddings, and cell–gene distances |

The parameter values shown for clustering and integration match the examples. They are explicit example settings, not all of the underlying function defaults. DEC fine-tuning is disabled by default in `Model`.

## Detection and disease data

`spapath_utils.detection(adata, embed, section_ids, ...)` returns an AnnData copy with `.obs["pred_label"]`. The last entry in `section_ids` identifies the disease sample. The maintained notebooks explicitly set:

```python
label_core="Pathological regions"
label_other="Healthy-like regions"
embed="cell_embed"
strategy="individual"
neighbors=30
threshold=0.05
```

The underlying function retains legacy label defaults, so pass the labels explicitly as above. `core_types` and `celltype_key` enable ground-truth comparison; set both to `None` when annotations are unavailable.

`spapath_utils.build_disease_data(adata_full, disease_adata_all_genes, disease_section)` aligns the detected observations to the all-gene disease data and copies their observation metadata and embeddings. By default it uses `pre_embed` and the disease graph to recalculate `.uns["gene_embed"]` and `.layers["cell_gene_dist"]`. Gene-expression plotting uses the copied spatial coordinates.

## Characterization

| Function in `spapath_utils` | Purpose |
| --- | --- |
| `calculate_signature_genes(adata, target_label="Pathological regions", topk=100, n_print=10)` | Return a gene list and print the leading genes; use the reconstructed disease dataset as input. |
| `run_signature_enrichment(genes, ...)` | Query Enrichr for GO/Hallmark enrichment and return terms passing adjusted-P-value and overlap filters. Requires network access. |
| `run_ccc_analysis(adata, platform="cosmx", include_reverse=True, ...)` | Calculate directed spatial ligand–receptor communication; both directions are included by default. Return a compact summary and store detailed results in the AnnData. |
| `calculate_tc_le_enrichment(adata, gene_table, ...)` | Calculate TC and LE scores in pathological regions, write `TC_score`, `LE_score`, and `TC_to_LE_score` to `.obs`, and return enrichment details. |

For TC-to-LE analysis, observations outside pathological regions receive missing score values. The gradient plot displays those healthy-like observations with a fixed background color.

## Plotting

| Function in `spapath_utils` | Plot |
| --- | --- |
| `plot_detection_umap(...)` | Detection UMAP with optional cell-type comparison and notebook-defined palettes |
| `plot_prediction_on_he(...)` | Predicted region labels over a cropped H&E image |
| `plot_score_on_he(adata, image_path, score_key, ...)` | A continuous score in the selected region over H&E; `cmap="Reds"` by default |
| `plot_gene_expression(adata, genes, ...)` | Spatial expression of selected genes |
| `plot_enrichment_bubble(...)` | GO/Hallmark enrichment bubble plot |
| `plot_tc_le_violin(...)` | TC-to-LE scores across tissue classes, with comparison statistics |

The H&E functions accept `coordinate_scale`, `figsize`, `point_size`, `point_scale`, and `save`. With `point_size=None`, marker diameter is derived from median nearest-neighbor spacing; `point_scale=0.85` and `figsize=(3, 3)` are the defaults. To reuse a figure without displaying it, pass `show=False`. The [notebooks](tutorials.md) contain the complete plotting calls for each dataset.
