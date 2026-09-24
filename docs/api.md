# Core API

The notebooks import `spapath_model` and `spapath_utils` from `scripts/`. See the [tutorials](tutorials.md) for complete calls and parameter settings.

## Main workflow

| Function | Purpose |
| --- | --- |
| `spapath_utils.attach_image_embedding` | Align image embeddings to observation names. |
| `spapath_utils.preprocess` | Prepare the combined data and retain the all-gene disease dataset. |
| `spapath_utils.build_graph_GAT_plus` | Build sample-specific graphs. |
| `spapath_model.Model.initial_embedding` | Learn initial `pre_embed` embeddings. |
| `spapath_model.Model.clustering` | Cluster observations and merge related clusters. |
| `spapath_model.Model.integrate` | Integrate references and disease into `cell_embed`. |
| `spapath_utils.detection` | Assign predicted region labels in `.obs["pred_label"]`. |
| `spapath_utils.build_disease_data` | Transfer predictions to the all-gene disease data and recalculate gene embeddings and cell–gene distances using `pre_embed`. |

Keep the disease sample last during preprocessing and detection. Pass `label_core="Pathological regions"` and `label_other="Healthy-like regions"` to `detection`; its defaults retain legacy names. Without ground-truth annotations, set `core_types=None` and `celltype_key=None`.

## Characterization and plotting

| Function in `spapath_utils` | Purpose |
| --- | --- |
| `calculate_signature_genes` | Select signature genes from the reconstructed disease dataset. |
| `run_signature_enrichment` | Perform GO/Hallmark enrichment through Enrichr. |
| `run_ccc_analysis` | Analyze spatial cell–cell communication; both directions are included by default. |
| `calculate_tc_le_enrichment` | Calculate TC, LE, and TC-to-LE scores. |
| `plot_detection_umap` | Show region predictions and optional cell-type labels. |
| `plot_prediction_on_he` / `plot_score_on_he` | Overlay predicted regions or continuous scores on H&E. |
| `plot_gene_expression` | Plot spatial expression of selected genes. |
| `plot_enrichment_bubble` / `plot_tc_le_violin` | Display enrichment results. |
