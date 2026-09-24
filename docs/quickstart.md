# Quick start

The [BC notebook](https://github.com/edawu11/SpaPath/blob/main/notebook/BC.ipynb) provides a complete example, from healthy and disease inputs to signature genes and enrichment. Run its cells in order after [installation](installation.md) and [data preparation](tutorials.md#data-layout).

## Input requirements

- Use one AnnData object per sample, with observations as rows and genes as columns. `.X` contains expression counts or scATAC gene-activity values before the notebook's preprocessing step.
- Supply gene identifiers in `.var_names` and unique observation identifiers in `.obs_names`. Gene identifiers must be compatible across samples.
- Provide `.obsm["spatial"]` for ST samples. The supplied non-spatial references also contain spatial placeholders; no spatial graph is built for `sc` or `scATAC` reference types.
- Each sample must contain `.obs["batch"]` matching its key in `adata_type_map`. Keep samples and dictionary entries in the same order, with the disease sample **last**.
- `ST_with_HE` inputs need `.obsm["image_embedding"]`. BC loads these embeddings from CSV files indexed by observation identifiers, with embedding columns named `dim_*`. Raw H&E images are used for plotting, not for generating embeddings in these notebooks.
- Ground-truth cell-type columns are used for comparison and plotting in the supplied examples. Detection can be run without them by setting `core_types=None` and `celltype_key=None`.

## Minimal BC workflow

Run this example from the repository root after placing the BC files in `data/BC/`:

```python
from pathlib import Path
import sys
import scanpy as sc

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
import spapath_model
import spapath_utils

input_dir = root / "data" / "BC"
healthy = sc.read_h5ad(input_dir / "V07.h5ad")
disease = sc.read_h5ad(input_dir / "H1.h5ad")
for section, adata in [("V07", healthy), ("H1", disease)]:
    spapath_utils.attach_image_embedding(
        adata, input_dir / "spatial" / f"{section}_image_embeddings.csv"
    )

sample_types = {"V07": "ST_with_HE", "H1": "ST_with_HE"}
adata_full, all_genes = spapath_utils.preprocess(
    adata_list=[healthy, disease],
    adata_type_map=sample_types,
    full_num_hvgs=3000,
    min_genes_qc=10,
    min_cells_qc=10,
)
adata_full = spapath_utils.build_graph_GAT_plus(
    adata_full, sample_types, K=8, img_threshold=0.0
)
model = spapath_model.Model(
    adata_full=adata_full,
    adata_type_map=sample_types,
    lr_pre=1e-4,
    lr=1e-4,
    n_pre_training_steps=500,
    n_training_steps=300,
    device="cuda",
    seed=123,
)
adata_full = model.initial_embedding()
adata_full = model.clustering(init_res=1.5, intopk=40)
adata_full = model.integrate(topk=40)
adata_full = spapath_utils.detection(
    adata=adata_full,
    embed="cell_embed",
    section_ids=list(sample_types),
    label_core="Pathological regions",
    label_other="Healthy-like regions",
    core_types=None,
    celltype_key=None,
    batch_key="batch",
    seed=123,
    neighbors=30,
    threshold=0.05,
    strategy="individual",
)
disease_data = spapath_utils.build_disease_data(
    adata_full=adata_full,
    disease_adata_all_genes=all_genes,
    disease_section="H1",
)
genes = spapath_utils.calculate_signature_genes(
    disease_data, topk=100, n_print=10
)
```

This example omits ground-truth evaluation. The full BC notebook includes its annotated cell types, plots, and GO/Hallmark enrichment.

## Inspect the result

| Location | Content |
| --- | --- |
| `adata_full.obs["pred_label"]` | Predicted region labels |
| `adata_full.obsm["pre_embed"]` | Initial cell/spot embedding |
| `adata_full.obsm["cell_embed"]` | Integrated cell/spot embedding |
| `disease_data.X` | Normalized expression across all retained disease genes |
| `disease_data.uns["gene_embed"]` | Gene embedding recalculated using `pre_embed` |
| `disease_data.layers["cell_gene_dist"]` | Recalculated cell–gene distances for downstream analysis |

“All genes” means genes retained after quality control, rather than only those shared across reference datasets. Keep the returned `all_genes` object from preprocessing for this step. The full workflows and plotting calls are available in the [tutorials](tutorials.md).
