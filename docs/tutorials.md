# Tutorials

Follow the notebooks directly below. Start with BC for a complete workflow from reference integration to signature genes and enrichment.

```{toctree}
:maxdepth: 1

Breast cancer (BC) <tutorials/BC>
Non-small-cell lung cancer (NSCLC) <tutorials/NSCLC>
Oral squamous cell carcinoma (OSCC) <tutorials/OSCC>
Multiple myeloma (MM) <tutorials/MM>
Crohn's disease (CD) <tutorials/CD>
```

## Data

**Public example-data download links are pending.** Data are not included in the repository or website. Place the following processed inputs in `data/<dataset>/` (a local directory or symlink). Results are written to `outputs/<dataset>/`.

| Dataset | Healthy reference files | Disease file |
| --- | --- | --- |
| BC | `V07.h5ad` | `H1.h5ad` |
| NSCLC | `scRNA_Sample1.h5ad` | `FOV_7.h5ad` |
| OSCC | `GM241.h5ad`, `BM169.h5ad` | `GSM6339632_s2.h5ad` |
| MM | `hHBM_scATAC.h5ad`, `hHBM_scRNA.h5ad`, `hHBM1_ST.h5ad` | `hMM2.h5ad` |
| CD | `V11Y24-011_B_adata.h5ad` | `V10A14_143_D_adata.h5ad` |

BC also needs `spatial/V07_image_embeddings.csv` and `spatial/H1_image_embeddings.csv`, indexed by observation names with `dim_*` embedding columns. The processed OSCC, MM ST, and CD files contain image embeddings; OSCC can alternatively load `spatial/GSM6339632_s2_image_embeddings.csv`.

For H&E overlays, add `BC/H1.jpg`, `OSCC/GSM6339632_s2_tissue_hires_image.png`, and `CD/V10A14_143_D.tissue_hires_image.png` under `data/`. The OSCC gene table is supplied as `notebook/TC_LE_genes.csv`.

For your own data, provide shared gene identifiers, spatial coordinates for ST, and a `batch` column matching the sample names in `adata_type_map`. Keep healthy references first and the disease sample last. Adjust image coordinate scales and cell-type annotations to match your samples.
