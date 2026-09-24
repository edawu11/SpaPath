# Tutorials

The five notebooks below are the maintained examples. Open them in Jupyter with the SpaPath kernel and execute cells in order. Outputs are saved under `outputs/<dataset>/`; they are not part of the repository.

**Public example-data download links are pending.** This repository and the documentation website do not include datasets. Supply the processed files locally using the layout below. Raw public-study downloads may require preprocessing before they can be used in these notebooks.

## Data layout

```text
data/
├── BC/
│   ├── V07.h5ad
│   ├── H1.h5ad
│   ├── H1.jpg
│   └── spatial/
│       ├── V07_image_embeddings.csv
│       └── H1_image_embeddings.csv
├── NSCLC/
│   ├── scRNA_Sample1.h5ad
│   └── FOV_7.h5ad
├── OSCC/
│   ├── GM241.h5ad
│   ├── BM169.h5ad
│   ├── GSM6339632_s2.h5ad
│   └── GSM6339632_s2_tissue_hires_image.png
├── MM/
│   ├── hHBM_scATAC.h5ad
│   ├── hHBM_scRNA.h5ad
│   ├── hHBM1_ST.h5ad
│   ├── hMM2.h5ad
│   └── tissue_hires_image.png
└── CD/
    ├── V11Y24-011_B_adata.h5ad
    ├── V10A14_143_D_adata.h5ad
    └── V10A14_143_D.tissue_hires_image.png
```

`data/` may be a local directory or a symlink to your data storage. It is ignored by Git. The MM H&E image is listed as an optional plotting asset; the current MM notebook ends at disease-data construction without an H&E overlay.

The processed CD, MM ST, and OSCC disease files contain image embeddings. If the OSCC disease file lacks them, supply `data/OSCC/spatial/GSM6339632_s2_image_embeddings.csv`. Single-cell references receive zero-filled image embeddings where needed so that concatenation retains the disease embedding.

## Breast cancer (BC)

[Open BC.ipynb](https://github.com/edawu11/SpaPath/blob/main/notebook/BC.ipynb)

Compare the H1 disease section against the V07 healthy ST reference. Both samples use image embeddings. The notebook performs detection, plots UMAP and predictions over H&E, constructs the all-gene disease dataset, identifies the top 100 pathological-region signature genes, and performs GO and Hallmark enrichment.

The first ten signature genes are printed. Selected gene-expression plots and enrichment plots are saved with the other figures in `outputs/BC/fig/`. The Enrichr step needs network access.

## Non-small-cell lung cancer (NSCLC)

[Open NSCLC.ipynb](https://github.com/edawu11/SpaPath/blob/main/notebook/NSCLC.ipynb)

Compare the FOV_7 disease data against `scRNA_Sample1`. After detection and disease-data construction, calculate spatial ligand–receptor communication between pathological and healthy-like regions using CosMx settings.

`include_reverse=True` evaluates both pathological → healthy-like and healthy-like → pathological directions. The summary reports cell counts, tested and significant ligand–receptor pair counts, and the smallest adjusted P value for each direction. This example analyzes one disease FOV, not a pooled multi-FOV dataset.

## Oral squamous cell carcinoma (OSCC)

[Open OSCC.ipynb](https://github.com/edawu11/SpaPath/blob/main/notebook/OSCC.ipynb)

Use both GM241 and BM169 single-cell references for the GSM6339632_s2 disease section. The notebook creates the disease dataset, then uses the included `notebook/TC_LE_genes.csv` gene table to calculate TC and LE enrichment scores in pathological regions.

The transition score is `LE_score - TC_score`, stored as `TC_to_LE_score`. A separate H&E plot shows pathological spots with a red score gradient and healthy-like spots in blue. A violin plot compares the scores across ground-truth core, transitory, and edge classes. The hires H&E overlay uses coordinate scale `0.08828852` for this sample.

## Multiple myeloma (MM)

[Open MM.ipynb](https://github.com/edawu11/SpaPath/blob/main/notebook/MM.ipynb)

The disease section is hMM2. Run four reference configurations: `hHBM_scATAC` alone, `hHBM_scRNA` alone, `hHBM1_ST` alone, and all three together. Each workflow performs detection and returns the integrated data and all-gene disease dataset. It does not run signature-gene or enrichment analysis.

## Crohn's disease (CD)

[Open CD.ipynb](https://github.com/edawu11/SpaPath/blob/main/notebook/CD.ipynb)

Compare V10A14_143_D against the V11Y24-011_B healthy ST reference. This workflow validates image embeddings, performs detection, plots UMAP and an H&E overlay, and ends after constructing the all-gene disease dataset. The input filenames retain the `_adata.h5ad` suffix.

## Adapting an example

Update sample identifiers, `adata_type_map`, and the input files together. Keep healthy references first and the disease sample last. The notebooks use `sc` for scRNA-seq, `scATAC` for gene-activity inputs, `ST` for spatial inputs, and `ST_with_HE` for image-assisted graphs.

Adjust `CELLTYPE_PALETTE` and `REGION_PALETTE` in the notebook when present. Supply the coordinate scale appropriate to your H&E image instead of reusing a scale from a different sample. For images where stored spatial coordinates already match image pixels, use `coordinate_scale=1.0`.
