# Overview

SpaPath identifies and characterizes pathological regions in disease spatial transcriptomics using one or more healthy references. Healthy references may come from ST, scRNA-seq, or scATAC-seq and may be used separately or together. For scATAC-seq, the input is a gene-activity matrix with gene identifiers shared with the disease dataset.

## Workflow

![SpaPath workflow: pathological-region detection and characterization](_static/workflow.png)

[View the original figure (PDF)](https://github.com/edawu11/SpaPath/blob/main/Fig_1.pdf)

1. **Co-embedding.** Learn cell/spot embeddings and derive gene embeddings from expression or gene-activity values. Spatial graphs support ST inputs; precomputed image embeddings can refine graph construction when available.
2. **Automated clustering.** Initialize clusters and merge them using their signature-gene relationships.
3. **Integration and detection.** Use cluster-specific signature genes to guide healthy–disease alignment, then assign pathological-region and healthy-like-region labels.
4. **Characterization.** Reconstruct the disease dataset across all retained genes and use its co-embedding for signature-gene selection, enrichment, spatial communication, or gene-program scoring.

## Outputs

The example workflows use two prediction labels: **Pathological regions** (`#B6473F`) and **Healthy-like regions** (`#6DBBD1`). Detection operates on integrated cell embeddings. Downstream disease-data construction uses the initial `pre_embed` embedding to recalculate gene embeddings and cell–gene distances over the full retained gene set.

The [tutorials](tutorials.md) cover breast cancer, non-small-cell lung cancer, oral squamous cell carcinoma, multiple myeloma, and Crohn's disease. Each notebook specifies its own references, plotting palette, and downstream analyses.

## Project information

The implementation is available on [GitHub](https://github.com/edawu11/SpaPath) under the [MIT License](https://github.com/edawu11/SpaPath/blob/main/LICENSE). Use [GitHub Issues](https://github.com/edawu11/SpaPath/issues) for questions and bug reports. A manuscript citation will be added when a public reference is available.
