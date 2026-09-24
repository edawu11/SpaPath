# Overview

SpaPath identifies pathological regions in disease spatial transcriptomics by comparison with healthy references. References can be ST, scRNA-seq, or scATAC-seq gene-activity data, used individually or together.

```{image} _static/workflow.png
:alt: SpaPath workflow
:width: 600px
:align: center
```

The workflow learns cell/spot–gene co-embeddings, performs automated clustering and reference integration, and predicts **Pathological regions** or **Healthy-like regions**. Downstream analyses include signature genes, functional enrichment, spatial cell–cell communication, and gene-program scores.

Start with [installation](installation.md), then follow a [tutorial](tutorials.md) for your dataset.
