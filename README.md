# SpaPath

**Pathological-region detection and characterization in disease spatial transcriptomics using multi-source healthy references.**

SpaPath compares disease spatial transcriptomics with healthy references from spatial transcriptomics (ST), scRNA-seq, or scATAC-seq. References can be used individually or jointly. Cell/spot–gene co-embeddings and cluster-specific signature genes support integration and the identification of pathological and healthy-like regions.

Downstream analyses include pathological-region signature genes, functional enrichment, spatial ligand–receptor communication, and variation in gene-program scores within pathological regions.

![SpaPath workflow: co-embedding, automated clustering, integration and detection, and pathological-region characterization](docs/_static/workflow.png)

[View the original figure (PDF)](Fig_1.pdf)

## Installation

The current workflows use Python 3.10 and PyTorch 2.1.2 with CUDA 12.1 on Linux. An NVIDIA GPU with a compatible driver is recommended. The requirements file records the versions used in the development environment; it is not a complete environment lockfile.

```bash
git clone https://github.com/edawu11/SpaPath.git
cd SpaPath
conda create -n spapath python=3.10 pip -y
conda activate spapath
python -m pip install -r requirements.txt
python -m ipykernel install --user --name spapath --display-name "Python (SpaPath)"
jupyter lab
```

Select the **Python (SpaPath)** kernel. The notebooks load the modules from `scripts/`; no separate package installation is required. See [installation details](docs/installation.md) for device settings and verification.

## Quick start

1. Place the processed input files in `data/<dataset>/`, following the [data layout](docs/tutorials.md#data-layout).
2. Open [notebook/BC.ipynb](notebook/BC.ipynb) and run its cells in order.
3. Inspect the predicted regions, H&E overlay, signature genes, and enrichment results. Figures are saved under `outputs/BC/fig/`.

The shared workflow is: load inputs → attach image embeddings when needed → preprocess and build graphs → initial embedding → automated clustering → integration → detection → construct the all-gene disease dataset.

**Example data are not included in this repository. Public download links are pending.** The notebooks require processed `.h5ad` files; the BC example also requires two image-embedding CSV files. H&E images are separate inputs for overlay plots. See [input requirements](docs/quickstart.md#input-requirements).

## Examples

| Notebook | Disease sample | Healthy reference(s) | Analysis |
| --- | --- | --- | --- |
| [BC](notebook/BC.ipynb) | H1 | V07, ST | Detection, H&E overlay, signature genes, GO/Hallmark enrichment |
| [NSCLC](notebook/NSCLC.ipynb) | FOV_7 | scRNA_Sample1, scRNA-seq | Detection and bidirectional spatial communication |
| [OSCC](notebook/OSCC.ipynb) | GSM6339632_s2 | GM241 and BM169, scRNA-seq | Detection, H&E overlays, TC-to-LE scores and violin plots |
| [MM](notebook/MM.ipynb) | hMM2 | scATAC, scRNA, ST, and their combination | Four reference configurations through disease-data construction |
| [CD](notebook/CD.ipynb) | V10A14_143_D | V11Y24-011_B, ST | Detection and H&E overlay through disease-data construction |

## Documentation

Read the documentation source: [Overview](docs/overview.md) · [Installation](docs/installation.md) · [Quick start](docs/quickstart.md) · [Tutorials](docs/tutorials.md) · [API reference](docs/api.md).

The Read the Docs configuration is included. The hosted documentation URL will be added after the first successful deployment.

To build and preview the documentation locally in a separate environment:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000`. Documentation builds do not run the notebooks or require example data. Deployment instructions are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Support and license

Please report questions and reproducible issues through [GitHub Issues](https://github.com/edawu11/SpaPath/issues).

SpaPath is distributed under the [MIT License](LICENSE). A manuscript citation will be added when a public reference is available.
