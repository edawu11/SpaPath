# Installation

## Environment

The development environment uses **Linux, Python 3.10.18, and PyTorch 2.1.2 with CUDA 12.1**. The example notebooks select `DEVICE = "cuda"`. Use an NVIDIA GPU with a driver compatible with CUDA 12.1, or set `DEVICE = "cpu"` for CPU execution. CPU execution can be substantially slower; no cross-platform performance benchmark is provided.

The [runtime requirements](https://github.com/edawu11/SpaPath/blob/main/requirements.txt) pin the main packages used by the workflows, including Scanpy, AnnData, NumPy, SciPy, scikit-learn, LIANA, GSEApy, and plotting libraries. They record the development environment rather than a fully resolved dependency lockfile.

## Install from source

```bash
git clone https://github.com/edawu11/SpaPath.git
cd SpaPath
conda create -n spapath python=3.10 pip -y
conda activate spapath
python -m pip install -r requirements.txt
python -m ipykernel install --user --name spapath --display-name "Python (SpaPath)"
```

The requirements file includes the PyTorch CUDA 12.1 wheel index. Installing a different PyTorch build requires adapting that requirement to your platform.

Verify the core imports and GPU availability from the repository root:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); import spapath_model, spapath_utils; import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

SpaPath currently consists of source modules in `scripts/`. The notebooks add that directory to Python's import path; they do not require `pip install spapath`.

## Open a notebook

```bash
jupyter lab
```

Choose the **Python (SpaPath)** kernel and open a notebook from `notebook/`. Launch Jupyter from the repository root or the `notebook/` directory so that the notebook can locate the project.

Prepare the [example inputs](tutorials.md#data-layout) before running. Figures are saved in `outputs/<dataset>/fig/`. Image overlays require the matching H&E image and spatial coordinate scale. BC enrichment uses the online Enrichr service and therefore needs network access.

## Build documentation only

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Visit `http://localhost:8000`. The documentation environment is separate from the analysis environment and does not install CUDA dependencies, load data, or execute notebooks.
