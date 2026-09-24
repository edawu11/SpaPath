# Installation

The workflows use **Linux, Python 3.10, and PyTorch 2.1.2 with CUDA 12.1**. Use an NVIDIA GPU with a compatible driver, or change `DEVICE = "cuda"` to `"cpu"` in the notebook for slower CPU execution.

```bash
git clone https://github.com/edawu11/SpaPath.git
cd SpaPath
conda create -n spapath python=3.10 pip -y
conda activate spapath
python -m pip install -r requirements.txt
python -m ipykernel install --user --name spapath --display-name "Python (SpaPath)"
jupyter lab
```

The notebooks import the source modules from `scripts/`; no separate package installation is required. Launch Jupyter from the repository root, select the **Python (SpaPath)** kernel, and open a notebook from `notebook/`.

Prepare the [example data](tutorials.md#data) before running. Figures are saved under `outputs/<dataset>/fig/`. BC functional enrichment requires internet access to Enrichr.
