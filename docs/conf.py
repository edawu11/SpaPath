from pathlib import Path
from shutil import copyfile


DOCS_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = DOCS_DIR / "tutorials"
TUTORIAL_DIR.mkdir(exist_ok=True)
for name in ("BC", "NSCLC", "OSCC", "MM", "CD"):
    copyfile(DOCS_DIR.parent / "notebook" / f"{name}.ipynb", TUTORIAL_DIR / f"{name}.ipynb")

project = "SpaPath"
author = "SpaPath contributors"
copyright = "2026, SpaPath contributors"
language = "en"

extensions = ["myst_nb"]
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
nb_execution_mode = "off"
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = "SpaPath documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_depth": 2, "collapse_navigation": False}
html_show_sourcelink = False
