project = "SpaPath"
author = "SpaPath contributors"
copyright = "2026, SpaPath contributors"
language = "en"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "DEPLOYMENT.md", "README.md"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = "SpaPath documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"navigation_depth": 2, "collapse_navigation": False}
html_context = {
    "display_github": True,
    "github_user": "edawu11",
    "github_repo": "SpaPath",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
