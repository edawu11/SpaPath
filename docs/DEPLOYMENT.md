# Documentation deployment

The repository builds a static Sphinx website using `.readthedocs.yaml` and `docs/requirements.txt`. The documentation builder uses Python 3.12 independently of the Python 3.10 analysis environment. It does not import SpaPath or execute notebooks.

## Local validation

From the repository root, use a separate environment:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000`. Data and analysis outputs are unnecessary for this build.

## First deployment

1. Sign in to [Read the Docs](https://app.readthedocs.org/).
2. Connect GitHub and grant the Read the Docs GitHub App access to `edawu11/SpaPath`.
3. Add that repository as a project. Prefer the project name `spapath` if available; select `main` as the default branch and English as the language.
4. Use the root `.readthedocs.yaml` configuration and start the `latest` build.
5. After the build succeeds, open the URL shown by Read the Docs. Check `overview.html`, installation, quick start, tutorials, API reference, and search.
6. Replace the pending-deployment sentence in the root README with the actual verified documentation URL. Do not advertise an assumed project URL before the site exists.

See the official [project import guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html). With the repository integration enabled, subsequent pushes trigger documentation builds.

Only documentation source and its workflow image are published in the site. Keep local datasets, generated analysis outputs, and private manuscript correspondence out of Git. The deployment instructions themselves are excluded from the website navigation and Sphinx build.
