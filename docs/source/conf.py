# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "eindir"
copyright = "2023--present, eindir Developers"
author = "eindir Developers"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_contributors",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.spelling",
]

autodoc2_render_plugin = "myst"
autodoc2_packages = [
    "../../eindir",
]

myst_enable_extensions = [
    "deflist",
]

intersphinx_mapping = {
    "eindir": ("https://asv.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/HaoZeke/eindir/",
    "source_branch": "main",
    "source_directory": "docs/",
}
