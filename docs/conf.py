# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../src"))


project = "jorbit"
author = "Ben Cassese"
release = "0.1.5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_automodapi.automodapi",
    "myst_parser",
    "sphinxcontrib.video",
    # "sphinx.ext.pngmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "visualizations"]
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "jorbit"
html_favicon = "_static/saturn.png"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ben-cassese/jorbit",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": False,
    "show_prev_next": False,
    "logo": {
        "text": "",
        "image_light": "_static/jorbit_logo_bright.png",
        "image_dark": "_static/jorbit_logo_dark.png",
    },
}

html_context = {"default_mode": "light"}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}

autodoc_typehints = "description"
python_use_unqualified_type_names = True
autodoc_typehints_format = "fully-qualified"
napoleon_use_param = True
napoleon_use_rtype = True
