import os
import sys
sys.path.insert(
    0, os.path.abspath("..")
)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jorbit'
copyright = '2023, Ben Cassese'
author = 'Ben Cassese'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # allows Google style-guide docs to render more prettily
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_automodapi.automodapi",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "jorbit"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ben-cassese/jorbit",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
}