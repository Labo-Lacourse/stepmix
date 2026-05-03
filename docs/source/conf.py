# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "StepMix"
copyright = "2022, Labo-Lacourse"
author = "Sacha Morin, Robin Legault"

release = "0.0"
version = "2.2.3"


# -- General configuration

extensions = [
    "sphinx.ext.napoleon",
    # 'numpydoc',
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
# numpydoc_show_class_members = False
# numpydoc_class_members_toctree = False
autodoc_default_flags = ["members"]
autosummary_generate = True
