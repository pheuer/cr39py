# Configuration file for the Sphinx documentation builder.

import os
import sys

# print(os.path.abspath(".."))
# Tell autodoc where to find the source code
sys.path.insert(0, os.path.abspath(".."))


# -- Project information

project = "cr39py"
copyright = "2024, Peter Heuer"
author = "Peter Heuer"

release = "0.1"
version = "2024.11"

# -- General configuration

default_role = "py:obj"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

# sphinx-autoapi
autoapi_dirs = ["../../src/cr39py"]
autoapi_type = ["python"]
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
    "special-members",
    "imported-members",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"
html_logo = "./_static/logo_190px.png"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
    "includehidden": False,
}

# -- Options for EPUB output
epub_show_urls = "footnote"


# sphinx.ext.autodoc

autoclass_content = "both"
autodoc_typehints_format = "short"
