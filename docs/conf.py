# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
import os
from pathlib import Path

# sys.path.insert(0, str(Path('..', 'src').resolve())) # Add the source directory to the path
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../src/gensbi'))

project = "GenSBI"
copyright = "2025, Aurelio Amerio"
author = "Aurelio Amerio"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
    # For extension examples and demos
    "myst_parser",
    "ablog",
    "jupyter_sphinx",
    "nbsphinx",
    "numpydoc",
    "sphinx_togglebutton",
    "jupyterlite_sphinx",
    "sphinx_favicon",
]


# -- MyST options ------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "linkify", "substitution","dollarmath"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# -- Mermaid options ---------------------------------------------------------
mermaid_version = "latest"
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'neutral'
});
"""

nbsphinx_execute = 'never'

# -- Internationalization ----------------------------------------------------
language = "en"

# -- Sphinx-copybutton options ---------------------------------------------
copybutton_exclude = ".linenos, .gp"
copybutton_selector = ":not(.prompt) > div.highlight pre"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_show_sourcelink = False
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo_small.png"
html_favicon = "_static/logo_small.png"

html_theme_options = {
    
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/aurelio-amerio/GenSBI",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "",
        # You can add "image_dark" if you have a dark mode logo
    },
    "use_edit_page_button": False,
    "show_toc_level": 1,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "search_as_you_type": True,
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src/gensbi"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"

pygments_style = "sphinx"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
