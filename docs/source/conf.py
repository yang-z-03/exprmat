# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

project = 'exprmat'
copyright = '2025 (c) Zheng Yang'
author = 'Zheng Yang'
release = '0.1.31'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_design'
    # 'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ["page.css"]
html_title = 'exprmat'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# html_theme = "sphinx_rtd_theme"
# import sphinx_rtd_theme
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# one of: [
#     'abap', 'algol', 'algol_nu', 'arduino', 'autumn', 'bw', 'borland', 'coffee', 
#     'colorful', 'default', 'dracula', 'emacs', 'friendly_grayscale', 'friendly', 
#     'fruity', 'github-dark', 'gruvbox-dark', 'gruvbox-light', 'igor', 'inkpot', 
#     'lightbulb', 'lilypond', 'lovelace', 'manni', 'material', 'monokai', 'murphy', 
#     'native', 'nord-darker', 'nord', 'one-dark', 'paraiso-dark', 'paraiso-light', 
#     'pastie', 'perldoc', 'rainbow_dash', 'rrt', 'sas', 'solarized-dark', 
#     'solarized-light', 'staroffice', 'stata-dark', 'stata-light', 'tango', 'trac', 
#     'vim', 'vs', 'xcode', 'zenburn'
# ]

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinxawesome_theme'
# Select a color scheme for light mode
pygments_style = "vs"
# Select a different color scheme for dark mode
pygments_style_dark = "github-dark"