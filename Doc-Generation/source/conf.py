# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Proyecto:_PIA_AA'
copyright = '2025, Noel Freire Mahía, Iván Hermida Mella'
author = 'Noel Freire Mahía \n Iván Hermida Mella'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# Añadimos el path de nuestra app
import os
import sys
sys.path.insert(0, os.path.abspath("C:/Users/Noel/Desktop/TrabajoPIA_AA/Code"))
# Activamos que incluya los TODO (veremos ejemplo)
todo_include_todos = True
# Podemos cambiar el tema para la generación html
html_theme = "sphinx_rtd_theme"

# Añadimos extensiones
extensions = [
 "sphinx.ext.autodoc",
 "sphinx.ext.todo",
 "sphinx.ext.viewcode",
]