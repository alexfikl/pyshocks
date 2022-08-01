# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# {{{ project information

from importlib import metadata

m = metadata.metadata("pyshocks")
project = m["Name"]
author = m["Author"]
copyright = f"2021-2022 {author}"
version = m["Version"]
release = version

# }}}

# {{{ general configuration

# needed extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# extension for source files
source_suffix = ".rst"
# name of the main (master) document
master_doc = "index"
# min sphinx version
needs_sphinx = "4.0"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

# }}}

# {{{ internationalization

language = "en"

# }}

# {{{ extension settings

autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": None,
}

# autodoc_typehints = "description"

# }}}

# {{{ links

intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable": None,
    "https://jax.readthedocs.io/en/latest": None,
}

# }}}
