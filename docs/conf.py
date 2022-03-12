# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# {{{ project information

project = "pyshocks"
copyright = "2021, Et Al"
author = "Et Al"
release = "2021.1"

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
needs_sphinx = "3.3"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

# }}}

# {{{ internationalization

language = "en"

# }}

# {{{ output

# html
html_theme = "furo"
html_theme_options = {
    "page_width": "75%",
}

# }}}

# {{{ extension settings

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": None,
}

autodoc_typehints = "description"
autodoc_type_aliases = {
    "VelocityFun": "pyshocks.continuity.VelocityFun",
}

# }}}

# {{{ links

intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "https://numpy.org/doc/stable": None,
    "https://jax.readthedocs.io/en/latest": None,
}

# }}}
