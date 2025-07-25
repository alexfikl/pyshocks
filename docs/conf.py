# SPDX-FileCopyrightText: 2022-2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: CC0-1.0

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# {{{ project information

from importlib import metadata

m = metadata.metadata("pyshocks")
project = m["Name"]
author = m["Author-email"]
copyright = f"2021-2022 {author}"  # noqa: A001
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
needs_sphinx = "6.0"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "show_toc_level": 2,
    "use_source_button": True,
    "use_repository_button": True,
    "navigation_with_keys": True,
    "repository_url": "https://github.com/alexfikl/pyshocks",
    "repository_branch": "main",
    "icon_links": [
        # {
        #     "name": "Release",
        #     "url": "https://github.com/alexfikl/pyshocks/releases",
        #     "icon": "https://img.shields.io/github/v/release/alexfikl/pyshocks",
        #     "type": "url",
        # },
        {
            "name": "License",
            "url": "https://github.com/alexfikl/pyshocks/tree/main/LICENSES",
            "icon": "https://img.shields.io/badge/License-GPL_2.0-blue.svg",
            "type": "url",
        },
        {
            "name": "CI",
            "url": "https://github.com/alexfikl/pyshocks",
            "icon": "https://github.com/alexfikl/pyshocks/actions/workflows/ci.yml/badge.svg",
            "type": "url",
        },
        {
            "name": "Issues",
            "url": "https://github.com/alexfikl/pyshocks/issues",
            "icon": "https://img.shields.io/github/issues/alexfikl/pyshocks",
            "type": "url",
        },
    ],
}

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
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# }}}
