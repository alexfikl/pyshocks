.. image:: https://github.com/alexfikl/pyshocks/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/pyshocks/actions?query=branch%3Amain+workflow%3ACI

.. image:: https://readthedocs.org/projects/pyshocks/badge/?version=latest
    :alt: Documentation
    :target: https://pyshocks.readthedocs.io/en/latest/?badge=latest

Installation
============

Common practice in the community is to set up a virtual environment

    python -m venv --system-site-packages /path/to/env/pyshocks

Activate the environment with

    source /path/to/env/pyshocks/bin/activate

Finally, just install the package with

    python -m pip install -e .[dev]

which should download all the dependencies. See the
`official documentation <https://docs.python.org/3/tutorial/venv.html>`__
for more details.

Conda
=====

TODO for anyone using `conda <https://github.com/conda-forge/miniforge>`__.

Documentation
=============

Documentation can be generated using `Sphinx <https://github.com/sphinx-doc/sphinx>`__.
For example, to generate a nice HTML-based variant go

    cd docs && make html
    firefox _build/html/index.html

Sphinx ca can also generate LaTeX documentation with

    make latex && cd _build/latex && latexrun pyshocks.tex
    xdg-open pyshocks.pdf
