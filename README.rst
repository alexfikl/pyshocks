.. image:: https://github.com/alexfikl/pyshocks/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/pyshocks/actions?query=branch%3Amain+workflow%3ACI

.. image:: https://readthedocs.org/projects/pyshocks/badge/?version=latest
    :alt: Documentation
    :target: https://pyshocks.readthedocs.io/en/latest/?badge=latest

About
=====

This repository contains some preliminary experiments on performing adjoint-based
optimization for systems with shocks by using automatic differentiation. The
main goal is to get it working for the one-dimensional unsteady Euler equations
with common WENO schemes.

It is currently quite far from that goal, so this is **very experimental**.

Requirements
============

The project currently supports Python 3.8 and later. The requirements are
listed in ``setup.cfg``

* ``jax`` and ``jaxlib``: base numeric and automatica differentiation package
  used throughout. See
  `JAX support policy <https://jax.readthedocs.io/en/latest/deprecation.html?highlight=nep>`__
  for details on supported versions.
* ``rich``: recommended for nicer logging.
* ``matplotlib`` and ``SciencePlots``: recommended for nicer plotting.

Additional development dependencies are listed in ``setup.cfg``.

Documentation
=============

Documentation can be generated using `Sphinx <https://github.com/sphinx-doc/sphinx>`__.
For example, to generate a nice HTML-based variant go

.. code:: bash

    cd docs && make html
    firefox _build/html/index.html

Sphinx ca can also generate LaTeX documentation with

.. code:: bash

    make latex && cd _build/latex && latexrun pyshocks.tex
    xdg-open pyshocks.pdf

The documentation is also hosted on
`readthedocs <https://pyshocks.readthedocs.io/en/latest/index.html>`__.
