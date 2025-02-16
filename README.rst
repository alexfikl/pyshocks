.. image:: https://github.com/alexfikl/pyshocks/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/pyshocks/actions?query=branch%3Amain+workflow%3ACI

.. image:: https://readthedocs.org/projects/pyshocks/badge/?version=latest
    :alt: Documentation
    :target: https://pyshocks.readthedocs.io/en/latest/?badge=latest

About
=====

This repository contains some preliminary experiments on performing adjoint-based
optimization of systems with shocks by using automatic differentiation. The
main goal is to get it working for the one-dimensional unsteady Euler equations
with common WENO schemes.

It is currently quite far from that goal, so this is **very experimental**.

**Caution**: As you might be able to tell, development on this has pretty much
stopped at this point. The code is still updated for newer versions of things
(``jax`` and ``numpy``), but no new features are added.

Requirements
============

The project currently supports Python 3.10 and later. The (full) requirements are
listed in ``pyproject.toml``.

* ``jax`` and ``jaxlib``: base numeric and automatic differentiation package
  used throughout. See
  `JAX support policy <https://jax.readthedocs.io/en/latest/deprecation.html?highlight=nep>`__
  for details on supported versions.
* ``rich``: recommended for nicer logging.
* ``matplotlib`` and ``SciencePlots``: recommended for nicer plotting.

For development, it is recommended to run

.. code:: bash

    python -m pip install -e '.[dev,vis]'

A pinned version of all the requirements is kept in ``requirements.txt``.
For a quick install with the versions that are currently being tested on the CI

.. code:: bash

   python -m pip install -r requirements.txt -e .

or use ``.github/requirements-dev.txt`` to also install development packages
(e.g. ``ruff``).

Documentation
=============

Documentation can be generated using `Sphinx <https://github.com/sphinx-doc/sphinx>`__.
For example, to generate a nice HTML-based variant go

.. code:: bash

    cd docs
    just build
    just view [BROWSER]

The documentation is also hosted on
`readthedocs <https://pyshocks.readthedocs.io/en/latest/index.html>`__.
