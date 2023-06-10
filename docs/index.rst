pyshocks documentation
======================

.. toctree::
    :maxdepth: 2
    :hidden:

    tutorial.rst
    equations.rst
    schemes.rst
    timestepping.rst
    misc.rst
    references.rst

This repository contains some preliminary experiments on performing adjoint-based
optimization of systems with shocks by using automatic differentiation. The
main goal is to get it working for the one-dimensional unsteady Euler equations
with common WENO schemes.

It is currently quite far from that goal, so this is **very experimental**.

It currently supports some standard equations like the linear advection equation,
the heat equation and Burgers' equation. Several numerical methods are implemented

* standard first-order finite volume methods;
* MUSCL reconstructions with a wide family of limiters;
* WENO reconstructions (standard Jiang-Shu, energy stable, etc);
* Summation-by-Parts operators (up to second order with variable coefficients).
* Arbitrary finite difference construction and flux splitting methods.

Not all of these methods and equations work well together. All the infrastructure
is implemented through ``jax`` and can be automatically differentiated
for use in optimization routines.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
