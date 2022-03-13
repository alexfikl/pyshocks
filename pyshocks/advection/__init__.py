r"""
Linear Advection Equation
-------------------------

This module implements schemes for the linear advection equation

.. math::

    \frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = 0,

where :math:`a(t, x)` is a known velocity field. For a conservative version
of the same equation see :mod:`pyshocks.continuity`, which also provides
initial conditions.

.. autoclass:: Scheme
.. autoclass:: Godunov
"""

from pyshocks.advection.schemes import Scheme, Godunov, WENOJS, WENOJS32, WENOJS53

__all__ = (
    "Scheme",
    "Godunov",
    "WENOJS",
    "WENOJS32",
    "WENOJS53",
)
