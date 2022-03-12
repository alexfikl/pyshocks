r"""
Burgers' Equation
-----------------

This module implements schemes for the inviscid Burgers' equation

.. math::

    \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial}{\partial x} (u^2)  = 0.

.. autoclass:: Scheme
.. autoclass:: LaxFriedrichs
.. autoclass:: EngquistOsher
.. autoclass:: WENOJS
.. autoclass:: WENOJS32
.. autoclass:: WENOJS53

Data
^^^^

.. autofunction:: ic_tophat
.. autofunction:: ic_rarefaction
.. autofunction:: ic_sine
.. autofunction:: ex_shock
"""

from pyshocks.burgers.schemes import (
        Scheme, LaxFriedrichs, EngquistOsher,
        WENOJS, WENOJS32, WENOJS53)

import numpy as np


# {{{ initial conditions

def ic_tophat(grid, x):
    # TODO: this looks like it should have a solution for all t
    size = grid.b - grid.a
    lb = grid.a + size / 3.0
    rb = grid.b - size / 3.0

    return (lb < x < rb).astype(np.float64)


def ic_rarefaction(grid, x):
    # TODO: this looks like it should have a solution for all t
    mid = 0.5 * (grid.a + grid.b)

    return (x > mid).astype(np.float64)


def ic_sine(grid, x):
    import jax.numpy as jnp
    dx = (grid.b - grid.a) / 3.0
    lb = grid.a + dx
    rb = grid.b - dx

    return jnp.where(
            jnp.logical_and(lb < x, x < rb),
            1.0 + jnp.sin(2.0 * jnp.pi * (x - lb) / dx),
            1.0)

# }}}


# {{{ forward analytic solutions

def ex_shock(grid, t, x):
    # shock velocity
    s = 0.5
    # initial shock location
    x0 = 0.5 * (grid.a + grid.b)

    return (x < (x0 + s * t)).astype(np.float64)

# }}}


__all__ = (
    "Scheme", "LaxFriedrichs", "EngquistOsher",
    "WENOJS", "WENOJS32", "WENOJS53",

    "ic_tophat", "ic_rarefaction", "ic_sine",
    "ex_shock",
)
