# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Burgers' Equation
-----------------

This module implements schemes for the inviscid Burgers' equation

.. math::

    \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial}{\partial x} (u^2)  = 0.

Schemes
^^^^^^^

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

from typing import Any, Tuple

from pyshocks.burgers.schemes import (
    Scheme,
    LaxFriedrichs,
    EngquistOsher,
    WENOJS,
    WENOJS32,
    WENOJS53,
)

import numpy as np


_SCHEMES = {
    "lf": LaxFriedrichs,
    "eo": EngquistOsher,
    "wenojs32": WENOJS32,
    "wenojs53": WENOJS53,
}


def scheme_ids() -> Tuple[str, ...]:
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve Burgers' equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """
    cls = _SCHEMES.get(name)
    if cls is None:
        raise ValueError(
            f"scheme '{name}' not found; try one of {tuple(_SCHEMES.keys())}"
        )

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


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
        1.0,
    )


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
    "Scheme",
    "LaxFriedrichs",
    "EngquistOsher",
    "WENOJS",
    "WENOJS32",
    "WENOJS53",
    "scheme_ids",
    "make_scheme_from_name",
    "ic_tophat",
    "ic_rarefaction",
    "ic_sine",
    "ex_shock",
)
