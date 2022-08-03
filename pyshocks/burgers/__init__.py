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
.. autoclass:: Godunov
.. autoclass:: Rusanov
.. autoclass:: LaxFriedrichs
.. autoclass:: EngquistOsher
.. autoclass:: ESWENO32
.. autoclass:: SSWENO242
.. autoclass:: SBP

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name

Data
^^^^

.. autofunction:: ic_tophat
.. autofunction:: ic_rarefaction
.. autofunction:: ic_sine
.. autofunction:: ex_shock
"""

from typing import Any, Dict, Tuple, Type

import jax.numpy as jnp

from pyshocks import Grid
from pyshocks.burgers.schemes import (
    Scheme,
    Godunov,
    Rusanov,
    LaxFriedrichs,
    EngquistOsher,
    ESWENO32,
    SSWENO242,
    SBP,
)


_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": LaxFriedrichs,
    "godunov": Godunov,
    "rusanov": Rusanov,
    "lf": LaxFriedrichs,
    "eo": EngquistOsher,
    "esweno32": ESWENO32,
}


def scheme_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available schemes.
    """
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve Burgers' equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """
    cls = _SCHEMES.get(name)
    if cls is None:
        raise ValueError(f"scheme '{name}' not found; try one of {scheme_ids()}")

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# {{{ initial conditions


def ic_tophat(grid: Grid, x: jnp.ndarray) -> jnp.ndarray:
    # TODO: this looks like it should have a solution for all t
    size = grid.b - grid.a
    lb = grid.a + size / 3.0
    rb = grid.b - size / 3.0

    return (lb < x < rb).astype(x.dtype)


def ic_rarefaction(grid: Grid, x: jnp.ndarray) -> jnp.ndarray:
    # TODO: this looks like it should have a solution for all t
    mid = 0.5 * (grid.a + grid.b)

    return (x > mid).astype(x.dtype)


def ic_sine(grid: Grid, x: jnp.ndarray) -> jnp.ndarray:
    dx = (grid.b - grid.a) / 3.0
    lb = grid.a + dx
    rb = grid.b - dx

    return jnp.where(  # type: ignore[no-untyped-call]
        jnp.logical_and(lb < x, x < rb),
        1.0 + jnp.sin(2.0 * jnp.pi * (x - lb) / dx),
        1.0,
    )


# }}}


# {{{ forward analytic solutions


def ex_shock(grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
    # shock velocity
    s = 0.5
    # initial shock location
    x0 = 0.5 * (grid.a + grid.b)

    return (x < (x0 + s * t)).astype(x.dtype)


# }}}


__all__ = (
    "Scheme",
    "Godunov",
    "Rusanov",
    "LaxFriedrichs",
    "EngquistOsher",
    "ESWENO32",
    "SSWENO242",
    "SBP",
    "scheme_ids",
    "make_scheme_from_name",
    "ic_tophat",
    "ic_rarefaction",
    "ic_sine",
    "ex_shock",
)
