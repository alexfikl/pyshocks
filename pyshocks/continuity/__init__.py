# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Continuity Equation
-------------------

This module implements schemes for the conservative continuity equation

.. math::

    \frac{\partial u}{\partial t} + \frac{\partial}{\partial x} (a u) = 0,

where :math:`a(t, x)` is a known velocity field.

Schemes
^^^^^^^

.. autoclass:: Scheme
.. autoclass:: FiniteVolumeScheme
.. autoclass:: FiniteDifferenceScheme

.. autoclass:: Godunov

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name
.. autofunction:: check_oslc
"""

from typing import Any, Dict, Tuple, Type

import jax.numpy as jnp

from pyshocks.continuity.schemes import (
    FiniteDifferenceScheme,
    FiniteVolumeScheme,
    Godunov,
    Scheme,
)
from pyshocks.grid import Grid
from pyshocks.tools import SpatialFunction

# NOTE: just providing an alias for common usage
Upwind = Godunov


# {{{ make_scheme_from_name

_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": Godunov,
    "godunov": Godunov,
    "upwind": Godunov,
}


def scheme_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available schemes.
    """
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve the continuity equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """

    cls = _SCHEMES.get(name)
    if cls is None:
        from pyshocks.tools import join_or

        raise ValueError(
            f"Scheme {name!r} not found. Try one of {join_or(scheme_ids())}."
        )

    from dataclasses import fields

    if "velocity" not in kwargs:
        kwargs["velocity"] = None

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}


# {{{ check_oslc


def check_oslc(grid: Grid, velocity: SpatialFunction, *, n: int = 512) -> jnp.ndarray:
    r"""Check the One-sided Lipschitz Continuous condition.

    A function :math:`f(t, x)` is called one-sided Lipschitz continuous if
    there exists an integrable function :math:`L(t)` such that for every
    :math:`x_1, x_2 \in \mathbb{R}`

    .. math::

        \frac{f(t, x_1) - f(t, x_2)}{x_1 - x_2} \le L(t)

    This computes the maximum velocity variation on the given *grid*. The
    user can then check this against a reasonable tolerance to determine if
    the OSLC is satisfied.

    :returns: the maximum over all points :math:`(x_1, x_2)` on *grid*.
    """

    x = jnp.linspace(grid.a, grid.b, n, dtype=jnp.float64)
    a = velocity(x)

    dadx = (a[1:] - a[:-1]) / (x[1:] - x[:-1])
    return jnp.max(dadx)


# }}}


__all__ = (
    "Scheme",
    "FiniteVolumeScheme",
    "FiniteDifferenceScheme",
    "Godunov",
    "scheme_ids",
    "make_scheme_from_name",
    "check_oslc",
)
