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

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name

Data
^^^^

.. autofunction:: ic_tophat
.. autofunction:: ic_rarefaction
.. autofunction:: ic_sine
.. autofunction:: ex_shock
"""

from typing import Any, Dict, Optional, Tuple, Type

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
)


_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": LaxFriedrichs,
    "godunov": Godunov,
    "rusanov": Rusanov,
    "lf": LaxFriedrichs,
    "eo": EngquistOsher,
    "esweno32": ESWENO32,
    "ssweno242": SSWENO242,  # type: ignore[dict-item]
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


def ex_shock(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    ul: float = 1.0,
    ur: float = 0.0,
    x0: Optional[float] = None,
) -> jnp.ndarray:
    r"""Construct a pure shock exact solution of the form

    .. math::

        u(t, x) =
        \begin{cases}
        u_L, & \quad x < x_0 + s t, \\
        u_R, & x \ge x_0 + s t,
        \end{cases}

    where :math:`s` is the shock velocity.

    :arg ul: left state value.
    :arg ur: right state value.
    :arg x0: initial location of the shock.
    """

    if ul <= ur:
        raise ValueError("ul > ur is required for a shock")

    if x0 is None:
        x0 = 0.5 * (grid.a + grid.b)

    if not grid.a < x0 < grid.b:
        raise ValueError("x0 must be in the domain [a, b]")

    # shock velocity
    s = (ul + ur) / 2.0
    # Heaviside indicator for left / right
    h = (x < (x0 + s * t)).astype(x.dtype)

    return h * ul + (1 - h) * ur


def ex_linear_shock(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    ul: float = 1.0,
    ur: float = 0.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""Construct a shock formed at a later time of the form

    .. math::

        u(t, x) =
        \begin{cases}
        u_L, & \quad x \le u_L t, \\
        \frac{u_L - \alpha x}{1 - \alpha t}, &
            \quad u_L t < x < x_b + u_R t \\
        u_R & \quad x > x_b + u_R t,
        \end{cases}
    """
    raise NotImplementedError


def ex_tophat(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    us: float = 0.0,
    uc: float = 1.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""Constructs an rarefaction-shock exact solution of the form

    .. math::

        u(t, x) =
        \begin{cases}
        u_C, & \quad x < x_0 + s t, \\
        u_R, & \quad x \ge x_0 + s t,
        \end{cases}

    where :math:`s` is the shock velocity.
    """
    if uc <= us:
        raise ValueError("uc should be larger for a right shock solution")

    xm = (grid.b + grid.a) / 2
    dx = grid.b - grid.a

    if xa is None:
        xa = xm - 0.25 * dx

    if xb is None:
        xb = xm + 0.25 * dx

    if xa >= xb:
        raise ValueError("invalid sides (must be xa < xb)")

    if not grid.a < xa < grid.b:
        raise ValueError("xa must be in the domain [a, b]")

    if not grid.a < xb < grid.b:
        raise ValueError("xb must be in the domain [a, b]")

    # shock velocity
    s = (uc + us) / 2

    # if t > ((xb - xa) / (uc - s)):
    #     raise NotImplementedError(
    #         "cannot evaluate past the time when the rarefaction hits the shock"
    #     )

    return (
        us * (x < xa + us * t)
        + (x - xa) / (t + 1.0e-15) * jnp.logical_and(xa + us * t < x, x < xa + uc * t)
        + uc * jnp.logical_and(xa + uc * t < x, x < xb + s * t)
        + us * (x > xb + s * t)
    )


# }}}


__all__ = (
    "Scheme",
    "Godunov",
    "Rusanov",
    "LaxFriedrichs",
    "EngquistOsher",
    "ESWENO32",
    "SSWENO242",
    "scheme_ids",
    "make_scheme_from_name",
    "ic_rarefaction",
    "ic_sine",
    "ex_shock",
    "ex_tophat",
)
