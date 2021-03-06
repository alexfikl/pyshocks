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
.. autoclass:: Godunov

.. autoclass:: WENOJS
.. autoclass:: WENOJS32
.. autoclass:: WENOJS53

Helpers
^^^^^^^

.. autofunction:: check_oslc

Data
^^^^

.. autofunction:: velocity_const
.. autofunction:: velocity_sign
.. autofunction:: ex_constant_velocity_field
.. autofunction:: ic_sine
"""

from typing import Optional

from pyshocks.grid import Grid
from pyshocks.schemes import SpatialFunction
from pyshocks.continuity.schemes import Scheme, Godunov, WENOJS, WENOJS32, WENOJS53

import jax.numpy as jnp


def check_oslc(grid: Grid, velocity: SpatialFunction, *, n: int = 512) -> float:
    r"""Check the One-sided Lipschitz Continuous condition.

    A function :math:`f(t, x)` is called one-sided Lipschitz continuous if
    there exists an integrable function :math:`L(t)` such that for every
    :math:`x_1, x_2 \in \mathbb{R}^d` such that

    .. math

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


# {{{ velocity fields


def velocity_const(grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluates a constant velocity field on *grid*."""
    return jnp.ones_like(x)


def velocity_sign(grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
    r"""Evaluates a sign velocity field on *grid*.

    .. math::

        a(x) =
        \begin{cases}
        -1, & \quad x < (a + b) / 2, \\
        +1, & \quad x > (a + b) / 2.
        \end{cases}
    """

    x0 = 0.5 * (grid.a + grid.b)
    a0 = jnp.ones_like(x)

    return jnp.where(x - x0 < 0, -a0, a0)


# }}}


# {{{ exact solutions


def ex_constant_velocity_field(
    t: float, x: jnp.ndarray, *, a: float, u0: SpatialFunction
) -> jnp.ndarray:
    """Evaluates exact solution for a constant velocity field.

    The exact solution is simply given by the traveling wave

    .. math::

        u(t, x) = u_0(x - a t)
    """

    return u0(x - a * t)


# }}}


# {{{ initial conditions


def ic_gaussian(grid: Grid, x: jnp.ndarray, *, sigma: float = 0.1) -> jnp.ndarray:
    r"""Gaussian initial condition given by

    .. math::

        u_0(x) = \exp\left(-\frac{(x - x_c)^2}{2 \sigma^2}\right),

    where :math:`x_c` is the domain midpoint.
    """
    mu = (grid.b - grid.a) / 2.0
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def ic_sine(grid: Grid, x: jnp.ndarray, *, k: float = 1) -> jnp.ndarray:
    r"""Initial condition given by

    .. math::

        u_0(x) = \sin k \pi x
    """
    return jnp.sin(k * jnp.pi * x)


def ic_sine_sine(
    grid: Grid, x: jnp.ndarray, *, k1: float = 1, k2: float = 1
) -> jnp.ndarray:
    r"""Initial condition given by

    .. math::

        u_0(x) = \sin \left(k_1 \pi x - \frac{\sin k_2 \pi x}{\pi}\right).
    """
    return jnp.sin(k1 * jnp.pi * x + jnp.sin(k2 * jnp.pi * x) / jnp.pi)


def ic_tophat(
    grid: Grid, x: jnp.ndarray, *, x0: Optional[float] = None, width: float = 0.25
) -> jnp.ndarray:
    r"""Initial condition given by

    .. math::

        u_0(x) =
        \begin{cases}
        1, & \quad (a + b - w) / 2 < x < (a + b + w) / 2, \\
        0, & \quad \text{otherwise}.
        \end{cases}
    """
    if x0 is None:
        x0 = 0.5 * (grid.a + grid.b)

    width = width * (grid.b - grid.a)
    value = jnp.ones_like(x)

    return jnp.where(
        jnp.logical_and((x0 - width / 2) < x, x < (x0 + width / 2)), value, -value
    )


# }}}


__all__ = (
    "Scheme",
    "Godunov",
    "WENOJS",
    "WENOJS32",
    "WENOJS53",
    "velocity_const",
    "velocity_sign",
    "ic_sine",
    "ex_constant_velocity_field",
)
