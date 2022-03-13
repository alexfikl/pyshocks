r"""
Continuity Equation
-------------------

This module implements schemes for the conservative continuity equation

.. math::

    \frac{\partial u}{\partial t} + \frac{\partial}{\partial x} (a u) = 0,

where :math:`a(t, x)` is a known velocity field.

.. class:: VelocityFun

    A callable ``Callable[[float, jnp.ndarray], jnp.ndarray]`` for velocity fields.

.. autoclass:: Scheme
.. autoclass:: Godunov

.. autoclass:: WENOJS
.. autoclass:: WENOJS32
.. autoclass:: WENOJS53

Data
^^^^

.. autofunction:: check_oslc
.. autofunction:: velocity_const
.. autofunction:: velocity_sign
.. autofunction:: ex_constant_velocity_field
.. autofunction:: ic_sine
"""

from pyshocks.continuity.schemes import Scheme, Godunov, WENOJS, WENOJS32, WENOJS53

import jax.numpy as jnp


def check_oslc(grid, velocity, *, n=512):
    x = jnp.linspace(grid.a, grid.b, n)
    a = velocity(x)

    dadx = (a[1:] - a[:-1]) / (x[1:] - x[:-1])
    return jnp.max(dadx)


# {{{ velocity fields


def velocity_const(grid, t, x):
    return jnp.ones_like(x)


def velocity_sign(grid, t, x):
    x0 = 0.5 * (grid.a + grid.b)
    a0 = jnp.ones_like(x)

    return jnp.where(x - x0 < 0, -a0, a0)


# }}}


# {{{ exact solutions


def ex_constant_velocity_field(t, x, *, a, u0):
    return u0(x - a * t)


# }}}


# {{{ initial conditions


def ic_gaussian(grid, x, *, sigma=0.1):
    r"""Gaussian initial condition given by

    .. math::

        u_0(x) = \exp\left(-\frac{(x - x_c)^2}{2 \sigma^2}\right),

    where :math:`x_c` is the domain midpoint.
    """
    mu = (grid.b - grid.a) / 2.0
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def ic_sine(grid, x, *, k=1):
    r"""Initial condition given by

    .. math::

        u_0(x) = \sin k \pi x
    """
    return jnp.sin(k * jnp.pi * x)


def ic_sine_sine(grid, x, *, k1=1, k2=1):
    r"""Initial condition given by

    .. math::

        u_0(x) = \sin \left(k_1 \pi x - \frac{\sin k_2 \pi x}{\pi}\right).
    """
    return jnp.sin(k1 * jnp.pi * x + jnp.sin(k2 * jnp.pi * x) / jnp.pi)


def ic_tophat(grid, x, *, x0=None, width=0.25):
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
