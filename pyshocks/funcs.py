# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Initial and Exact Solutions
---------------------------

This module contains a set of initial and exact solutions for the various
equations.

Advection Equation Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a constant velocity field :math:`a`, exact solutions can be obtained from
the initial condition :math:`u_0(x)` as

.. math::

    u(t, x) = u_0(x - a t).

Diffusion Equation Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: diffusion_expansion
.. autofunction:: diffusion_tophat

Inviscid Burgers' Equation Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: burgers_riemann
.. autofunction:: burgers_linear_shock
.. autofunction:: burgers_tophat

Misc Functions
^^^^^^^^^^^^^^

.. autofunction:: ic_constant
.. autofunction:: ic_rarefaction
.. autofunction:: ic_sine
.. autofunction:: ic_sine_sine
.. autofunction:: ic_cut_sine
.. autofunction:: ic_gaussian
.. autofunction:: ic_boxcar
"""

from typing import Optional, Tuple

import jax.numpy as jnp

from pyshocks.grid import Grid

# {{{ misc


def ic_constant(grid: Grid, x: jnp.ndarray, *, c: float = 1.0) -> jnp.ndarray:
    """A constant function.

    .. math::

        g(x) = c.
    """
    return jnp.full_like(x, c)  # type: ignore[no-untyped-call]


def ic_rarefaction(
    grid: Grid,
    x: jnp.ndarray,
    *,
    ul: float = 0.0,
    ur: float = 1.0,
    x0: Optional[float] = None,
) -> jnp.ndarray:
    r"""A step function.

    .. math::

        g(x) =
        \begin{cases}
        u_L, & \quad x < x_0, \\
        u_R, & \quad x \ge x_0.
        \end{cases}
    """
    if x0 is None:
        x0 = 0.5 * (grid.a + grid.b)

    h = (x < x0).astype(x.dtype)

    return ul * h + (1 - h) * ur


def ic_sign(grid: Grid, x: jnp.ndarray, *, x0: Optional[float] = None) -> jnp.ndarray:
    r"""A step function like :func:`ic_rarefaction` with :math:`u_L = -1`
    and :math:`u_R = 1`.
    """
    return ic_rarefaction(grid, x, ul=-1.0, ur=1.0, x0=x0)


def ic_sine(grid: Grid, x: jnp.ndarray, *, k: float = 2) -> jnp.ndarray:
    r"""A standard sine function.

    .. math::

        g(x) = \sin \pi k \hat{x},

    where :math:`\hat{x}` is centered in the *grid*, i.e.

    .. math::

        \hat{x} = \frac{x - (b + a) / 2}{b - a}.
    """
    # NOTE: this makes the solution periodic on any domain for even k
    xm = (grid.b + grid.a) / 2
    x = (x - xm) / (grid.b - grid.a)

    return jnp.sin(jnp.pi * k * x)


def ic_sine_sine(
    grid: Grid, x: jnp.ndarray, *, k1: float = 2, k2: float = 2
) -> jnp.ndarray:
    r"""A nested sine function.

    .. math::

        g(x) = \sin \left(k_1 \pi \hat{x} - \frac{\sin k_2 \pi \hat{x}}{\pi}\right).

    where :math:`\hat{x}` is defined in :func:`ic_sine`.
    """
    # NOTE: this makes the solution periodic on any domain for even k
    xm = (grid.b + grid.a) / 2
    x = (x - xm) / (grid.b - grid.a)

    return jnp.sin(k1 * jnp.pi * x + jnp.sin(k2 * jnp.pi * x) / jnp.pi)


def ic_cut_sine(
    grid: Grid,
    x: jnp.ndarray,
    *,
    k: int = 1,
    us: float = 1.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""A discontinuous sine function.

    .. math::

        g(x) =
        \begin{cases}
        \sin (2 \pi k (x - x_a) / (x_b - x_a)), & \quad x_a < x < x_b, \\
        u_s, & \quad \text{otherwise}.
        \end{cases}
    """
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

    return jnp.where(  # type: ignore[no-untyped-call]
        jnp.logical_and(xa < x, x < xb),
        1.0 + jnp.sin(2.0 * jnp.pi * (x - xa) / (xb - xa)),
        us,
    )


def ic_gaussian(
    grid: Grid,
    x: jnp.ndarray,
    *,
    sigma: float = 0.1,
    xc: Optional[float] = None,
    amplitude: Optional[float] = None,
) -> jnp.ndarray:
    r"""A standard Gaussian function.

    .. math::

        g(x) = A \exp\left(-\frac{(x - x_c)^2}{2 \sigma^2}\right).
    """
    if xc is None:
        xc = (grid.b + grid.a) / 2.0

    if amplitude is None:
        amplitude = 1 / jnp.sqrt(2 * jnp.pi * sigma**2)

    return amplitude * jnp.exp(-((x - xc) ** 2) / (2 * sigma**2))


def ic_boxcar(
    grid: Grid,
    x: jnp.ndarray,
    *,
    amplitude: float = 1.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""A boxcar function.

    .. math::

        g(x) =
        \begin{cases}
        A, & \quad x_a < x < x_b, \\
        0, & \quad \text{otherwise}.
        \end{cases}
    """
    return burgers_tophat(grid, 0.0, x, us=0.0, uc=amplitude, xa=xa, xb=xb)


# }}}


# {{{ advection


# }}}


# {{{ diffusion


def diffusion_expansion(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    modes: Tuple[float, ...] = (1.0,),
    amplitudes: Tuple[float, ...] = (1.0,),
    diffusivity: float = 1.0,
) -> jnp.ndarray:
    r"""A series expansion solution for the heat equation.

    .. math::

        u(t, x) = \sum_n A_n \sin \lambda_n x e^{-d \lambda_n^2 t},

    where :math:`A_n` is given by *amplitudes*, :math:`d` is the *diffusivity*
    and :math:`\omega_n` denotes the *modes* in

    .. math::

        \lambda_n = \frac{\pi \omega_n}{L}.
    """
    assert len(modes) > 0
    assert len(modes) == len(amplitudes)
    assert diffusivity > 0

    L = grid.b - grid.a
    return sum(
        [
            a
            * jnp.sin(jnp.pi * n * x / L)
            * jnp.exp(-diffusivity * (n * jnp.pi / L) ** 2 * t)
            for a, n in zip(amplitudes, modes)
        ],
        jnp.zeros_like(x),  # type: ignore[no-untyped-call]
    )


def diffusion_tophat(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    x0: Optional[float] = None,
    diffusivity: float = 1.0,
) -> jnp.ndarray:
    r"""A tophat exact solution for the diffusion equation.

    .. math::

        u(t, x) = \frac{1}{2} \left(
            \operatorname{erf} \left(\frac{1 - 2 x}{4 \sqrt{d t}}\right)
            + \operatorname{erf} \left(\frac{1 + 2 x}{4 \sqrt{d t}}\right)
        \right).
    """
    if x0 is None:
        x0 = (grid.b + grid.a) / 2
        x = x - x0

    from jax.scipy.special import erf

    td = 4 * jnp.sqrt(diffusivity * t) + 1.0e-15
    return (erf((1 - 2 * x) / td) + erf((1 + 2 * x) / td)) / 2  # type: ignore


# }}}


# {{{ burgers


def burgers_riemann(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    ul: float = 1.0,
    ur: float = 0.0,
    x0: Optional[float] = None,
) -> jnp.ndarray:
    r"""Construct a solution for the pure Burgers Riemann problem.

    .. math::

        u(t, x) =
        \begin{cases}
        u_L, & \quad x < x_0, \\
        u_R, & \quad x \ge x_0,
        \end{cases}

    where :math:`x_0` is a given point in :math:`[a, b]`. If :math:`u_L > u_R`,
    we have a shock solutions and otherwise a rarefaction.

    :arg ul: left state value.
    :arg ur: right state value.
    :arg x0: initial location of the shock.
    """
    if x0 is None:
        x0 = 0.5 * (grid.a + grid.b)

    if not grid.a < x0 < grid.b:
        raise ValueError("x0 must be in the domain [a, b]")

    if ul <= ur:
        x_t = (x - x0) / (t + 1.0e-15)
        r = (
            ul * (x < x0 + ul * t)
            + x_t * jnp.logical_and(x0 + ul * t < x, x0 + ur * t > x)
            + ur * (x0 + ur * t < x)
        )
    else:
        # shock velocity
        s = (ul + ur) / 2.0
        # Heaviside indicator for left / right
        h = (x < (x0 + s * t)).astype(x.dtype)

        r = h * ul + (1 - h) * ur

    return r


def burgers_linear_shock(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    ul: float = 1.0,
    ur: float = 0.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""Construct a shock solution for the initial condition

    .. math::

        u(0, x) =
        \begin{cases}
        u_L, & \quad x \le x_a, \\
        u_L + (u_R - u_L) \frac{x - x_a}{x_b - x_a}, & \quad x_a < x < x_b \\
        u_R & \quad x \ge x_b,
        \end{cases}
    """
    xm = (grid.b + grid.a) / 2
    dx = grid.b - grid.a

    if xa is None:
        xa = xm - 0.125 * dx

    if xb is None:
        xb = xm + 0.125 * dx

    if xa >= xb:
        raise ValueError("invalid sides (must be xa < xb)")

    if not grid.a < xa < grid.b:
        raise ValueError("xa must be in the domain [a, b]")

    if not grid.a < xb < grid.b:
        raise ValueError("xb must be in the domain [a, b]")

    if ul <= ur:
        raise NotImplementedError

    # line is given by `a * x + b`
    a = (ul - ur) / (xb - xa)
    b = ul + xa * a

    # time at which the strong solution breaks down
    tmax = 1 / a

    # strong solutions valid for t < tmax
    xt = (b - a * x) / (1 - a * t)
    s0 = (
        ul * (x < xa + ul * t)
        + xt * jnp.logical_and(xa + ul * t < x, xb + ur * t > x)
        + ur * (x > xb + ur * t)
    )

    # weak solution valid for t > tmax
    s = (ul + ur) / 2
    x0 = xb + ur * tmax
    h = (x < (x0 + s * (t - tmax))).astype(x.dtype)
    s1 = ul * h + ur * (1 - h)

    h = t < tmax
    return s0 * h + (1 - h) * s1


def burgers_tophat(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    us: float = 0.0,
    uc: float = 1.0,
    xa: Optional[float] = None,
    xb: Optional[float] = None,
) -> jnp.ndarray:
    r"""Constructs an rarefaction-shock exact solution for the initial condition

    .. math::

        u(t, x) =
        \begin{cases}
        u_C, & \quad x_a < x < x_b, \\
        u_S, & \quad \text{otherwise},
        \end{cases}
    """
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

    if uc <= us:
        raise NotImplementedError

    # shock velocity
    s = (uc + us) / 2

    return (
        us * (x < xa + us * t)
        + (x - xa) / (t + 1.0e-15) * jnp.logical_and(xa + us * t < x, x < xa + uc * t)
        + uc * jnp.logical_and(xa + uc * t < x, x < xb + s * t)
        + us * (x > xb + s * t)
    )


# }}}
