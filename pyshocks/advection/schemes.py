# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase, Boundary
from pyshocks import numerical_flux, apply_operator, predict_timestep
from pyshocks.weno import WENOJSMixin, WENOJS32Mixin, WENOJS53Mixin


# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the linear advection equation.

    .. attribute:: velocity

        Advection velocity at cell centers.

    .. automethod:: __init__
    """

    # NOTE: this is Optional just for mypy, but should never be `None` in practice
    # FIXME: we want this to be a function so that we can evaluate it at
    # (t, x) every time
    velocity: Optional[jnp.ndarray]


@predict_timestep.register(Scheme)
def _predict_timestep_advection(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> float:
    assert scheme.velocity is not None

    # NOTE: keep in sync with pyshocks.continuity.schemes.predict_timestep
    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@apply_operator.register(Scheme)
def _apply_operator_advection(
    scheme: Scheme, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
):
    assert scheme.velocity is not None

    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -scheme.velocity * (f[1:] - f[:-1]) / grid.dx


# }}}


# {{{ upwind


@dataclass(frozen=True)
class Godunov(Scheme):
    """First-order Godunov (upwind) scheme for the advection equation.

    .. automethod:: __init__
    """

    @property
    def order(self):
        return 1

    @property
    def stencil_width(self):
        return 1


@numerical_flux.register(Godunov)
def _numerical_flux_advection_godunov(
    scheme: Godunov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert u.shape[0] == grid.x.size

    a = scheme.velocity
    aavg = (a[1:] + a[:-1]) / 2
    fnum = jnp.where(aavg > 0, u[:-1], u[1:])

    return jnp.pad(fnum, 1)


# }}}


# {{{ WENOJS


@dataclass(frozen=True)
class WENOJS(Scheme, WENOJSMixin):  # pylint: disable=abstract-method
    """See :class:`pyshocks.burgers.WENOJS`."""


@dataclass(frozen=True)
class WENOJS32(WENOJS32Mixin, WENOJS):
    """See :class:`pyshocks.burgers.WENOJS32`.

    .. automethod:: __init__
    """

    eps: float = 1.0e-6

    def __post_init__(self):
        self.set_coefficients()


@dataclass(frozen=True)
class WENOJS53(WENOJS53Mixin, WENOJS):
    """See :class:`pyshocks.burgers.WENOJS53`.

    .. automethod:: __init__
    """

    eps: float = 1.0e-12

    def __post_init__(self):
        self.set_coefficients()


@numerical_flux.register(WENOJS)
def _numerical_flux_advection_wenojs(
    scheme: WENOJS, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert u.size == grid.x.size

    from pyshocks.weno import reconstruct

    up = reconstruct(grid, scheme, u)
    ap = reconstruct(grid, scheme, scheme.velocity)

    um = reconstruct(grid, scheme, u[::-1])[::-1]
    am = reconstruct(grid, scheme, scheme.velocity[::-1])[::-1]

    # NOTE: this uses an upwind flux
    aavg = (ap[:-1] + am[1:]) / 2
    fnum = jnp.where(aavg > 0, up[:-1], um[1:])

    return jnp.pad(fnum, 1)


# }}}
