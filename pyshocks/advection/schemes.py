# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase, Boundary
from pyshocks import numerical_flux, apply_operator, predict_timestep


# {{{ base


@dataclass(frozen=True)
class SpatialVelocity:
    """A placeholder for a spatially-varying velocity field."""

    velocity: jnp.ndarray

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.velocity


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
) -> jnp.ndarray:
    assert scheme.velocity is not None

    # NOTE: keep in sync with pyshocks.continuity.schemes.predict_timestep
    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@apply_operator.register(Scheme)
def _apply_operator_advection(
    scheme: Scheme, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None

    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -scheme.velocity * (f[1:] - f[:-1]) / grid.dx


# }}}


# {{{ upwind


@dataclass(frozen=True)
class Godunov(Scheme):
    """A Godunov (upwind) scheme for the advection equation.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_advection_godunov(
    scheme: Godunov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    # NOTE: the values are given at the cell boundaries as follows
    #
    #       i - 1             i           i + 1
    #   --------------|--------------|--------------
    #           u^R_{i - 1}      u^R_i
    #                   u^L_i         u^L_{i + 1}

    ul, ur = reconstruct(scheme.rec, grid, u)
    al, ar = reconstruct(scheme.rec, grid, scheme.velocity)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, ur[:-1], ul[1:])  # type: ignore[no-untyped-call]

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}
