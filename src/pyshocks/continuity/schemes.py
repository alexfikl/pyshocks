# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from pyshocks import (
    Boundary,
    ConservationLawScheme,
    FiniteDifferenceSchemeBase,
    Grid,
    SchemeBase,
    flux,
    numerical_flux,
    predict_timestep,
)
from pyshocks.tools import Array, ScalarLike

# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the continuity equation.

    .. automethod:: __init__
    """

    # NOTE: this is Optional just for mypy, but should never be `None` in practice
    #: Advection velocity
    velocity: Array | None


@flux.register(Scheme)
def _flux_continuity(scheme: Scheme, t: ScalarLike, x: Array, u: Array) -> Array:
    assert scheme.velocity is not None

    return scheme.velocity * u


@predict_timestep.register(Scheme)
def _predict_timestep_continuity(
    scheme: Scheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    assert scheme.velocity is not None

    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@dataclass(frozen=True)
class FiniteVolumeScheme(Scheme, ConservationLawScheme):
    """Base class for finite volume-based numerical schemes for the continuity
    equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(Scheme, FiniteDifferenceSchemeBase):
    """Base class for finite difference-based numerical schemes for the continuity
    equation.

    .. automethod:: __init__
    """


# }}}


# {{{ upwind


@dataclass(frozen=True)
class Godunov(FiniteVolumeScheme):
    """A Godunov (upwind) scheme for the continuity equation.

    The flux of the Godunov scheme is given by
    :func:`~pyshocks.scalar.scalar_flux_upwind`.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_continuity_godunov(
    scheme: Godunov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    a = scheme.velocity
    ul, ur = reconstruct(scheme.rec, grid, bc.boundary_type, u, u, a)
    al, ar = reconstruct(scheme.rec, grid, bc.boundary_type, a, a, a)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, ar[:-1] * ur[:-1], al[1:] * ul[1:])

    return jnp.pad(fnum, 1)


# }}}
