# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import (
    Grid,
    Boundary,
    SchemeBase,
    FiniteDifferenceSchemeBase,
    ConservationLawScheme,
)
from pyshocks import flux, numerical_flux, predict_timestep


# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the continuity equation.

    .. attribute:: velocity

        Advection velocity.

    .. automethod:: __init__
    """

    # NOTE: this is Optional just for mypy, but should never be `None` in practice
    velocity: Optional[jnp.ndarray]


@flux.register(Scheme)
def _flux_continuity(
    scheme: Scheme, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None

    return scheme.velocity * u


@predict_timestep.register(Scheme)
def _predict_timestep_continuity(
    scheme: Scheme, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
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
    scheme: Godunov, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    ul, ur = reconstruct(scheme.rec, grid, bc.boundary_type, u)
    al, ar = reconstruct(scheme.rec, grid, bc.boundary_type, scheme.velocity)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(
        aavg > 0, ar[:-1] * ur[:-1], al[1:] * ul[1:]  # type: ignore[no-untyped-call]
    )

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}
