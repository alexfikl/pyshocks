# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import Grid, ConservationLawSchemeV2
from pyshocks import flux, numerical_flux, predict_timestep


# {{{ base


@dataclass(frozen=True)
class Scheme(ConservationLawSchemeV2):  # pylint: disable=abstract-method
    """Base class for numerical schemes for the continuity equation.

    .. attribute:: velocity

        Advection velocity at cell centers.

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
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None

    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


# }}}


# {{{ upwind


@dataclass(frozen=True)
class Godunov(Scheme):
    """A Godunov (upwind) scheme for the continuity equation.

    The flux of the Godunov scheme is given by
    :func:`~pyshocks.scalar.scalar_flux_upwind`.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_continuity_godunov(
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
    fnum = jnp.where(
        aavg > 0, ar[:-1] * ur[:-1], al[1:] * ul[1:]  # type: ignore[no-untyped-call]
    )

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}
