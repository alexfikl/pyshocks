# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase, FiniteDifferenceSchemeBase, ConservationLawScheme
from pyshocks import numerical_flux, predict_timestep


# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """
    .. attribute:: diffusivity
    """

    diffusivity: Optional[jnp.ndarray]


@predict_timestep.register(Scheme)
def _predict_timestep_diffusion(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.diffusivity is not None

    dmax = jnp.max(jnp.abs(scheme.diffusivity[grid.i_]))
    return 0.5 * grid.dx_min**2 / dmax


@dataclass(frozen=True)
class FiniteVolumeScheme(Scheme, ConservationLawScheme):
    """Base class for finite volume-based numerical schemes the heat equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(Scheme, FiniteDifferenceSchemeBase):
    """Base class for finite difference-based numerical schemes for the heat equation.

    .. automethod:: __init__
    """


# }}}


# {{{ polynomial


@dataclass(frozen=True)
class CenteredScheme(FiniteVolumeScheme):
    pass


@numerical_flux.register(CenteredScheme)
def _numerical_flux_diffusion_centered_scheme(
    scheme: CenteredScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.diffusivity is not None

    # FIXME: higher order?
    d = scheme.diffusivity
    davg = (d[1:] + d[:-1]) / 2

    fnum = -davg * (u[1:] - u[:-1]) / grid.df

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}
