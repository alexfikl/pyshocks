# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import jax.numpy as jnp

from pyshocks import Grid, ConservationLawScheme
from pyshocks import numerical_flux, predict_timestep


# {{{ base


@dataclass(frozen=True)
class Scheme(ConservationLawScheme):
    """
    .. attribute:: diffusivity
    """

    diffusivity: jnp.ndarray


@predict_timestep.register(Scheme)
def _predict_timestep_diffusion(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> float:
    dmax = jnp.max(jnp.abs(scheme.diffusivity[grid.i_]))

    return 0.5 * grid.dx_min**2 / dmax


# }}}


# {{{ polynomial


@dataclass(frozen=True)
class CenteredScheme(Scheme):
    @property
    def order(self):
        return 2

    @property
    def stencil_width(self):
        return 1


@numerical_flux.register(CenteredScheme)
def _numerical_flux_diffusion_centered_scheme(
    scheme: CenteredScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    # FIXME: higher order?
    d = scheme.diffusivity
    davg = (d[1:] + d[:-1]) / 2

    fnum = -davg * (u[1:] - u[:-1]) / grid.df

    return jnp.pad(fnum, 1)


# }}}
