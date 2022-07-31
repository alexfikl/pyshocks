# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import jax.numpy as jnp

from pyshocks import Grid, ConservationLawSchemeV2
from pyshocks import flux, numerical_flux, predict_timestep


# {{{ base


@dataclass(frozen=True)
class Scheme(ConservationLawSchemeV2):  # pylint: disable=abstract-method
    """Base class for numerical schemes for Burgers' equation.

    .. automethod:: __init__
    """


@flux.register(Scheme)
def _flux_burgers(
    scheme: Scheme, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    return u**2 / 2


@predict_timestep.register(Scheme)
def _predict_timestep_burgers(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    # largest wave speed i.e. max |f'(u)|
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min / smax


# }}}


# {{{ Rusanov


@dataclass(frozen=True)
class Rusanov(Scheme):
    r"""Modified Rusanov scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_rusanov`.

    .. attribute:: alpha

    .. automethod:: __init__
    """

    alpha: float = 1.0


@numerical_flux.register(Rusanov)
def _numerical_flux_burgers_rusanov(
    scheme: Rusanov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_rusanov

    return scalar_flux_rusanov(scheme, grid, t, u, u, alpha=scheme.alpha)


@predict_timestep.register(Rusanov)
def _predict_timestep_burgers_rusanov(
    scheme: Rusanov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min ** (2 - scheme.alpha) / smax


# }}}


# {{{ Lax-Friedrichs


@dataclass(frozen=True)
class LaxFriedrichs(Rusanov):
    r"""Modified Lax-Friedrichs scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_lax_friedrichs`.

    .. attribute:: alpha

    .. automethod:: __init__
    """


@numerical_flux.register(LaxFriedrichs)
def _numerical_flux_burgers_lax_friedrichs(
    scheme: LaxFriedrichs, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_lax_friedrichs

    return scalar_flux_lax_friedrichs(scheme, grid, t, u, u, alpha=scheme.alpha)


# }}}


# {{{ Engquist-Osher


@dataclass(frozen=True)
class EngquistOsher(Scheme):
    r"""Classic Engquist-Osher scheme with the flux

    .. math::

        f(u_l, u_r) = \frac{1}{2} (f(u_l) + f(u_r)) -
            \frac{1}{2} \int_{u_l}^{u_r} |f'(u)| \,\mathrm{d}u.

    In the case of a convex flux, the flux can be simplified to

    .. math

        f(u_l, u_r) = f(u_l^+) + f(u_r^-) - f(\omega),

    where :math:`u_l^+ = \max(u_l, \omega)` and :math:`u_r^- = \min(u_r, \omega)`.
    Here, :math:`\omega = 0` is the point at which the Burgers flux attains its
    minimum.

    .. attribute:: omega

    .. automethod:: __init__
    """

    omega: float = field(default=0, init=False, repr=False)


@numerical_flux.register(EngquistOsher)
def _numerical_flux_burgers_engquist_osher(
    scheme: EngquistOsher, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_engquist_osher

    return scalar_flux_engquist_osher(scheme, grid, t, u, omega=scheme.omega)


# }}}

# vim: fdm=marker
