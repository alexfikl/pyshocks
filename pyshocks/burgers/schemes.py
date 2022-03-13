from dataclasses import dataclass, field

import jax.numpy as jnp

from pyshocks import Grid, ConservationLawScheme
from pyshocks import flux, numerical_flux, predict_timestep
from pyshocks.weno import WENOJSMixin, WENOJS32Mixin, WENOJS53Mixin


# {{{ base


@dataclass(frozen=True)
class Scheme(ConservationLawScheme):
    """Base class for numerical schemes for Burgers' equation."""


@flux.register(Scheme)
def _flux_burgers(
    scheme: Scheme, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    return u**2 / 2


@predict_timestep.register(Scheme)
def _predict_timestep_burgers(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> float:
    # largest wave speed i.e. max |f'(u)|
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min / smax


# }}}


# {{{ Lax-Friedrichs


@dataclass(frozen=True)
class LaxFriedrichs(Scheme):
    r"""Modified Lax-Friedrichs scheme with the flux

    .. math::

        f(u_l, u_r) = \frac{1}{2 \Delta x_{lr}} (f(u_l) + f(u_r))
            + \frac{1}{2} \frac{\Delta x^\alpha}{\Delta x^2} (u_r - u_l),

    where :math:`\alpha \le 1`. The limit case of :math:`\alpha = 1` gives the
    classic scheme. An :math:`\alpha < 1` acts as additional artificial
    viscosity.

    .. attribute:: alpha
    .. attribute:: order

    .. automethod:: __init__
    """

    alpha: float = 1.0

    @property
    def order(self):
        return self.alpha


@numerical_flux.register(LaxFriedrichs)
def _numerical_flux_burgers_lax_friedrichs(
    scheme: LaxFriedrichs, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_lax_friedrichs

    return scalar_flux_lax_friedrichs(scheme, grid, t, u, u, alpha=scheme.alpha)


@predict_timestep.register(LaxFriedrichs)
def _predict_timestep_burgers_lax_friedrichs(
    scheme: LaxFriedrichs, grid: Grid, t: float, u: jnp.ndarray
) -> float:
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min ** (2 - scheme.alpha) / smax


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
    Here, :math:`\omega = 0` is the point at which the flux attains its
    minimum.

    .. attribute:: order
    .. automethod:: __init__
    """

    omega: float = field(default=0, init=False, repr=False)

    @property
    def order(self):
        return 1


@numerical_flux.register(EngquistOsher)
def _numerical_flux_burgers_engquist_osher(
    scheme: EngquistOsher, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_engquist_osher

    return scalar_flux_engquist_osher(scheme, grid, t, u, omega=scheme.omega)


# }}}


# {{{ WENO


@dataclass(frozen=True)
class WENOJS(Scheme, WENOJSMixin):  # pylint: disable=abstract-method
    """Classic (finite volume) WENO schemes by Jiang and Shu. Implementation
    follows the steps from [Shu2009]_.

    .. [Shu2009] C.-W. Shu, *High Order Weighted Essentially Nonoscillatory
        Schemes for Convection Dominated Problems*,
        SIAM Review, Vol. 51, pp. 82--126, 2009,
        `DOI <http://dx.doi.org/10.1137/070679065>`__.

    .. attribute:: eps
    .. automethod:: __init__
    """

    def __post_init__(self):
        # pylint: disable=no-member
        self.set_coefficients()


@dataclass(frozen=True)
class WENOJS32(WENOJS32Mixin, WENOJS):
    """Third-order WENO scheme based on :class:`WENOJS`."""

    eps: float = 1.0e-6


@dataclass(frozen=True)
class WENOJS53(WENOJS53Mixin, WENOJS):
    """Fifth-order WENO scheme based on :class:`WENOJS`."""

    eps: float = 1.0e-12


@numerical_flux.register(WENOJS)
def _numerical_flux_burgers_wenojs(
    scheme: WENOJS, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    # FIXME: use scalar_flux_lax_friedrichs? somehow?
    # WENO should probably have a Riemann solver-based flux thing passed to it

    from pyshocks.weno import reconstruct

    up = reconstruct(grid, scheme, u)
    fp = flux(scheme, t, grid.f, up)

    um = reconstruct(grid, scheme, u[::-1])[::-1]
    fm = flux(scheme, t, grid.f, um)

    # NOTE: using the *global* Lax-Friedrichs flux
    umax = jnp.max(jnp.abs(u[grid.i_]))
    fnum = 0.5 * (fp[:-1] + fm[1:] + umax * (up[:-1] - um[1:]))

    return jnp.pad(fnum, 1)


# }}}

# vim: fdm=marker
