# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase, ConservationLawScheme, Boundary
from pyshocks import flux, numerical_flux, predict_timestep, apply_operator


# {{{ base


@dataclass(frozen=True)
class Scheme(ConservationLawScheme):  # pylint: disable=abstract-method
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


# {{{ Godunov


@dataclass(frozen=True)
class Godunov(Scheme):
    r"""Standard Godunov (upwind) scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_upwind`.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_burgers_godunov(
    scheme: Godunov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind

    return scalar_flux_upwind(scheme, grid, t, u, u)


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

# {{{ ESWENO32


@dataclass(frozen=True)
class ESWENO32(Scheme):
    """Third-order Energy Stable WENO (ESWENO) scheme by [Yamaleev2009]_."""


@numerical_flux.register(ESWENO32)
def _numerical_flux_burgers_esweno32(
    scheme: ESWENO32, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind
    from pyshocks.weno import es_weno_weights
    from pyshocks import reconstruction

    rec = scheme.rec
    if not isinstance(rec, reconstruction.ESWENO32):
        raise TypeError("ESWENO32 scheme requires the ESWENO32 reconstruction")

    # {{{ compute dissipative flux of ESWENO

    # NOTE: computing these twice :(
    omega = es_weno_weights(u, rec.a, rec.b, rec.d, eps=rec.eps)[0, :]

    # NOTE: see Equation 37 in [Yamaleev2009] for mu expression
    mu = jnp.sqrt((omega[1:] - omega[:-1]) ** 2 + rec.delta**2) / 8.0

    # NOTE: see Equation  in [Yamaleev2009] for flux expression
    gnum = -(mu + (omega[1:] - omega[:-1]) / 8.0) * (u[1:] - u[:-1])

    gnum = jnp.pad(gnum, 1)  # type: ignore[no-untyped-call]

    # }}}

    fnum = scalar_flux_upwind(scheme, grid, t, u, u)
    return fnum + gnum


# }}}


# {{{ SSWENO242


@dataclass(frozen=True)
class SSWENO242(Scheme):
    """Fourth-order Energy Stable WENO (ESWENO) scheme by [Yamaleev2009]_.

    .. [Fisher2013] T. C. Fisher, M. H. Carpenter, *High-Order Entropy Stable
        Finite Difference Schemes for Nonlinear Conservation Laws: Finite Domains*,
        Journal of Computational Physics, Vol. 252, pp. 518--557, 2013,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2013.06.014>`__.
    """


@numerical_flux.register(SSWENO242)
def _numerical_flux_burgers_ssweno242(
    scheme: SSWENO242, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    raise NotImplementedError


# }}}


# {{{ SBP


@dataclass(frozen=True)
class SBP(SchemeBase):  # pylint: disable=abstract-method
    """Implements a finite difference Summation-by-Parts (SBP) scheme with
    a Simultaneous-Approximation-Term (SAT) for boundary conditions.

    .. attribute:: tau

        Weight used for the SAT term.
    """

    tau: float

    def __post_init__(self) -> None:
        assert self.tau >= 0.5

    @property
    def stencil_width(self) -> int:
        return 0


@flux.register(SBP)
def _flux_burgers_sbp(
    scheme: SBP, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    return u**2 / 2


@predict_timestep.register(SBP)
def _predict_timestep_burgers_sbp(
    scheme: SBP, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    # largest wave speed i.e. max |f'(u)|
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min / smax


@apply_operator.register(SBP)
def _apply_operator_sbp(
    scheme: SBP, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    raise NotImplementedError


@dataclass(frozen=True)
class SBP21(SBP):
    @property
    def order(self) -> int:
        return 2


@dataclass(frozen=True)
class SBP42(SBP):
    @property
    def order(self) -> int:
        return 4


# }}}

# vim: fdm=marker
