# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field, replace

import jax.numpy as jnp

from pyshocks import (
    Grid,
    Boundary,
    SchemeBase,
    FiniteDifferenceSchemeBase,
    ConservationLawScheme,
)
from pyshocks import reconstruction
from pyshocks import (
    bind,
    flux,
    numerical_flux,
    predict_timestep,
)


# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
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
    scheme: Scheme, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    # largest wave speed i.e. max |f'(u)|
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min / smax


@dataclass(frozen=True)
class FiniteVolumeScheme(Scheme, ConservationLawScheme):
    """Base class for finite volume-based numerical schemes for Burgers' equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(Scheme, FiniteDifferenceSchemeBase):
    """Base class for finite difference-based numerical schemes for Burgers' equation.

    .. automethod:: __init__
    """


# }}}


# {{{ Godunov


@dataclass(frozen=True)
class Godunov(FiniteVolumeScheme):
    r"""Standard Godunov (upwind) scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_upwind`.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_burgers_godunov(
    scheme: Godunov, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind

    return scalar_flux_upwind(scheme, grid, bc.boundary_type, t, u, u)


# }}}


# {{{ Rusanov


@dataclass(frozen=True)
class Rusanov(FiniteVolumeScheme):
    r"""Modified Rusanov scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_rusanov`.

    .. attribute:: alpha

    .. automethod:: __init__
    """

    alpha: float = 1.0


@numerical_flux.register(Rusanov)
def _numerical_flux_burgers_rusanov(
    scheme: Rusanov, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_rusanov

    return scalar_flux_rusanov(
        scheme, grid, bc.boundary_type, t, u, u, alpha=scheme.alpha
    )


@predict_timestep.register(Rusanov)
def _predict_timestep_burgers_rusanov(
    scheme: Rusanov, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
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
    scheme: LaxFriedrichs, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_lax_friedrichs

    return scalar_flux_lax_friedrichs(
        scheme, grid, bc.boundary_type, t, u, u, alpha=scheme.alpha
    )


# }}}


# {{{ Engquist-Osher


@dataclass(frozen=True)
class EngquistOsher(FiniteVolumeScheme):
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
    scheme: EngquistOsher, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_engquist_osher

    return scalar_flux_engquist_osher(
        scheme, grid, bc.boundary_type, t, u, omega=scheme.omega
    )


# }}}

# {{{ ESWENO32


@dataclass(frozen=True)
class ESWENO32(FiniteVolumeScheme):
    r"""Third-order Energy Stable WENO (ESWENO) scheme by [Yamaleev2009]_.

    This scheme requires the :class:`~pyshocks.reconstruction.ESWENO32`
    reconstruction. It adds a diffusive flux to ensure the energy stability
    of the method, as described in [Yamaleev2009]_.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.ESWENO32):
            raise TypeError("ESWENO32 scheme requires the ESWENO32 reconstruction")


@bind.register(ESWENO32)
def _bind_burgers_esweno32(  # type: ignore[misc]
    scheme: ESWENO32, grid: Grid, bc: Boundary
) -> ESWENO32:
    from pyshocks.weno import es_weno_parameters

    # NOTE: prefer the parameters recommended by Carpenter!
    eps, delta = es_weno_parameters(grid, 1.0)
    return replace(scheme, rec=replace(scheme.rec, eps=eps, delta=delta))


@numerical_flux.register(ESWENO32)
def _numerical_flux_burgers_esweno32(
    scheme: ESWENO32, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind
    from pyshocks.weno import es_weno_weights

    rec = scheme.rec
    assert isinstance(rec, reconstruction.ESWENO32)

    # {{{ compute dissipative flux of ESWENO

    # NOTE: computing these twice :(
    omega = es_weno_weights(rec.s, u, eps=rec.eps)[0, :]

    # NOTE: see Equation 37 in [Yamaleev2009] for mu expression
    mu = jnp.sqrt((omega[1:] - omega[:-1]) ** 2 + rec.delta**2) / 8.0

    # NOTE: see Equation  in [Yamaleev2009] for flux expression
    gnum = -(mu + (omega[1:] - omega[:-1]) / 8.0) * (u[1:] - u[:-1])

    gnum = jnp.pad(gnum, 1)  # type: ignore[no-untyped-call]

    # }}}

    fnum = scalar_flux_upwind(scheme, grid, bc.boundary_type, t, u, u)
    return fnum + gnum


# }}}


# {{{ SSMUSCL


@dataclass(frozen=True)
class SSMUSCL(FiniteVolumeScheme):
    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.MUSCLS):
            raise TypeError("SSMUSCL scheme requires the MUSCLS reconstruction")


@numerical_flux.register(SSMUSCL)
def _numerical_flux_burgers_ssmuscl(
    scheme: SSMUSCL, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind

    return scalar_flux_upwind(scheme, grid, bc.boundary_type, t, u, u)


# }}}


# vim: fdm=marker
