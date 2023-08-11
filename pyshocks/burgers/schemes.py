# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field, replace
from typing import ClassVar, cast

import jax.numpy as jnp

import pyshocks.finitedifference as fd
from pyshocks import (
    Boundary,
    ConservationLawScheme,
    FiniteDifferenceSchemeBase,
    Grid,
    SchemeBase,
    UniformGrid,
    apply_operator,
    bind,
    flux,
    numerical_flux,
    predict_timestep,
    reconstruction,
)
from pyshocks.tools import Array, Scalar, ScalarLike

# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for Burgers' equation.

    .. automethod:: __init__
    """


@flux.register(Scheme)
def _flux_burgers(scheme: Scheme, t: ScalarLike, x: Array, u: Array) -> Array:
    return u**2 / 2


@predict_timestep.register(Scheme)
def _predict_timestep_burgers(
    scheme: Scheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Scalar:
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
    scheme: Godunov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    from pyshocks.scalar import scalar_flux_upwind

    return scalar_flux_upwind(scheme, grid, bc.boundary_type, t, u, u)


# }}}


# {{{ Rusanov


@dataclass(frozen=True)
class Rusanov(FiniteVolumeScheme):
    r"""Modified Rusanov scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_rusanov`.

    .. automethod:: __init__
    """

    #: Coefficient controlling the amount of artificial dissipation.
    alpha: float = 1.0


@numerical_flux.register(Rusanov)
def _numerical_flux_burgers_rusanov(
    scheme: Rusanov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    from pyshocks.scalar import scalar_flux_rusanov

    return scalar_flux_rusanov(
        scheme, grid, bc.boundary_type, t, u, u, alpha=scheme.alpha
    )


@predict_timestep.register(Rusanov)
def _predict_timestep_burgers_rusanov(
    scheme: Rusanov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Scalar:
    smax = jnp.max(jnp.abs(u[grid.i_]))

    return 0.5 * grid.dx_min ** (2 - scheme.alpha) / smax


# }}}


# {{{ Lax-Friedrichs


@dataclass(frozen=True)
class LaxFriedrichs(Rusanov):
    r"""Modified Lax-Friedrichs scheme with the flux given by
    :func:`~pyshocks.scalar.scalar_flux_lax_friedrichs`.

    .. automethod:: __init__
    """


@numerical_flux.register(LaxFriedrichs)
def _numerical_flux_burgers_lax_friedrichs(
    scheme: LaxFriedrichs, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
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

    .. automethod:: __init__
    """

    #: Point at which the flux attains a global minimum.
    omega: float = field(default=0, init=False, repr=False)


@numerical_flux.register(EngquistOsher)
def _numerical_flux_burgers_engquist_osher(
    scheme: EngquistOsher, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    from pyshocks.scalar import scalar_flux_engquist_osher

    return scalar_flux_engquist_osher(
        scheme, grid, bc.boundary_type, t, u, u, omega=scheme.omega
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
            raise TypeError("ESWENO32 scheme requires the ESWENO32 reconstruction.")


@bind.register(ESWENO32)
def _bind_burgers_esweno32(  # type: ignore[misc]
    scheme: ESWENO32, grid: Grid, bc: Boundary
) -> ESWENO32:
    from pyshocks.weno import es_weno_parameters

    # NOTE: prefer the parameters recommended by Carpenter!
    eps, delta = es_weno_parameters(grid, jnp.ones_like(grid.x))

    assert isinstance(scheme.rec, reconstruction.ESWENO32)
    return replace(scheme, rec=replace(scheme.rec, eps=eps, delta=delta))


@numerical_flux.register(ESWENO32)
def _numerical_flux_burgers_esweno32(
    scheme: ESWENO32, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
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

    gnum = jnp.pad(gnum, 1)

    # }}}

    fnum = scalar_flux_upwind(scheme, grid, bc.boundary_type, t, u, u)
    return fnum + gnum


# }}}


# {{{ SSMUSCL


@dataclass(frozen=True)
class SSMUSCL(FiniteVolumeScheme):
    """Implements the entropy stable MUSCL scheme described in [Hesthaven2018]_.

    This implements the entropy stable schemes described in Example 8.15
    from [Hesthaven2018]_. The variants define a first-order and a second-order
    scheme.
    """

    #: An integer denoting the version of the schemes described in
    #: [Hesthaven2018]_. First variant denotes the limiter defined on
    #: page 193 and the second variant denotes the limiter defined on
    #: page 194.
    variant: int = 2

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_v{self.variant}".lower()

    @property
    def order(self) -> int:
        return self.variant


def hesthaven_limiter(u: Array, *, variant: int = 1) -> Array:
    # [Hesthaven2018] Page 193
    # gives phi_{i + 1/2} for i in [0, n - 1]
    phi = jnp.where(
        u[:-1] * u[1:] > 0,
        1 - jnp.sign(u[:-1]),
        jnp.where(
            u[:-1] >= u[1:],
            1 - jnp.sign(u[:-1] + u[1:]),
            -2 * u[:-1] / (u[1:] - u[:-1]),
        ),
    )

    if variant == 1:
        pass
    elif variant == 2:
        from pyshocks.limiters import local_slope_ratio

        # [Hesthaven2018] Page 194
        # r_i for i in [1, n - 1]
        r = jnp.pad(local_slope_ratio(u), 1)

        phi = jnp.where(
            # if u_i * u_{i + 1} have different signs, revert to variant 1
            u[:-1] * u[1:] <= 0,
            phi,
            # otherwise use second-order scheme
            jnp.where(
                u[:-1] > 0,
                jnp.minimum(1, r[:-1]),
                jnp.maximum(1, 2 - r[:-1]),
            ),
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return cast(Array, phi)


@numerical_flux.register(SSMUSCL)
def _numerical_flux_burgers_ssmuscl(
    scheme: SSMUSCL, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    # FIXME: any way to include this in the MUSCL reconstruction?
    phi = hesthaven_limiter(u, variant=scheme.variant)
    up = u[:-1] + 0.5 * phi * (u[1:] - u[:-1])

    fnum = flux(scheme, t, grid.f, up)
    return jnp.pad(fnum, 1)


# }}}


# {{{ flux splitting Rusanov


@dataclass(frozen=True)
class FluxSplitRusanov(FiniteDifferenceScheme):
    """A standard Rusanov scheme using flux-splitting techniques.

    .. automethod:: __init__
    """

    #: Order of the one sided flux finite difference approximations.
    sorder: int

    #: Finite difference stencil biased to the right.
    sp: ClassVar[fd.Stencil]
    #: Finite difference stencil biased to the left.
    sm: ClassVar[fd.Stencil]

    @property
    def order(self) -> int:
        return self.sorder

    @property
    def stencil_width(self) -> int:
        return self.sorder


@bind.register(FluxSplitRusanov)
def _bind_advection_flux_split_rusanov(  # type: ignore[misc]
    scheme: FluxSplitRusanov, grid: UniformGrid, bc: Boundary
) -> FluxSplitRusanov:
    assert isinstance(grid, UniformGrid)
    assert scheme.order % 2 == 1

    km = (scheme.order - 2) if scheme.order > 1 else 0
    kp = scheme.order

    sm = fd.make_taylor_approximation(1, (-km, kp))
    sp = fd.make_taylor_approximation(1, (-kp, km))

    object.__setattr__(scheme, "sm", sm)
    object.__setattr__(scheme, "sp", sp)

    return scheme


@apply_operator.register(FluxSplitRusanov)
def _apply_operator_flux_split_rusanov(
    scheme: FluxSplitRusanov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    f = flux(scheme, t, grid.x, u)
    a = jnp.abs(u)

    dfp = fd.apply_derivative(scheme.sm, (f + a * u) / 2, grid.dx)
    dfm = fd.apply_derivative(scheme.sp, (f - a * u) / 2, grid.dx)

    # FIXME: why isn't this `-dflux`?
    return dfp + dfm


# }}}

# vim: fdm=marker
