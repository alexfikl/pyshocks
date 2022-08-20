# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from pyshocks import (
    Grid,
    UniformGrid,
    SchemeBase,
    FiniteVolumeSchemeBase,
    FiniteDifferenceSchemeBase,
    Boundary,
)
from pyshocks import (
    bind,
    apply_operator,
    numerical_flux,
    predict_timestep,
    evaluate_boundary,
)
from pyshocks import sbp


# {{{ base


@dataclass(frozen=True)
class SpatialVelocity:
    """A placeholder for a spatially-varying velocity field."""

    velocity: jnp.ndarray

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        assert self.velocity.shape == x.shape
        return self.velocity


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the linear advection equation.

    .. attribute:: velocity

        Advection velocity.

    .. automethod:: __init__
    """

    velocity: jnp.ndarray


@predict_timestep.register(Scheme)
def _predict_timestep_advection(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None

    # NOTE: keep in sync with pyshocks.continuity.schemes.predict_timestep
    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@apply_operator.register(Scheme)
def _apply_operator_advection(
    scheme: Scheme, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.velocity is not None

    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -scheme.velocity * (f[1:] - f[:-1]) / grid.dx


@dataclass(frozen=True)
class FiniteVolumeScheme(Scheme, FiniteVolumeSchemeBase):
    """Base class for finite volume-based numerical schemes for the linear
    advection equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(Scheme, FiniteDifferenceSchemeBase):
    """Base class for finite difference-based numerical schemes for the linear
    advection equation.

    .. automethod:: __init__
    """


# }}}


# {{{ godunov


def upwind_flux(scheme: Scheme, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    ul, ur = reconstruct(scheme.rec, grid, u)
    al, ar = reconstruct(scheme.rec, grid, scheme.velocity)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, ur[:-1], ul[1:])  # type: ignore[no-untyped-call]

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class Godunov(FiniteVolumeScheme):
    """A Godunov (upwind) scheme for the advection equation.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_advection_godunov(
    scheme: Godunov, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return upwind_flux(scheme, grid, u)


# }}}


# {{{ ESWENO


def esweno_lf_flux(scheme: Scheme, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    f = scheme.velocity * u
    ul, ur = reconstruct(scheme.rec, grid, u)
    fl, fr = reconstruct(scheme.rec, grid, f)

    a = jnp.max(jnp.abs(scheme.velocity))
    fnum = 0.5 * (fl[1:] + fr[:-1]) - 0.5 * a * (ul[1:] - ur[:-1])
    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class ESWENO32(FiniteVolumeScheme):
    """Third-order Energy Stable WENO (ESWENO) scheme by [Yamaleev2009]_.

    .. [Yamaleev2009] N. K. Yamaleev, M. H. Carpenter, *Third-Order Energy
        Stable WENO Scheme*,
        Journal of Computational Physics, Vol. 228, pp. 3025--3047, 2009,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2009.01.011>`__.
    """


@numerical_flux.register(ESWENO32)
def _apply_derivative_burgers_esweno32(
    scheme: ESWENO32, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.weno import es_weno_weights
    from pyshocks import reconstruction

    rec = scheme.rec

    if isinstance(rec, reconstruction.ESWENO32):
        # {{{ compute dissipative flux of ESWENO

        # NOTE: computing these twice :(
        _, omega1 = es_weno_weights(u, rec.a, rec.b, rec.d, eps=rec.eps)

        # NOTE: see Equation 37 in [Yamaleev2009] for mu expression
        mu = jnp.sqrt((omega1[1:] - omega1[:-1]) ** 2 + rec.delta**2) / 8.0

        # NOTE: see Equation  in [Yamaleev2009] for flux expression
        gnum = -(mu + (omega1[1:] - omega1[:-1]) / 8.0) * (u[1:] - u[:-1])

        gnum = jnp.pad(gnum, 1)  # type: ignore[no-untyped-call]

        # }}}
    else:
        gnum = 0.0

    return esweno_lf_flux(scheme, grid, u) + gnum


# }}}


# {{{ SBP-SAT


@dataclass(frozen=True)
class SBPSAT(FiniteDifferenceScheme):
    op: sbp.SBPOperator

    P: ClassVar[jnp.ndarray]
    D1: ClassVar[jnp.ndarray]

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{type(self.op).__name__}".lower()

    @property
    def order(self) -> int:
        return self.op.boundary_order

    @property
    def stencil_width(self) -> int:
        return 0


@bind.register(SBPSAT)
def _bind_advection_sbp(  # type: ignore[misc]
    scheme: SBPSAT, grid: UniformGrid, bc: Boundary
) -> SBPSAT:
    from pyshocks.scalar import SATBoundary

    assert isinstance(grid, UniformGrid)
    assert isinstance(bc, SATBoundary)
    assert scheme.velocity is not None

    P = sbp.sbp_norm_matrix(scheme.op, grid, bc.boundary_type)
    D1 = sbp.sbp_first_derivative_matrix(scheme.op, grid, bc.boundary_type)

    # FIXME: make these into sparse matrices
    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "D1", D1)

    return scheme


@apply_operator.register(SBPSAT)
def _apply_operator_sbp_sat_21(
    scheme: SBPSAT, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import SATBoundary

    assert isinstance(bc, SATBoundary)

    assert bc.left is not None
    ga = evaluate_boundary(bc.left, grid, t, u)
    ga = (scheme.velocity[0] > 0) * ga

    assert bc.right is not None
    gb = evaluate_boundary(bc.right, grid, t, u)
    gb = (scheme.velocity[-1] < 0) * gb

    return -scheme.velocity * (scheme.D1 @ u) - (ga + gb) / scheme.P


# }}}
