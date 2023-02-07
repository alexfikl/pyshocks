# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, cast

import jax.numpy as jnp

import pyshocks.finitedifference as fd
from pyshocks import (
    Boundary,
    FiniteDifferenceSchemeBase,
    FiniteVolumeSchemeBase,
    Grid,
    SchemeBase,
    UniformGrid,
    apply_operator,
    bind,
    evaluate_boundary,
    numerical_flux,
    predict_timestep,
    sbp,
)
from pyshocks.tools import Array, Scalar, ScalarLike

# {{{ base


@dataclass(frozen=True)
class SpatialVelocity:
    """A placeholder for a spatially-varying velocity field."""

    velocity: Array

    def __call__(self, t: ScalarLike, x: Array) -> Array:
        assert self.velocity.shape == x.shape
        return self.velocity


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the linear advection equation.

    .. attribute:: velocity

        Advection velocity.

    .. automethod:: __init__
    """

    velocity: Array


@predict_timestep.register(Scheme)
def _predict_timestep_advection(
    scheme: Scheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Scalar:
    assert scheme.velocity is not None

    # NOTE: keep in sync with pyshocks.continuity.schemes.predict_timestep
    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@apply_operator.register(Scheme)
def _apply_operator_advection(
    scheme: Scheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    assert scheme.velocity is not None

    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, bc, t, u)

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


def upwind_flux(scheme: Scheme, grid: Grid, bc: Boundary, u: Array) -> Array:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    a = scheme.velocity
    ul, ur = reconstruct(scheme.rec, grid, bc.boundary_type, u, u, a)
    al, ar = reconstruct(scheme.rec, grid, bc.boundary_type, a, a, a)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, ur[:-1], ul[1:])

    return jnp.pad(fnum, 1)


@dataclass(frozen=True)
class Godunov(FiniteVolumeScheme):
    """A Godunov (upwind) scheme for the advection equation.

    .. automethod:: __init__
    """


@numerical_flux.register(Godunov)
def _numerical_flux_advection_godunov(
    scheme: Godunov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    return upwind_flux(scheme, grid, bc, u)


# }}}


# {{{ ESWENO


def esweno_lf_flux(scheme: Scheme, grid: Grid, bc: Boundary, u: Array) -> Array:
    assert scheme.velocity is not None
    assert scheme.rec is not None
    assert u.shape[0] == grid.x.size

    from pyshocks.reconstruction import reconstruct

    # FIXME: what the hell is this? Why are we reconstructing f?
    f = scheme.velocity * u
    ul, ur = reconstruct(scheme.rec, grid, bc.boundary_type, u, u, u)
    fl, fr = reconstruct(scheme.rec, grid, bc.boundary_type, f, u, scheme.velocity)

    a = jnp.max(jnp.abs(scheme.velocity))
    fnum = 0.5 * (fl[1:] + fr[:-1]) - 0.5 * a * (ul[1:] - ur[:-1])
    return jnp.pad(fnum, 1)


@dataclass(frozen=True)
class ESWENO32(FiniteVolumeScheme):
    """Third-order Energy Stable WENO (ESWENO) scheme by [Yamaleev2009]_."""


@numerical_flux.register(ESWENO32)
def _numerical_flux_advection_esweno32(
    scheme: ESWENO32, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    from pyshocks import reconstruction
    from pyshocks.weno import es_weno_weights

    rec = scheme.rec

    if isinstance(rec, reconstruction.ESWENO32):
        # {{{ compute dissipative flux of ESWENO

        # NOTE: computing these twice :(
        _, omega1 = es_weno_weights(rec.s, u, eps=rec.eps)

        # NOTE: see Equation 37 in [Yamaleev2009] for mu expression
        mu = jnp.sqrt((omega1[1:] - omega1[:-1]) ** 2 + rec.delta**2) / 8.0

        # NOTE: see Equation  in [Yamaleev2009] for flux expression
        gnum = -(mu + (omega1[1:] - omega1[:-1]) / 8.0) * (u[1:] - u[:-1])

        gnum = jnp.pad(gnum, 1)

        # }}}
    else:
        gnum = jnp.array(0.0)

    return cast(Array, esweno_lf_flux(scheme, grid, bc, u) + gnum)


# }}}


# {{{ biased splitting


@dataclass(frozen=True)
class FluxSplitGodunov(FiniteDifferenceScheme):
    sorder: int

    sp: ClassVar[fd.Stencil]
    sm: ClassVar[fd.Stencil]

    @property
    def order(self) -> int:
        return self.sorder

    @property
    def stencil_width(self) -> int:
        return self.sorder


@bind.register(FluxSplitGodunov)
def _bind_advection_flux_split_godunov(  # type: ignore[misc]
    scheme: FluxSplitGodunov, grid: UniformGrid, bc: Boundary
) -> FluxSplitGodunov:
    assert isinstance(grid, UniformGrid)
    assert scheme.velocity is not None
    assert scheme.order % 2 == 1

    km = (scheme.order - 2) if scheme.order > 1 else 0
    kp = scheme.order

    sm = fd.make_taylor_approximation(1, (-km, kp))
    assert sm.order >= scheme.sorder, (scheme.sorder, sm)
    sp = fd.make_taylor_approximation(1, (-kp, km))
    assert sp.order >= scheme.sorder, (scheme.sorder, sp)

    object.__setattr__(scheme, "sm", sm)
    object.__setattr__(scheme, "sp", sp)

    return scheme


@apply_operator.register(FluxSplitGodunov)
def _apply_operator_flux_split_godunov(
    scheme: FluxSplitGodunov, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    assert scheme.velocity is not None

    from pyshocks import apply_boundary

    u = apply_boundary(bc, grid, t, u)
    cond = scheme.velocity < 0

    du_p = fd.apply_derivative(scheme.sp, jnp.where(cond, u, 0), grid.dx)
    du_m = fd.apply_derivative(scheme.sm, jnp.where(cond, 0, u), grid.dx)

    # FIXME: why is this not `-velocity * du`?
    return scheme.velocity * (du_p + du_m)


# }}}


# {{{ SBP-SAT


@dataclass(frozen=True)
class SBPSAT(FiniteDifferenceScheme):
    op: sbp.SBPOperator

    P: ClassVar[Array]
    D1: ClassVar[Array]

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
    # {{{ scheme

    assert isinstance(grid, UniformGrid)
    assert scheme.velocity is not None

    P = sbp.sbp_norm_matrix(scheme.op, grid, bc.boundary_type)
    D1 = sbp.sbp_first_derivative_matrix(scheme.op, grid, bc.boundary_type)

    # FIXME: make these into sparse matrices
    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "D1", D1)

    # }}}

    # {{{ boundary

    from pyshocks.scalar import PeriodicBoundary, SATBoundary

    if isinstance(bc, SATBoundary):
        from pyshocks.scalar import OneSidedAdvectionSATBoundary

        if isinstance(bc.left, OneSidedAdvectionSATBoundary):
            assert isinstance(bc.right, OneSidedAdvectionSATBoundary)

            object.__setattr__(bc.left, "velocity", scheme.velocity[0])
            object.__setattr__(bc.right, "velocity", scheme.velocity[-1])
        else:
            assert not isinstance(bc.right, OneSidedAdvectionSATBoundary)
    elif isinstance(bc, PeriodicBoundary):
        pass
    else:
        raise TypeError(f"Unsupported boundary type: {type(bc).__name__!r}.")

    # }}}

    return scheme


@apply_operator.register(SBPSAT)
def _apply_operator_sbp_sat_21(
    scheme: SBPSAT, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    gb = evaluate_boundary(bc, grid, t, u)
    return -scheme.velocity * (scheme.D1 @ u) - gb / scheme.P


# }}}
