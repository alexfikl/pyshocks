# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, cast

import jax
import jax.numpy as jnp

from pyshocks import (
    Boundary,
    BoundaryType,
    Grid,
    apply_operator,
    bind,
    flux,
    predict_timestep,
    reconstruction,
    sbp,
)
from pyshocks.burgers.schemes import FiniteDifferenceScheme
from pyshocks.sbp import Stencil
from pyshocks.tools import Array, Scalar, ScalarLike

# {{{ two-point entropy conservative flux


@jax.jit
def two_point_entropy_flux_42(qi: Array, u: Array) -> Array:
    # FIXME: this only works for the 4-2 scheme
    # FIXME: boundary stencil is just plain old wrong

    def fs(ul: Scalar, ur: Scalar) -> Scalar:
        return (ul * ul + ul * ur + ur * ur) / 6

    qr = qi[qi.size // 2 + 1 :]
    fss = jnp.zeros(u.size + 1, dtype=u.dtype)

    i = 1
    fss = fss.at[i].set(
        2 * qr[0] * fs(u[i - 1], u[i]) + 2 * qr[1] * fs(u[i - 1], u[i + 1])
    )
    i = u.size - 1
    fss = fss.at[i].set(
        2 * qr[0] * fs(u[i - 1], u[i + 1]) + 2 * qr[1] * fs(u[i - 2], u[i + 1])
    )

    def body(i: int, fss: Array) -> Array:
        result = fss.at[i].set(
            2 * qr[1] * fs(u[i - 2], u[i])
            + 2 * qr[0] * fs(u[i - 1], u[i])
            + 2 * qr[1] * fs(u[i - 1], u[i + 1])
        )
        return cast(Array, result)

    return cast(
        Array,
        jax.lax.fori_loop(2, u.size - 1, body, fss),  # type: ignore[no-untyped-call]
    )


# }}}


# {{{ scheme


@dataclass(frozen=True)
class SSWENO242(FiniteDifferenceScheme):
    """Fourth-order Entropy Stable WENO (SS-WENO) scheme by [Fisher2013]_."""

    #: Reconstruction method used by the SS-WENO method.
    rec: reconstruction.SSWENO242
    #: SBP scheme used to approximate the diffusive terms, if any.
    sbp: sbp.SBP42

    #: Constant diffusivity coefficient. If zero, no diffusion is used.
    nu: ScalarLike = 0.0
    #: Offset used in computing the entropy stable flux in Equation 3.42
    #: from [Fisher2013]_.
    c: ScalarLike = 1.0e-12

    #: Diagonal norm operator corresponding to :attr:`sbp`.
    P: ClassVar[Array]
    #: Stencil of the operator used to construct the first-order derivative in
    #: the SBP scheme. This stencil is used to compute the two-point entropy
    #: flux for high-order schemes.
    q: ClassVar[Stencil]
    #: Second-order derivative SBP operator.
    DD: ClassVar[Array]

    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.SSWENO242):
            raise TypeError("SSWENO242 scheme requires the SSWENO242 reconstruction.")

    @property
    def order(self) -> int:
        return self.rec.order

    @property
    def stencil_width(self) -> int:
        return self.rec.stencil_width


@bind.register(SSWENO242)
def _bind_diffusion_sbp(  # type: ignore[misc]
    scheme: SSWENO242, grid: Grid, bc: Boundary
) -> SSWENO242:
    from pyshocks import UniformGrid
    from pyshocks.scalar import OneSidedBurgersBoundary, SATBoundary

    assert isinstance(grid, UniformGrid)
    if bc.boundary_type == BoundaryType.Periodic:
        pass
    else:
        assert isinstance(bc, SATBoundary)
        assert isinstance(bc.left, OneSidedBurgersBoundary)
        assert isinstance(bc.right, OneSidedBurgersBoundary)

    object.__setattr__(scheme, "nu", 1.0e-3)

    q = sbp.make_sbp_42_first_derivative_q_stencil(dtype=grid.dtype)

    P = sbp.sbp_norm_matrix(scheme.sbp, grid, bc.boundary_type)
    D2 = sbp.sbp_second_derivative_matrix(scheme.sbp, grid, bc.boundary_type, scheme.nu)

    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "q", q)
    object.__setattr__(scheme, "DD", D2)

    return scheme


@flux.register(SSWENO242)
def _flux_burgers_ssweno242(
    scheme: SSWENO242, t: ScalarLike, x: Array, u: Array
) -> Array:
    return flux.dispatch(FiniteDifferenceScheme)(scheme, t, x, u)


@predict_timestep.register(SSWENO242)
def _predict_timestep_burgers_ssweno242(
    scheme: SSWENO242, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Scalar:
    dt = predict_timestep.dispatch(FiniteDifferenceScheme)(scheme, grid, bc, t, u)

    if scheme.nu > 0:
        dt = jnp.minimum(dt, 0.5 * grid.dx_min**2 / scheme.nu)

    return dt


@apply_operator.register(SSWENO242)
def _apply_operator_burgers_ssweno242(
    scheme: SSWENO242,
    grid: Grid,
    bc: Boundary,
    t: ScalarLike,
    u: Array,
) -> Array:
    assert u.shape == grid.x.shape

    from pyshocks.reconstruction import reconstruct

    # NOTE: w is taken to be the entropy variable ([Fisher2013] Section 4.1)
    w = u

    # NOTE: use a global Lax-Friedrichs splitting as recommended in [Frenzel2021]
    f = flux(scheme, t, grid.x, u)
    alpha = jnp.max(jnp.abs(u))
    fp = (f + alpha * u) / 2
    fm = (f - alpha * u) / 2

    # reconstruct
    fp, _ = reconstruct(scheme.rec, grid, bc.boundary_type, fp, u, u)
    _, fm = reconstruct(scheme.rec, grid, bc.boundary_type, fm, u, u)

    # {{{ inviscid flux

    # standard WENO reconstructed flux
    fw = jnp.pad(fp[1:] + fm[:-1], 1)

    # two-point entropy conservative flux ([Fisher2013] Equation 4.7)
    fs = cast(Array, two_point_entropy_flux_42(scheme.q.int, u))
    assert fs.shape == grid.f.shape

    # entropy stable flux ([Fisher2013] Equation 3.42)
    b = jnp.pad((w[1:] - w[:-1]) * (fs[1:-1] - fw[1:-1]), 1)
    delta = (jnp.sqrt(b**2 + scheme.c**2) - b) / jnp.sqrt(b**2 + scheme.c**2)
    fssw = fw + delta * (fs - fw)
    assert fssw.shape == grid.f.shape

    # }}}

    # {{{ viscous dissipation for entropy stability

    # NOTE: the viscous part is described only in terms of the SBP matrices,
    # not as a flux, in [Fisher2013], so we're keeping it that way here too

    if scheme.nu > 0:
        gssw = scheme.DD @ w
    else:
        gssw = jnp.array(0.0)

    # }}}

    # {{{ handle boundary conditions

    from pyshocks import evaluate_boundary

    gb = evaluate_boundary(bc, grid, t, u)

    # }}}

    # [Fisher2013] Equation 3.45
    r = -(fssw[1:] - fssw[:-1]) / scheme.P + gb / scheme.P + gssw

    # NOTE: a bit hacky, but cleans any boundary goo
    if bc.boundary_type == BoundaryType.Periodic:
        from pyshocks import apply_boundary

        r = apply_boundary(bc, grid, t, r)

    return r


# }}}
