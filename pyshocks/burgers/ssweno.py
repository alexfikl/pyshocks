# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from pyshocks import Grid, Boundary, BoundaryType
from pyshocks import bind, apply_operator, predict_timestep, flux
from pyshocks import reconstruction, sbp
from pyshocks.burgers.schemes import FiniteDifferenceScheme
from pyshocks.sbp import Stencil

# {{{ two-point entropy conservative flux


@jax.jit
def two_point_entropy_flux_42(qi: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    # FIXME: this only works for the 4-2 scheme
    # FIXME: boundary stencil is just plain old wrong

    def fs(ul: jnp.ndarray, ur: jnp.ndarray) -> jnp.ndarray:
        return (ul * ul + ul * ur + ur * ur) / 6

    qr = qi[qi.size // 2 + 1 :]
    fss = jnp.zeros(u.size + 1, dtype=u.dtype)  # type: ignore[no-untyped-call]

    i = 1
    fss = fss.at[i].set(
        2 * qr[0] * fs(u[i - 1], u[i]) + 2 * qr[1] * fs(u[i - 1], u[i + 1])
    )
    i = u.size - 1
    fss = fss.at[i].set(
        2 * qr[0] * fs(u[i - 1], u[i + 1]) + 2 * qr[1] * fs(u[i - 2], u[i + 1])
    )

    def body(i: int, fss: jnp.ndarray) -> jnp.ndarray:
        return fss.at[i].set(
            2 * qr[1] * fs(u[i - 2], u[i])
            + 2 * qr[0] * fs(u[i - 1], u[i])
            + 2 * qr[1] * fs(u[i - 1], u[i + 1])
        )

    return jax.lax.fori_loop(2, u.size - 1, body, fss)


# }}}


# {{{ scheme


@dataclass(frozen=True)
class SSWENO242(FiniteDifferenceScheme):
    """Fourth-order Energy Stable WENO (ESWENO) scheme by [Fisher2013]_.

    .. [Fisher2013] T. C. Fisher, M. H. Carpenter, *High-Order Entropy Stable
        Finite Difference Schemes for Nonlinear Conservation Laws: Finite Domains*,
        Journal of Computational Physics, Vol. 252, pp. 518--557, 2013,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2013.06.014>`__.

    .. attribute:: c

        Offset used in computing the entropy stable flux in Equation 3.42
        from [Fisher2013]_.
    """

    rec: reconstruction.SSWENO242
    sbp: sbp.SBP42

    nu: float = 0.0
    c: float = 1.0e-12

    # sbp operators
    P: ClassVar[jnp.ndarray]
    q: ClassVar[Stencil]
    DD: ClassVar[jnp.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.SSWENO242):
            raise TypeError("SSWENO242 scheme requires the SSWENO242 reconstruction")

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
    from pyshocks.scalar import SATBoundary, OneSidedBurgersBoundary

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
    scheme: SSWENO242, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    return flux.dispatch(FiniteDifferenceScheme)(scheme, t, x, u)


@predict_timestep.register(SSWENO242)
def _predict_timestep_burgers_ssweno242(
    scheme: SSWENO242, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    dt = predict_timestep.dispatch(FiniteDifferenceScheme)(scheme, grid, bc, t, u)

    if scheme.nu > 0:
        dt = jnp.minimum(dt, 0.5 * grid.dx_min**2 / scheme.nu)

    return dt


@apply_operator.register(SSWENO242)
def _apply_operator_burgers_ssweno242(
    scheme: SSWENO242,
    grid: Grid,
    bc: Boundary,
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
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
    fp, _ = reconstruct(scheme.rec, grid, bc.boundary_type, fp)
    _, fm = reconstruct(scheme.rec, grid, bc.boundary_type, fm)

    # {{{ inviscid flux

    # standard WENO reconstructed flux
    fw = jnp.pad(fp[1:] + fm[:-1], 1)  # type: ignore[no-untyped-call]

    # two-point entropy conservative flux ([Fisher2013] Equation 4.7)
    fs = two_point_entropy_flux_42(scheme.q.int, u)
    assert fs.shape == grid.f.shape

    # entropy stable flux ([Fisher2013] Equation 3.42)
    b = jnp.pad((w[1:] - w[:-1]) * (fs[1:-1] - fw[1:-1]), 1)  # type: ignore
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
        gssw = 0.0

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
