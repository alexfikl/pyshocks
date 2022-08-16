# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, Tuple

import jax
import jax.numpy as jnp

from pyshocks import Grid, Boundary
from pyshocks import apply_operator, predict_timestep, flux
from pyshocks import reconstruction
from pyshocks.burgers.schemes import FiniteDifferenceScheme


# {{{ two-point entropy conservative flux


@jax.jit
def two_point_entropy_flux(q: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    uu = jnp.tile((u * u).reshape(-1, 1), u.size)  # type: ignore[no-untyped-call]
    qfs = q * (jnp.outer(u, u) + uu + uu.T) / 3

    # NOTE: u is numbered [1, N] and fluxes are number [0, N] ([Fisher2013] Fig. 1)
    fss = jnp.zeros(u.size + 1, dtype=u.dtype)  # type: ignore[no-untyped-call]

    # FIXME: jax unrolls this to `u.size` statements, which is horrible!
    for i in range(1, u.size):
        assert 0 <= i < fss.size

        # NOTE: computes the following sum
        #   sum(k, i, N) sum(l, 1, i + 1) 2 q[l, k] f(u_l, u_k)
        # where the python indexing is exclusive, not inclusive!
        fss = fss.at[i].set(jnp.sum(qfs[:i, i:]))

    return fss


# }}}


# {{{ make_ss_weno_242_matrices


def make_ss_weno_242_matrices(
    grid: Grid,
    bc: Boundary,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from pyshocks import weno

    n = grid.x.size
    dtype = grid.x.dtype

    qi, hi = weno.ss_weno_242_operator_coefficients(dtype=dtype)
    p, qb, hb = weno.ss_weno_242_operator_boundary_coefficients(dtype=dtype)

    from pyshocks.tools import circulant_matrix
    from pyshocks.scalar import PeriodicBoundary, SSWENOBurgersBoundary

    if isinstance(bc, PeriodicBoundary):
        p = jnp.ones_like(p)  # type: ignore[no-untyped-call]
        P = weno.ss_weno_norm_matrix(p, n)
        Q = circulant_matrix(qi, n)
        H = circulant_matrix(hi, n)
        Qs = weno.ss_weno_derivative_matrix(qi, None, n)
    elif isinstance(bc, SSWENOBurgersBoundary):
        P = weno.ss_weno_norm_matrix(p, n)
        Q = weno.ss_weno_derivative_matrix(qi, qb, n)
        H = weno.ss_weno_interpolation_matrix(hi, hb, n)
        Qs = Q
    else:
        raise TypeError(f"unsupported boundary conditions: '{type(bc).__name__}'")

    return P, Q, H, Qs


def make_ss_weno_242_sbp_matrix(
    grid: Grid,
    bc: Boundary,
    *,
    nu: float,
) -> jnp.ndarray:
    from pyshocks.scalar import PeriodicBoundary, SSWENOBurgersBoundary

    if not isinstance(bc, (SSWENOBurgersBoundary, PeriodicBoundary)):
        raise TypeError(f"unsupported boundary conditions: '{type(bc).__name__}'")

    # NOTE: See [Mattsson2012] for details
    n = grid.x.size
    dx = grid.dx_min
    dtype = grid.x.dtype

    from pyshocks import sbp

    # get metric
    P = dx * sbp.make_sbp_21_norm_matrix(n, dtype=dtype)

    invP = jnp.diag(1 / P)  # type: ignore[no-untyped-call]
    P = jnp.diag(P)  # type: ignore[no-untyped-call]

    # get first-order derivative
    Q = sbp.make_sbp_21_first_derivative_q_matrix(n, dtype=dtype)
    D = invP @ Q

    # get R matrix
    (D22,) = sbp.make_sbp_21_second_derivative_d_matrices(n, dtype=dtype)
    (C22,) = sbp.make_sbp_21_second_derivative_c_matrices(n, dtype=dtype)
    B = nu * jnp.eye(n)  # type: ignore[no-untyped-call]
    R = dx**3 / 4 * D22.T @ C22 @ B @ D22

    # get Bbar matrix
    Bbar = jnp.zeros_like(B)  # type: ignore[no-untyped-call]
    Bbar = Bbar.at[0, 0].set(-B[0, 0])
    Bbar = Bbar.at[-1, -1].set(B[-1, -1])

    # get S matrix
    S = sbp.make_sbp_21_second_derivative_s_matrix(n, dx, dtype=dtype)

    # put it all together
    M = D.T @ P @ B @ D + R

    return -M + Bbar @ S


# }}}


# {{{ scheme


def prepare_ss_weno_242_scheme(
    scheme: "SSWENO242", grid: Grid, bc: Boundary
) -> "SSWENO242":
    object.__setattr__(scheme, "nu", grid.dx_min ** (4 / 3))

    P, _, _, Qs = make_ss_weno_242_matrices(grid, bc)

    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "Qs", Qs)

    # FIXME: replace with sbp.SBP42 operators
    DD = make_ss_weno_242_sbp_matrix(grid, bc, nu=scheme.nu)

    object.__setattr__(scheme, "DD", DD)

    return scheme


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

    nu: float = 1.0e-2
    c: float = 1.0e-12

    # first derivative
    P: ClassVar[jnp.ndarray]
    DD: ClassVar[jnp.ndarray]

    Qs: ClassVar[jnp.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.SSWENO242):
            raise TypeError("SSWENO242 scheme requires the SSWENO242 reconstruction")

    @property
    def order(self) -> int:
        return self.rec.order

    @property
    def stencil_width(self) -> int:
        return self.rec.stencil_width


@flux.register(SSWENO242)
def _flux_burgers_ssweno242(
    scheme: SSWENO242, t: float, x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    return flux.dispatch(FiniteDifferenceScheme)(scheme, t, x, u)


@predict_timestep.register(SSWENO242)
def _predict_timestep_burgers_ssweno242(
    scheme: SSWENO242, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return jnp.minimum(
        predict_timestep.dispatch(FiniteDifferenceScheme)(scheme, grid, t, u),
        0.5 * grid.dx_min**2 / scheme.nu,
    )


@apply_operator.register(SSWENO242)
def _apply_operator_burgers_ssweno242(
    scheme: SSWENO242,
    grid: Grid,
    bc: Boundary,
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    from pyshocks.scalar import PeriodicBoundary, SSWENOBurgersBoundary

    if not isinstance(bc, (SSWENOBurgersBoundary, PeriodicBoundary)):
        raise TypeError(f"boundary has unsupported type: '{type(bc).__name__}'")

    assert u.shape == grid.x.shape

    from pyshocks.reconstruction import reconstruct

    # NOTE: w is taken to be the entropy variable ([Fisher2013] Section 4.1)
    w = u
    dx = scheme.P * grid.dx_min

    # NOTE: use a global Lax-Friedrichs splitting as recommended in [Frenzel2021]
    f = flux(scheme, t, grid.x, u)
    alpha = jnp.max(jnp.abs(u))
    fp = (f + alpha * u) / 2
    fm = (f - alpha * u) / 2

    # reconstruct
    _, fp = reconstruct(scheme.rec, grid, fp)
    fm, _ = reconstruct(scheme.rec, grid, fm)

    # {{{ inviscid flux

    # standard WENO reconstructed flux
    fw = jnp.pad(fp[1:] + fm[:-1], 1)  # type: ignore[no-untyped-call]

    # two-point entropy conservative flux ([Fisher2013] Equation 4.7)
    fs = two_point_entropy_flux(scheme.Qs, u)
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

    gssw = scheme.DD @ w

    # }}}

    # {{{ handle boundary conditions

    from pyshocks import evaluate_boundary

    gb = evaluate_boundary(bc, grid, t, u)

    # }}}

    # [Fisher2013] Equation 3.45
    r = (-(fssw[1:] - fssw[:-1]) + gssw + gb) / dx

    # NOTE: a bit hacky, but cleans any boundary goo
    if isinstance(bc, PeriodicBoundary):
        from pyshocks import apply_boundary

        r = apply_boundary(bc, grid, t, r)

    return r


# }}}
