# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import jax
import jax.numpy as jnp

from pyshocks import (
    Grid,
    SchemeBase,
    ConservationLawScheme,
    Boundary,
)
from pyshocks import reconstruction
from pyshocks import (
    flux,
    numerical_flux,
    predict_timestep,
    apply_operator,
    apply_boundary,
)


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

    def __post_init__(self) -> None:
        if not isinstance(self.rec, reconstruction.ESWENO32):
            raise TypeError("ESWENO32 scheme requires the ESWENO32 reconstruction")


@numerical_flux.register(ESWENO32)
def _numerical_flux_burgers_esweno32(
    scheme: ESWENO32, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from pyshocks.scalar import scalar_flux_upwind
    from pyshocks.weno import es_weno_weights

    rec = scheme.rec
    assert isinstance(rec, reconstruction.ESWENO32)

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


def make_ss_weno_242_matrices(
    bc: Boundary, n: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    import pyshocks.weno as w

    qi, hi = w.ss_weno_242_operator_coefficients()
    p, qb, hb = w.ss_weno_242_operator_boundary_coefficients()

    from pyshocks.scalar import PeriodicBoundary, SSWENOBurgersBoundary

    if isinstance(bc, PeriodicBoundary):
        p = jnp.ones_like(p)  # type: ignore[no-untyped-call]
        P = w.ss_weno_norm_matrix(p, n)  # noqa: N806
        Q = w.ss_weno_circulant(qi, n)  # noqa: N806
        H = w.ss_weno_circulant(hi, n)  # noqa: N806
        Qs = w.ss_weno_derivative_matrix(qi, None, n)  # noqa: N806
    elif isinstance(bc, SSWENOBurgersBoundary):
        P = w.ss_weno_norm_matrix(p, n)  # noqa: N806
        Q = w.ss_weno_derivative_matrix(qi, qb, n)  # noqa: N806
        H = w.ss_weno_interpolation_matrix(hi, hb, n)  # noqa: N806
        Qs = Q  # noqa: N806
    else:
        raise TypeError(f"unsupported boundary conditions: '{type(bc).__name__}'")

    return P, Q, H, Qs


def prepare_ss_weno_242_scheme(
    scheme: "SSWENO242", grid: Grid, bc: Boundary
) -> "SSWENO242":
    P, Q, H, Qs = make_ss_weno_242_matrices(bc, grid.x.size)  # noqa: N806
    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "Q", Q)
    object.__setattr__(scheme, "H", H)
    object.__setattr__(scheme, "Qs", Qs)

    return scheme


@dataclass(frozen=True)
class SSWENO242(SchemeBase):
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
    c: float = 1.0e-12

    P: ClassVar[jnp.ndarray]
    Q: ClassVar[jnp.ndarray]
    H: ClassVar[jnp.ndarray]

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
    return flux.dispatch(Scheme)(scheme, t, x, u)


@predict_timestep.register(SSWENO242)
def _predict_timestep_burgers_ssweno242(
    scheme: SSWENO242, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return predict_timestep.dispatch(Scheme)(scheme, grid, t, u)


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

    # NOTE: use a global Lax-Friedrichs splitting as recommended in [Frenzel2021]
    f = flux(scheme, t, grid.x, u)
    alpha = jnp.max(jnp.abs(u))
    fp = (f + alpha * u) / 2
    fm = (f - alpha * u) / 2

    # reconstruct
    fp, _ = reconstruct(scheme.rec, grid, fp)
    _, fm = reconstruct(scheme.rec, grid, fm)

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

    gssw = jnp.zeros_like(fssw)  # type: ignore[no-untyped-call]

    # }}}

    # {{{ handle boundary conditions

    from pyshocks import evaluate_boundary

    gb = evaluate_boundary(bc, grid, t, u)

    # }}}

    # [Fisher2013] Equation 3.45
    dx = scheme.P * grid.dx_min
    r = (-(fssw[1:] - fssw[:-1]) + (gssw[1:] - gssw[:-1]) + gb) / dx

    # NOTE: a bit hacky, but cleans any boundary goo
    if isinstance(bc, PeriodicBoundary):
        r = apply_boundary(bc, grid, t, r)

    return r


# }}}


# vim: fdm=marker
