# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.linalg as la

from pyshocks.logging import get_logger
from pyshocks.tools import set_recommended_matplotlib

logger = get_logger("burgers_ssweno_numpy")
set_recommended_matplotlib()


# FIXME: should probably use np.ndarray[Any, np.float64] or something
Array = Any
Matrix = Any
LL, L, C, R, RR = 0, 1, 2, 3, 4


# {{{ initial conditions


def ic_riemann(x: Array, *, ul: float, ur: float, xc: float | None = None) -> Array:
    if xc is None:
        dx = x[-1] - x[0]
        xc = x[0] + 0.25 * dx

    H = np.array(x < xc, dtype=x.dtype)
    return ur + (ul - ur) * H


def ic_tophat(t: float, x: Array, *, us: float, uc: float) -> Array:
    dx = x[-1] - x[0]
    xc = 0.5 * (x[0] + x[-1])
    xa = xc - 0.25 * dx
    xb = xc + 0.25 * dx

    s = (uc + us) / 2
    return (
        us * (x <= xa + us * t)
        + (x - xa) / (t + 1.0e-15) * np.logical_and(xa + us * t < x, x < xa + uc * t)
        + uc * np.logical_and(xa + uc * t < x, x < xb + s * t)
        + us * (x >= xb + s * t)
    )


def ic_sine(x: Array, *, k: float, dx: float | None = None) -> Array:
    if dx is None:
        dx = x[-1] - x[0]

    return np.sin(2.0 * np.pi * k * x / dx)


def ic_poly(x: Array, *, a: float, b: float, c: float, d: float) -> Array:
    return x**5 + a * x**3 + b * x**2 + c * x + d


def ic_linear(x: Array, *, ul: float, ur: float) -> Array:
    a, b = x[0], x[-1]
    return ul + (ur - ul) * (x - a) / (b - a)


def ic_gaussian(x: Array, *, sigma: float = 1.0e-2, xm: float | None = None) -> Array:
    if xm is None:
        xm = (x[-1] + x[0]) / 2
    return np.exp(-((x - xm) ** 2) / (2 * sigma))


# }}}


# {{{ weno 2-4-2

# [Fisher2011] T. C. Fisher, M. H. Carpenter, N. K. Yamaleev, S. H. Frankel,
#   Boundary Closures for Fourth-Order Energy Stable Weighted Essentially
#   Non-Oscillatory Finite-Difference Schemes,
#   Journal of Computational Physics, Vol. 230, pp. 3727--3752, 2011,
#   https://dx.doi.org/10.1016/j.jcp.2011.01.043.


def weno_242_flux_points(x: Array) -> Array:
    y = np.empty(x.size + 1, dtype=x.dtype)

    y[0] = x[0]
    y[1:] = (x[1] - x[0]) * sbp_42_p_matrix(x.size, x.dtype)

    return np.cumsum(y)


# {{{ interpolation


@lru_cache
def weno_242_ops_interp(n: int, dtype: Any) -> tuple[Array, Array, Array, Array, Array]:
    """
    :returns: a :class:`tuple` of arrays of shape ``(2, n + 1)``, corresponding
        to :math:`(I_{LL}, I_L, I_C, I_R, I_{RR})`.
    """
    n = n + 1
    b = 4

    # [Fisher2011] Equation 77
    # ILL
    ILL = np.zeros((2, n), dtype=dtype)
    ILL[0, :b] = [0, 0, 0, -71 / 48]
    ILL[1, :b] = [0, 0, 0, 119 / 48]

    # IL
    IL = np.stack([
        np.full(n, -1 / 2, dtype=dtype),
        np.full(n, 3 / 2, dtype=dtype),
    ])
    IL[0, :b] = [0, 0, -7 / 12, -23 / 48]
    IL[1, :b] = [0, 0, 19 / 12, 71 / 48]

    # IC
    IC = np.stack([
        np.full(n, 1 / 2, dtype=dtype),
        np.full(n, 1 / 2, dtype=dtype),
    ])
    IC[0, :b] = [1, 31 / 48, 5 / 12, 25 / 48]
    IC[1, :b] = [0, 17 / 48, 7 / 12, 23 / 48]

    # IR
    IR = np.stack([
        np.full(n, 3 / 2, dtype=dtype),
        np.full(n, -1 / 2, dtype=dtype),
    ])
    IR[0, :b] = [0, 79 / 48, 17 / 12, 73 / 48]
    IR[1, :b] = [0, -31 / 48, -5 / 12, -25 / 48]

    # IRR
    IRR = np.zeros((2, n), dtype=dtype)
    IRR[0, :b] = [0, 127 / 48, 29 / 12, 121 / 48]
    IRR[1, :b] = [0, -79 / 48, -17 / 12, -73 / 48]

    # symmetrize
    Is = [ILL, IL, IC, IR, IRR]
    m = len(Is)
    for i in range(m):
        Is[i][0, -b:] = Is[m - i - 1][1, :b][::-1]
        Is[i][1, -b:] = Is[m - i - 1][0, :b][::-1]

    assert np.linalg.norm(np.sum(ILL[:, 3], axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(ILL[:, -b:-1], axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(IL[:, 2:-3], axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(IC, axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(IR[:, 3:-2], axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(IRR[:, -b], axis=0) - 1) < 1.0e-15
    assert np.linalg.norm(np.sum(IRR[:, 1:b], axis=0) - 1) < 1.0e-15

    return ILL, IL, IC, IR, IRR


def weno_242_interp(f: Array, *, stencil: int | None = None) -> Array:
    n = f.size
    ILL, IL, IC, IR, IRR = weno_242_ops_interp(n, f.dtype)

    if stencil is None:
        fbar = np.zeros((5, n + 1), dtype=f.dtype)

        j = np.s_[3:n]
        fbar[LL, j] = ILL[0, j] * f[:-3] + ILL[1, j] * f[1:-2]
        j = np.s_[2:n]
        fbar[L, j] = IL[0, j] * f[:-2] + IL[1, j] * f[1:-1]
        j = np.s_[1:n]
        fbar[C, j] = IC[0, j] * f[:-1] + IC[1, j] * f[1:]
        j = np.s_[1 : n - 1]
        fbar[R, j] = IR[0, j] * f[1:-1] + IR[1, j] * f[2:]
        j = np.s_[1 : n - 2]
        fbar[RR, 1 : n - 2] = IRR[0, j] * f[2:-1] + IRR[1, j] * f[3:]

        fbar[C, 0] = f[0]
        fbar[C, -1] = f[-1]
    else:
        assert stencil == C
        fbar = np.zeros(n + 1, dtype=f.dtype)

        j = np.s_[1:n]
        fbar[j] = IC[0, j] * f[:-1] + IC[1, j] * f[1:]
        fbar[0] = f[0]
        fbar[-1] = f[-1]

    return fbar


# }}}


# {{{ smoothness


def beta_avg(beta: Array, k: int = 4) -> Array:
    return np.sum(beta**k) ** (1 / k)


def weno_242_downwind_stencil(u: Array) -> Array:
    # NOTE: stencils match non-zero elements in `d`
    n = u.size
    dw = np.full(n + 1, -1, dtype=np.int32)

    dw[3:-3] = np.where((u[3:-2] + u[2:-3]) > 0, R, L)

    # left boundary
    j = 1
    dw[j] = RR if (u[j] + u[j - 1]) > 0 else C
    j = 2
    dw[j] = R if (u[j] + u[j - 1]) > 0 else L
    j = 3
    dw[j] = R if (u[j] + u[j - 1]) > 0 else LL

    # right boundary
    j = n - 3
    dw[j] = RR if (u[j] + u[j - 1]) > 0 else L
    j = n - 2
    dw[j] = R if (u[j] + u[j - 1]) > 0 else L
    j = n - 1
    dw[j] = C if (u[j] + u[j - 1]) > 0 else LL

    return dw


def weno_242_smoothness(
    u: Array,
    *,
    p: int = 2,
    k: int | None = None,
    dw: Array | None = None,
) -> tuple[Array, Array]:
    n = u.size
    b = 3

    beta = np.zeros((5, n + 1), dtype=u.dtype)
    tau = np.zeros(n + 1, dtype=u.dtype)

    # beta: Equation 71
    beta[LL, 3:n] = (u[1:-2] - u[:-3]) ** p
    beta[L, 2:n] = (u[1:-1] - u[:-2]) ** p
    beta[C, 1:n] = (u[1:] - u[:-1]) ** p
    beta[R, 1 : n - 1] = (u[2:] - u[1:-1]) ** p
    beta[RR, 1 : n - 2] = (u[3:] - u[2:-1]) ** p

    # tau: Equation 74
    tau[2 : n - 1] = (u[3:] - 3 * u[2:-1] + 3 * u[1:-2] - u[:-3]) ** p
    j = 1
    tau[j] = (u[j + 2] - 3 * u[j + 1] + 3 * u[j] - u[j - 1]) ** p
    j = n - 1
    tau[j] = (u[j] - 3 * u[j - 1] + 3 * u[j - 2] - u[j - 3]) ** p

    # NOTE: this should have no effect
    mask = weno_242_weight_mask(n + 1, u.dtype)
    beta[~mask] = 0.0

    # downwind: Equation 72
    # set the downwind stencil to ensure stability
    if k is not None:
        assert k % 2 == 0, f"k must be even: {k}"
        assert dw is not None

        # interior
        for j in range(b + 1, n - b):
            beta[dw[j], j] = beta_avg(beta[L : R + 1, j], k)

        # left boundary
        j = 1
        beta[dw[j], j] = beta_avg(beta[C : RR + 1, j], k)
        j = 2
        beta[dw[j], j] = beta_avg(beta[L : R + 1, j], k)
        j = 3
        beta[dw[j], j] = beta_avg(beta[LL : R + 1, j], k)

        # right boundary
        j = n - 3
        beta[dw[j], j] = beta_avg(beta[L : RR + 1, j], k)
        j = n - 2
        beta[dw[j], j] = beta_avg(beta[L : R + 1, j], k)
        j = n - 1
        beta[dw[j], j] = beta_avg(beta[LL : C + 1, j], k)

        if __debug__:
            # NOTE: the downwind weight is supposed to be smaller
            flags = np.array([
                np.all(beta[dw[i], i] >= beta[:, i]) for i in range(dw.size)
            ])
            assert np.all(flags), flags

    return beta, tau


# }}}


# {{{ weights


@lru_cache
def weno_242_ops_weight(n: int, dtype: Any) -> Array:
    """
    :returns: an array of shape ``(5, n + 1)`` containing the ideal weights for
        the 2-4-2 WENO scheme.
    """
    # [Fisher2011] Equation 78
    d = np.zeros((5, n), dtype=dtype)
    d[1, :] = 1 / 6
    d[2, :] = 2 / 3
    d[3, :] = 1 / 6

    db = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 24 / 31, 1013 / 4898, 3 / 158],
            [0, 11 / 56, 51 / 70, 3 / 40, 0],
            [3 / 142, 357 / 3266, 408 / 575, 4 / 25, 0],
        ],
        dtype=dtype,
    ).T
    d[:, :4] = db
    d[:, -4:] = db[::-1, ::-1]

    assert np.linalg.norm(np.sum(d, axis=0) - 1) < 5.0e-15

    return d


@lru_cache
def weno_242_weight_mask(n: int, dtype: Any) -> Array:
    return weno_242_ops_weight(n, dtype) > 0


@lru_cache
def weno_242_downwind(n: int, dtype: Any) -> Array:
    mask = weno_242_weight_mask(n, dtype)
    return mask


@lru_cache
def weno_242_stencil_size(n: int, dtype: Any) -> Array:
    return np.sum(weno_242_weight_mask(n, dtype), axis=0)


def weno_242_weights(beta: Array, tau: Array, *, epsilon: float) -> Array:
    n = tau.size

    d = weno_242_ops_weight(n, tau.dtype)
    alpha = d * (1 + tau / (epsilon + beta))
    omega = alpha / np.sum(alpha, axis=0, keepdims=True)

    return omega, d


# }}}


# {{{ reconstruct


def weno_242_reconstruct(
    u: Array,
    f: Array,
    *,
    epsilon: float = 1.0e-8,
    k: int | None = None,
    dw: Array | None = None,
) -> Array:
    """
    :arg epsilon: tolerance used in computing the WENO weights.
    :arg k: power used when updating the downstream WENO weights.
    """

    # NOTE: details in [Fisher2011] Appendix A

    # NOTE: geometry from [Fisher2011] Figure 1 (0-indexed for everyone)
    #
    # y0 y1      y2   y3          yi-1   yi   yi+1        yn-3  yn-2   yn-1 yn
    #   |-+---|---+-|--+--| .... |--+--|--+--|--+--| ... |--+--|-+---|---+-|
    #   x0   x1     x2   x3     xi-2  xi-1   xi  xi+1        xn-3  xn-2    xn-1
    #
    # where `u` is defined at the `n` `x_i` points. The reconstructed values are
    # defined at the `n + 1` `y_i` points, which includes `beta` (smoothness),
    # `omega` (weights) and `fhat` (ENO stencils).
    #
    # The stencils under consideration are for the i-th flux
    #
    #   S_LL = {i - 2, i - 3}
    #   S_L  = {i - 1, i - 2}
    #   S_C  = {i    , i - 1}
    #   S_R  = {i + 1, i    }
    #   S_RR = {i + 2, i + 1}

    # NOTE: The flux is not reconstructed at `y_0` and `y_n` because it needs
    # to be consistent with SBP rules, so the reconstructed values are just
    # ([Fisher2013] Equation 2.39)
    #
    #   \hat{f}_0 = f_0
    #   \hat{f}_n = f_n

    # smoothness coefficients: [Fisher2011]
    beta, tau = weno_242_smoothness(u, k=k, dw=dw)

    sym_error = la.norm(beta - beta[::-1], axis=1) / la.norm(beta, axis=1)
    logger.info("sym beta %.12e %.12e %.12e %.12e %.12e", *sym_error)
    sym_error = la.norm(tau - tau[::-1]) / la.norm(tau)
    logger.info("sym tau  %.12e", sym_error)

    # nonlinear weights: [Fisher2011]
    omega, _ = weno_242_weights(beta, tau, epsilon=epsilon)

    sym_error = la.norm(omega - omega[::-1], axis=1) / la.norm(omega, axis=1)
    logger.info("sym omega %.12e %.12e %.12e %.12e %.12e", *sym_error)

    # interpolate flux: [Fisher2011] Equation 68
    fbar = weno_242_interp(f)

    sym_error = la.norm(fbar - fbar[::-1], axis=1) / la.norm(fbar, axis=1)
    logger.info("sym fbar %.12e %.12e %.12e %.12e %.12e", *sym_error)

    # boundary: sbp consistency Equation 2.39 [Fisher2013]
    fbar[:, 0] = f[0]
    fbar[:, -1] = f[-1]

    return np.sum(omega * fbar, axis=0)


# }}}

# }}}


# {{{ entropy conservative flux


@lru_cache
def sbp_42_p_stencil(dtype: Any) -> tuple[Array, Array]:
    """
    :returns: a :class:`tuple` containing the interior and the left boundary
        stencils. The right boundary stencil is obtained by symmetry.
    """
    return (
        np.array([1], dtype=dtype),
        np.array([17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype=dtype),
    )


@lru_cache
def sbp_42_q_stencil(dtype: Any) -> tuple[Array, Array, Array]:
    """
    :returns: a :class:`tuple` containing the interior and the left boundary
        stencils. The right boundary stencil is obtained by symmetry.
    """
    qi = np.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype)

    qb_l = np.array(
        [
            [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
            [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
            [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
            [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
        ],
        dtype=dtype,
    )
    qb_r = -qb_l[::-1, ::-1]

    return qi, qb_l, qb_r


@lru_cache
def sbp_42_p_matrix(n: int, dtype: Any) -> Array:
    """
    :return: the diagonal of the :math:`P` norm matrix.
    """
    P, Pb = sbp_42_p_stencil(dtype)

    P = np.full(n, P[0], dtype=dtype)
    P[: Pb.size] = Pb
    P[-Pb.size :] = Pb[::-1]

    return P


# [Fisher2013] T. C. Fisher, M. H. Carpenter,
#   High-Order Entropy Stable Finite Difference Schemes for Nonlinear
#   Conservation Laws: Finite Domains,
#   Journal of Computational Physics, Vol. 252, pp. 518--557, 2013,
#   https://dx.doi.org/10.1016/j.jcp.2013.06.014.


def two_point_entropy_flux(u: float, v: float) -> float:
    return (u * u + u * v + v * v) / 6


def two_point_entropy_flux_121(u: Array) -> Array:
    fs = np.zeros(u.size + 1, dtype=u.dtype)

    fs[1:-1] = two_point_entropy_flux(u[1:], u[:-1])
    fs[0] = u[0] ** 2 / 2
    fs[-1] = u[-1] ** 2 / 2

    return fs


def two_point_entropy_flux_242(u: Array) -> Array:
    n = u.size
    qi, qb_l, qb_r = sbp_42_q_stencil(u.dtype)
    fs = np.zeros(n + 1, dtype=u.dtype)

    # interior: [Fisher2013] Equation 3.9, simplified for a pentadiagonal system
    # FIXME: any way to vectorize these better? jax will jit it, so not horrible
    for i in range(qb_l.shape[0], n - qb_l.shape[0] + 1):
        fs[i] = (
            2 * qi[3] * two_point_entropy_flux(u[i - 1], u[i])
            + 2 * qi[4] * two_point_entropy_flux(u[i - 1], u[i + 1])
            + 2 * qi[4] * two_point_entropy_flux(u[i - 2], u[i])
        )

    # boundary: [Fisher2013] Equation 3.9
    # FIXME: write these as a dense loop? this sort of hardcodes the zeros in qb
    i, j = 1, 0
    fs[i] = (
        2 * qb_l[j, j + 1] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_l[j, j + 2] * two_point_entropy_flux(u[i - 1], u[i + 1])
        + 2 * qb_l[j, j + 3] * two_point_entropy_flux(u[i - 1], u[i + 2])
    )
    i, j = 2, 1
    fs[i] = (
        2 * qb_l[j, j + 1] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_l[j - 1, j + 1] * two_point_entropy_flux(u[i - 2], u[i])
        + 2 * qb_l[j - 1, j + 2] * two_point_entropy_flux(u[i - 2], u[i + 1])
    )
    i, j = 3, 2
    fs[i] = (
        2 * qb_l[j, j + 1] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_l[j, j + 2] * two_point_entropy_flux(u[i - 1], u[i + 1])
        + 2 * qb_l[j - 2, j + 1] * two_point_entropy_flux(u[i - 3], u[i])
    )

    i, j = n - 1, 2
    fs[i] = (
        2 * qb_r[j, j + 3] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_r[j - 1, j + 3] * two_point_entropy_flux(u[i - 2], u[i])
        + 2 * qb_r[j - 2, j + 3] * two_point_entropy_flux(u[i - 3], u[i])
    )
    i, j = n - 2, 1
    fs[i] = (
        2 * qb_r[j, j + 3] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_r[j, j + 4] * two_point_entropy_flux(u[i - 1], u[i + 1])
        + 2 * qb_r[j - 1, j + 4] * two_point_entropy_flux(u[i - 2], u[i + 1])
    )
    i, j = n - 3, 0
    fs[i] = (
        2 * qb_r[j, j + 3] * two_point_entropy_flux(u[i - 1], u[i])
        + 2 * qb_r[j, j + 5] * two_point_entropy_flux(u[i - 1], u[i + 2])
        - 2 * qb_r[j, j] * two_point_entropy_flux(u[i - 2], u[i])
    )

    # domain boundary
    fs[0] = u[0] ** 2 / 2
    fs[-1] = u[-1] ** 2 / 2

    return fs


# }}}


# {{{ rk


@dataclass(frozen=True)
class Solution:
    y: Array


def ssprk33(
    fn: Callable[[float, Array], Array],
    u0: Array,
    t_eval: Array,
    *,
    callback: Callable[[float, Array], None] | None,
) -> Solution:
    y = np.empty((u0.size, t_eval.size), dtype=u0.dtype)
    y[:, 0] = u0

    if callback is not None:
        callback(t_eval[0], u0)

    for n in range(t_eval.size - 1):
        t = t_eval[n]
        dt = t_eval[n + 1] - t_eval[n]

        u = y[:, n]
        k1 = u + dt * fn(t, u)
        k2 = 3 / 4 * u + 1 / 4 * (k1 + dt * fn(t + dt, k1))
        u = 1 / 3 * u + 2 / 3 * (k2 + dt * fn(t + 0.5 * dt, k2))

        y[:, n + 1] = u
        if callback is not None:
            callback(t_eval[n + 1], u)

        assert np.all(np.isfinite(u)), np.where(~np.isfinite(u))

    return Solution(y=y)


def ckrk45(
    fn: Callable[[float, Array], Array],
    u0: Array,
    t_eval: Array,
    *,
    callback: Callable[[float, Array], None] | None,
) -> Solution:
    y = np.empty((u0.size, t_eval.size), dtype=u0.dtype)
    y[:, 0] = u0

    a = np.array(
        [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
        ],
        dtype=u0.dtype,
    )

    b = np.array(
        [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
        ],
        dtype=u0.dtype,
    )

    c = np.array(
        [
            0.0,
            1432997174477 / 9575080441755,
            2526269341429 / 6820363962896,
            2006345519317 / 3224310063776,
            2802321613138 / 2924317926251,
        ],
        dtype=u0.dtype,
    )

    if callback is not None:
        callback(t_eval[0], u0)

    for n in range(t_eval.size - 1):
        t = t_eval[n]
        dt = t_eval[n + 1] - t_eval[n]

        p = k = y[:, n]
        for i in range(a.size):
            k = a[i] * k + dt * fn(t + c[i] * dt, p)
            p = p + b[i] * k

        y[:, n + 1] = p
        if callback is not None:
            callback(t_eval[n + 1], p)

        assert np.all(np.isfinite(p)), np.where(~np.isfinite(p))

    return Solution(y=y)


# }}}


# {{{ flux


def weno_lf_flux_242(u: Array, *, epsilon: float, k: int | None) -> Array:
    from numpy.lib.stride_tricks import sliding_window_view

    a = np.abs(u)
    a[1:-1] = np.max(sliding_window_view(a, window_shape=3), axis=1)

    f = u**2 / 2
    f_m = (f - a * u) / 2
    f_p = (f + a * u) / 2

    dw_p = weno_242_downwind_stencil(u)
    dw_m = weno_242_downwind_stencil(-u)

    fw_m = weno_242_reconstruct(u[::-1], f_m[::-1], epsilon=epsilon, k=k, dw=dw_m)[::-1]
    fw_p = weno_242_reconstruct(u, f_p, epsilon=epsilon, k=k, dw=dw_p)
    fw = fw_p + fw_m

    sym_error = la.norm(u + u[::-1]) / la.norm(u)
    logger.info("sym u %.12e", sym_error)
    sym_error = la.norm(fw - fw[::-1]) / la.norm(fw)
    logger.info("sym fw %.12e", sym_error)

    if sym_error > 1.0e-10:
        raise SystemExit(1)

    return fw


def ssweno_limiter(w: Array, fw: Array, fs: Array, *, epsilon: float) -> Array:
    # limiter (Equation 3.42 in [Fisher213])
    b = np.zeros_like(fs)
    b[1:-1] = (w[1:] - w[:-1]) * (fs[1:-1] - fw[1:-1])

    return 1 - b / np.sqrt(b**2 + epsilon**2)


def ssweno_boundary(
    t: float,
    u: Array,
    *,
    method: str = "ssweno",
    ul: Callable[[float], Array],
    ur: Callable[[float], Array],
) -> Array:
    g = np.zeros_like(u)

    # boundary conditions (Equation 4.8 in [Fisher2013])
    if u[0] > 0:
        g[0] = -((u[0] + abs(u[0])) / 3 * u[0] - ul(t))

    if u[-1] < 0:
        g[-1] = +((u[-1] - abs(u[-1])) / 3 * u[-1] + ur(t))

    return g


def burgers_rhs(
    t: float,
    u: Array,
    *,
    dx: Array,
    ul: Callable[[float], Array],
    ur: Callable[[float], Array],
    c: float,
    epsilon: float,
    k: int | None,
    method: str = "ssweno",
) -> Array:
    sym_error = la.norm(u + u[::-1]) / la.norm(u)
    logger.info(
        "t = %.7e max %.12e norm = %.12e sym %.12e",
        t,
        np.max(np.abs(u)),
        np.sqrt(u @ (dx * u)),
        sym_error,
    )

    # w - entropy variable (Equation 4.2 in [Fisher2013])
    w = u
    fw = weno_lf_flux_242(u, epsilon=epsilon, k=k)

    if method == "ssweno":
        fs = two_point_entropy_flux_242(u)
        delta = ssweno_limiter(w, fw, fs, epsilon=c)

        fssw = fw + delta * (fs - fw)
        assert np.all((w[1:] - w[:-1]) * (fssw[1:-1] - fs[1:-1]) <= 0.0)
    elif method == "weno":
        fssw = fw
    elif method == "ssweno121":
        fs = two_point_entropy_flux_121(u)
        delta = ssweno_limiter(w, fw, fs, epsilon=c)

        fssw = fw + delta * (fs - fw)
        assert np.all((w[1:] - w[:-1]) * (fssw[1:-1] - fs[1:-1]) <= 0.0)
    else:
        raise ValueError(f"Unknown method: {method!r}.")

    g = ssweno_boundary(t, u, ul=ul, ur=ur, method=method)
    return -(fssw[1:] - fssw[:-1]) / dx + g / dx


# }}}


# {{{ main driver


def main(
    *,
    nx: int = 65,
    nt: int | None = None,
    a: float = -1.0,
    b: float = 1.0,
    t0: float = 0.0,
    tf: float = 1.75,
    theta: float = 0.75,
    ul: float = +1.0,
    ur: float = -1.0,
    alpha: float = 1.0,
    c: float = 1.0e-12,
    k: int | None = 8,
    ivp: str = "ckrk45",
    ic: str = "linear",
    suffix: str = "",
    animate: bool = False,
    visualize: bool = True,
) -> None:
    """
    :arg nx: number of spatial grid cells.
    :arg nt: number of time steps.
    :arg a: left grid boundary.
    :arg b: right grid boundary.
    :arg t0: initial time.
    :arg tf: final time.
    :arg ul: initial left state.
    :arg ur: initial right state,
    """
    # {{{ setup

    x = np.linspace(a, b, nx)
    # xbar = weno_242_flux_points(x)

    dx = x[1] - x[0]
    P = dx * sbp_42_p_matrix(x.size, x.dtype)

    if ic == "sine":
        gl, gr = ul, ur
    else:
        gl = np.sign(ul) * np.sqrt(3 / 2 * abs(ul))
        gr = np.sign(ur) * np.sqrt(3 / 2 * abs(ur))

    if ic == "riemann":
        u0 = ic_riemann(x, ul=gl, ur=gr, xc=0.0)
    elif ic == "linear":
        u0 = ic_linear(x, ul=gl, ur=gr)
    elif ic == "tophat":
        u0 = ic_tophat(0.0, x, us=gr, uc=gl)
    elif ic == "sine":
        u0 = ic_sine(x, k=2, dx=b - a)
    else:
        raise ValueError(f"Unknown initial condition: {ic!r}.")

    epsilon = dx**4 * np.sqrt(u0 @ (P * u0))
    cfl_dt = 0.5 * theta * dx / np.max(np.abs(u0))
    if nt is None:
        nt = int((tf - t0) / cfl_dt) + 2

    tspan = np.linspace(t0, tf, nt)
    logger.info("dx %.12e dt %.12e cfl %.12e", dx, tspan[1] - tspan[0], cfl_dt)

    assert cfl_dt > (tspan[1] - tspan[0])

    # }}}

    # {{{ integrate

    @dataclass(frozen=True)
    class Callback:
        beta: list[Array] = field(default_factory=list)
        tau: list[Array] = field(default_factory=list)
        omega: list[Array] = field(default_factory=list)
        delta: list[Array] = field(default_factory=list)

        energy: list[float] = field(default_factory=list)
        energy_per_h: list[float] = field(default_factory=list)
        entropy: list[float] = field(default_factory=list)

        symmetry: list[float] = field(default_factory=list)

        def __call__(self, t: float, u: Array) -> None:
            w = u
            fw = weno_lf_flux_242(u, epsilon=epsilon, k=4)
            fs = two_point_entropy_flux_242(u)

            delta = ssweno_limiter(w, fw, fs, epsilon=c)
            self.delta.append(delta)

            dw = weno_242_downwind_stencil(u)
            beta, tau = weno_242_smoothness(u, k=k, dw=dw)
            omega, _ = weno_242_weights(beta, tau, epsilon=epsilon)
            self.beta.append(beta.T)
            self.tau.append(tau)
            self.omega.append(omega.T)

            self.energy.append(u @ u / 2)
            self.energy_per_h.append(u @ (P * u) / 2)
            self.entropy.append(P @ (u**2 / 2))

            sym_error = la.norm(u + u[::-1]) / la.norm(u)
            self.symmetry.append(float(sym_error))

    def ul_func(t: float) -> Array:
        return ul + alpha * abs(ur - ul) / 2 * np.sin(np.pi * t)

    def ur_func(t: float) -> Array:
        return ur - alpha * abs(ur - ul) / 2 * np.sin(np.pi * t)

    def rhs_func(t: float, u: Array) -> Array:
        return burgers_rhs(
            t,
            u,
            dx=P,
            ul=ul_func,
            ur=ur_func,
            c=c,
            k=k,
            epsilon=epsilon,
        )

    callback = Callback()

    if ivp == "scipy":
        import scipy.integrate as si

        logger.info("Starting 'scipy.integrate.solve_ivp' ...")
        solution = si.solve_ivp(
            rhs_func,
            [t0, tf],
            u0,
            t_eval=tspan,
            rtol=1.0e-6,
        )
    elif ivp == "ssprk33":
        logger.info("Starting 'ssprk33' ...")
        solution = ssprk33(rhs_func, u0, t_eval=tspan, callback=callback)
    elif ivp == "ckrk45":
        logger.info("Starting 'ckrk45' ...")
        solution = ckrk45(rhs_func, u0, t_eval=tspan, callback=callback)
    else:
        raise ValueError(f"Unknown ivp: {ivp!r}.")

    logger.info("Done ...")

    # }}}

    # {{{ plot

    logger.info("Plotting...")

    # animate the solution
    # code originally lifted and modified from
    # https://matplotlib.org/examples/animation/simple_anim.html

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    def gca(f: Figure) -> Axes:
        ax = f.gca()
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x, t)$")
        ax.set_xlim((a, b))
        ax.set_ylim((np.min(solution.y) - 0.25, np.max(solution.y) + 0.25))
        # ax.set_ylim((-2.0, 2.0))
        ax.margins(0.05)

        return ax

    if ic == "tophat":
        uf = ic_tophat(tf, x, us=gr, uc=gl)
    else:
        uf = None

    # {{{ animation

    suffix = f"{ic}_{suffix}" if suffix else ic

    if animate:
        from pyshocks.tools import save_animation

        save_animation(
            # f"burgers_ssweno_{suffix}.mp4",
            None,
            x,
            (solution.y,),
            fig_kwargs={"dpi": 100, "layout": "tight"},
            plot_kwargs={"linewidth": 2},
            ymin=-2,
            ymax=+2,
        )

        # tau = np.array(callback.tau).T
        # beta = np.array(callback.beta)
        # betas = tuple(beta[:, :, i].T for i in range(beta.shape[-1]))

        # save_animation(
        #     f"burgers_ssweno_{suffix}_smoothness.mp4",
        #     xbar,
        #     betas[1:-1],
        #     legends=(r"$\beta_L$", r"$\beta_C$", r"$\beta_R$"),
        #     fig_kwargs={"dpi": 100, "layout": "tight"},
        #     plot_kwargs={"linewidth": 2},
        #     xlabel=r"$\bar{x}$",
        #     ylabel=r"$\beta$",
        #     ymin=np.nanmin(betas[1:-1]),
        #     ymax=np.nanmax(betas[1:-1]),
        # )

        # omega = np.array(callback.omega)
        # omegas = tuple(omega[:, :, i].T for i in range(omega.shape[-1]))

        # save_animation(
        #     f"burgers_ssweno_{suffix}_weights.mp4",
        #     xbar,
        #     omegas[1:-1],
        #     legends=(r"$\omega_L$", r"$\omega_C$", r"$\omega_R$"),
        #     fig_kwargs={"dpi": 100, "layout": "tight"},
        #     plot_kwargs={"linewidth": 2},
        #     xlabel=r"$\bar{x}$",
        #     ylabel=r"$\omega$",
        #     ymin=np.nanmin(omegas[1:-1]),
        #     ymax=np.nanmax(omegas[1:-1]),
        # )

        # delta = np.array(callback.delta).T

        # save_animation(
        #     f"burgers_ssweno_{suffix}_delta.mp4",
        #     xbar,
        #     (delta,),
        #     fig_kwargs={"dpi": 100, "layout": "tight"},
        #     plot_kwargs={"linewidth": 2},
        #     xlabel=r"$\bar{x}$",
        #     ylabel=r"$\delta$",
        #     ymin=np.nanmin(delta),
        #     ymax=np.nanmax(delta),
        # )
    else:
        fig = plt.figure(figsize=(10, 10), layout="tight", dpi=300)
        ax = gca(fig)
        ax.plot(x, u0, "k:", linewidth=2, markersize=10)

        if uf is not None:
            ax.plot(x, uf, "k--")
        ax.plot(x, solution.y[:, -1], "o-", linewidth=2, markersize=10)

        fig.savefig(f"burgers_ssweno_{suffix}")
        fig.clf()

    # }}}

    # {{{ energy

    if not animate:
        energy = np.array(callback.energy)
        ax = fig.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$K$")

        ax.plot(tspan, energy)
        ax.axhline(energy[0], color="k", ls="--")
        ax.axhline(np.min(energy), color="k", ls="--")
        ax.axhline(np.max(energy), color="k", ls="--")
        fig.savefig(f"burgers_ssweno_{suffix}_energy")
        fig.clf()

        entropy = np.array(callback.entropy)
        ax = fig.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$S$")

        ax.plot(tspan, entropy)
        ax.axhline(entropy[0], color="k", ls="--")
        ax.axhline(np.min(entropy), color="k", ls="--")
        ax.axhline(np.max(entropy), color="k", ls="--")
        fig.savefig(f"burgers_ssweno_{suffix}_entropy")
        fig.clf()

        symmetry = np.array(callback.symmetry)
        ax = fig.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$Error$")

        ax.plot(tspan, symmetry)
        ax.axhline(symmetry[0], color="k", ls="--")
        ax.axhline(np.min(symmetry), color="k", ls="--")
        ax.axhline(np.max(symmetry), color="k", ls="--")
        fig.savefig(f"burgers_ssweno_{suffix}_symmetry")
        fig.clf()

    # }}}

    # {{{ zoomed in

    if not animate:
        # NOTE: this needs updating when the domain changes
        ax = gca(fig)
        # ax.set_xlim((1.7, 2.1))
        # ax.set_ylim((1.2, 1.55))
        ax.set_xlim((-0.15, +0.15))
        ax.set_ylim((0.75, 1.05))

        if uf is not None:
            ax.plot(x, uf, "k--")
        ax.plot(x, solution.y[:, -1], "o-")

        fig.savefig(f"burgers_ssweno_{suffix}_top")
        fig.clf()

        ax = gca(fig)
        # ax.set_xlim((-1.5, -1.1))
        # ax.set_ylim((0.4, 0.6))
        ax.set_xlim((-0.15, +0.15))
        ax.set_ylim((-1.05, -0.75))

        if uf is not None:
            ax.plot(x, uf, "k--")
        ax.plot(x, solution.y[:, -1], "o-")

        fig.savefig(f"burgers_ssweno_{suffix}_bottom")
        fig.clf()

    # }}}

    if not animate:
        plt.close(fig)

    # }}}


# }}}


# {{{ experiments


def run_fisher2013_test0(suffix: str = "", *, animate: bool = True) -> None:
    main(
        nx=64,
        nt=None,
        a=-1.0,
        b=1.0,
        t0=0.0,
        tf=1.0,
        theta=0.5,
        ul=+1.0,
        ur=-1.0,
        alpha=0.0,
        ivp="ckrk45",
        ic="linear",
        suffix=suffix,
        animate=animate,
        visualize=True,
    )


def run_fisher2013_test1(suffix: str = "", *, animate: bool = True) -> None:
    main(
        nx=65,
        nt=None,
        a=-1.0,
        b=1.0,
        t0=0.0,
        tf=1.0,
        theta=0.1,
        ul=+1.0,
        ur=-1.0,
        alpha=0.5,
        ivp="ckrk45",
        ic="linear",
        suffix=suffix,
        animate=animate,
        visualize=True,
    )


# }}}


# {{{ tests


def cell_averaged(x: Array, f: Callable[[Array], Array], *, order: int) -> Array:
    from numpy.polynomial.legendre import leggauss

    xi, wi = leggauss(order)

    dm = 0.5 * (x[1:] - x[:-1]).reshape(1, -1)
    xm = 0.5 * (x[1:] + x[:-1]).reshape(1, -1)

    xi = xm + dm * xi.reshape(-1, 1)
    wi = dm * wi.reshape(-1, 1)

    favg = np.sum(f(xi) * wi, axis=0) / np.diff(x)

    return favg


def run_checks(*, visualize: bool = False) -> None:
    check_weno_242_smoothness(visualize=visualize)
    check_weno_242_interp(visualize=visualize)
    check_weno_242_entropy_flux(visualize=visualize)


def check_weno_242_smoothness(
    a: float = -3.0, b: float = 3.0, *, visualize: bool = False
) -> None:
    from pyshocks import EOCRecorder

    stencils = ["LL", "L", "C", "R", "RR"]
    eocs = [EOCRecorder(name=f"beta_{stencils[i]}") for i in range(5)] + [
        EOCRecorder(name="tau")
    ]

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    k = 4.0 * np.pi / (b - a)

    for nx in (128, 256, 384, 512, 768, 1024):
        x = np.linspace(a, b, nx + 1)
        y = weno_242_flux_points(x)
        # y = np.concatenate([[a], (x[1:] + x[:-1]) / 2, [b]])
        h = x[1] - x[0]

        u = np.sin(k * (x + 1))
        du = np.stack([
            np.zeros_like(y),
            h * k * np.roll(np.cos(k * (y + 1)), 1),
            h * k * np.roll(np.cos(k * (y + 1)), 0),
            h * k * np.roll(np.cos(k * (y + 1)), -1),
            np.zeros_like(y),
        ])
        # NOTE: these are separate because we can't nicely roll it
        du[LL, 3] = h * k * np.cos(k * (y[1] + 1))
        du[LL, -2] = h * k * np.cos(k * (y[-4] + 1))
        du[RR, 1] = h * k * np.cos(k * (y[3] + 1))
        du[RR, -4] = h * k * np.cos(k * (y[-2] + 1))

        ddu = -(h**3) * k**3 * np.cos(k * (y + 1))

        beta, tau = weno_242_smoothness(u, p=1)
        mask = weno_242_weight_mask(y.size, x.dtype)
        mask[:, 0] = mask[:, -1] = False

        for s in (LL, L, C, R, RR):
            du_norm = la.norm(du[s, mask[s]], ord=1)
            error = la.norm(du[s, mask[s]] - beta[s, mask[s]], ord=1) / du_norm
            eocs[s].add_data_point(float(h), float(error))

        error = la.norm(tau[1:-1] - ddu[1:-1], ord=1) / la.norm(ddu[1:-1], ord=1)
        eocs[-1].add_data_point(float(h), float(error))

        if visualize:
            du = du * mask
            beta = beta * mask

            # logger.info("%.12e %.12e", du[LL, 0], beta[LL, 3])
            # ax.plot(y, beta[L] * mask[L], y, du[L] * mask[L], "k--")
            # ax.plot(y, tau, y, ddu, "k--")
            ax.semilogy(y, np.abs(du[L] - beta[L]) + 1.0e-16)

    if visualize:
        ax.set_xlabel("$y$")
        ax.set_ylabel("$Error$")

        fig.savefig("burgers_ssweno_stencil_smoothness")
        mp.close(fig)

    for eoc in eocs:
        logger.info("\n%s", eoc)
        # assert eoc.satisfied(2.0, slack=0.25)


def check_weno_242_interp(
    a: float = -3.0, b: float = 3.0, *, visualize: bool = False
) -> None:
    from pyshocks import EOCRecorder

    stencils = ["LL", "L", "C", "R", "RR"]
    eocs = [EOCRecorder(name=f"{stencils[i]}") for i in range(5)] + [
        EOCRecorder(name="WENO")
    ]

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    from functools import partial

    func = partial(ic_sine, k=2, dx=b - a)
    func = partial(ic_poly, a=0, b=0, c=1, d=0)
    # func = partial(ic_gaussian, sigma=5.0e-1, xm=(a + b) / 2)

    for nx in (128, 192, 256, 384, 512, 768, 1024):
        x = np.linspace(a, b, nx + 1)
        y = weno_242_flux_points(x)
        # y = np.concatenate([[a], (x[1:] + x[:-1]) / 2, [b]])
        dx = x[1] - x[0]

        f = cell_averaged(y, func, order=4)
        fbar = func(y)

        fhat = weno_242_interp(f)
        mask = weno_242_weight_mask(y.size, x.dtype)

        for s in (L, C, R, RR):
            fbar_norm = la.norm(fbar[mask[s]], ord=1)
            error = la.norm(fbar[mask[s]] - fhat[s, mask[s]], ord=1) / fbar_norm
            eocs[s].add_data_point(float(dx), float(error))

        error = abs(fbar[4] - fhat[LL, 3])
        eocs[LL].add_data_point(float(dx), float(error))

        fhathat = weno_242_reconstruct(f, f, epsilon=dx**2)
        error = la.norm(fbar - fhathat, ord=1) / la.norm(fbar, ord=1)
        # j = 10
        # error = abs(fbar[j] - fhathat[j])
        eocs[-1].add_data_point(float(dx), float(error))

        if visualize:
            # print((fbar * mask[LL])[:10])
            # print(fhat[LL, :10])
            # if nx == 1024:
            #     breakpoint()
            # ax.plot(y, fbar * mask[LL], y, fhat[LL], "--")
            ax.plot(y, fhat[C], "--")
            # ax.semilogy(y, np.abs(uhat - uy) + 1.0e-16)

    if visualize:
        ax.plot(y, fbar, "--")
        ax.set_xlabel("$y$")
        ax.set_ylabel("$Error$")

        fig.savefig("burgers_ssweno_stencil_interp")
        mp.close(fig)

    for eoc in eocs:
        logger.info("\n%s", eoc)
        # assert eoc.satisfied(2.0, slack=0.25)


def check_weno_242_entropy_flux(
    a: float = -3.0, b: float = 3.0, *, visualize: bool = False
) -> None:
    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

    from functools import partial

    func = partial(ic_sine, k=2)
    # func = partial(ic_poly, a=1, b=1, c=0, d=0)

    # for nx in [128, 192, 256, 384, 512, 768, 1024]:
    for nx in [128, 192, 256]:
        x = np.linspace(a, b, nx + 1)
        y = weno_242_flux_points(x)

        u = func(x)
        fs_121 = two_point_entropy_flux_121(u)
        fs_242 = two_point_entropy_flux_242(u)

        w = u
        psi = w**3 / 6
        error_121 = (w[1:] - w[:-1]) * fs_121[1:-1] - (psi[1:] - psi[:-1])
        logger.info("121: %.12e", la.norm(error_121))
        error_242 = (w[1:] - w[:-1]) * fs_242[1:-1] - (psi[1:] - psi[:-1])
        logger.info("242: %.12e", la.norm(error_242))

        if visualize:
            ax.plot(y[1:-1], error_242, label="$f_s^{2-4-2}$")
            ax.plot(y[1:-1], error_121, label="$f_s^{1-2-1}$")

    if visualize:
        ax.set_xlabel("$y$")
        ax.set_ylabel("$f_s$")
        ax.legend()

        fig.savefig("burgers_ssweno_entropy_flux")
        mp.close(fig)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_checks()
