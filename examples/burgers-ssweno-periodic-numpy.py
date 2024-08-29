# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

# NOTE: some inspiration for similar codes
#  https://github.com/surajp92/CFD_Julia/blob/master/07_Inviscid_Burgers_Flux_Splitting/burgers_flux_splitting.jl
#  https://github.com/omersan/5.9.Burgers-Equation-WENO5-FluxSplitting/blob/master/burger_sol.f90
#  https://github.com/wme7/WENO/blob/master/Non-lineard1/WENO3resAdv1d.m

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import scipy.ndimage as snd

from pyshocks.logging import get_logger
from pyshocks.tools import set_recommended_matplotlib

logger = get_logger("burgers_ssweno_periodic_numpy")
set_recommended_matplotlib()


# FIXME: should probably use np.ndarray[Any, np.float64] or something
Array = Any
Matrix = Any
L, C, R = 0, 1, 2


# {{{ initial conditions


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


def ic_sine(x: Array, *, k: float, dx: float) -> Array:
    return np.sin(2.0 * np.pi * k * x / dx)


# }}}


# {{{ weno 2-4-2

# [Fisher2011] T. C. Fisher, M. H. Carpenter, N. K. Yamaleev, S. H. Frankel,
#   Boundary Closures for Fourth-Order Energy Stable Weighted Essentially
#   Non-Oscillatory Finite-Difference Schemes,
#   Journal of Computational Physics, Vol. 230, pp. 3727--3752, 2011,
#   https://dx.doi.org/10.1016/j.jcp.2011.01.043.


# {{{ interpolation


def weno_242_interp(f: Array) -> Array:
    fbar = np.zeros((3, f.size), dtype=f.dtype)
    fbar[L] = snd.convolve1d(f, [3 / 2, -1 / 2, 0], origin=-1, mode="wrap")
    fbar[C] = snd.convolve1d(f, [1 / 2, 1 / 2, 0], origin=0, mode="wrap")
    fbar[R] = snd.convolve1d(f, [-1 / 2, 3 / 2, 0], origin=1, mode="wrap")

    return fbar


# }}}


# {{{ smoothness


def weno_242_downwind(
    u: Array, beta: Array, *, dw: Array, k: int, weighted: bool = False
) -> Array:
    # compute some average for beta
    # NOTE: [Fisher2011] has `(sum(beta^k) / 3)^(1/k)` but that seems to lead
    # to values of `beta[d] < beta` e.g. beta = (1, 0, 0) gives
    #
    #   bmax = (1 / 3)^(1 / k) < 1, but -> 1 as k -> infinity
    #
    # so we don't really get what we need in the common case

    n = beta.shape[0] if weighted else 1
    bmax = (np.sum(np.abs(beta) ** k, axis=0) / n) ** (1 / k)

    i = np.arange(dw.size)
    beta[dw, i] = bmax

    if __debug__ and not weighted:
        # NOTE: the downwind weight is supposed to be smaller
        flags = np.array([np.all(beta[dw[i], i] >= beta[:, i]) for i in range(dw.size)])
        assert np.all(flags), flags

    return beta


def weno_242_smoothness(
    u: Array,
    *,
    dw: Array | None = None,
    k: int | None = None,
    weighted: bool = False,
) -> tuple[Array, Array]:
    n = u.size

    # beta: Equation 71
    beta = np.zeros((3, n), dtype=u.dtype)
    beta[L] = snd.convolve1d(u, [1, -1, 0], origin=-1, mode="wrap") ** 2
    beta[C] = snd.convolve1d(u, [1, -1, 0], origin=0, mode="wrap") ** 2
    beta[R] = snd.convolve1d(u, [1, -1, 0], origin=1, mode="wrap") ** 2

    # tau: Equation 74
    tau = snd.convolve1d(u, [-1, 3, -3, 1], origin=0, mode="wrap") ** 2

    # downwind: Equation 72
    # set the downwind stencil to ensure stability
    if k is not None:
        assert dw is not None
        beta = weno_242_downwind(u, beta, dw=dw, k=k, weighted=weighted)

    return beta, tau


# }}}


# {{{ weights


def weno_242_weights(beta: Array, tau: Array, *, epsilon: float) -> Array:
    d = np.array([[1 / 6, 2 / 3, 1 / 6]], dtype=beta.dtype).T
    alpha = d * (1 + tau / (epsilon + beta))
    omega = alpha / np.sum(alpha, axis=0, keepdims=True)

    return omega


# }}}


# {{{ reconstruct


def weno_242_reconstruct(
    u: Array,
    f: Array,
    *,
    k: int | None = None,
    dw: Array | None = None,
    weighted: bool = True,
    epsilon: float | None = None,
    dx: float | None = None,
) -> Array:
    """
    :arg epsilon: tolerance used in computing the WENO weights.
    :arg k: power used when updating the downstream WENO weights.
    """
    # smoothness coefficients: [Fisher2011]
    beta, tau = weno_242_smoothness(u, dw=dw, k=k, weighted=weighted)

    if epsilon is None:
        u_sqr = snd.convolve1d(u**2, [1, 1, 1], mode="wrap")
        b_sqr = snd.convolve1d(
            np.linalg.norm(beta, axis=0) ** 2, [1, 1, 1], mode="wrap"
        )
        c = 1 + u_sqr / 3 + np.sqrt(b_sqr) / 4

        assert dx is not None
        epsilon = np.minimum(c * dx**4, dx**2)

    # nonlinear weights: [Fisher2011]
    omega = weno_242_weights(beta, tau, epsilon=epsilon)

    # interpolate flux: [Fisher2011] Equation 68
    fbar = weno_242_interp(f)

    if __debug__ and k is not None and not weighted:
        # NOTE: the downwind weight is supposed to be smaller
        assert dw is not None
        flags = np.array([
            np.all(omega[dw[i], i] <= omega[:, i]) for i in range(dw.size)
        ])

        assert np.all(flags), flags

    return np.sum(omega * fbar, axis=0)


# }}}

# }}}


# {{{ entropy conservative flux

# [Fisher2013] T. C. Fisher, M. H. Carpenter,
#   High-Order Entropy Stable Finite Difference Schemes for Nonlinear
#   Conservation Laws: Finite Domains,
#   Journal of Computational Physics, Vol. 252, pp. 518--557, 2013,
#   https://dx.doi.org/10.1016/j.jcp.2013.06.014.


@lru_cache
def sbp_42_p_matrix(n: int, dtype: Any) -> Array:
    """
    :return: the diagonal of the :math:`P` norm matrix.
    """
    return np.full(n, 1.0, dtype=dtype)


def two_point_entropy_flux(u: float, v: float) -> float:
    return (u * u + u * v + v * v) / 6


def two_point_entropy_flux_242(u: Array) -> Array:
    n = u.size
    fs = np.zeros(n, dtype=u.dtype)

    for i in range(n):
        im = i - 1
        ip = (i + 1) % n
        ipp = (i + 2) % n

        fs[i] = (
            4 / 3 * two_point_entropy_flux(u[i], u[ip])
            - 1 / 6 * two_point_entropy_flux(u[i], u[ipp])
            - 1 / 6 * two_point_entropy_flux(u[im], u[ip])
        )
        # fs[i] = two_point_entropy_flux(u[i], u[ip])

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


def weno_lf_flux_242(
    u: Array,
    *,
    k: int | None,
    epsilon: float | None,
    dx: float | None,
) -> Array:
    # Burgers flux
    f = u**2 / 2

    # local Lax-Friedrichs numerical flux
    a = np.abs(u)
    a = snd.maximum_filter1d(a, size=3, mode="wrap")

    offset_m = -1
    u_m = np.roll(u, offset_m)
    f_m = np.roll((f - a * u) / 2, offset_m)

    offset_p = 0
    u_p = np.roll(u, offset_p)
    f_p = np.roll((f + a * u) / 2, offset_p)

    # get downwind stencils
    if k is not None:
        ubar = snd.convolve1d(u, [0.5, 0.5, 0], origin=0, mode="wrap")
        dw_p: Array = np.where(ubar > 0, R, L)
        dw_p = np.roll(dw_p, offset_p)

        dw_m: Array = np.where(ubar <= 0, R, L)
        dw_m = np.roll(dw_m, offset_m)
    else:
        dw_m = dw_p = None

    # weno reconstruction
    fw_m = weno_242_reconstruct(
        u_m[::-1], f_m[::-1], dw=dw_m[::-1], epsilon=epsilon, k=k, dx=dx
    )[::-1]
    fw_p = weno_242_reconstruct(u_p, f_p, dw=dw_p, epsilon=epsilon, k=k, dx=dx)

    return fw_p + fw_m


def ssweno_limiter(w: Array, fw: Array, fs: Array, *, epsilon: float) -> Array:
    # limiter (Equation 3.42 in [Fisher213])
    b = np.zeros_like(fs)
    b[:-1] = (w[1:] - w[:-1]) * (fs[:-1] - fw[:-1])
    b[-1] = (w[0] - w[-1]) * (fs[-1] - fw[-1])

    bhat = np.sqrt(b**2 + epsilon**2)
    return (bhat - b) / (bhat)


def burgers_rhs(
    t: float,
    u: Array,
    *,
    dx: Array,
    ul: Callable[[float], Array],
    ur: Callable[[float], Array],
    c: float,
    epsilon: float | None = None,
    k: int | None = None,
) -> Array:
    logger.info(
        "t = %.7e max %.12e norm = %.12e", t, np.max(np.abs(u)), np.sqrt(u @ (dx * u))
    )

    # w - entropy variable (Equation 4.2 in [Fisher2013])
    w = u
    fssw = np.zeros(u.size + 1)

    fw = weno_lf_flux_242(u, epsilon=epsilon, k=k, dx=dx[0])
    fs = two_point_entropy_flux_242(u)
    delta = ssweno_limiter(w, fw, fs, epsilon=c)

    fssw[1:] = fw + delta * (fs - fw)
    # fssw[1:] = fs
    # fssw[1:] = fw

    fssw[0] = fssw[-1]

    return -(fssw[1:] - fssw[:-1]) / dx


# }}}


# {{{ main driver


def evaluate_at(t: Array, x: Array, fn: Callable[[Array, Array], Array]) -> Array:
    result = np.empty((x.size, t.size), dtype=x.dtype)

    for n in range(t.size):
        result[:, n] = fn(t[n], x)

    return result


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
    c: float = 1.0e-12,
    k: int | None = 4,
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

    x = np.linspace(a, b, nx, endpoint=False)

    dx = x[1] - x[0]
    P = np.full_like(x, dx)
    # xbar = x + dx / 2

    if ic == "tophat":
        u0 = ic_tophat(0.0, x, us=ur, uc=ul)
    elif ic == "sine":
        u0 = ic_sine(x, k=2, dx=b - a)
    else:
        raise ValueError(f"Unknown initial condition: {ic!r}.")

    epsilon = dx**4 * np.max(np.abs(u0))
    # epsilon = None
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

        def __call__(self, t: float, u: Array) -> None:
            # get downwind stencils
            ubar = snd.convolve1d(u, [0.5, 0.5, 0], origin=0, mode="wrap")
            dw = np.where(ubar > 0, R, L)

            beta, tau = weno_242_smoothness(u, dw=dw, k=k)
            omega = weno_242_weights(beta, tau, epsilon=dx**4)
            self.beta.append(beta.T)
            self.tau.append(tau)
            self.omega.append(omega.T)

            self.energy.append(u @ u / 2)
            self.energy_per_h.append(u @ (P * u) / 2)
            self.entropy.append(P @ (u**2 / 2))

    def ul_func(t: float) -> Array:
        return u0[0]

    def ur_func(t: float) -> Array:
        return u0[1]

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
        uf = ic_tophat(tf, x, us=ur, uc=ul)
    else:
        uf = None

    # {{{ animation

    suffix = f"{ic}_{suffix}" if suffix else ic

    if animate:
        from pyshocks.tools import save_animation

        plotted_solution: tuple[Any, ...] = (solution.y,)
        if ic == "tophat":
            plotted_solution = (
                *plotted_solution,
                evaluate_at(tspan, x, lambda t, x: ic_tophat(t, x, us=ur, uc=ul)),
            )

        save_animation(
            # f"burgers_ssweno_{suffix}.mp4",
            None,
            x,
            plotted_solution,
            fig_kwargs={"dpi": 200, "layout": "tight"},
            plot_kwargs={"linewidth": 2, "markersize": 5},
        )

        # # tau = np.array(callback.tau).T
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
        # )

        omega = np.array(callback.omega)
        omegas = tuple(omega[:, :, i].T for i in range(omega.shape[-1]))

        save_animation(
            # f"burgers_ssweno_{suffix}_weights.mp4",
            None,
            x,
            omegas,
            legends=(r"$\omega^L$", r"$\omega^C$", r"$\omega^R$"),
            fig_kwargs={"dpi": 100, "layout": "tight"},
            plot_kwargs={"linewidth": 2},
            xlabel=r"$\bar{x}$",
            ylabel=r"$\omega$",
        )

        # delta = np.array(callback.delta).T

        # save_animation(
        #     f"burgers_ssweno_{suffix}_delta.mp4",
        #     xbar,
        #     (delta,),
        #     fig_kwargs={"dpi": 100, "layout": "tight"},
        #     plot_kwargs={"linewidth": 2},
        #     xlabel=r"$\bar{x}$",
        #     ylabel=r"$\delta$",
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
        fig.savefig(f"burgers_ssweno_{ic}_energy")
        fig.clf()

        entropy = np.array(callback.entropy)
        ax = fig.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$S$")

        ax.plot(tspan, entropy)
        ax.axhline(entropy[0], color="k", ls="--")
        ax.axhline(np.min(entropy), color="k", ls="--")
        ax.axhline(np.max(entropy), color="k", ls="--")
        fig.savefig(f"burgers_ssweno_{ic}_entropy")
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

        fig.savefig(f"burgers_ssweno_{ic}_top")
        fig.clf()

        ax = gca(fig)
        # ax.set_xlim((-1.5, -1.1))
        # ax.set_ylim((0.4, 0.6))
        ax.set_xlim((-0.15, +0.15))
        ax.set_ylim((-1.05, -0.75))

        if uf is not None:
            ax.plot(x, uf, "k--")
        ax.plot(x, solution.y[:, -1], "o-")

        fig.savefig(f"burgers_ssweno_{ic}_bottom")
        fig.clf()

    # }}}

    if not animate:
        plt.close(fig)

    # }}}


# }}}


# {{{ experiments


def run_periodic_test0(suffix: str = "", *, animate: bool = True) -> None:
    main(
        nx=65,
        nt=None,
        a=-1.0,
        b=1.0,
        t0=0.0,
        tf=0.75,
        theta=0.5,
        ul=1.5,
        ur=0.5,
        ivp="ckrk45",
        ic="tophat",
        suffix=suffix,
        animate=animate,
        visualize=True,
    )


def run_periodic_test1(suffix: str = "", *, animate: bool = True) -> None:
    main(
        nx=127,
        nt=None,
        a=-1.0,
        b=1.0,
        t0=0.0,
        tf=1.5,
        theta=0.5,
        ul=+1.0,
        ur=-1.0,
        ivp="ckrk45",
        ic="sine",
        suffix=suffix,
        animate=animate,
        visualize=True,
    )


# }}}


# {{{ tests


def cell_averaged(x: Array, f: Callable[[Array], Array], *, order: int) -> Array:
    dx = x[1] - x[0]

    from numpy.polynomial.legendre import leggauss

    xi, wi = leggauss(order)

    dm = 0.5 * (x[1:] - x[:-1]).reshape(1, -1)
    xm = 0.5 * (x[1:] + x[:-1]).reshape(1, -1)

    xi = xm + dm * xi.reshape(-1, 1)
    wi = dm * wi.reshape(-1, 1)

    favg = np.sum(f(xi) * wi, axis=0) / dx

    return favg


def test_interpolation(*, visualize: bool = True) -> None:
    r"""Tests the WENO interpolation on a smooth function:
    * for the individual stencils :math:`S_L, S_C` and :math:`S_R`.
    * for the full stencil :math:`S_L \cup S_C \cup S_R`.
    """
    from pyshocks import EOCRecorder

    eoc = [EOCRecorder() for _ in range(3)] + [EOCRecorder()]

    import numpy.linalg as la

    a, b = -1.0, 1.0

    for n in (16, 32, 64, 128, 256, 384, 512):
        x = np.linspace(a, b, n, endpoint=False)
        dx = x[1] - x[0]
        y = x + dx / 2

        # NOTE: to get high-order full stencil interpolation, we need to take
        # the average here, since that is the quantity getting reconstructed
        ybar = np.concatenate([[y[0] - dx], y])
        f = cell_averaged(ybar, lambda z: ic_sine(z, k=2, dx=b - a), order=6)
        fbar = ic_sine(y, k=2, dx=b - a)

        # individual stencil interpolation
        fhat = weno_242_interp(f)

        error = la.norm(fbar - fhat, axis=1, ord=1) / la.norm(fbar, ord=1)
        eoc[L].add_data_point(dx, error[L])
        eoc[C].add_data_point(dx, error[C])
        eoc[R].add_data_point(dx, error[R])

        # full stencil interpolation (both sides should work!)
        fhathat = weno_242_reconstruct(f, f, epsilon=dx**4)

        error_f = la.norm(fbar - fhathat, ord=np.inf) / la.norm(fbar, ord=np.inf)
        eoc[-1].add_data_point(float(dx), float(error_f))

    logger.info("\n%s\n%s\n%s\n%s", *eoc)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        # ax.plot(y, f, ":")
        ax.plot(y, fbar, "k--")
        ax.plot(y, fhathat)
        # ax.plot(y, fhat[L], label="$f^L$")
        # ax.plot(y, fhat[C], label="$f^C$")
        # ax.plot(y, fhat[R], label="$f^R$")
        # ax.semilogy(np.abs(fbar - fhathat), "o-")
        ax.set_xlabel(r"$\bar{x}$")  # noqa: RUF027
        # ax.set_ylim((1.0e-16, 1))
        # ax.legend()

        fig.savefig("burgers_ssweno_interp_convergence")
        mp.close(fig)


def test_smoothness(*, visualize: bool = True) -> None:
    from pyshocks import EOCRecorder

    stencils = ["L", "C", "R"]
    eoc = [EOCRecorder(name=rf"beta_{stencils[i]}") for i in range(3)] + [
        EOCRecorder(name="tau")
    ]

    import numpy.linalg as la

    a, b = -1.0, 1.0
    k = 4.0 * np.pi / (b - a)
    for n in (16, 32, 64, 128, 256, 384, 512):
        x = np.linspace(a, b, n, endpoint=False)
        dx = x[1] - x[0]
        y = x + dx / 2

        u = np.sin(k * x)
        du = np.stack([
            np.roll(dx * k * np.cos(k * y), +1) ** 2,
            np.roll(dx * k * np.cos(k * y), +0) ** 2,
            np.roll(dx * k * np.cos(k * y), -1) ** 2,
        ])
        ddu = (dx**3 * k**3 * np.cos(k * y)) ** 2

        beta, tau = weno_242_smoothness(u, k=None)

        error = la.norm(beta - du, axis=1, ord=1) / la.norm(du[C], ord=1)
        eoc[L].add_data_point(dx, error[L])
        eoc[C].add_data_point(dx, error[C])
        eoc[R].add_data_point(dx, error[R])

        error = la.norm(tau - ddu, ord=1) / la.norm(ddu, ord=1)
        eoc[-1].add_data_point(dx, error)

    logger.info("\n%s\n%s\n%s\n%s", *eoc)

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        # ax.plot(y, du[C], "k--")
        # ax.plot(y, beta[L], label=r"$\beta^L$")
        # ax.plot(y, beta[C], label=r"$\beta^C$")
        # ax.plot(y, beta[R], label=r"$\beta^R$")
        ax.plot(y, ddu, "k--")
        ax.plot(y, tau, label=r"$\tau$")
        ax.set_xlabel(r"$\bar{x}$")  # noqa: RUF027
        ax.legend()

        fig.savefig("burgers_ssweno_smooth_convergence")
        mp.close(fig)


def test_stencil_pick(*, visualize: bool = False) -> None:
    n = 32
    a, b = -1.0, 1.0
    xm = (b + a) / 2
    dx = b - a
    xa, xb = xm - dx / 4, xm + dx / 4

    x = np.linspace(a, b, n, endpoint=False)
    dx = x[1] - x[0]
    y = x + dx / 2

    u = 1.0 + np.logical_and(x > xa, x < xb).astype(x.dtype)

    # u_p
    u_p = u
    ubar = snd.convolve1d(u, [0.5, 0.5, 0], origin=0, mode="wrap")
    dw = np.where(ubar > 0, R, L)

    beta, tau = weno_242_smoothness(u_p, k=4, dw=dw, weighted=True)
    omega = weno_242_weights(beta, tau, epsilon=dx**4)
    ubar = weno_242_interp(u_p)
    uhat_p = np.sum(omega * ubar, axis=0)

    # u_m
    u_m = np.roll(-u[::-1], 0)
    ubar = snd.convolve1d(u_m, [0.5, 0.5, 0], origin=0, mode="wrap")
    dw = np.where(ubar > 0, R, L)

    beta, tau = weno_242_smoothness(u_m, k=4, dw=dw, weighted=True)
    omega = weno_242_weights(beta, tau, epsilon=dx**4)
    ubar = weno_242_interp(u_m)
    uhat_m = -np.sum(omega * ubar, axis=0)[::-1]

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        ax.plot(y, omega[L], "x-", label=r"$\omega^L$")
        ax.plot(y, omega[C], "o-", label=r"$\omega^C$")
        ax.plot(y, omega[R], "+-", label=r"$\omega^R$")
        ax.set_xlabel(r"$\bar{x}$")  # noqa: RUF027
        ax.legend()

        fig.savefig("burgers_ssweno_jump_weights")
        fig.clf()

    if visualize:
        ax = fig.gca()

        ax.plot(x, u, "k-", ms=3, label="$u$")
        ax.plot(y, uhat_p, "x-", ms=5, label=r"$\bar{u}^+$")  # noqa: RUF027
        ax.plot(y, uhat_m, "s-", ms=3, label=r"$\bar{u}^-$")  # noqa: RUF027
        ax.plot(x, u, "ko", ms=5)
        ax.set_xlabel("$x$")
        ax.legend()

        fig.savefig("burgers_ssweno_jump_interp")
        fig.clf()


def run_tests(*, visualize: bool = False) -> None:
    test_interpolation(visualize=visualize)
    test_smoothness(visualize=visualize)
    test_stencil_pick(visualize=visualize)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_tests()
