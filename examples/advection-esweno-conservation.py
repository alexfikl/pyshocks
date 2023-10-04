# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from typing import cast

import jax
import jax.numpy as jnp

from pyshocks import (
    advection,
    apply_operator,
    get_logger,
    make_uniform_cell_grid,
    predict_timestep,
    reconstruction,
)
from pyshocks.tools import Array, ScalarLike

logger = get_logger("advection")


def cosine_pulse(
    x: Array,
    *,
    w: float = 5.0 * jnp.pi,
    xc: float = 0.5,
    sigma: float = 0.2,
    p: int = 4,
) -> Array:
    """Compute a cosine pulse using the parameters from [Yamaleev2009] Eq. 83."""
    r = (0.5 + 0.5 * jnp.cos(w * (x - xc))) ** p
    mask = (jnp.abs(x - xc) < sigma).astype(x.dtype)

    return cast(Array, r * mask)


def main(
    rec_name: str,
    *,
    outdir: pathlib.Path,
    n: int = 256,
    periods: int = 5,
    theta: float = 0.45,
    interactive: bool = False,
    visualize: bool = True,
    verbose: bool = True,
) -> None:
    r"""
    :arg n: number of cells the discretize the domain.
    :arg theta: Courant number used in time step estimation as
        :math:`\Delta t = \theta \Delta \tilde{t}`.
    :arg periods: number of time periods.
    """
    if visualize or interactive:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            interactive = visualize = False

    # {{{ setup

    from pyshocks import cell_average, make_leggauss_quadrature
    from pyshocks.scalar import PeriodicBoundary

    grid = make_uniform_cell_grid(a=0.0, b=1.0, n=n, nghosts=3)
    quad = make_leggauss_quadrature(grid, order=5)

    velocity = jnp.ones_like(grid.x)
    u0 = cell_average(quad, cosine_pulse)

    from pyshocks.weno import es_weno_parameters

    eps, delta = es_weno_parameters(grid, u0)
    logger.info("esweno: eps %.12e delta %.12e", eps, delta)

    boundary = PeriodicBoundary()
    rec = reconstruction.make_reconstruction_from_name(rec_name)
    scheme = advection.make_scheme_from_name(
        "esweno32", rec=rec, velocity=velocity, eps=eps, delta=delta
    )

    # }}}

    # {{{ plotting

    s = grid.i_
    if interactive:
        fig = plt.figure()
        ax = fig.gca()
        plt.ion()

        (ln0,) = ax.plot(grid.x[s], u0[s], "o-", ms=1)

        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_ylim((float(jnp.min(u0) - 1), float(jnp.max(u0) + 1)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

    # }}}

    # {{{ right-hand side

    def _predict_timestep(_t: ScalarLike, _u: Array) -> Array:
        return theta * predict_timestep(scheme, grid, boundary, _t, _u)

    def _apply_operator(_t: ScalarLike, _u: Array) -> Array:
        return apply_operator(scheme, grid, boundary, _t, _u)

    # }}}

    # {{{ evolution

    from pyshocks import norm, timestepping

    tfinal = 1.0 * periods
    method = timestepping.SSPRK33(
        predict_timestep=jax.jit(_predict_timestep),
        source=jax.jit(_apply_operator),
        checkpoint=None,
    )

    l2_norm_tmp = []
    tv_norm_tmp = []
    event = None

    for event in timestepping.step(method, u0, tfinal=tfinal):
        l2_norm_tmp.append(norm(grid, event.u, p=2))
        tv_norm_tmp.append(norm(grid, event.u, p="tvd"))

        logger.info(
            "%s || norm l2 %.5e tv %.5e", event, l2_norm_tmp[-1], tv_norm_tmp[-1]
        )

        if interactive:
            ln0.set_ydata(event.u[s])
            plt.pause(0.01)

    # }}}

    # {{{ visualize

    if interactive:
        plt.ioff()
        plt.close(fig)

    if visualize:
        l2_norm = jnp.array(l2_norm_tmp, dtype=u0.dtype)
        tv_norm = jnp.array(tv_norm_tmp, dtype=u0.dtype)

        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s], "o-")
        ax.plot(grid.x[s], u0[s], "k--")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

        fig.savefig(outdir / f"advection_esweno_{scheme.name}_{n:05d}_solution")
        fig.clf()

        ax = fig.gca()
        ax.plot((l2_norm - l2_norm[0]) / l2_norm[0])
        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$(\|u^n\|_2 - \|u_0\|_2) / \|u_0\|_2$")

        fig.savefig(outdir / f"advection_esweno_{scheme.name}_{n:05d}_l2")
        fig.clf()

        ax = fig.gca()
        ax.plot(jnp.diff(tv_norm))
        ax.set_xlabel("$n$")
        ax.set_ylabel("$TV(u^{n + 1}) - TV(u^n)$")

        fig.savefig(outdir / f"advection_esweno_{scheme.name}_{n:05d}_tv")
        fig.clf()

        plt.close(fig)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reconstruct",
        default="esweno32",
        type=str.lower,
        choices=reconstruction.reconstruction_ids(),
    )
    parser.add_argument("-n", "--numcells", type=int, default=200)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()
    main(
        args.reconstruct,
        n=args.numcells,
        outdir=args.outdir,
        interactive=args.interactive,
    )
