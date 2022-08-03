# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import jax
import jax.numpy as jnp

from pyshocks import (
    make_uniform_grid,
    apply_operator,
    predict_timestep,
)
from pyshocks import advection, reconstruction, get_logger

logger = get_logger("advection")


def cosine_pulse(
    x: jnp.ndarray,
    *,
    w: float = 5.0 * jnp.pi,
    xc: float = 0.5,
    sigma: float = 0.2,
    p: int = 4,
) -> jnp.ndarray:
    """Compute a cosine pulse using the parameters from [Yamaleev2009] Eq. 83."""
    r = (0.5 + 0.5 * jnp.cos(w * (x - xc))) ** p
    mask = (jnp.abs(x - xc) < sigma).astype(x.dtype)

    return r * mask


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

    # {{{ setup

    from pyshocks.scalar import PeriodicBoundary
    from pyshocks import Quadrature, cell_average

    grid = make_uniform_grid(a=0.0, b=1.0, n=n, nghosts=3)
    quad = Quadrature(grid=grid, order=5)

    velocity = jnp.ones_like(grid.x)  # type: ignore[no-untyped-call]
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

    if interactive or visualize:
        import matplotlib.pyplot as plt

    s = grid.i_
    if interactive:
        fig = plt.figure()
        ax = fig.gca()
        plt.ion()

        (ln0,) = ax.plot(grid.x[s], u0[s], "o-", ms=1)

        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1, jnp.max(u0) + 1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

        ax.grid(True)

    # }}}

    # {{{ right-hand side

    def _predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, _t, _u)

    def _apply_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, _t, _u)

    # }}}

    # {{{ evolution

    from pyshocks import norm
    from pyshocks import timestepping

    tfinal = 1.0 * periods
    method = timestepping.SSPRK33(
        predict_timestep=jax.jit(_predict_timestep),
        source=jax.jit(_apply_operator),
        checkpoint=None,
    )

    l2_norm = []
    tv_norm = []
    event = None

    for event in timestepping.step(method, u0, tfinal=tfinal):
        l2_norm.append(norm(grid, event.u, p=2))
        tv_norm.append(norm(grid, event.u, p="tvd"))

        logger.info("%s || norm l2 %.5e tv %.5e", event, l2_norm[-1], tv_norm[-1])

        if interactive:
            ln0.set_ydata(event.u[s])
            plt.pause(0.01)

    # }}}

    # {{{ visualize

    if interactive:
        plt.close(fig)

    if visualize:
        l2_norm = jnp.array(l2_norm, dtype=u0.dtype)  # type: ignore
        tv_norm = jnp.array(tv_norm, dtype=u0.dtype)  # type: ignore

        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s], "o-")
        ax.plot(grid.x[s], u0[s], "k--")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlim([grid.a, grid.b])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.grid(True)

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
