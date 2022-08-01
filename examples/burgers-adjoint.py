# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as mp

from pyshocks import (
    UniformGrid,
    Boundary,
    ConservationLawScheme,
    timeme,
)
from pyshocks import burgers, reconstruction, limiters, get_logger
from pyshocks.checkpointing import InMemoryCheckpoint
from pyshocks.timestepping import step, adjoint_step, Stepper

logger = get_logger("burgers-adjoint")


@dataclass
class Simulation:
    scheme: ConservationLawScheme
    grid: UniformGrid
    bc: Boundary
    stepper: Stepper

    tfinal: float

    @property
    def name(self) -> str:
        n = self.grid.n
        return f"{self.scheme.name}_{type(self.stepper).__name__}_{n:05d}".lower()


@timeme
def evolve_forward(
    sim: Simulation,
    u0: jnp.ndarray,
    *,
    dirname: pathlib.Path,
    interactive: bool,
    visualize: bool,
) -> jnp.ndarray:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ plotting

    s = grid.i_

    if interactive:
        fig = mp.figure()
        ax = fig.gca()
        mp.ion()

        _, ln1 = ax.plot(grid.x[s], u0[s], "k--", grid.x[s], u0[s], "o-", ms=1)
        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1.0, jnp.max(u0) + 1.0])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

    # }}}

    # {{{ evolve

    from pyshocks import norm

    event = None
    for event in step(stepper, u0, tfinal=sim.tfinal):
        umax = norm(grid, event.u, p=jnp.inf)
        logger.info(
            "[%4d] t = %.5e / %.5e dt %.5e umax = %.5e",
            event.iteration,
            event.t,
            sim.tfinal,
            event.dt,
            umax,
        )

        if interactive:
            ln1.set_ydata(event.u[s])
            mp.pause(0.01)

    # }}}

    # {{{ plot

    if interactive:
        mp.close(fig)

    if visualize:
        umax = jnp.max(event.u[grid.i_])
        umin = jnp.min(event.u[grid.i_])
        umag = jnp.max(jnp.abs(event.u[grid.i_]))

        fig = mp.figure()
        ax = fig.gca()
        ax.plot(grid.x[grid.i_], event.u[grid.i_], label="$u(T)$")
        ax.plot(grid.x[grid.i_], u0[grid.i_], "k--", label="$u(0)$")

        ax.set_ylim([umin - 0.1 * umag, umax + 0.1 * umag])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"$T = {event.t:.3f}$")

        fig.savefig(dirname / f"burgers_forward_{sim.name}")
        mp.close(fig)

    # }}}

    return event.u, event.iteration


@timeme
def evolve_adjoint(
    sim: Simulation,
    uf: jnp.ndarray,
    p0: jnp.ndarray,
    *,
    dirname: pathlib.Path,
    maxit: int,
    interactive: bool,
    visualize: bool,
) -> None:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ setup

    from pyshocks import apply_boundary
    from pyshocks.scalar import dirichlet_boundary

    bc = dirichlet_boundary(
        lambda t, x: jnp.zeros_like(x)  # type: ignore[no-untyped-call]
    )

    @jax.jit
    def _apply_boundary(t: float, u: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        return apply_boundary(bc, grid, t, p)

    # }}}

    # {{{ evolve

    s = grid.i_

    if interactive or visualize:
        pmax = jnp.max(p0[s])
        pmin = jnp.min(p0[s])
        pmag = jnp.max(jnp.abs(p0[s]))

    if interactive:
        fig = mp.figure()
        ax = fig.gca()
        mp.ion()

        ln0, ln1 = ax.plot(grid.x[s], uf[s], "k--", grid.x[s], p0[s], "o-", ms=1)

        # NOTE: This is where the plateau should be for the initial condition
        # chosen in `main`; modify as needed
        ax.axhline(0.5, color="k", linestyle="--", lw=1)

        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
        ax.set_xlabel("$x$")

    from pyshocks import norm

    event = None
    for event in adjoint_step(stepper, p0, maxit=maxit, apply_boundary=_apply_boundary):
        pmax = norm(grid, event.p, p=jnp.inf)
        logger.info(
            "[%4d] t = %.5e / %.5e dt %.5e pmax = %.5e",
            event.iteration,
            event.t,
            event.tfinal,
            event.dt,
            pmax,
        )

        if interactive:
            ln0.set_ydata(event.u[s])
            ln1.set_ydata(event.p[s])
            mp.pause(0.01)

    # }}}

    # {{{ plot

    if interactive:
        mp.close(fig)

    if not visualize:
        return

    fig = mp.figure()
    ax = fig.gca()

    ax.plot(grid.x[s], event.p[s])
    ax.plot(grid.x[s], p0[s], "k--")
    ax.axhline(0.5, color="k", linestyle=":", lw=1)

    ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p$")

    fig.savefig(dirname / f"burgers_adjoint_{sim.name}")

    # }}}


def main(
    scheme: ConservationLawScheme,
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 257,
    tfinal: float = 1.0,
    theta: float = 1.0,
    interactive: bool = False,
    visualize: bool = True,
) -> None:
    if not outdir.exists():
        outdir.mkdir()

    # {{{ setup

    from pyshocks import make_uniform_grid

    grid = make_uniform_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)

    from pyshocks import Quadrature

    order = int(max(scheme.order, 1.0)) + 1
    quad = Quadrature(grid=grid, order=order)

    # }}}

    # {{{ initial condition

    from pyshocks import cell_average

    solution = partial(burgers.ex_shock, grid)
    u0 = cell_average(quad, lambda x: solution(0.0, x))

    from pyshocks.scalar import dirichlet_boundary

    boundary = dirichlet_boundary(solution)

    # }}}

    # {{{ time stepping

    if isinstance(scheme.rec, reconstruction.ESWENO32):
        # NOTE: prefer the parameters recommended by Carpenter!
        eps, delta = reconstruction.es_weno_from_grid(grid, u0)
        scheme = replace(scheme, rec=replace(rec, eps=eps, delta=delta))

    from pyshocks import predict_timestep, apply_operator

    def forward_predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, _t, _u)

    def forward_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, _t, _u)

    from pyshocks.timestepping import SSPRK33

    stepper = SSPRK33(
        predict_timestep=jax.jit(forward_predict_timestep),
        source=jax.jit(forward_operator),
        checkpoint=InMemoryCheckpoint(basename="Iteration"),
    )

    # }}}

    # {{{ evolve forward

    sim = Simulation(
        scheme=scheme,
        grid=grid,
        bc=boundary,
        stepper=stepper,
        tfinal=tfinal,
    )

    uf, maxit = evolve_forward(
        sim,
        u0,
        dirname=outdir,
        interactive=False,
        visualize=visualize,
    )

    # }}}

    # {{{ evolve adjoint

    evolve_adjoint(
        sim,
        uf,
        uf,
        maxit=maxit,
        dirname=outdir,
        interactive=interactive,
        visualize=visualize,
    )

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scheme",
        default="lf",
        type=str.lower,
        choices=burgers.scheme_ids(),
    )
    parser.add_argument(
        "-r",
        "--reconstruct",
        default="default",
        type=str.lower,
        choices=reconstruction.reconstruction_ids(),
    )
    parser.add_argument(
        "-l",
        "--limiter",
        default="default",
        type=str.lower,
        choices=limiters.limiter_ids(),
    )
    parser.add_argument(
        "--alpha", default=0.995, type=float, help="Lax-Friedrichs scheme parameter"
    )
    parser.add_argument("-n", "--numcells", type=int, default=256)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    lm = limiters.make_limiter_from_name(args.limiter, theta=1.0, variant=1)
    rec = reconstruction.make_reconstruction_from_name(args.reconstruct, lm=lm)
    ascheme = burgers.make_scheme_from_name(args.scheme, rec=rec, alpha=args.alpha)

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()

    main(
        ascheme,
        n=args.numcells,
        outdir=args.outdir,
        interactive=args.interactive,
    )
