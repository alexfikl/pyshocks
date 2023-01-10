# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyshocks import (
    Boundary,
    FiniteVolumeSchemeBase,
    Grid,
    SchemeBase,
    bind,
    burgers,
    get_logger,
    limiters,
    reconstruction,
    sbp,
    timeit,
)
from pyshocks.checkpointing import InMemoryCheckpoint
from pyshocks.timestepping import Stepper, adjoint_step, step

logger = get_logger("burgers-adjoint")


@dataclass
class Simulation:
    scheme: SchemeBase
    grid: Grid
    bc: Boundary
    stepper: Stepper

    u0: jnp.ndarray
    tfinal: float

    @property
    def name(self) -> str:
        n = self.grid.n
        return f"{self.scheme.name}_{type(self.stepper).__name__}_{n:05d}".lower()


def make_time_stepper(
    scheme: SchemeBase, grid: Grid, bc: Boundary, *, theta: float
) -> Stepper:
    from pyshocks import apply_operator, predict_timestep

    def forward_predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, bc, _t, _u)

    def forward_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, bc, _t, _u)

    from pyshocks.timestepping import SSPRK33

    return SSPRK33(
        predict_timestep=jax.jit(forward_predict_timestep),
        source=jax.jit(forward_operator),
        checkpoint=InMemoryCheckpoint(basename="Iteration"),
    )


def make_finite_volume_simulation(
    scheme: SchemeBase,
    *,
    a: float,
    b: float,
    n: int,
    theta: float,
    tfinal: float,
) -> Simulation:
    assert isinstance(scheme, FiniteVolumeSchemeBase)

    from pyshocks import make_leggauss_quadrature, make_uniform_cell_grid

    order = int(max(scheme.order, 1)) + 1
    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)
    quad = make_leggauss_quadrature(grid, order=order)

    from pyshocks import cell_average, funcs

    u0 = cell_average(quad, lambda x: funcs.burgers_tophat(grid, 0.0, x))

    from pyshocks.scalar import make_dirichlet_boundary

    bc = make_dirichlet_boundary(ga=lambda t, x: funcs.burgers_tophat(grid, t, x))
    stepper = make_time_stepper(scheme, grid, bc, theta=theta)

    scheme = bind(scheme, grid, bc)
    return Simulation(
        scheme=scheme, grid=grid, bc=bc, stepper=stepper, u0=u0, tfinal=tfinal
    )


def make_finite_difference_simulation(
    scheme: SchemeBase,
    *,
    a: float,
    b: float,
    n: int,
    theta: float,
    tfinal: float,
) -> Simulation:
    from pyshocks import make_uniform_point_grid
    from pyshocks.scalar import PeriodicBoundary

    grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)
    bc = PeriodicBoundary()

    from pyshocks import funcs

    u0 = funcs.burgers_tophat(grid, 0.0, grid.x)
    stepper = make_time_stepper(scheme, grid, bc, theta=theta)

    scheme = bind(scheme, grid, bc)
    return Simulation(
        scheme=scheme, grid=grid, bc=bc, stepper=stepper, u0=u0, tfinal=tfinal
    )


@timeit
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
        import matplotlib.pyplot as mp

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

    if interactive:
        mp.ioff()
        mp.close(fig)

    # }}}

    return event.u, event.iteration


@timeit
def evolve_adjoint(
    sim: Simulation,
    uf: jnp.ndarray,
    p0: jnp.ndarray,
    *,
    dirname: pathlib.Path,
    maxit: int,
    interactive: bool,
    visualize: bool,
) -> jnp.ndarray:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ setup

    from pyshocks import apply_boundary
    from pyshocks.scalar import make_neumann_boundary

    bc = make_neumann_boundary(lambda t: 0.0)

    @jax.jit
    def _apply_boundary(t: float, u: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        return apply_boundary(bc, grid, t, p)

    # }}}

    # {{{ evolve

    s = grid.i_

    if interactive or visualize:
        import matplotlib.pyplot as mp

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

    if interactive:
        mp.ioff()
        mp.close(fig)

    # }}}

    return event.p


def main(
    scheme: SchemeBase,
    *,
    outdir: pathlib.Path,
    a: float = -1.5,
    b: float = +1.5,
    n: int = 128,
    tfinal: float = 0.75,
    theta: float = 1.0,
    interactive: bool = False,
    visualize: bool = True,
) -> None:
    if visualize or interactive:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            interactive = visualize = False

    if not outdir.exists():
        outdir.mkdir()

    if isinstance(scheme, FiniteVolumeSchemeBase):
        factory = make_finite_volume_simulation
    else:
        factory = make_finite_difference_simulation

    sim = factory(scheme, a=a, b=b, n=n, theta=theta, tfinal=tfinal)

    # {{{ evolve forward <-> backward

    uf, maxit = evolve_forward(
        sim,
        sim.u0,
        dirname=outdir,
        interactive=False,
        visualize=visualize,
    )

    p0 = evolve_adjoint(
        sim,
        uf,
        uf,
        maxit=maxit,
        dirname=outdir,
        interactive=interactive,
        visualize=visualize,
    )

    # }}}

    # {{{ plot

    if not visualize:
        return

    import matplotlib.pyplot as mp

    fig = mp.figure()
    grid = sim.grid
    i = sim.grid.i_

    # {{{ forward solution

    umax = jnp.max(uf[i])
    umin = jnp.min(uf[i])
    umag = jnp.max(jnp.abs(uf[i]))

    ax = fig.gca()
    ax.plot(grid.x[i], uf[i], label="$u(T)$")
    ax.plot(grid.x[i], sim.u0[i], "k--", label="$u(0)$")

    ax.set_ylim([umin - 0.1 * umag, umax + 0.1 * umag])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.legend()

    fig.savefig(outdir / f"burgers_forward_{sim.name}")
    fig.clf()

    # }}}

    # {{{ adjoint solution

    pmax = jnp.max(p0[i])
    pmin = jnp.min(p0[i])
    pmag = jnp.max(jnp.abs(p0[i]))

    ax = fig.gca()

    ax.plot(grid.x[i], p0[i], label="$p(0)$")
    ax.plot(grid.x[i], sim.u0[i], "k--", label="$u(0)$")
    ax.axhline(0.5, color="k", linestyle=":", lw=1)

    ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
    ax.set_xlabel("$x$")
    ax.legend()

    fig.savefig(outdir / f"burgers_adjoint_{sim.name}")

    # }}}

    mp.close(fig)

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
        "-p",
        "--sbp",
        default="default",
        type=str.lower,
        choices=sbp.operator_ids(),
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

    op = sbp.make_operator_from_name(args.sbp)
    lm = limiters.make_limiter_from_name(args.limiter, theta=1.0, variant=1)
    rec = reconstruction.make_reconstruction_from_name(args.reconstruct, lm=lm)
    ascheme = burgers.make_scheme_from_name(
        args.scheme, rec=rec, sbp=op, alpha=args.alpha
    )

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()

    main(
        ascheme,
        n=args.numcells,
        outdir=args.outdir,
        interactive=args.interactive,
    )
