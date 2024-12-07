# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp

from pyshocks import (
    Boundary,
    Grid,
    VectorFunction,
    advection,
    apply_operator,
    get_logger,
    limiters,
    make_uniform_cell_grid,
    predict_timestep,
    reconstruction,
)
from pyshocks.tools import Array, ScalarLike

logger = get_logger("advection")


def make_solution(
    name: str, grid: Grid, a: float = 1.0, offset: float = 1.0
) -> tuple[VectorFunction, VectorFunction]:
    from pyshocks import funcs

    if name == "const":

        def ic_sine(x: Array) -> Array:
            return offset + funcs.ic_sine(grid, x, k=1)

        def velocity(t: ScalarLike, x: Array) -> Array:
            return funcs.ic_constant(grid, x, c=a)

        def solution(t: ScalarLike, x: Array) -> Array:
            return ic_sine(x - a * t)

    elif name == "sign":

        def velocity(t: ScalarLike, x: Array) -> Array:
            return a * funcs.ic_sign(grid, x)

        def solution(t: ScalarLike, x: Array) -> Array:
            # FIXME: this has an known solution for all time!
            return offset + funcs.ic_sine(grid, x, k=1)

    elif name == "double_sign":

        def velocity(t: ScalarLike, x: Array) -> Array:
            return a * funcs.ic_sign(grid, x)

        def solution(t: ScalarLike, x: Array) -> Array:
            return a * funcs.ic_sign(grid, x)

    else:
        raise ValueError(f"Unknown example: {name!r}.")

    return jax.jit(velocity), jax.jit(solution)


def make_boundary_conditions(
    name: str, solution: VectorFunction, *, a: float = 1.0
) -> Boundary:
    from pyshocks.scalar import PeriodicBoundary, make_dirichlet_boundary

    bc: Boundary
    if name == "const":
        bc = PeriodicBoundary()
    elif name in {"sign", "double_sign"}:
        bc = make_dirichlet_boundary(ga=jax.jit(solution))
    else:
        raise ValueError(f"Unknown example: {name!r}.")

    return bc


def main(
    scheme: advection.Scheme,
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 512,
    tfinal: float = 0.5,
    theta: float = 1.0,
    example_name: str = "sign",
    interactive: bool = False,
    visualize: bool = False,
    verbose: bool = True,
) -> tuple[Array, Array, Array]:
    r"""
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of cells the discretize the domain.
    :arg tfinal: final time horizon :math:`[0, t_{final}]`.
    :arg theta: Courant number used in time step estimation as
        :math:`\Delta t = \theta \Delta \tilde{t}`.
    """
    if visualize or interactive:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            interactive = visualize = False

    # {{{ setup

    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)

    from dataclasses import replace

    velocity, solution = make_solution(example_name, grid, a=0.5, offset=1.0)
    boundary = make_boundary_conditions(example_name, solution, a=0.5)

    # initial condition
    from pyshocks import cell_average, make_leggauss_quadrature

    order = int(max(scheme.order, 1.0)) + 1
    quad = make_leggauss_quadrature(grid, order=order)
    u0 = cell_average(quad, lambda x: solution(0.0, x))

    # update coefficients (i.e. velocity field)
    v = cell_average(quad, lambda x: velocity(0.0, x))
    scheme = replace(scheme, velocity=v)

    # }}}

    # {{{ plotting

    s = grid.i_
    if interactive:
        fig = plt.figure()
        ax = fig.gca()
        plt.ion()

        ln0, ln1 = ax.plot(grid.x[s], u0[s], "k--", grid.x[s], u0[s], "o-", ms=1)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_ylim((float(jnp.min(u0) - 1), float(jnp.max(u0) + 1)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

    # }}}

    # {{{ right-hand side

    def _predict_timestep(t_: ScalarLike, u_: Array) -> Array:
        return theta * predict_timestep(scheme, grid, boundary, t_, u_)

    def _apply_operator(t_: ScalarLike, u_: Array) -> Array:
        return apply_operator(scheme, grid, boundary, t_, u_)

    # }}}

    # {{{ evolution

    from pyshocks import timestepping

    method = timestepping.SSPRK33(
        predict_timestep=jax.jit(_predict_timestep),
        source=jax.jit(_apply_operator),
        checkpoint=None,
    )
    step = timestepping.step(method, u0, tfinal=tfinal)

    from pyshocks import IterationTimer

    timer = IterationTimer(name="advection")

    try:
        while True:
            with timer.tick():
                event = next(step)

            if verbose:
                umax = jnp.max(jnp.abs(event.u[s]))
                logger.info("%s umax %.5e", event, umax)

            if interactive:
                uhat = cell_average(quad, lambda x, event=event: solution(event.t, x))
                ln0.set_ydata(uhat[s])
                ln1.set_ydata(event.u[s])
                plt.pause(0.01)
    except StopIteration:
        pass

    if interactive:
        plt.ioff()
        plt.close(fig)

    # }}}

    # {{{ visualize

    if verbose:
        logger.info("%s", timer.stats())

    from pyshocks import rnorm

    uhat = cell_average(quad, lambda x: solution(tfinal, x))
    error = rnorm(grid, event.u, uhat, p=1)

    if verbose:
        logger.info("error: %.12e", error)

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s])
        ax.plot(grid.x[s], uhat[s])
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

        fig.savefig(outdir / f"advection_{scheme.name}_{example_name}_{n:05d}")
        plt.close(fig)

    # }}}

    return grid.x[s], event.u[s], uhat[s]


def convergence(
    scheme: advection.Scheme,
    *,
    outdir: pathlib.Path,
    example_name: str = "sign",
    visualize: bool = True,
) -> None:
    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=type(scheme).__name__)

    # NOTE:
    # If ncells is even, the discontinuity in the velocity field will
    # always be at a cell face and the convergence will look very nice.
    #
    # However, if it's odd, the exact position of the discontinuity will
    # jump around in a cell and the convergence will be a lot more chaotic
    #
    # All of this seems to be solved if we compute the cell averages to higher
    # order, likely because it actually averages something.

    ncells = 3 + jnp.array([64, 128, 256, 512, 1024, 2048])
    for n in ncells:
        x, u, uhat = main(
            scheme,
            outdir=outdir,
            n=n,
            example_name=example_name,
            interactive=False,
            visualize=False,
            verbose=False,
        )

        imid = len(u) // 2
        umid = u[imid]
        h_max = jnp.max(jnp.diff(x))

        error = abs(umid - 1.0)
        eoc.add_data_point(h_max, error)

        logger.info("plateau: n %5d u %.5e error %.5e", n, umid, error)

        if visualize:
            ax.plot(x, u)
            if n == ncells[0]:
                ax.plot(x, uhat, "ko--")

    logger.info("\n%s", eoc)

    if visualize:
        ax.axhline(1.0, color="k", ls=":", lw=2)
        ax.set_xlim((-1.0, 1.0))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        fig.savefig(outdir / f"advection_{scheme.name}_convergence")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scheme",
        default="godunov",
        type=str.lower,
        choices=advection.scheme_ids(),
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
        "--example",
        default="sign",
        type=str.lower,
        choices=["const", "sign", "double_sign"],
    )
    parser.add_argument("-n", "--numcells", type=int, default=256)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    lm = limiters.make_limiter_from_name(args.limiter, theta=1.0)
    rec = reconstruction.make_reconstruction_from_name(args.reconstruct, lm=lm)
    ascheme = advection.make_scheme_from_name(args.scheme, rec=rec, velocity=None)

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()
    if args.convergence:
        convergence(ascheme, outdir=args.outdir, example_name=args.example)
    else:
        main(
            ascheme,
            n=args.numcells,
            outdir=args.outdir,
            example_name=args.example,
            interactive=args.interactive,
        )
