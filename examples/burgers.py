# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp

from pyshocks import (
    Boundary,
    FiniteVolumeSchemeBase,
    Grid,
    SchemeBase,
    VectorFunction,
    apply_operator,
    burgers,
    get_logger,
    limiters,
    predict_timestep,
    reconstruction,
    sbp,
)
from pyshocks.tools import Array, ScalarLike

logger = get_logger("burgers")


def ic_func(grid: Grid, t: ScalarLike, x: Array, *, variant: int = 1) -> Array:
    from pyshocks import funcs

    if variant == 1:
        return funcs.burgers_tophat(grid, t, x)

    if variant == 2:
        return 0.5 - funcs.ic_sine(grid, x)

    raise ValueError(f"Unknown initial condition: {variant!r}.")


def make_finite_volume(
    order: float, sw: int, *, a: float, b: float, n: int, periodic: bool = True
) -> tuple[Grid, Boundary, VectorFunction]:
    from pyshocks import cell_average, make_leggauss_quadrature, make_uniform_cell_grid
    from pyshocks.scalar import PeriodicBoundary, make_dirichlet_boundary

    order = int(max(order, 1.0)) + 1
    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=sw)
    quad = make_leggauss_quadrature(grid, order=order)

    if periodic:
        boundary: Boundary = PeriodicBoundary()
    else:
        boundary = make_dirichlet_boundary(ga=lambda t, x: ic_func(grid, t, x))

    def make_solution(t: ScalarLike, x: Array) -> Array:
        return cell_average(quad, lambda x: ic_func(grid, t, x))

    return grid, boundary, make_solution


def make_finite_difference(
    order: float,
    sw: int,
    *,
    a: float,
    b: float,
    n: int,
    periodic: bool = True,
) -> tuple[Grid, Boundary, VectorFunction]:
    from pyshocks import make_uniform_point_grid
    from pyshocks.scalar import PeriodicBoundary, make_burgers_sat_boundary

    if periodic:
        grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=3)
        boundary: Boundary = PeriodicBoundary()
    else:
        grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=0)
        boundary = make_burgers_sat_boundary(
            ga=lambda t: ic_func(grid, t, jnp.array(grid.a)),
            gb=lambda t: ic_func(grid, t, jnp.array(grid.b)),
        )

    def make_solution(t: ScalarLike, x: Array) -> Array:
        return ic_func(grid, t, x)

    return grid, boundary, make_solution


def main(
    scheme: SchemeBase,
    *,
    outdir: pathlib.Path,
    a: float = -1.5,
    b: float = 1.5,
    n: int = 512,
    tfinal: float = 0.5,
    theta: float = 1.0,
    interactive: bool = False,
    visualize: bool = True,
    verbose: bool = True,
) -> None:
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

    if isinstance(scheme, FiniteVolumeSchemeBase):
        grid, boundary, solution = make_finite_volume(
            scheme.order, scheme.stencil_width, a=a, b=b, n=n
        )
    else:
        grid, boundary, solution = make_finite_difference(
            scheme.order, scheme.stencil_width, a=a, b=b, n=n
        )

    from pyshocks import bind

    u0 = solution(0.0, grid.x)
    scheme = bind(scheme, grid, boundary)

    # }}}

    # {{{ plotting

    s = grid.i_
    if interactive:
        fig = plt.figure()
        ax = fig.gca()
        plt.ion()

        ln0, ln1 = ax.plot(grid.x[s], u0[s], "k--", grid.x[s], u0[s], "o-", ms=1)
        if isinstance(scheme, burgers.SSMUSCL):
            from pyshocks.burgers.schemes import hesthaven_limiter

            phi = hesthaven_limiter(u0, variant=scheme.variant)
            (ln2,) = ax.plot(grid.f[1:-1], phi, "k", lw=1)

        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_ylim((float(jnp.min(u0) - 1.5), float(jnp.max(u0) + 1)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"t = {0.0:.3f}")

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
        # predict_timestep=_predict_timestep,
        # source=_apply_operator,
        checkpoint=None,
    )
    step = timestepping.step(method, u0, tfinal=tfinal)

    from pyshocks import IterationTimer, norm

    timer = IterationTimer(name="burgers")

    times = []
    norm_energy = []
    norm_tv = []

    try:
        while True:
            with timer.tick():
                event = next(step)

            times.append(event.t)
            norm_energy.append(norm(grid, event.u, p=2, weighted=True))
            norm_tv.append(norm(grid, event.u, p="tvd"))

            if verbose:
                logger.info(
                    "%s energy %.5e tv(u) %.5e", event, norm_energy[-1], norm_tv[-1]
                )

            if interactive:
                if isinstance(scheme, burgers.SSMUSCL):
                    phi = hesthaven_limiter(event.u, variant=scheme.variant)
                    ln2.set_ydata(phi)

                uhat = solution(event.t, grid.x)
                ln0.set_ydata(uhat[s])
                ln1.set_ydata(event.u[s])
                ax.set_title(f"t = {event.t:.3f}")
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

    uhat = solution(tfinal, grid.x)
    error = rnorm(grid, event.u, uhat, p=1)

    if verbose:
        logger.info("error %.12e", error)

    if visualize:
        fig = plt.figure()

        t = jnp.array(times)
        energy = jnp.array(norm_energy)
        tv = jnp.array(norm_tv)

        # {{{ plot solution

        ax = fig.gca()
        ax.plot(grid.x[s], event.u[s])
        ax.plot(grid.x[s], uhat[s])
        ax.set_xlim((float(grid.a), float(grid.b)))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

        fig.savefig(outdir / f"burgers_{scheme.name}_{n:05d}")
        fig.clf()

        # {{{ plot total variation differences

        ax = fig.gca()
        ax.semilogy(t[1:], jnp.abs(jnp.diff(tv)))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$TV(u)$")
        fig.savefig(outdir / f"burgers_{scheme.name}_{n:05d}_norm_tv")
        fig.clf()

        # }}}

        # {{{ plot energy

        ax = fig.gca()
        ax.plot(t, energy)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\|u\|_{2, h}$")
        fig.savefig(outdir / f"burgers_{scheme.name}_{n:05d}_norm_energy")
        fig.clf()

        # }}}

        plt.close(fig)

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
        "--alpha", default=1.0, type=float, help="Lax-Friedrichs scheme parameter"
    )
    parser.add_argument("-n", "--numcells", type=int, default=256)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    op = sbp.make_operator_from_name(args.sbp)
    lm = limiters.make_limiter_from_name(args.limiter, variant=1, theta=1.0)
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
