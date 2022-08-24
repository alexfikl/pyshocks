# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from typing import Tuple

import jax
import jax.numpy as jnp

from pyshocks import (
    Grid,
    Boundary,
    SchemeBase,
    FiniteVolumeSchemeBase,
    VectorFunction,
    apply_operator,
    predict_timestep,
)
from pyshocks import burgers, limiters, reconstruction, sbp, get_logger

logger = get_logger("burgers")


def ic_func(grid: Grid, t: float, x: jnp.ndarray, *, variant: int = 1) -> jnp.ndarray:
    from pyshocks import funcs

    if variant == 1:
        return funcs.burgers_tophat(grid, t, x)
    elif variant == 2:
        return 0.5 - funcs.ic_sine(grid, x)
    else:
        raise ValueError(f"unknown initial condition: '{variant}'")


def make_finite_volume(
    order: float, sw: int, *, a: float, b: float, n: int, periodic: bool = True
) -> Tuple[Grid, Boundary, VectorFunction]:
    from pyshocks.scalar import PeriodicBoundary, make_dirichlet_boundary
    from pyshocks import make_uniform_cell_grid, make_leggauss_quadrature, cell_average

    order = int(max(order, 1.0)) + 1
    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=sw)
    quad = make_leggauss_quadrature(grid, order=order)

    if periodic:
        boundary: Boundary = PeriodicBoundary()
    else:
        boundary = make_dirichlet_boundary(ga=lambda t, x: ic_func(grid, t, x))

    def make_solution(t: float, x: jnp.ndarray) -> jnp.ndarray:
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
) -> Tuple[Grid, Boundary, VectorFunction]:
    from pyshocks.scalar import PeriodicBoundary, make_burgers_sat_boundary
    from pyshocks import make_uniform_point_grid

    if periodic:
        grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=3)
        boundary: Boundary = PeriodicBoundary()
    else:
        grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=0)
        boundary = make_burgers_sat_boundary(
            ga=lambda t: ic_func(grid, t, grid.a),
            gb=lambda t: ic_func(grid, t, grid.b),
        )

    def make_solution(t: float, x: jnp.ndarray) -> jnp.ndarray:
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

    if interactive or visualize:
        import matplotlib.pyplot as plt

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

        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1.5, jnp.max(u0) + 1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"t = {0.0:.3f}")

    # }}}

    # {{{ right-hand side

    def _predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, boundary, _t, _u)

    def _apply_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, _t, _u)

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

    # }}}

    # {{{ visualize

    if verbose:
        t_total, t_mean, t_std = timer.stats()
        logger.info("total %.3fs iteration %.3fs ± %.5f", t_total, t_mean, t_std)

    from pyshocks import rnorm

    uhat = solution(tfinal, grid.x)
    error = rnorm(grid, event.u, uhat, p=1)

    if verbose:
        logger.info("error %.12e", error)

    if interactive:
        plt.close(fig)

    if visualize:
        fig = plt.figure()

        t = jnp.array(times)  # type: ignore[no-untyped-call]
        energy = jnp.array(norm_energy)  # type: ignore[no-untyped-call]
        tv = jnp.array(norm_tv)  # type: ignore[no-untyped-call]

        # {{{ plot solution

        ax = fig.gca()
        ax.plot(grid.x[s], event.u[s])
        ax.plot(grid.x[s], uhat[s])
        ax.set_xlim([grid.a, grid.b])
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
