# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial

import jax
import jax.numpy as jnp

from pyshocks import make_uniform_grid, apply_operator, predict_timestep
from pyshocks import burgers, get_logger

logger = get_logger("burgers")


def main(
    scheme: burgers.Scheme,
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 256,
    tfinal: float = 1.0,
    theta: float = 1.0,
    interactive: bool = False,
    visualize: bool = True,
    verbose: bool = True,
):
    r"""
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of cells the discretize the domain.
    :arg tfinal: final time horizon :math:`[0, t_{final}]`.
    :arg theta: Courant number used in time step estimation as
        :math:`\Delta t = \theta \Delta \tilde{t}`.
    """
    # {{{ setup

    grid = make_uniform_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)
    solution = partial(burgers.ex_shock, grid)

    from pyshocks import Quadrature, cell_average

    order = int(max(scheme.order, 1.0)) + 1
    quad = Quadrature(grid=grid, order=order)

    from pyshocks.scalar import dirichlet_boundary

    u0 = cell_average(quad, lambda x: solution(0.0, x))
    boundary = dirichlet_boundary(solution)

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
        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1, jnp.max(u0) + 1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.grid(True)

    # }}}

    # {{{ right-hand side

    @jax.jit
    def _predict_timestep(_t, _u):
        return theta * predict_timestep(scheme, grid, _t, _u)

    @jax.jit
    def _apply_operator(_t, _u):
        return apply_operator(scheme, grid, boundary, _t, _u)

    # }}}

    # {{{ evolution

    from pyshocks import timestepping

    method = timestepping.SSPRK33(
        predict_timestep=_predict_timestep,
        source=_apply_operator,
    )
    step = timestepping.step(method, u0, tfinal=tfinal)

    from pyshocks import IterationTimer

    timer = IterationTimer(name="burgers")

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

    # }}}

    # {{{ visualize

    if verbose:
        t_total, t_mean, t_std = timer.stats()
        logger.info("total %.3fs iteration %.3fs ± %.5f", t_total, t_mean, t_std)

    from pyshocks import rnorm

    uhat = cell_average(quad, lambda x: solution(tfinal, x))
    error = rnorm(grid, event.u, uhat, p=1)

    if verbose:
        logger.info("error %.12e", error)

    if interactive:
        plt.close(fig)

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s])
        ax.plot(grid.x[s], uhat[s])
        ax.set_xlim([grid.a, grid.b])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.grid(True)

        scheme_name = type(scheme).__name__.lower()
        fig.savefig(outdir / f"burgers_{scheme_name}_{n:05d}")
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
        choices=["lf", "eo", "wenojs32", "wenojs53"],
    )
    parser.add_argument(
        "--alpha", default=1.0, type=float, help="Lax-Friedrichs scheme parameter"
    )
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    name_to_scheme = {
        "lf": burgers.LaxFriedrichs(alpha=args.alpha),
        "eo": burgers.EngquistOsher(),
        "wenojs32": burgers.WENOJS32(),
        "wenojs53": burgers.WENOJS53(),
    }

    main(name_to_scheme[args.scheme], outdir=args.outdir)
