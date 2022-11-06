# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from functools import partial

import jax
import jax.numpy as jnp

from pyshocks import (
    Boundary,
    make_uniform_point_grid,
    bind,
    apply_operator,
    predict_timestep,
)
from pyshocks import advection, sbp, get_logger
from pyshocks.scalar import PeriodicBoundary, make_advection_sat_boundary

logger = get_logger("advection-sbp-sat")


def sine_wave(t: float, x: jnp.ndarray, *, k: int = 1) -> jnp.ndarray:
    return jnp.sin(2.0 * jnp.pi * k * (t + 1) * x)


def main(
    sbp_op_name: str,
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 512,
    tfinal: float = 3.0,
    theta: float = 1.0,
    is_periodic: bool = True,
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

    # set up grid
    grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=0, is_periodic=is_periodic)

    # set up user data
    velocity = jnp.ones_like(grid.x)
    func = partial(sine_wave, k=1)
    u0 = func(0.0, grid.x)

    # set up boundary conditions
    if is_periodic:
        boundary: Boundary = PeriodicBoundary()
    else:
        boundary = make_advection_sat_boundary(
            ga=lambda t: func(t, grid.a), gb=lambda t: func(t, grid.b)
        )

    # set up scheme
    op = sbp.make_operator_from_name(sbp_op_name)
    scheme = advection.SBPSAT(rec=None, op=op, velocity=velocity)
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

        _, ln1 = ax.plot(grid.x[s], u0[s], "k--", grid.x[s], u0[s], "o-", ms=1)
        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1, jnp.max(u0) + 1])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

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

    event = None
    for event in timestepping.step(method, u0, tfinal=tfinal):
        if verbose:
            umax = jnp.max(jnp.abs(event.u[s]))
            usqr = jnp.sqrt(event.u[s] @ (scheme.P * event.u[s]))
            logger.info("%s umax %.5e usqr %.5e", event, umax, usqr)

        if interactive:
            ln1.set_ydata(event.u[s])
            plt.pause(0.01)

    if interactive:
        plt.ioff()
        plt.close(fig)

    # }}}

    # {{{ visualize

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s])
        ax.plot(grid.x[s], u0[s], "k--")
        ax.plot(grid.x[s], func(tfinal, grid.x), "k:")
        ax.set_xlim([grid.a, grid.b])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

        fig.savefig(outdir / f"advection_{scheme.name}_{n:05d}")
        plt.close(fig)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sbp",
        default="sbp21",
        type=str.lower,
        choices=sbp.operator_ids(),
    )
    parser.add_argument("-n", "--numcells", type=int, default=256)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()

    main(
        sbp_op_name=args.sbp,
        n=args.numcells,
        outdir=args.outdir,
        interactive=args.interactive,
    )
