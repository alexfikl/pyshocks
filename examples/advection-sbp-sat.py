# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib

import jax
import jax.numpy as jnp

from pyshocks import (
    make_uniform_point_grid,
    apply_operator,
    predict_timestep,
)
from pyshocks import advection, get_logger

logger = get_logger("advection-sbp")


def main(
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 256,
    tfinal: float = 1.0,
    theta: float = 1.0,
    example_name: str = "sign",
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

    from pyshocks.scalar import make_sat_boundary

    grid = make_uniform_point_grid(a=a, b=b, n=n, nghosts=0)

    velocity = jnp.ones_like(grid.x)  # type: ignore
    scheme = advection.SBPSAT21(rec=None, velocity=velocity)

    boundary = make_sat_boundary(
        ga=lambda t: jnp.sin(2.0 * jnp.pi * (1 - t)), gb=lambda t: 0.0
    )

    u0 = jnp.zeros_like(grid.x)  # type: ignore
    u0 = jnp.sin(2.0 * jnp.pi * grid.x)

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
        ax.grid(True)

    # }}}

    # {{{ right-hand side

    def _predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, _t, _u)

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
            logger.info("%s umax %.5e", event, umax)

        if interactive:
            ln1.set_ydata(event.u[s])
            plt.pause(0.01)

    if interactive:
        plt.close(fig)

    # }}}

    # {{{ visualize

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], event.u[s])
        ax.set_xlim([grid.a, grid.b])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.grid(True)

        fig.savefig(outdir / f"advection_{scheme.name}_{example_name}_{n:05d}")
        plt.close(fig)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numcells", type=int, default=256)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    args = parser.parse_args()

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()

    main(
        n=args.numcells,
        outdir=args.outdir,
        interactive=args.interactive,
    )
