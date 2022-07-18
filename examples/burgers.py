# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import os
from functools import partial

import jax
import jax.numpy as jnp

from pyshocks import UniformGrid, apply_operator, predict_timestep
from pyshocks import burgers


def main(
    scheme,
    a=-1.0,
    b=+1.0,
    n=256,
    tfinal=1.0,
    theta=1.0,
    interactive=False,
    visualize=True,
    verbose=True,
):
    # {{{ setup

    order = int(max(scheme.order, 1.0)) + 1
    grid = UniformGrid(a=a, b=b, n=n, nghosts=order)

    solution = partial(burgers.ex_shock, grid)

    from pyshocks import Quadrature, cell_average

    quad = Quadrature(grid=grid, order=order)
    u = cell_average(quad, lambda x: solution(0.0, x))

    from pyshocks.scalar import dirichlet_boundary

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

        ln0, ln1 = ax.plot(grid.x[s], u[s], "k--", grid.x[s], u[s], "o-", ms=1)
        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u) - 1, jnp.max(u) + 1])
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
    step = timestepping.step(method, u, tfinal=tfinal)

    from pyshocks import IterationTimer

    timer = IterationTimer(name="burgers")

    try:
        while True:
            with timer.tick():
                event = next(step)

            if verbose:
                umax = jnp.max(jnp.abs(event.u[s]))
                print(f"{event} umax {umax:.5e}")

            if interactive:
                uhat = cell_average(quad, lambda x, event=event: solution(event.t, x))
                ln0.set_ydata(uhat[s])
                ln1.set_ydata(event.u[s])
                plt.pause(0.01)
    except StopIteration:
        pass

    # }}}

    # {{{ visualize

    t_total, t_mean, t_std = timer.stats()
    print(f"total {t_total:.3f}s iteration {t_mean:.3f}s ± {t_std}")

    from pyshocks import rnorm

    uhat = cell_average(quad, lambda x: solution(tfinal, x))
    error = rnorm(grid, event.u, uhat, p=1)

    print(f"error {error}")

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
        filename = os.path.join(
            os.path.dirname(__file__), f"burgers_{scheme_name}_{n:05d}"
        )
        fig.savefig(filename)

    # }}}


if __name__ == "__main__":
    main(
        scheme=burgers.LaxFriedrichs(alpha=1),
        # scheme=burgers.EngquistOsher(),
        # scheme=burgers.WENOJS32(),
        # scheme=burgers.WENOJS53(),
    )
