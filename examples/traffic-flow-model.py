# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyshocks import Boundary, ConservationLawScheme, Grid, flux, numerical_flux
from pyshocks.tools import Array, Scalar, ScalarLike


@dataclass(frozen=True)
class TrafficFlowScheme(ConservationLawScheme):
    pass


@flux.register(TrafficFlowScheme)
def _flux_traffic_flow(
    scheme: TrafficFlowScheme,
    t: ScalarLike,
    x: Array,
    u: Array,
) -> Array:
    return (1 - u) * u


@numerical_flux.register(TrafficFlowScheme)
def _numerical_flux_traffic_flow(
    scheme: TrafficFlowScheme,
    grid: Grid,
    bc: Boundary,
    t: ScalarLike,
    u: Array,
) -> Array:
    u_avg = (u[1:] + u[:-1]) / 2
    f = (1 - u_avg) * u[:-1]

    return jnp.pad(f, 1)


def main(
    a: float = -1.0,
    b: float = 1.0,
    n: int = 256,
) -> None:
    # {{{ setup

    from pyshocks.reconstruction import make_reconstruction_from_name

    rec = make_reconstruction_from_name("constant")
    scheme = TrafficFlowScheme(rec=rec)

    from pyshocks import make_uniform_cell_grid

    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)

    from pyshocks.scalar import make_dirichlet_boundary

    boundary = make_dirichlet_boundary(
        ga=lambda t, x: jnp.ones_like(x),
        gb=lambda t, x: jnp.zeros_like(x),
    )

    mid = (grid.a + grid.b) / 2.0
    u0 = (grid.x < mid).astype(grid.x.dtype)

    # }}}

    # {{{ time stepping

    from pyshocks import apply_operator

    def predict_timestep(t: ScalarLike, u: Array) -> Scalar:
        return jnp.array(1.0e-2)

    def right_hand_side(t: ScalarLike, u: Array) -> Array:
        return apply_operator(scheme, grid, boundary, t, u)

    from pyshocks import timestepping

    integrator = timestepping.ForwardEuler(
        predict_timestep=jax.jit(predict_timestep),
        source=jax.jit(right_hand_side),
        checkpoint=None,
    )

    for event in timestepping.step(integrator, u0, tfinal=1.0):
        umax = jnp.linalg.norm(event.u[grid.i_])
        print(f"[{event.iteration:04d}] t = {event.t:.5e} umax {umax:.5e}")

    # }}}


if __name__ == "__main__":
    main()
