from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pyshocks import ConservationLawScheme, Grid, flux, numerical_flux


@dataclass(frozen=True)
class TrafficFlowScheme(ConservationLawScheme):
    pass


@flux.register(TrafficFlowScheme)
def _flux_traffic_flow(
    scheme: TrafficFlowScheme,
    t: float,
    x: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    return (1 - u) * u


@numerical_flux.register(TrafficFlowScheme)
def _numerical_flux_traffic_flow(
    scheme: TrafficFlowScheme,
    grid: Grid,
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    u_avg = (u[1:] + u[:-1]) / 2
    f = (1 - u_avg) * u[:-1]

    return jnp.pad(f, 1)  # type: ignore[no-untyped-call]


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
        ga=lambda t, x: jnp.ones_like(x),  # type: ignore[no-untyped-call]
        gb=lambda t, x: jnp.zeros_like(x),  # type: ignore[no-untyped-call]
    )

    mid = (grid.a + grid.b) / 2.0
    u0 = (grid.x < mid).astype(grid.x.dtype)

    # }}}

    # {{{ time stepping

    from pyshocks import apply_operator

    def right_hand_side(t: float, u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, t, u)

    from pyshocks import timestepping

    integrator = timestepping.ForwardEuler(
        predict_timestep=lambda t, u: 1.0e-2,
        source=jax.jit(right_hand_side),
        checkpoint=None,
    )

    for event in timestepping.step(integrator, u0, tfinal=1.0):
        umax = jnp.linalg.norm(event.u[grid.i_])
        print(f"[{event.iteration:04d}] t = {event.t:.5e} umax {umax:.5e}")

    # }}}


if __name__ == "__main__":
    main()
