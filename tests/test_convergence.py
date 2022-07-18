from dataclasses import replace
from functools import partial

from pyshocks import timeme, get_logger
from pyshocks import burgers, advection, continuity

import jax
import jax.numpy as jnp

import pytest

logger = get_logger("test_convergence")


@timeme
def evolve(
    scheme,
    solution,
    n: int,
    *,
    timestep: float,
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 0.5,
    order: int = None,
    finalize=None,
    visualize: bool = True,
):
    if visualize:
        try:
            import matplotlib.pyplot as mp
        except ImportError:
            visualize = False

    # {{{ grid

    if order is None:
        order = int(max(scheme.order, 1)) + 1

    from pyshocks import UniformGrid, Boundary

    grid = UniformGrid(a=a, b=b, n=n, nghosts=order)

    from pyshocks import Quadrature

    quad = Quadrature(grid=grid, order=order)

    if finalize is not None:
        scheme = finalize(scheme, grid, quad)

    if visualize:
        equation_name = scheme.__module__.split(".")[-2]
        scheme_name = type(scheme).__name__.lower()
        s = grid.i_

    # }}}

    # {{{ initial/boundary conditions

    from pyshocks import cell_average

    solution = partial(solution, grid)
    u0 = cell_average(quad, lambda x: solution(0.0, x))

    if isinstance(scheme, advection.Scheme):
        from pyshocks.scalar import PeriodicBoundary

        boundary: Boundary = PeriodicBoundary()
    else:
        from pyshocks.scalar import dirichlet_boundary

        boundary = dirichlet_boundary(solution)

    # }}}

    # {{{ time stepping

    from pyshocks import apply_operator

    @jax.jit
    def _apply_operator(_t, _u):
        return apply_operator(scheme, grid, boundary, _t, _u)

    import pyshocks.timestepping as ts

    maxit, dt = ts.predict_maxit_from_timestep(tfinal, timestep)

    stepper = ts.SSPRK33(
        predict_timestep=lambda _t, _u: dt,
        source=_apply_operator,
    )

    t = 0.0
    for event in ts.step(stepper, u0, maxit=maxit):
        u = event.u
        t = event.t
        # logger.info("[%05d] t = %.5e / %.5e dt = %.5e",
        #         event.iteration, event.t, tfinal, event.dt)

    # exact solution
    uhat = cell_average(quad, lambda x: solution(t, x))

    # }}}

    # {{{ plot

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        ax.grid()
        ax.plot(grid.x[s], jnp.log10(jnp.abs(u[s] - uhat[s])))
        ax.plot(grid.x[s], u0[s], "--")
        ax.plot(grid.x[s], u[s], "--")
        ax.plot(grid.x[s], uhat[s], "k--")

        filename = f"convergence_{equation_name}_{scheme_name}_{n:04}"
        fig.savefig(filename)
        mp.close(fig)

    # }}}

    from pyshocks import rnorm

    h_max = jnp.max(jnp.diff(grid.f))
    error = rnorm(grid, u, uhat, p=1)

    return h_max, error


# {{{ burgers


@pytest.mark.parametrize(
    ("scheme", "resolutions"),
    [
        (burgers.LaxFriedrichs(alpha=0.98), list(range(64, 128 + 1, 16))),
        (burgers.EngquistOsher(), list(range(32, 128 + 1, 16))),
    ],
)
def test_burgers_convergence(scheme, resolutions, a=-1.0, b=1.0, tfinal=1.0):
    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=type(scheme).__name__)

    from pyshocks.timestepping import predict_timestep_from_resolutions

    dt = predict_timestep_from_resolutions(a, b, resolutions, umax=10.0)

    for n in resolutions:
        h_max, error = evolve(
            scheme, burgers.ex_shock, n, timestep=dt, a=a, b=b, tfinal=tfinal
        )

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= scheme.order - 0.1


# }}}


# {{{ advection


@pytest.mark.parametrize(
    ("scheme", "order", "resolutions"),
    [
        (advection.Godunov(velocity=None), 1, list(range(80, 160 + 1, 16))),
        (advection.WENOJS32(velocity=None), 3, list(range(192, 384 + 1, 32))),
        (advection.WENOJS53(velocity=None), 5, list(range(32, 256 + 1, 32))),
    ],
)
def test_advection_convergence(scheme, order, resolutions, a=-1.0, b=+1, tfinal=1.0):
    # NOTE: WENOJS53 convergence is very finicky with respect to what initial
    # conditions / time steps / whatever we give it. This has to do with the
    # critical points in the solutions, eps, and other things (JS is not the
    # most robust of the WENO family of schemes). The choice here seems to work!

    def solution(grid, t, x):
        u0 = partial(continuity.ic_sine_sine, grid)
        return continuity.ex_constant_velocity_field(t, x, a=1.0, u0=u0)

    def finalize(s, grid, quad):
        from pyshocks import cell_average

        velocity = cell_average(
            quad, lambda x: 1.0 * continuity.velocity_const(grid, 0, x)
        )

        return replace(s, velocity=velocity)

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=type(scheme).__name__)

    for n in resolutions:
        # NOTE: SSPRK33 is order dt^3, so this makes it dt^3 ~ dx^5
        dt = 8.0 * ((b - a) / n) ** (5.0 / 3.0)

        h_max, error = evolve(
            scheme,
            solution,
            n,
            timestep=dt,
            order=order,
            a=a,
            b=b,
            tfinal=tfinal,
            finalize=finalize,
        )

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= order - 0.5


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
