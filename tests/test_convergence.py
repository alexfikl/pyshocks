# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import replace
from functools import partial
from typing import Callable, List, Optional, Tuple

from pyshocks import (
    SchemeBase,
    Grid,
    Quadrature,
    T,
    timeme,
    get_logger,
    set_recommended_matplotlib,
)
from pyshocks.reconstruction import make_reconstruction_from_name
from pyshocks.limiters import make_limiter_from_name
from pyshocks import burgers, advection, continuity, diffusion

import jax
import jax.numpy as jnp

import pytest

logger = get_logger("test_convergence")
set_recommended_matplotlib()


@timeme
def evolve(
    scheme: SchemeBase,
    solution: Callable[[Grid, float, jnp.ndarray], jnp.ndarray],
    n: int,
    *,
    timestep: float,
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 0.5,
    order: Optional[int] = None,
    finalize: Optional[Callable[[SchemeBase, Grid, Quadrature], SchemeBase]] = None,
    visualize: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if visualize:
        try:
            import matplotlib.pyplot as mp
        except ImportError:
            visualize = False

    # {{{ grid

    if order is None:
        order = scheme.order
    order = int(max(order, 1.0))

    from pyshocks import make_uniform_cell_grid, make_leggauss_quadrature, Boundary

    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)
    quad = make_leggauss_quadrature(grid, order=order + 1)

    if finalize is not None:
        scheme = finalize(scheme, grid, quad)

    if visualize:
        equation_name = scheme.__module__.split(".")[-2]
        s = grid.i_

    # }}}

    # {{{ initial/boundary conditions

    def solution_wrap(t: float, x: jnp.ndarray) -> jnp.ndarray:
        return solution(grid, t, x)

    from pyshocks import cell_average

    u0 = cell_average(quad, lambda x: solution_wrap(0.0, x))

    if isinstance(scheme, advection.Scheme):
        from pyshocks.scalar import PeriodicBoundary

        boundary: Boundary = PeriodicBoundary()
    else:
        from pyshocks.scalar import make_dirichlet_boundary

        boundary = make_dirichlet_boundary(solution_wrap)

    # }}}

    # {{{ time stepping

    from pyshocks import apply_operator

    @jax.jit
    def _apply_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, _t, _u)

    import pyshocks.timestepping as ts

    maxit, dt = ts.predict_maxit_from_timestep(tfinal, timestep)

    stepper = ts.SSPRK33(
        predict_timestep=lambda _t, _u: dt,
        source=_apply_operator,
        checkpoint=None,
    )

    u = u0
    for event in ts.step(stepper, u0, maxit=maxit):
        u = event.u

    # exact solution
    uhat = cell_average(quad, lambda x: solution_wrap(tfinal, x))

    # }}}

    # {{{ plot

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        ax.plot(grid.x[s], u0[s], "--", label="$u(0, x)$")
        ax.plot(grid.x[s], u[s], "-", label="$u(T, x)$")
        ax.plot(grid.x[s], uhat[s], "k--", label=r"$\hat{u}(T, x)$")
        ax.set_xlabel("$x$")
        ax.legend()

        filename = f"convergence_{equation_name}_{scheme.name}_{n:04}_solution"
        fig.savefig(filename)
        fig.clf()

        ax = fig.gca()
        ax.semilogy(grid.x[s], jnp.abs(u[s] - uhat[s]))
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$|u - \hat{u}|$")

        filename = f"convergence_{equation_name}_{scheme.name}_{n:04}_error"
        fig.savefig(filename)
        fig.clf()

        mp.close(fig)

    # }}}

    from pyshocks import rnorm

    h_max = jnp.max(jnp.diff(grid.f))
    error = rnorm(grid, u, uhat, p=1)

    return h_max, error


# {{{ burgers


@pytest.mark.parametrize(
    ("scheme_name", "resolutions"),
    [
        ("rusanov", list(range(64, 128 + 1, 16))),
        ("lf", list(range(64, 128 + 1, 16))),
        ("eo", list(range(32, 128 + 1, 16))),
    ],
)
def test_burgers_convergence(
    scheme_name: str,
    resolutions: List[int],
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 1.0,
) -> None:
    from pyshocks import EOCRecorder

    rec = make_reconstruction_from_name("constant")
    scheme = burgers.make_scheme_from_name(scheme_name, rec=rec, alpha=0.98)

    eoc = EOCRecorder(name=scheme.name)

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
    ("scheme_name", "rec_name", "order", "resolutions"),
    [
        ("godunov", "constant", 1, list(range(80, 160 + 1, 16))),
        ("godunov", "muscl", 2, list(range(80, 160 + 1, 16))),
        ("godunov", "wenojs32", 3, list(range(192, 384 + 1, 32))),
        ("godunov", "wenojs53", 5, list(range(32, 256 + 1, 32))),
        ("godunov", "esweno32", 3, list(range(32, 256 + 1, 32))),
        ("godunov", "ssweno242", 4, list(range(192, 384 + 1, 32))),
    ],
)
def test_advection_convergence(
    scheme_name: str,
    rec_name: str,
    order: int,
    resolutions: List[int],
    a: float = -1.0,
    b: float = +1.0,
    tfinal: float = 1.0,
) -> None:
    # NOTE: WENOJS53 convergence is very finicky with respect to what initial
    # conditions / time steps / whatever we give it. This has to do with the
    # critical points in the solutions, eps, and other things (JS is not the
    # most robust of the WENO family of schemes). The choice here seems to work!

    def solution(grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        u0 = partial(continuity.ic_sine_sine, grid)
        return continuity.ex_constant_velocity_field(t, x, a=1.0, u0=u0)

    def finalize(s: T, grid: Grid, quad: Quadrature) -> T:
        from pyshocks import cell_average

        velocity = cell_average(
            quad, lambda x: 1.0 * continuity.velocity_const(grid, 0, x)
        )

        return replace(s, velocity=velocity)

    lm = make_limiter_from_name("default", theta=1.0)
    rec = make_reconstruction_from_name(rec_name, lm=lm)
    scheme = advection.make_scheme_from_name(scheme_name, rec=rec, velocity=None)

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=scheme.name)

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


# {{{ test_diffusion_convergence


@pytest.mark.parametrize(
    ("scheme_name", "order", "resolutions"),
    [
        ("centered", 2, list(range(80, 160 + 1, 16))),
    ],
)
def test_diffusion_convergence(
    scheme_name: str,
    order: int,
    resolutions: List[int],
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 1.0,
    diffusivity: float = 1.0,
) -> None:
    def ex_sin_exp(grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return diffusion.ex_expansion(grid, t, x, diffusivity=diffusivity)

    def finalize(s: T, grid: Grid, quad: Quadrature) -> T:
        d = jnp.full_like(grid.x, diffusivity)  # type: ignore[no-untyped-call]
        return replace(s, diffusivity=d)

    rec = make_reconstruction_from_name("constant")
    scheme = diffusion.make_scheme_from_name(scheme_name, rec=rec)

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=scheme.name)

    from pyshocks.timestepping import predict_timestep_from_resolutions

    dt = 0.5 * predict_timestep_from_resolutions(a, b, resolutions, umax=1.0, p=2)

    for n in resolutions:
        h_max, error = evolve(
            scheme,
            ex_sin_exp,
            n,
            timestep=dt,
            a=a,
            b=b,
            tfinal=tfinal,
            finalize=finalize,
        )

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= scheme.order - 0.1


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
