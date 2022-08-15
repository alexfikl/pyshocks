# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

from pyshocks import (
    SchemeBase,
    Grid,
    Boundary,
    SpatialFunction,
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


# {{{ evolve


@dataclass(frozen=True)
class ConvergenceTestCase:
    scheme_name: str

    def make_grid(self, a: float, b: float, n: int) -> Grid:
        raise NotImplementedError

    def make_boundary(self, grid: Grid) -> Boundary:
        raise NotImplementedError

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        raise NotImplementedError

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class FiniteVolumeTestCase(ConvergenceTestCase):  # pylint: disable=abstract-method
    def make_grid(self, a: float, b: float, n: int) -> Grid:
        from pyshocks import make_uniform_cell_grid

        # NOTE: number of ghosts cells should depend on the scheme?
        return make_uniform_cell_grid(a=a, b=b, n=n, nghosts=3)

    def cell_average(self, grid: Grid, func: SpatialFunction) -> jnp.ndarray:
        from pyshocks import make_leggauss_quadrature, cell_average

        quad = make_leggauss_quadrature(grid, order=5)
        return cell_average(quad, func)


@dataclass(frozen=True)
class FiniteDifferenceTestCase(ConvergenceTestCase):  # pylint: disable=abstract-method
    def make_grid(self, a: float, b: float, n: int) -> Grid:
        from pyshocks import make_uniform_point_grid

        return make_uniform_point_grid(a=a, b=b, n=n, nghosts=0)


@timeme
def evolve(
    case: ConvergenceTestCase,
    n: int,
    *,
    order: int,
    dt: float,
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 0.5,
    visualize: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if visualize:
        try:
            import matplotlib.pyplot as mp
        except ImportError:
            visualize = False

    # {{{ setup

    from pyshocks import bind

    grid = case.make_grid(a, b, n)
    bc = case.make_boundary(grid)

    scheme = case.make_scheme(grid, bc)
    scheme = bind(scheme, grid, bc)

    u0 = case.evaluate(grid, 0.0, grid.x)

    # }}}

    # {{{ time stepping

    if visualize:
        equation_name = scheme.__module__.split(".")[-2]
        s = grid.i_

    from pyshocks import apply_operator

    @jax.jit
    def _apply_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, bc, _t, _u)

    import pyshocks.timestepping as ts

    maxit, dt = ts.predict_maxit_from_timestep(tfinal, dt)

    stepper = ts.SSPRK33(
        predict_timestep=lambda _t, _u: dt,
        source=_apply_operator,
        checkpoint=None,
    )

    u = u0
    for event in ts.step(stepper, u0, maxit=maxit):
        u = event.u

    # exact solution
    uhat = case.evaluate(grid, tfinal, grid.x)

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


# }}}


# {{{ burgers


@dataclass(frozen=True)
class BurgersTestCase(FiniteVolumeTestCase):
    def make_boundary(self, grid: Grid) -> Boundary:
        from pyshocks.scalar import make_dirichlet_boundary

        return make_dirichlet_boundary(
            ga=lambda t, x: burgers.ex_shock(grid, t, x),
            gb=lambda t, x: burgers.ex_shock(grid, t, x),
        )

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        rec = make_reconstruction_from_name("constant")
        return burgers.make_scheme_from_name(self.scheme_name, rec=rec, alpha=0.98)

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.cell_average(grid, partial(burgers.ex_shock, grid, t))


@pytest.mark.parametrize(
    ("case", "order", "resolutions"),
    [
        (BurgersTestCase("rusanov"), 1, list(range(64, 128 + 1, 16))),
        (BurgersTestCase("lf"), 1, list(range(64, 128 + 1, 16))),
        (BurgersTestCase("eo"), 1, list(range(32, 128 + 1, 16))),
    ],
)
def test_burgers_convergence(
    case: BurgersTestCase,
    order: int,
    resolutions: List[int],
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 1.0,
) -> None:
    from pyshocks import EOCRecorder
    from pyshocks.timestepping import predict_timestep_from_resolutions

    eoc = EOCRecorder(name=case.scheme_name)
    dt = predict_timestep_from_resolutions(a, b, resolutions, umax=10.0)

    for n in resolutions:
        h_max, error = evolve(case, n, order=order, dt=dt, a=a, b=b, tfinal=tfinal)

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= order - 0.1


# }}}


# {{{ advection


@dataclass(frozen=True)
class AdvectionTestCase(FiniteVolumeTestCase):
    rec_name: str
    a: float = 1.0

    def make_boundary(self, grid: Grid) -> Boundary:
        from pyshocks.scalar import PeriodicBoundary

        return PeriodicBoundary()

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        velocity = jnp.full_like(grid.x, self.a)  # type: ignore[no-untyped-call]

        lm = make_limiter_from_name("default", theta=1.0)
        rec = make_reconstruction_from_name(self.rec_name, lm=lm)
        return advection.make_scheme_from_name(
            self.scheme_name, rec=rec, velocity=velocity
        )

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        # NOTE: WENOJS53 convergence is very finicky with respect to what initial
        # conditions / time steps / whatever we give it. This has to do with the
        # critical points in the solutions, eps, and other things (JS is not the
        # most robust of the WENO family of schemes). The choice here seems to work!

        def func(x: jnp.ndarray) -> jnp.ndarray:
            u0 = partial(continuity.ic_sine_sine, grid)
            return continuity.ex_constant_velocity_field(t, x, a=self.a, u0=u0)

        return self.cell_average(grid, func)


@dataclass(frozen=True)
class SATAdvectionTestCase(FiniteDifferenceTestCase):
    sbp_name: str
    a: float = 1.0

    def make_boundary(self, grid: Grid) -> Boundary:
        from pyshocks.scalar import make_sat_boundary

        return make_sat_boundary(
            ga=partial(self.evaluate, grid, x=grid.a),
            gb=partial(self.evaluate, grid, x=grid.b),
        )

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        from pyshocks.sbp import make_operator_from_name

        velocity = jnp.full_like(grid.x, self.a)  # type: ignore[no-untyped-call]
        op = make_operator_from_name(self.sbp_name)

        return advection.make_scheme_from_name(
            self.scheme_name, rec=None, op=op, velocity=velocity
        )

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        u0 = partial(continuity.ic_sine_sine, grid)
        return continuity.ex_constant_velocity_field(t, x, a=self.a, u0=u0)


@pytest.mark.parametrize(
    ("case", "order", "resolutions"),
    [
        (AdvectionTestCase("godunov", "constant"), 1, list(range(80, 160 + 1, 16))),
        (AdvectionTestCase("godunov", "muscl"), 2, list(range(80, 160 + 1, 16))),
        (AdvectionTestCase("godunov", "wenojs32"), 3, list(range(192, 384 + 1, 32))),
        (AdvectionTestCase("godunov", "wenojs53"), 5, list(range(32, 256 + 1, 32))),
        (AdvectionTestCase("godunov", "esweno32"), 3, list(range(32, 256 + 1, 32))),
        (AdvectionTestCase("godunov", "ssweno242"), 4, list(range(192, 384 + 1, 32))),
        (SATAdvectionTestCase("sbp", "sbp21"), 2, list(range(80, 160 + 1, 16))),
        # FIXME: why is this 3rd order?
        (SATAdvectionTestCase("sbp", "sbp42"), 3, list(range(192, 384 + 1, 32))),
    ],
)
def test_advection_convergence(
    case: AdvectionTestCase,
    order: int,
    resolutions: List[int],
    a: float = -1.0,
    b: float = +1.0,
    tfinal: float = 1.0,
) -> None:

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=f"{case.scheme_name}_{case.rec_name}")

    for n in resolutions:
        # NOTE: SSPRK33 is order dt^3, so this makes it dt^3 ~ dx^5
        dt = 8.0 * ((b - a) / n) ** (5.0 / 3.0)

        h_max, error = evolve(case, n, order=order, dt=dt, a=a, b=b, tfinal=tfinal)

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= order - 0.5


# }}}


# {{{ test_diffusion_convergence


@dataclass(frozen=True)
class DiffusionTestCase(FiniteVolumeTestCase):
    d: float = 1.0

    def make_boundary(self, grid: Grid) -> Boundary:
        from pyshocks.scalar import make_dirichlet_boundary

        return make_dirichlet_boundary(
            ga=lambda t, x: diffusion.ex_expansion(grid, t, x, diffusivity=self.d),
            gb=lambda t, x: diffusion.ex_expansion(grid, t, x, diffusivity=self.d),
        )

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        diffusivity = jnp.full_like(grid.x, self.d)  # type: ignore[no-untyped-call]

        rec = make_reconstruction_from_name("constant")
        return diffusion.make_scheme_from_name(
            self.scheme_name, rec=rec, diffusivity=diffusivity
        )

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.cell_average(
            grid, partial(diffusion.ex_expansion, grid, t, diffusivity=self.d)
        )


@dataclass(frozen=True)
class SATDiffusionTestCase(FiniteDifferenceTestCase):
    sbp_name: str
    d: float = 1.0

    def make_boundary(self, grid: Grid) -> Boundary:
        from pyshocks.scalar import make_sat_boundary

        return make_sat_boundary(
            ga=partial(self.evaluate, grid, x=grid.a),
            gb=partial(self.evaluate, grid, x=grid.b),
        )

    def make_scheme(self, grid: Grid, bc: Boundary) -> SchemeBase:
        from pyshocks.sbp import make_operator_from_name

        diffusivity = jnp.full_like(grid.x, self.d)  # type: ignore[no-untyped-call]
        op = make_operator_from_name(self.sbp_name)

        return diffusion.make_scheme_from_name(
            self.scheme_name, rec=None, op=op, diffusivity=diffusivity
        )

    def evaluate(self, grid: Grid, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return diffusion.ex_expansion(grid, t, x)


@pytest.mark.parametrize(
    ("case", "order", "resolutions"),
    [
        (DiffusionTestCase("centered"), 2, list(range(80, 160 + 1, 16))),
        (SATDiffusionTestCase("sbp", "sbp21"), 2, list(range(80, 160 + 1, 16))),
        (SATDiffusionTestCase("sbp", "sbp42"), 4, list(range(80, 160 + 1, 16))),
    ],
)
def test_diffusion_convergence(
    case: DiffusionTestCase,
    order: int,
    resolutions: List[int],
    a: float = -1.0,
    b: float = 1.0,
    tfinal: float = 1.0,
    diffusivity: float = 1.0,
) -> None:
    from pyshocks.timestepping import predict_timestep_from_resolutions
    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=case.scheme_name)
    dt = 0.5 * predict_timestep_from_resolutions(a, b, resolutions, umax=1.0, p=2)

    for n in resolutions:
        h_max, error = evolve(case, n, order=order, dt=dt, a=a, b=b, tfinal=tfinal)

        eoc.add_data_point(h_max, error)
        logger.info("n %3d h_max %.3e error %.6e", n, h_max, error)

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= order - 0.1


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
