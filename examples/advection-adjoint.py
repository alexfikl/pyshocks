# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import pathlib
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as mp

from pyshocks import UniformGrid, Boundary, timeme
from pyshocks import advection, get_logger
from pyshocks.timestepping import Stepper, StepCompleted, step
from pyshocks.adjoint import InMemoryCheckpoint, save, load

logger = get_logger("advection-adjoint")


@dataclass
class Simulation:
    scheme: advection.Scheme
    grid: UniformGrid
    bc: Boundary
    stepper: Stepper

    tfinal: float
    chk: InMemoryCheckpoint

    @property
    def name(self) -> str:
        n = self.grid.n
        scheme = type(self.scheme).__name__.lower()
        stepper = type(self.stepper).__name__.lower()
        return f"{scheme}_{stepper}_{n:05}"

    def save_checkpoint(self, event: StepCompleted) -> None:
        from pyshocks import apply_boundary

        # NOTE: boundary conditions are applied before the time step, so they
        # are not actually enforced after, which seems to mess with the adjoint
        u = apply_boundary(self.bc, self.grid, event.t, event.u)

        save(
            self.chk,
            event.iteration,
            {
                "iteration": event.iteration,
                "t": event.t,
                "dt": event.dt,
                "u": u,
            },
        )

    def load_checkpoint(self) -> StepCompleted:
        self.chk.count -= 1
        data = load(self.chk, self.chk.count)

        return StepCompleted(tfinal=self.tfinal, **data)


@timeme
def evolve_forward(
    sim: Simulation,
    u0: jnp.ndarray,
    *,
    dirname: pathlib.Path,
    interactive: bool,
    visualize: bool,
) -> jnp.ndarray:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ plotting

    s = grid.i_

    if interactive:
        fig = mp.figure()
        ax = fig.gca()
        mp.ion()

        _, ln1 = ax.plot(grid.x[s], u0[s], "k--", grid.x[s], u0[s], "o-", ms=1)
        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([jnp.min(u0) - 1.0, jnp.max(u0) + 1.0])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")

    # }}}

    # {{{ evolve

    from pyshocks import norm

    event = None
    for event in step(stepper, u0, tfinal=sim.tfinal):
        umax = norm(grid, event.u, p=jnp.inf)
        logger.info(
            "[%4d] t = %.5e / %.5e dt %.5e umax = %.5e",
            event.iteration,
            event.t,
            sim.tfinal,
            event.dt,
            umax,
        )

        sim.save_checkpoint(event)
        if interactive:
            ln1.set_ydata(event.u[s])
            mp.pause(0.01)

    # }}}

    # {{{ plot

    if interactive:
        mp.close(fig)

    if not visualize:
        umax = jnp.max(event.u[grid.i_])
        umin = jnp.min(event.u[grid.i_])
        umag = jnp.max(jnp.abs(event.u[grid.i_]))

        fig = mp.figure()
        ax = fig.gca()
        ax.plot(grid.x[grid.i_], event.u[grid.i_], label="$u(T)$")
        ax.plot(grid.x[grid.i_], u0[grid.i_], "k--", label="$u(0)$")

        ax.set_ylim([umin - 0.1 * umag, umax + 0.1 * umag])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"$T = {event.t:.3f}$")

        fig.savefig(dirname / f"advection_forward_{sim.name}")
        mp.close(fig)

    # }}}

    return event.u


@timeme
def evolve_adjoint(
    sim: Simulation,
    p0: jnp.ndarray,
    *,
    dirname: pathlib.Path,
    interactive: bool,
    visualize: bool,
) -> None:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ setup

    from pyshocks.scalar import PeriodicBoundary, dirichlet_boundary

    if isinstance(sim.bc, PeriodicBoundary):
        bc: Boundary = sim.bc
    else:
        bc = dirichlet_boundary(
            lambda t, x: jnp.zeros_like(x)  # type: ignore[no-untyped-call]
        )

    chk = sim.load_checkpoint()
    p = p0

    maxit = chk.iteration
    t = chk.t
    dt = chk.dt

    assert jnp.linalg.norm(p0 - chk.u) < 1.0e-15
    assert abs(t - sim.tfinal) < 1.0e-15

    # }}}

    # {{{ jacobians

    # NOTE: jacfwd generates the whole Jacobian matrix of size `n x n`, but it
    # seems to be *a lot* faster than using vjp as a matrix free method;
    # possibly because this is all nicely jitted beforehand
    #
    # should not be too big of a problem because we don't plan to do huge
    # problems -- mostly n < 1024

    from pyshocks.timestepping import advance

    jac_fun = jax.jit(jax.jacfwd(partial(advance, stepper), argnums=2))

    # }}}

    # {{{ evolve

    s = grid.i_

    if interactive or visualize:
        pmax = jnp.max(p[s])
        pmin = jnp.min(p[s])
        pmag = jnp.max(jnp.abs(p[s]))

    if interactive:
        fig = mp.figure()
        ax = fig.gca()
        mp.ion()

        ln0, ln1 = ax.plot(grid.x[s], chk.u[s], "k--", grid.x[s], p[s], "o-", ms=1)

        ax.set_xlim([grid.a, grid.b])
        ax.set_ylim([pmin - 0.25 * pmag, pmax + 0.25 * pmag])
        ax.set_xlabel("$x$")

    from pyshocks import apply_boundary, norm
    p = apply_boundary(bc, grid, t, p)

    for n in range(maxit, 0, -1):
        # load next forward state
        chk = sim.load_checkpoint()
        t = chk.t

        # compute jacobian at current forward state
        jac = jac_fun(dt, t, chk.u)

        # evolve adjoint state
        p = jac.T @ p
        p = apply_boundary(bc, grid, t, p)

        dt = chk.dt
        pmax = norm(grid, p, p=jnp.inf)
        logger.info(
            "[%4d] t = %.5e / %.5e dt %.5e pmax = %.5e", n - 1, t, sim.tfinal, dt, pmax
        )

        if interactive:
            ln0.set_ydata(chk.u[s])
            ln1.set_ydata(p[s])
            mp.pause(0.01)

    # }}}

    # {{{ plot

    if not visualize:
        return

    fig = mp.figure()
    ax = fig.gca()
    ax.plot(grid.x[s], p[s])
    ax.plot(grid.x[s], p0[s], "k--")

    ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p$")

    fig.savefig(dirname / f"advection_adjoint_{sim.name}")

    # }}}


def main(
    scheme: advection.Scheme,
    *,
    outdir: pathlib.Path,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 256,
    tfinal: float = jnp.pi / 3,
    theta: float = 0.75,
    bctype: str = "dirichlet",
    interactive: bool = False,
    visualize: bool = True,
) -> None:
    if not outdir.exists():
        outdir.mkdir()

    # {{{ geometry

    from pyshocks import make_uniform_grid

    grid = make_uniform_grid(a=a, b=b, n=n, nghosts=scheme.stencil_width)

    from pyshocks import Quadrature, cell_average

    order = int(max(scheme.order, 1.0)) + 1
    quad = Quadrature(grid=grid, order=order)

    from pyshocks import continuity

    velocity = cell_average(quad, lambda x: continuity.velocity_const(grid, 0.0, x))
    scheme = replace(scheme, velocity=velocity)

    # }}}

    # {{{ boundary conditions

    ic = partial(continuity.ic_sine, grid)
    solution = partial(continuity.ex_constant_velocity_field, a=1.0, u0=ic)

    u0 = cell_average(quad, ic)

    boundary: Boundary
    if bctype == "periodic":
        from pyshocks.scalar import PeriodicBoundary

        boundary = PeriodicBoundary()
    elif bctype == "dirichlet":
        from pyshocks.scalar import dirichlet_boundary

        boundary = dirichlet_boundary(solution)
    else:
        raise ValueError(bctype)

    # }}}

    # {{{ forward time stepping

    from pyshocks import predict_timestep, apply_operator

    def forward_predict_timestep(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return theta * predict_timestep(scheme, grid, _t, _u)

    def forward_operator(_t: float, _u: jnp.ndarray) -> jnp.ndarray:
        return apply_operator(scheme, grid, boundary, _t, _u)

    from pyshocks.timestepping import SSPRK33

    stepper = SSPRK33(
        predict_timestep=jax.jit(forward_predict_timestep),
        source=jax.jit(forward_operator),
    )

    # }}}

    # {{{ evolve forward

    sim = Simulation(
        scheme=scheme,
        grid=grid,
        bc=boundary,
        stepper=stepper,
        tfinal=tfinal,
        chk=InMemoryCheckpoint(),
    )

    uf = evolve_forward(
        sim,
        u0,
        dirname=outdir,
        interactive=False,
        visualize=visualize,
    )

    # }}}

    # {{{ evolve adjoint

    evolve_adjoint(
        sim,
        uf,
        dirname=outdir,
        interactive=interactive,
        visualize=visualize,
    )

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scheme",
        default="godunov",
        type=str.lower,
        choices=advection.scheme_ids(),
    )
    parser.add_argument(
        "--outdir", type=pathlib.Path, default=pathlib.Path(__file__).parent
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
    )
    args = parser.parse_args()

    from pyshocks.tools import set_recommended_matplotlib

    set_recommended_matplotlib()

    main(
        advection.make_scheme_from_name(args.scheme),
        outdir=args.outdir,
        interactive=args.interactive,
    )
