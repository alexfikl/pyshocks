# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as mp

from pyshocks import UniformGrid, Boundary, norm, timeme, get_logger
from pyshocks import advection, apply_boundary, apply_operator, predict_timestep
from pyshocks.timestepping import Stepper, step
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

    def save_checkpoint(self, event):
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

    def load_checkpoint(self):
        self.chk.count -= 1
        data = load(self.chk, self.chk.count)

        from pyshocks.timestepping import StepCompleted

        return StepCompleted(tfinal=self.tfinal, **data)


def get_filename(sim: Simulation, basename: str, ext: str = ""):
    if ext:
        ext = f".{ext}" if ext[0] != "." else ext

    n = sim.grid.n
    scheme = type(sim.scheme).__name__.lower()
    stepper = type(sim.stepper).__name__.lower()
    suffix = f"{scheme}_{stepper}_{n:05}"

    import os

    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, f"{basename}_{suffix}{ext}")


@timeme
def evolve_forward(
    sim: Simulation,
    u0: jnp.ndarray,
    *,
    interactive: bool = False,
    visualize: bool = True,
) -> None:
    grid = sim.grid
    stepper = sim.stepper

    # {{{ evolve

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

    # }}}

    # {{{ plot

    if not visualize:
        return

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

    fig.savefig(get_filename(sim, "advection_forward"))
    mp.close(fig)

    # }}}


@timeme
def evolve_adjoint(
    sim: Simulation, *, interactive: bool = False, visualize: bool = True
):
    grid = sim.grid
    stepper = sim.stepper

    # {{{ setup

    from pyshocks.scalar import PeriodicBoundary, dirichlet_boundary

    if isinstance(sim.bc, PeriodicBoundary):
        bc: Boundary = sim.bc
    else:
        bc = dirichlet_boundary(lambda t, x: jnp.zeros_like(x))

    chk = sim.load_checkpoint()
    p = chk.u

    maxit = chk.iteration
    t = chk.t
    dt = chk.dt

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
        ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
        ax.set_xlabel("$x$")
        ax.grid(True)

    for n in range(maxit, 0, -1):
        # compute jacobian at current forward state
        jac = jac_fun(dt, t, chk.u)

        # evolve adjoint state
        p = apply_boundary(bc, grid, t, p)
        p = jac.T @ p

        # load next forward state
        chk = sim.load_checkpoint()
        t = chk.t
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

    ax.set_ylim([pmin - 0.1 * pmag, pmax + 0.1 * pmag])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p$")

    fig.savefig(get_filename(sim, "advection_adjoint"))

    # }}}


def main(
    scheme,
    a=-1.0,
    b=+1.0,
    n=1024,
    tfinal=1.0,
    theta=1.0,
    bctype="dirichlet",
    interactive=False,
    visualize=False,
):
    # {{{ geometry

    order = int(max(scheme.order, 1.0)) + 1
    grid = UniformGrid(a=a, b=b, n=n, nghosts=order)

    from pyshocks import Quadrature, cell_average

    quad = Quadrature(grid=grid, order=order)

    from pyshocks import continuity

    velocity = cell_average(quad, lambda x: continuity.velocity_const(grid, 0.0, x))
    scheme = replace(scheme, velocity=velocity)

    # }}}

    # {{{ boundary conditions

    ic = partial(continuity.ic_sine, grid)
    solution = partial(continuity.ex_constant_velocity_field, a=1.0, u0=ic)

    u0 = cell_average(quad, ic)

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

    @jax.jit
    def forward_predict_timestep(_t, _u):
        return theta * predict_timestep(scheme, grid, _t, _u)

    @jax.jit
    def forward_operator(_t, _u):
        return apply_operator(scheme, grid, boundary, _t, _u)

    from pyshocks.timestepping import ForwardEuler as TimeStepper

    stepper = TimeStepper(
        predict_timestep=forward_predict_timestep,
        source=forward_operator,
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

    evolve_forward(
        sim,
        u0,
        interactive=interactive,
        visualize=visualize,
    )

    # }}}

    # {{{ evolve adjoint

    evolve_adjoint(
        sim,
        interactive=False,
        visualize=visualize,
    )

    # }}}


if __name__ == "__main__":
    try:
        # https://github.com/nschloe/matplotx
        import matplotx

        mp.style.use(matplotx.styles.dufte)
    except ImportError:
        pass

    main(
        scheme=advection.Godunov(velocity=None),
        # scheme=advection.WENOJS32(velocity=None),
        # scheme=advection.WENOJS53(velocity=None),
    )
