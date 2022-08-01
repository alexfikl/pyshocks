# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Integrators
-----------

.. autoclass:: Stepper
    :no-show-inheritance:
.. autoclass:: ForwardEuler
.. autoclass:: SSPRK33
.. autoclass:: RK44

.. autoclass:: StepCompleted
    :no-show-inheritance:
.. autoclass:: AdjointStepCompleted

.. autofunction:: step
.. autofunction:: adjoint_step
.. autofunction:: advance
"""

from dataclasses import dataclass
from functools import partial, singledispatch
from typing import Callable, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp

from pyshocks.checkpointing import Checkpoint, save, load
from pyshocks.tools import ScalarFunction, VectorFunction


# {{{ interface


@dataclass(frozen=True)
class StepCompleted:
    """
    .. attribute:: t
    .. attribute:: tfinal
    .. attribute:: dt
    .. attribute:: iteration
    .. attribute:: u
    """

    t: float
    tfinal: float
    dt: float
    iteration: int
    u: jnp.ndarray

    def __str__(self) -> str:
        return (
            f"[{self.iteration:5d}] "
            f"t = {self.t:.5e} / {self.tfinal:.5e} dt {self.dt:.5e}"
        )


@dataclass(frozen=True)
class AdjointStepCompleted(StepCompleted):
    """
    .. attribute:: p
    """

    p: jnp.ndarray


@dataclass(frozen=True)
class Stepper:
    """Generic time stepping method for first-order ODEs.

    .. attribute:: predict_step

        A callable taking ``(t, u)`` and returning a maximum time step
        based on a priori estimates.

    .. attribute:: source

        A callable taking ``(t, u)`` that acts as a source term to the ODE.

    .. attribute:: checkpoint

        A :class:`~pyshocks.checkpointing.Checkpoint` used to save the solution
        values at every timestep.
    """

    predict_timestep: ScalarFunction
    source: VectorFunction
    checkpoint: Optional[Checkpoint]


def step(
    stepper: Stepper,
    u0: jnp.ndarray,
    *,
    maxit: Optional[int] = None,
    tstart: float = 0.0,
    tfinal: Optional[float] = None,
) -> Iterator[StepCompleted]:
    """Advance a given ODE description in time to *tfinal*.

    This function is a generator and is meant to be used as

    .. code::

        for event in step(stepper, u0, tfinal=1.0):
            # process event if desired

    :param stepper: time stepper description and source term.
    :param u0: initial condition at *tstart*.
    :param maxit: maximum number of iteration, by default taken to be infinity.
    :param tstart: initial time at which to start the ODE.
    :param tfinal: final time.

    :returns: a :class:`StepCompleted` at the end of each taken time step.
    """
    if tfinal is None:
        tfinal = jnp.inf

    m = 0
    t = tstart
    u = u0

    yield StepCompleted(t=t, tfinal=tfinal, dt=0.0, iteration=m, u=u)

    while True:
        # NOTE: this checkpoints both the initial condition and the final state
        if stepper.checkpoint is not None:
            save(stepper.checkpoint, m, {"m": m, "t": t, "u": u})

        if tfinal is not None and t >= tfinal:
            break

        if maxit is not None and m >= maxit:
            break

        dt = stepper.predict_timestep(t, u)
        dt = min(dt, tfinal - t) + 1.0e-15
        if not jnp.isfinite(dt):
            raise ValueError(f"time step is not finite: {dt}")

        u = advance(stepper, dt, t, u)

        m += 1
        t += dt

        yield StepCompleted(t=t, tfinal=tfinal, dt=dt, iteration=m, u=u)


def adjoint_step(
    stepper: Stepper,
    p0: jnp.ndarray,
    *,
    maxit: int,
    apply_boundary: Optional[
        Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ] = None,
) -> Iterator[AdjointStepCompleted]:
    if stepper.checkpoint is None:
        raise ValueError("adjoint time stepping requires a checkpoint")

    # {{{ construct jacobian

    # NOTE: jacfwd generates the whole Jacobian matrix of size `n x n`, but it
    # seems to be *a lot* faster than using vjp as a matrix free method;
    # possibly because this is all nicely jitted beforehand
    #
    # should not be too big of a problem because we don't plan to do huge
    # problems -- mostly n < 1024

    jac_fun = jax.jit(jax.jacfwd(partial(advance, stepper), argnums=2))

    # }}}

    # NOTE: chk["u"] was supposedly already used to compute p0, so this just
    # gets this checkpoints out of the way and checks consistency
    chk = load(stepper.checkpoint, maxit)
    assert chk["m"] == maxit

    t = tfinal = chk["t"]

    p = p0
    if apply_boundary is not None:
        p = apply_boundary(chk["t"], chk["u"], p)

    yield AdjointStepCompleted(
        t=t, tfinal=tfinal, dt=0.0, iteration=maxit, u=chk["u"], p=p
    )

    for m in range(maxit - 1, -1, -1):
        # load forward state
        chk = load(stepper.checkpoint, m)
        dt = t - chk["t"]
        assert chk["m"] == m

        # advance adjoint
        jac = jac_fun(dt, chk["t"], chk["u"])
        p = jac.T @ p

        if apply_boundary is not None:
            p = apply_boundary(chk["t"], chk["u"], p)

        # yield new solution
        t = chk["t"]
        yield AdjointStepCompleted(
            t=t, tfinal=tfinal, dt=dt, iteration=m, u=chk["u"], p=p
        )


@singledispatch
def advance(stepper: Stepper, dt: float, t: float, u: jnp.ndarray) -> jnp.ndarray:
    r"""Advances the ODE for a single time step, i.e.

    .. math::

        u^{n + 1} = \mathrm{advance}(u^{n + 1}, u^n, \dots).

    :param stepper: time stepping method description and source term.
    :param dt: time step to advance by.
    :param t: time at the beginning of the time step.
    :param u: variable value at the beginning of the time step.

    :returns: approximated value of *u* at :math:`t + \Delta t`.
    """
    raise NotImplementedError(type(stepper).__name__)


# }}}


# {{{ fixed time step


def predict_timestep_from_maxit(tfinal: float, maxit: int) -> Tuple[int, float]:
    """Determine time step from *tfinal* and *maxit*.

    :returns: a tuple of ``(maxit, dt)`` with the approximated values.
    """
    dt = tfinal / maxit + 1.0e-15
    return maxit, dt


def predict_maxit_from_timestep(tfinal: float, dt: float) -> Tuple[int, float]:
    """Determine the maximum number of iteration for a fixed time step *dt*.

    :returns: a tuple ``(maxit, dt)`` with the approximated values.
    """
    maxit = int(tfinal / dt) + 1
    dt = tfinal / maxit + 1.0e-15

    return maxit, dt


def predict_timestep_from_resolutions(
    a: float, b: float, resolutions: List[int], *, umax: float = 1.0, p: int = 1
) -> float:
    """Determine a maximum time step that is stable for the given domain
    and resolutions. The time step is computed based on the characteristic
    velocity *umax*.

    :returns: a time step small enough for all given resolutions.
    """
    dx = (b - a) / max(resolutions)

    return dx**p / umax


# }}}


# {{{ Forward Euler


@dataclass(frozen=True)
class ForwardEuler(Stepper):
    """Classic Forward Euler time stepping method."""


@advance.register(ForwardEuler)
def _advance_forward_euler(
    stepper: ForwardEuler, dt: float, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return u + dt * stepper.source(t, u)


# }}}


# {{{ SSPRK33


@dataclass(frozen=True)
class SSPRK33(Stepper):
    """The optimal third order SSP Runge-Kutta method with 3 stages."""


@advance.register(SSPRK33)
def _advance_ssprk33(
    stepper: SSPRK33, dt: float, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    fn = stepper.source

    k1 = u + dt * fn(t, u)
    k2 = 3.0 / 4.0 * u + 1.0 / 4.0 * (k1 + dt * fn(t + dt, k1))
    return 1.0 / 3.0 * u + 2.0 / 3.0 * (k2 + dt * fn(t + 0.5 * dt, k2))


# }}}


# {{{ RK44


@dataclass(frozen=True)
class RK44(Stepper):
    """The classic fourth order Runge-Kutta method with 4 stages."""


@advance.register(RK44)
def _advance_rk44(stepper: RK44, dt: float, t: float, u: jnp.ndarray) -> jnp.ndarray:
    fn = stepper.source

    k1 = dt * fn(t, u)
    k2 = dt * fn(t + 0.5 * dt, u + 0.5 * k1)
    k3 = dt * fn(t + 0.5 * dt, u + 0.5 * k2)
    k4 = dt * fn(t + dt, u + k3)

    return u + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# }}}
