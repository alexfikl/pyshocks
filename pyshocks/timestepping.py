# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autoclass:: Stepper
    :no-show-inheritance:
.. autoclass:: ForwardEuler
.. autoclass:: SSPRK33
.. autoclass:: RK44

.. autoclass:: StepCompleted
    :no-show-inheritance:

.. autofunction:: step
.. autofunction:: advance
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import List, Optional

import jax.numpy as jnp

from pyshocks.schemes import ScalarFunction, VectorFunction


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

    def __str__(self):
        return (
            f"[{self.iteration:5d}] "
            f"t = {self.t:.5e} / {self.tfinal:.5e} dt {self.dt:.5e}"
        )


@dataclass(frozen=True)
class Stepper:
    """Generic time stepping method for first-order ODEs.

    .. attribute:: predict_step

        A callable taking ``(t, u)`` and returning a maximum time step
        based on a priori estimates.

    .. attribute:: source

        A callable taking ``(t, u)`` that acts as a source term to the ODE.
    """

    predict_timestep: ScalarFunction
    source: VectorFunction


def step(
    stepper: Stepper,
    u0: jnp.ndarray,
    *,
    maxit: Optional[int] = None,
    tstart: float = 0.0,
    tfinal: Optional[float] = None,
):
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
        if tfinal is not None and t >= tfinal:
            break

        if maxit is not None and m >= maxit:
            break

        dt = stepper.predict_timestep(t, u)
        dt = min(dt, tfinal - t) + 1.0e-15

        u = advance(stepper, dt, t, u)

        m += 1
        t += dt

        yield StepCompleted(t=t, tfinal=tfinal, dt=dt, iteration=m, u=u)


@singledispatch
def advance(stepper: Stepper, dt: float, t: float, u: jnp.ndarray) -> jnp.ndarray:
    r"""Advances the ODE for a single time step, i.e.

    .. math::

        u^{n + 1} = \mathrm{advance}(u^{n + 1}, u^n, \dots).

    :param stepper: time stepping method description and source term.
    :param dt: time step to advance by.
    :param t: time at the begining of the time step.
    :param u: variable value at the begining of the time step.

    :returns: approximated value of *u* at :math:`t + \Delta t`.
    """
    raise NotImplementedError(type(stepper).__name__)


# }}}


# {{{ fixed time step


def predict_timestep_from_maxit(tfinal: float, maxit: int):
    """Determine time step from *tfinal* and *maxit*.

    :returns: a tuple of ``(maxit, dt)`` with the approximated values.
    """
    dt = tfinal / maxit + 1.0e-15
    return maxit, dt


def predict_maxit_from_timestep(tfinal: float, dt: float):
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
def _advance_ssprk33(stepper: SSPRK33, dt: float, t: float, u: jnp.ndarray):
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
def _advance_rk44(stepper: RK44, dt: float, t: float, u: jnp.ndarray):
    fn = stepper.source

    k1 = dt * fn(t, u)
    k2 = dt * fn(t + 0.5 * dt, u + 0.5 * k1)
    k3 = dt * fn(t + 0.5 * dt, u + 0.5 * k2)
    k4 = dt * fn(t + dt, u + k3)

    return u + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# }}}
