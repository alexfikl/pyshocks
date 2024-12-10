# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Integrators
-----------

.. autoclass:: Stepper
    :no-show-inheritance:
    :members:
.. autoclass:: ForwardEuler
.. autoclass:: SSPRK33
.. autoclass:: RK44
.. autoclass:: CKRK45

.. autoclass:: StepCompleted
    :no-show-inheritance:
    :members:
.. autoclass:: AdjointStepCompleted
    :members:

.. autofunction:: step
.. autofunction:: adjoint_step
.. autofunction:: advance
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, singledispatch
from typing import Callable, ClassVar, Iterator

import jax
import jax.numpy as jnp

from pyshocks.checkpointing import Checkpoint, load, save
from pyshocks.tools import Array, Scalar, ScalarFunction, ScalarLike, VectorFunction

# {{{ interface


@dataclass(frozen=True)
class StepCompleted:
    t: Scalar
    """Current time."""
    tfinal: Scalar
    """Final time of the simulation."""
    dt: Scalar
    """Time step used to get to :attr:`t` in the current step. Note that for
    multi-stage schemes, this is the step used by the all stages.
    """
    iteration: int
    """Current iteration."""
    u: Array
    """Solution at the time :attr:`t`."""

    def __str__(self) -> str:
        return (
            f"[{self.iteration:5d}] "
            f"t = {self.t:.5e} / {self.tfinal:.5e} dt {self.dt:.5e}"
        )


@dataclass(frozen=True)
class AdjointStepCompleted(StepCompleted):
    p: Array
    """Adjoint solution at the time :attr:`~StepCompleted.t`."""


@dataclass(frozen=True)
class Stepper:
    """Generic time stepping method for first-order ODEs."""

    predict_timestep: ScalarFunction
    """A callable taking ``(t, u)`` and returning a maximum time step
    based on a priori estimates."""
    source: VectorFunction
    """A callable taking ``(t, u)`` that acts as a source term to the ODE."""
    checkpoint: Checkpoint | None
    """A :class:`~pyshocks.checkpointing.Checkpoint` used to save the solution
    values at every timestep.
    """


def step(
    stepper: Stepper,
    u0: Array,
    *,
    maxit: int | None = None,
    tstart: ScalarLike = 0.0,
    tfinal: ScalarLike | None = None,
) -> Iterator[StepCompleted]:
    """Advance a given ODE description in time to *tfinal*.

    This function is a generator and is meant to be used as

    .. code::

        for event in step(stepper, u0, tfinal=1.0):
            # process event if desired

    :arg stepper: time stepper description and source term.
    :arg u0: initial condition at *tstart*.
    :arg maxit: maximum number of iteration, by default taken to be infinity.
    :arg tstart: initial time at which to start the ODE.
    :arg tfinal: final time.

    :returns: a :class:`StepCompleted` at the end of each taken time step.
    """
    if tfinal is None:
        # NOTE: can't set this to jnp.inf because we have a debug check for it
        tfinal = float(jnp.finfo(u0.dtype).max)  # type: ignore[no-untyped-call]

    m = 0
    t = jnp.array(tstart, dtype=u0.dtype)
    tfinal = jnp.array(tfinal, dtype=u0.dtype)
    u = u0

    assert isinstance(tfinal, jax.Array)
    yield StepCompleted(
        t=t, tfinal=tfinal, dt=jnp.array(0.0, dtype=u0.dtype), iteration=m, u=u
    )

    while True:
        # NOTE: this checkpoints both the initial condition and the final state
        if stepper.checkpoint is not None:
            save(stepper.checkpoint, m, {"m": m, "t": t, "u": u})

        if tfinal is not None and t >= tfinal:
            break

        if maxit is not None and m >= maxit:
            break

        dt = stepper.predict_timestep(t, u)
        if tfinal != jnp.inf:
            dt_min = tfinal - t
            dt = (dt if dt < dt_min else dt_min) + 1.0e-15

        if not jnp.isfinite(dt):
            raise ValueError(f"Time step is not finite: {dt!r}.")

        u = advance(stepper, dt, t, u)

        m += 1
        t += dt

        yield StepCompleted(t=t, tfinal=tfinal, dt=dt, iteration=m, u=u)


def adjoint_step(
    stepper: Stepper,
    p0: Array,
    *,
    maxit: int,
    apply_boundary: Callable[[Scalar, Array, Array], Array] | None = None,
) -> Iterator[AdjointStepCompleted]:
    if stepper.checkpoint is None:
        raise ValueError("Adjoint time stepping requires a checkpoint.")

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
        t=t,
        tfinal=tfinal,
        dt=jnp.array(0.0, dtype=p.dtype),
        iteration=maxit,
        u=chk["u"],
        p=p,
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
def advance(stepper: Stepper, dt: ScalarLike, t: ScalarLike, u: Array) -> Array:
    r"""Advances the ODE for a single time step, i.e.

    .. math::

        u^{n + 1} = \mathrm{advance}(u^{n + 1}, u^n, \dots).

    :arg stepper: time stepping method description and source term.
    :arg dt: time step to advance by.
    :arg t: time at the beginning of the time step.
    :arg u: variable value at the beginning of the time step.

    :returns: approximated value of *u* at :math:`t + \Delta t`.
    """
    raise NotImplementedError(type(stepper).__name__)


# }}}


# {{{ fixed time step


def predict_timestep_from_maxit(tfinal: ScalarLike, maxit: int) -> tuple[int, Scalar]:
    """Determine time step from *tfinal* and *maxit*.

    :returns: a tuple of ``(maxit, dt)`` with the approximated values.
    """
    dt = tfinal / maxit + 1.0e-15
    return maxit, jnp.array(dt, dtype=jnp.float64)


def predict_maxit_from_timestep(
    tfinal: ScalarLike, dt: ScalarLike
) -> tuple[int, Scalar]:
    """Determine the maximum number of iteration for a fixed time step *dt*.

    :returns: a tuple ``(maxit, dt)`` with the approximated values.
    """
    maxit = int(tfinal / dt)
    dt = tfinal / maxit + 1.0e-15

    return maxit, jnp.array(dt, dtype=jnp.float64)


def predict_timestep_from_resolutions(
    a: ScalarLike,
    b: ScalarLike,
    resolutions: list[int],
    *,
    umax: ScalarLike = 1.0,
    p: int = 1,
) -> Scalar:
    """Determine a maximum time step that is stable for the given domain
    and resolutions. The time step is computed based on the characteristic
    velocity *umax*.

    :returns: a time step small enough for all given resolutions.
    """
    dx = (b - a) / max(resolutions)

    return jnp.array(dx**p / umax, dtype=jnp.float64)


# }}}


# {{{ Forward Euler


@dataclass(frozen=True)
class ForwardEuler(Stepper):
    """Classic Forward Euler time stepping method."""


@advance.register(ForwardEuler)
def _advance_forward_euler(
    stepper: ForwardEuler, dt: ScalarLike, t: ScalarLike, u: Array
) -> Array:
    return u + dt * stepper.source(t, u)


# }}}


# {{{ SSPRK33


@dataclass(frozen=True)
class SSPRK33(Stepper):
    """The optimal third order SSP Runge-Kutta method with 3 stages."""


@advance.register(SSPRK33)
def _advance_ssprk33(
    stepper: SSPRK33, dt: ScalarLike, t: ScalarLike, u: Array
) -> Array:
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
def _advance_rk44(stepper: RK44, dt: ScalarLike, t: ScalarLike, u: Array) -> Array:
    fn = stepper.source

    k1 = dt * fn(t, u)
    k2 = dt * fn(t + dt / 2, u + k1 / 2)
    k3 = dt * fn(t + dt / 2, u + k2 / 2)
    k4 = dt * fn(t + dt, u + k3)

    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# }}}


# {{{ CKRK45


@dataclass(frozen=True)
class CKRK45(Stepper):
    """Low-storage, five-stage, fourth-order Runge-Kutta method of Carpenter
    and Kennedy [Carpenter1994]_.

    The coefficients are presented on page 13.
    """

    # NOTE: myth goes that this coefficients are accurate to 26 digits

    a: ClassVar[Array] = jnp.array(
        [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
        ],
        dtype=jnp.float64,
    )

    b: ClassVar[Array] = jnp.array(
        [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
        ],
        dtype=jnp.float64,
    )

    c: ClassVar[Array] = jnp.array(
        [
            0.0,
            1432997174477 / 9575080441755,
            2526269341429 / 6820363962896,
            2006345519317 / 3224310063776,
            2802321613138 / 2924317926251,
        ],
        dtype=jnp.float64,
    )


@advance.register(CKRK45)
def _advance_ckrk45(stepper: CKRK45, dt: ScalarLike, t: ScalarLike, u: Array) -> Array:
    fn = stepper.source

    p = k = u
    for i in range(stepper.a.size):
        k = stepper.a[i] * k + dt * fn(t + stepper.c[i] * dt, p)
        p = p + stepper.b[i] * k

    return p


# }}}
