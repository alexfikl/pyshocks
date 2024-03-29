# SPDX-FileCopyrightText: 2021 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Type, cast

import jax
import jax.numpy as jnp
import pytest

from pyshocks import get_logger
from pyshocks import timestepping as ts
from pyshocks.tools import Array, ScalarLike

logger = get_logger("test_timestepping")


@pytest.mark.parametrize(
    ("cls", "order"),
    [(ts.ForwardEuler, 1), (ts.SSPRK33, 3), (ts.RK44, 4), (ts.CKRK45, 4)],
)
def test_time_convergence(
    cls: Type[ts.Stepper], order: int, *, visualize: bool = False
) -> None:
    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    # {{{ ode

    @jax.jit
    def source(t: ScalarLike, u: Array) -> Array:
        return jnp.array([jnp.exp(-t)])

    def solution(t: ScalarLike) -> Array:
        return cast(Array, -source(t, jnp.array([0.0])))

    tfinal = 4.0
    u0 = solution(0.0)
    u_ex = solution(tfinal)

    if visualize:
        fig = plt.figure()
        ax = fig.gca()

    # }}}

    # {{{ convergence

    from pyshocks import EOCRecorder

    eoc = EOCRecorder(name=cls.__name__)

    for n in range(2, 7):
        dt: ScalarLike = 1.0 / 2.0**n
        maxit, dt = ts.predict_maxit_from_timestep(tfinal, dt)

        stepper = cls(
            predict_timestep=lambda t, u, dt=dt: dt,
            source=source,
            checkpoint=None,
        )

        u_acc = []
        t_acc = []
        for event in ts.step(stepper, u0, maxit=maxit):
            u_acc.append(event.u[0])
            t_acc.append(event.t)

        u = jnp.array(u_acc)
        t = jnp.array(t_acc)

        if visualize:
            ax.plot(t, u)

        u_ap = u[-1]
        error = jnp.linalg.norm(u_ap - u_ex) / jnp.linalg.norm(u_ex)

        eoc.add_data_point(dt, error)
        logger.info("n %2d dt %.5e error %.5e", n, dt, error)

    # }}}

    # {{{ visualize

    if visualize:
        ax.set_xlabel("$t$")
        ax.set_ylabel("$u$")
        fig.savefig(f"timestepping_{eoc.name.lower()}")
        plt.close(fig)

    # }}}

    logger.info("\n%s", eoc)
    assert eoc.estimated_order >= order - 0.1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
