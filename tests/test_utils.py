# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np

import jax
import jax.numpy as jnp

from pyshocks import get_logger, set_recommended_matplotlib
from pyshocks import make_uniform_point_grid

import pytest

logger = get_logger("test_utils")
set_recommended_matplotlib()


# {{{ test_ss_weno_burgers_matrices


@pytest.mark.parametrize("bc_type", ["periodic", "ssweno"])
def test_ss_weno_burgers_matrices(bc_type: str) -> None:
    from pyshocks import Boundary
    from pyshocks.scalar import PeriodicBoundary, make_ss_weno_boundary

    if bc_type == "periodic":
        bc: Boundary = PeriodicBoundary()
    elif bc_type == "ssweno":
        bc = make_ss_weno_boundary(ga=lambda t: 0.0)
    else:
        raise ValueError(f"unknown boundary type: '{bc_type}'")

    from pyshocks import EOCRecorder
    from pyshocks.burgers.schemes import make_ss_weno_242_matrices

    eoc = EOCRecorder(name="error")
    for n in range(192, 384 + 1, 32):
        grid = make_uniform_point_grid(a=-1.0, b=1.0, n=n, nghosts=0)
        P, Q, H = make_ss_weno_242_matrices(bc, n)  # noqa: N806

        u0 = jnp.sin(2.0 * jnp.pi * grid.x) ** 2
        int_u0 = (P @ u0) * grid.dx_min

        error = abs(int_u0 - 1)
        eoc.add_data_point(grid.dx_min, error)
        logger.info("error: %4d %.12e", n, error)

        assert jnp.linalg.norm(jnp.sum(Q, axis=1)) < 1.0e-12
        if bc_type == "periodic":
            assert jnp.linalg.norm(Q + Q.T) < 1.0e-12
        else:
            e_i = jnp.eye(1, n, 0).squeeze()  # type: ignore[no-untyped-call]
            e_n = jnp.eye(1, n, n - 1).squeeze()  # type: ignore[no-untyped-call]

            assert (
                jnp.linalg.norm(Q + Q.T + jnp.outer(e_i, e_i) - jnp.outer(e_n, e_n))
                < 1.0e-12
            )

        assert jnp.linalg.norm(jnp.sum(H, axis=1) - 1) < 1.0e-12

    logger.info("eoc:\n%s", eoc)
    assert eoc.satisfied(3.75)


# }}}


# {{{ test_ss_weno_burgers_two_point_flux


@pytest.mark.parametrize("bc_type", ["periodic"])
def test_ss_weno_burgers_two_point_flux(bc_type: str) -> None:
    from pyshocks.scalar import PeriodicBoundary

    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=64, nghosts=0)
    if bc_type == "periodic":
        bc = PeriodicBoundary()
    else:
        raise ValueError(f"unknown boundary type: '{bc_type}'")

    from pyshocks import BlockTimer
    from pyshocks.burgers.schemes import make_ss_weno_242_matrices

    _, Q, _ = make_ss_weno_242_matrices(bc, grid.x.size)  # noqa: N806

    def two_point_flux_numpy_v0(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)
        fs = np.zeros_like(w)

        for i in range(u.size):
            for j in range(i):
                for k in range(i, u.size):
                    fs[i] += 2 * q[k, j] * (w[k] * w[k] + w[k] * w[j] + w[j] * w[j]) / 6

        return jax.device_put(fs)

    def two_point_flux_numpy_v1(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)
        fs = np.zeros_like(w)

        ws = np.tile((w * w).reshape(-1, 1), w.size)
        gs = (np.outer(w, w) + ws + ws.T) / 6
        qgs = 2 * q * gs

        for i in range(u.size):
            for j in range(i):
                for k in range(i, u.size):
                    fs[i] += qgs[k, j]

        return jax.device_put(fs)

    def two_point_flux_numpy_v2(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)
        fs = np.zeros_like(w)

        ws = np.tile((w * w).reshape(-1, 1), w.size)
        gs = (np.outer(w, w) + ws + ws.T) / 6
        qgs = 2 * q * gs

        j = np.arange(w.size)
        for i in range(u.size):
            irow = j >= i
            icol = j < i

            fs[i] = np.sum(qgs[np.ix_(irow, icol)])

        return jax.device_put(fs)

    @jax.jit
    def two_point_flux_jax(u: jnp.ndarray) -> jnp.ndarray:
        us = jnp.tile((u * u).reshape(-1, 1), u.size)  # type: ignore
        fs = (jnp.outer(u, u) + us + us.T) / 6
        qfs = 2 * Q * fs

        fss = jnp.empty_like(u)  # type: ignore
        j = np.arange(u.size)

        for i in range(u.size):
            (irow,) = np.where(j >= i)
            (icol,) = np.where(j < i)

            fss = fss.at[i].set(jnp.sum(qfs[np.ix_(irow, icol)]))

        return fss

    # {{{ reference numpy version

    u0 = jnp.sin(2.0 * jnp.pi * grid.x) ** 2
    with BlockTimer() as bt:
        fs_v0 = two_point_flux_numpy_v0(u0)
    logger.info("%s", bt)

    # }}}

    # {{{ numpy v1

    with BlockTimer() as bt:
        fs_v1 = two_point_flux_numpy_v1(u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_v0 - fs_v1) / jnp.linalg.norm(fs_v0)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # }}}

    # {{{ numpy v2

    with BlockTimer() as bt:
        fs_v2 = two_point_flux_numpy_v2(u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_v0 - fs_v2) / jnp.linalg.norm(fs_v0)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # }}}

    # {{{ jax

    with BlockTimer() as bt:
        fs_jax = two_point_flux_jax(u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_v0 - fs_jax) / jnp.linalg.norm(fs_v0)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
