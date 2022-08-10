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


def test_ss_weno_burgers_two_point_flux_first_order() -> None:
    from pyshocks.weno import ss_weno_circulant

    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=64, nghosts=0)
    q = ss_weno_circulant(
        jnp.array([-0.5, 0, 0.5], dtype=grid.x.dtype), grid.x.size  # type: ignore
    )

    from pyshocks.burgers.schemes import two_point_entropy_flux

    u = jnp.full_like(grid.x, 1.0)  # type: ignore
    fs = two_point_entropy_flux(q, u)

    # NOTE: constant solutions should just do nothing
    error = jnp.linalg.norm(fs[2:-1] - u[1:-1] ** 2 / 2)
    assert error < 1.0e-15


@pytest.mark.skip
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
        fs = np.zeros(w.size + 2, dtype=w.dtype)

        for i in range(w.size):
            for j in range(i + 1):
                for k in range(i, u.size):
                    fs[i + 1] += q[j, k] * (w[k] * w[k] + w[k] * w[j] + w[j] * w[j]) / 3

        return jax.device_put(fs)

    def two_point_flux_numpy_v1(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)

        ww = np.tile((w * w).reshape(-1, 1), w.size)
        qfs = q * (np.outer(w, w) + ww + ww.T) / 3

        fs = np.empty(w.size + 2, dtype=w.dtype)
        for i in range(w.size - 1):
            fs[i + 1] = np.sum(qfs[: i + 1, i:])

        return jax.device_put(fs)

    def two_point_flux_numpy_v2(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)

        ww = np.tile((w * w).reshape(-1, 1), w.size)
        qfs = q * (np.outer(w, w) + ww + ww.T) / 3

        fs = np.empty(w.size + 2, dtype=w.dtype)
        for i in range(w.size - 1):
            fs[i + 1] = np.sum(qfs[: i + 1, i:])

        return jax.device_put(fs)

    @jax.jit
    def two_point_flux_jax_v1(u: jnp.ndarray) -> jnp.ndarray:
        uu = jnp.tile((u * u).reshape(-1, 1), u.size)  # type: ignore
        qfs = Q * (jnp.outer(u, u) + uu + uu.T) / 3

        def body(i: int, fss: jnp.ndarray) -> jnp.ndarray:
            return fss.at[i].set(jnp.sum(qfs[:i, i:]))

        return jax.lax.fori_loop(
            0, u.size, body, jnp.empty_like(u)  # type: ignore[no-untyped-call]
        )

    # {{{ reference numpy version

    u0 = jnp.sin(2.0 * jnp.pi * grid.x) ** 2
    with BlockTimer() as bt:
        fs_ref = two_point_flux_numpy_v0(u0)
    logger.info("%s", bt)

    # }}}

    # {{{ numpy v1

    with BlockTimer() as bt:
        fs_np = two_point_flux_numpy_v1(u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_ref - fs_np) / jnp.linalg.norm(fs_ref)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # }}}

    # {{{ numpy v2

    with BlockTimer() as bt:
        fs_np = two_point_flux_numpy_v2(u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_ref - fs_np) / jnp.linalg.norm(fs_ref)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # }}}

    # {{{ jax

    from pyshocks.burgers.schemes import two_point_entropy_flux

    with BlockTimer() as bt:
        fs_jax = two_point_entropy_flux(Q, u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_ref - fs_jax) / jnp.linalg.norm(fs_ref)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    # with BlockTimer() as bt:
    #     fs_jax = two_point_flux_jax_v1(u0)
    # logger.info("%s", bt)

    # error = jnp.linalg.norm(fs_ref - fs_jax) / jnp.linalg.norm(fs_ref)
    # logger.info("error: %.12e", error)
    # assert error < 1.0e-15

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
