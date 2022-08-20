# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np

import jax
import jax.numpy as jnp

from pyshocks import get_logger, set_recommended_matplotlib
from pyshocks import make_uniform_point_grid, BoundaryType

import pytest

logger = get_logger("test_utils")
set_recommended_matplotlib()


# {{{ test_sbp_matrices


@pytest.mark.parametrize("order", ["21", "42"])
def test_sbp_matrices(order: int, visualize: bool = False) -> None:
    from pyshocks import sbp

    bc = BoundaryType.Dirichlet
    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=64, nghosts=0)
    op = getattr(sbp, f"SBP{order}")()

    n = grid.x.size
    dtype = grid.x.dtype

    # {{{ P

    P = sbp.sbp_matrix_from_name(op, grid, bc, "P")
    assert P.shape == (n,)
    assert P.dtype == dtype

    # }}}

    # {{{ Q

    B = sbp.make_sbp_boundary_matrix(n, dtype=dtype)
    assert B.shape == (n, n)
    assert B.dtype == dtype

    Q = sbp.sbp_matrix_from_name(op, grid, bc, "Q")
    assert Q.shape == (n, n)
    assert Q.dtype == dtype
    assert jnp.linalg.norm(Q.T + Q - B) < 1.0e-15

    # }}}

    # {{{ R

    R = sbp.sbp_matrix_from_name(op, grid, bc, "R")
    assert R.shape == (n, n)
    assert R.dtype == dtype
    assert jnp.linalg.norm(R - R.T) < 1.0e-6

    s, _ = jnp.linalg.eig(R)  # type: ignore[no-untyped-call]
    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_rb_{order}_eigs")
        fig.clf()

    assert jnp.all(jnp.real(s)) > 0.0

    # }}}

    # {{{ P dx

    P = sbp.sbp_norm_matrix(op, grid, bc)
    assert P.shape == (n,)

    def dotp(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return x @ (P * y)

    # }}}

    # {{{ D1

    logger.info("D1")

    D1 = sbp.sbp_first_derivative_matrix(op, grid, bc)
    assert D1.shape == (n, n)
    assert D1.dtype == dtype

    # check SBP property
    v = jnp.cos(2.0 * jnp.pi * grid.x)
    error = abs(dotp(v, D1 @ v) + dotp(D1 @ v, v) - v @ (B @ v))

    logger.info("sbp error: %.12e", error)
    assert error < 5.0e-14

    # check order conditions
    for i in range(op.order + 2):
        dx_ref = i * grid.x ** max(0, i - 1)
        dx_app = D1 @ (grid.x**i)

        error_b = jnp.linalg.norm(dx_ref - dx_app)
        error_i = jnp.linalg.norm(dx_ref[4:-4] - dx_app[4:-4])
        logger.info(
            "error %2d / %2d: %.12e %.12e", i, op.boundary_order, error_b, error_i
        )

        if i <= op.boundary_order:
            assert error_b < 1.0e-13
        else:
            assert error_b > 1.0e-8

        if i <= op.order:
            assert error_i < 1.0e-13
        else:
            assert error_i > 1.0e-8

    # check eigenvalues: should all be imaginary
    s, _ = jnp.linalg.eig(D1)  # type: ignore[no-untyped-call]
    if visualize:
        ax = fig.gca()

        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_d1_{order}_eigs")
        fig.clf()

    assert jnp.linalg.norm(jnp.real(s)) < 1.0
    assert jnp.linalg.norm(jnp.imag(s)) > 1.0

    # }}}

    # {{{ D2

    logger.info("D2")

    D2 = sbp.sbp_second_derivative_matrix(op, grid, bc, 1.0)
    assert D2.shape == (n, n)
    assert D2.dtype == dtype

    # check SBP property [Mattsson2004] Equation 10
    S = sbp.sbp_matrix_from_name(op, grid, bc, "S")
    BS = B @ S
    error = abs(
        dotp(v, D2 @ v) + dotp(D2 @ v, v) + 2 * dotp(D1 @ v, D1 @ v) - 2 * v @ (BS @ v)
    )

    logger.info("sbp error: %.12e", error)
    # FIXME: why is this so large?
    assert error < 1.0e-6

    # check eigenvalues: should all be negative
    s, _ = jnp.linalg.eig(D2)  # type: ignore[no-untyped-call]
    if visualize:
        ax = fig.gca()
        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_d2_{order}_eigs")
        fig.clf()

    logger.info("max(eig): %.12e", jnp.max(jnp.real(s)))
    assert jnp.all(jnp.real(s) < 5.0e-13)

    # }}}

    if visualize:
        mp.close(fig)


# }}}


# {{{ test_ss_weno_burgers_two_point_flux


def test_ss_weno_burgers_two_point_flux_first_order(n: int = 64) -> None:
    from pyshocks.weno import ss_weno_derivative_matrix

    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=n, nghosts=0)
    q = ss_weno_derivative_matrix(
        jnp.array([-0.5, 0, 0.5], dtype=grid.x.dtype), None, grid.x.size  # type: ignore
    )

    from pyshocks.burgers.ssweno import two_point_entropy_flux

    # check constant
    u = jnp.full_like(grid.x, 1.0)  # type: ignore

    fs_ref = (u[1:] * u[1:] + u[1:] * u[:-1] + u[:-1] * u[:-1]) / 6
    fs = two_point_entropy_flux(q, u)
    assert fs.shape == grid.f.shape

    # NOTE: constant solutions should just do nothing
    error = jnp.linalg.norm(fs[1:-1] - fs_ref)
    assert error < 1.0e-15

    # check non-constant
    u = jnp.sin(2.0 * jnp.pi * grid.x)

    fs_ref = (u[1:] * u[1:] + u[1:] * u[:-1] + u[:-1] * u[:-1]) / 6
    fs = two_point_entropy_flux(q, u)
    assert fs.shape == grid.f.shape

    error = jnp.linalg.norm(fs[1:-1] - fs_ref)
    assert error < 1.0e-15


@pytest.mark.parametrize("bc_type", ["periodic"])
def test_ss_weno_burgers_two_point_flux(bc_type: str) -> None:
    from pyshocks.scalar import PeriodicBoundary

    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=64, nghosts=0)
    if bc_type == "periodic":
        bc = PeriodicBoundary()
    else:
        raise ValueError(f"unknown boundary type: '{bc_type}'")
    assert bc is not None

    from pyshocks import BlockTimer
    from pyshocks import weno

    qi, _ = weno.ss_weno_242_operator_coefficients()
    Q = weno.ss_weno_derivative_matrix(qi, None, grid.x.size)

    def two_point_flux_numpy_v0(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)
        fs = np.zeros(w.size + 1, dtype=w.dtype)

        for i in range(1, w.size):
            for j in range(i):
                for k in range(i, u.size):
                    fs[i] += q[j, k] * (w[k] * w[k] + w[k] * w[j] + w[j] * w[j]) / 3

        return jax.device_put(fs)

    def two_point_flux_numpy_v1(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)

        ww = np.tile((w * w).reshape(-1, 1), w.size)
        qfs = q * (np.outer(w, w) + ww + ww.T) / 3

        fs = np.zeros(w.size + 1, dtype=w.dtype)
        for i in range(1, w.size):
            fs[i] = np.sum(qfs[:i, i:])

        return jax.device_put(fs)

    def two_point_flux_numpy_v2(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)

        ww = np.tile((w * w).reshape(-1, 1), w.size)
        qfs = q * (np.outer(w, w) + ww + ww.T) / 3

        fs = np.zeros(w.size + 1, dtype=w.dtype)
        for i in range(1, w.size):
            fs[i] = np.sum(qfs[:i, i:])

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

    from pyshocks.burgers.ssweno import two_point_entropy_flux

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
