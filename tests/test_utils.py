# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import replace

import numpy as np

import jax
import jax.numpy as jnp

from pyshocks import get_logger, set_recommended_matplotlib
from pyshocks import make_uniform_point_grid, BoundaryType

import pytest

logger = get_logger("test_utils")
set_recommended_matplotlib()


# {{{ test_sbp_matrices


@pytest.mark.parametrize("name", ["21", "42"])
@pytest.mark.parametrize("bc", [BoundaryType.Dirichlet, BoundaryType.Periodic])
def test_sbp_matrices(name: str, bc: BoundaryType, visualize: bool = False) -> None:
    from pyshocks import sbp

    is_periodic = bc == BoundaryType.Periodic
    grid = make_uniform_point_grid(
        a=-1.0, b=1.0, n=64, nghosts=0, is_periodic=is_periodic
    )
    op = getattr(sbp, f"SBP{name}")(sbp.SecondDerivativeType.Compatible)

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

    if is_periodic:
        assert jnp.linalg.norm(Q.T + Q) < 1.0e-15
    else:
        assert jnp.linalg.norm(Q.T + Q - B) < 1.0e-15

    # }}}

    # {{{ R

    R = sbp.sbp_matrix_from_name(op, grid, bc, "R")
    assert R.shape == (n, n)
    assert R.dtype == dtype

    error = jnp.linalg.norm(R - R.T) / jnp.linalg.norm(R)
    logger.info("error(r): %.12e", error)
    assert error < 1.0e-6

    s, _ = jnp.linalg.eig(R)
    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()
        ax = fig.gca()

        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_rb_{op.ids}_eigs")
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
    D1.block_until_ready()

    assert D1.shape == (n, n)
    assert D1.dtype == dtype

    # check SBP property
    v = jnp.cos(2.0 * jnp.pi * grid.x)

    if is_periodic:
        error = abs(dotp(v, D1 @ v) + dotp(D1 @ v, v))
    else:
        error = abs(dotp(v, D1 @ v) + dotp(D1 @ v, v) - v @ (B @ v))

    logger.info("sbp error: %.12e", error)
    assert error < 5.0e-14

    # check order conditions
    for i in range(op.order + 2):
        dx_ref = i * grid.x ** max(0, i - 1)
        dx_app = D1 @ (grid.x**i)

        if is_periodic:
            # NOTE: x is not periodic, so the first element is incorrect
            error_b = jnp.nan
            error_i = jnp.linalg.norm(dx_ref[2:-2] - dx_app[2:-2])
        else:
            error_b = jnp.linalg.norm(dx_ref - dx_app)
            error_i = jnp.linalg.norm(dx_ref[4:-4] - dx_app[4:-4])

        logger.info(
            "error %2d / %2d: %.12e %.12e", i, op.boundary_order, error_b, error_i
        )

        if not is_periodic:
            if i <= op.boundary_order:
                assert error_b < 1.0e-13
            else:
                assert error_b > 1.0e-8

        if i <= op.order:
            assert error_i < 1.0e-13
        else:
            assert error_i > 1.0e-8

    # check eigenvalues: should all be imaginary
    s, _ = jnp.linalg.eig(D1)
    if visualize:
        ax = fig.gca()

        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_d1_{op.ids}_eigs")
        fig.clf()

    assert jnp.linalg.norm(jnp.real(s)) < 1.0
    assert jnp.linalg.norm(jnp.imag(s)) > 1.0

    # }}}

    # {{{ D2

    logger.info("D2")

    D2 = sbp.sbp_second_derivative_matrix(op, grid, bc, 1.0)
    D2.block_until_ready()

    assert D2.shape == (n, n)
    assert D2.dtype == dtype

    # check SBP compatibility property [Mattsson2004] Equation 10
    if op.second_derivative != sbp.SecondDerivativeType.Narrow:
        Rhat = jnp.diag(1 / P) @ R  # type: ignore[no-untyped-call]
        if is_periodic:
            error = abs(
                dotp(v, D2 @ v)
                + dotp(D2 @ v, v)
                + 2 * dotp(D1 @ v, D1 @ v)
                - 2 * dotp(Rhat @ v, v)
            )
        else:
            S = sbp.sbp_matrix_from_name(op, grid, bc, "S")
            BS = B @ S
            error = abs(
                dotp(v, D2 @ v)
                + dotp(D2 @ v, v)
                + 2 * dotp(Rhat @ v, v)
                - 2 * dotp(D1 @ v, D1 @ v)
                + 2 * v @ (BS @ v)
            )

        logger.info("sbp error: %.12e", error)
        # assert error < 1.0e-6

    # check eigenvalues: should all be negative
    s, _ = jnp.linalg.eig(D2)
    if visualize:
        ax = fig.gca()
        ax.plot(jnp.real(s), jnp.imag(s), "o")
        ax.set_xlabel(r"$\lambda_r$")
        ax.set_ylabel(r"$\lambda_i$")

        fig.savefig(f"test_sbp_matrices_d2_{op.ids}_eigs")
        fig.clf()

    logger.info("max(eig): %.12e", jnp.max(jnp.real(s)))
    # assert jnp.all(jnp.real(s) < 5.0e-13)

    # }}}

    if visualize:
        mp.close(fig)


# }}}


# {{{ test_sbp_matrices_convergence


def _sbp_rnorm(P: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    if x.shape == ():
        return abs(x - y) / abs(y)

    z = x - y
    return jnp.sqrt((P * z) @ z) / jnp.sqrt((P * y) @ y)


@pytest.mark.parametrize("name", ["21", "42"])
@pytest.mark.parametrize("bc", [BoundaryType.Dirichlet, BoundaryType.Periodic])
@pytest.mark.parametrize("sd_type", ["Compatible", "FullyCompatible", "Narrow"])
def test_sbp_matrices_convergence(
    name: str, bc: BoundaryType, sd_type: str, visualize: bool = True
) -> None:
    from pyshocks import sbp

    is_periodic = bc == BoundaryType.Periodic
    op = getattr(sbp, f"SBP{name}")(getattr(sbp.SecondDerivativeType, sd_type))

    from pyshocks import EOCRecorder

    eoc_pu = EOCRecorder(name="P")
    eoc_du = EOCRecorder(name="D_1")
    eoc_dd = EOCRecorder(name="D_2")

    for n in range(192, 384 + 1, 32):
        grid = make_uniform_point_grid(
            a=-1.0, b=1.0, n=n, nghosts=0, is_periodic=is_periodic
        )

        b = jnp.sin(2.0 * jnp.pi * grid.x)
        # b = jnp.ones_like(grid.x)
        u = jnp.sin(2.0 * jnp.pi * grid.x)

        # int(u^2)
        pu_ref = 1.0
        # du/dx
        du_ref = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * grid.x)
        # d/dx (b du/dx)
        dd_ref = 4.0 * jnp.pi**2 * jnp.cos(4.0 * jnp.pi * grid.x)
        # dd_ref = -4.0 * jnp.pi**2 * jnp.sin(2.0 * jnp.pi * grid.x)

        P = sbp.sbp_norm_matrix(op, grid, bc)
        D1 = sbp.sbp_first_derivative_matrix(op, grid, bc)
        D2 = sbp.sbp_second_derivative_matrix(op, grid, bc, b)

        pu_sbp = (P * u) @ u
        du_sbp = D1 @ u
        dd_sbp = D2 @ u

        pu_error = _sbp_rnorm(P, pu_sbp, pu_ref)
        du_error = _sbp_rnorm(P, du_sbp, du_ref)
        dd_error = _sbp_rnorm(P, dd_sbp, dd_ref)
        logger.info("error: norm %.12e d1 %.12e d2 %.12e", pu_error, du_error, dd_error)

        eoc_pu.add_data_point(grid.dx_min, pu_error)
        eoc_du.add_data_point(grid.dx_min, du_error)
        eoc_dd.add_data_point(grid.dx_min, dd_error)

    logger.info("\n%s\n%s\n%s", eoc_pu, eoc_du, eoc_dd)

    if is_periodic:
        order = op.order
    else:
        order = op.boundary_order

    assert eoc_pu.estimated_order >= order - 0.25
    assert eoc_du.estimated_order >= order - 0.25
    assert eoc_dd.estimated_order >= order - 0.25


# }}}


# {{{ test_ss_weno_burgers_two_point_flux


@jax.jit
def two_point_entropy_flux_21(qi: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    def fs(ul: jnp.ndarray, ur: jnp.ndarray) -> jnp.ndarray:
        return (ul * ul + ul * ur + ur * ur) / 6

    qr = qi[qi.size // 2 + 1 :]
    fss = jnp.zeros(u.size + 1, dtype=u.dtype)  # type: ignore[no-untyped-call]

    i = 1
    fss = fss.at[i].set(2 * qr[1] * fs(u[i - 1], u[i]))
    i = u.size - 1
    fss = fss.at[i].set(2 * qr[0] * fs(u[i - 1], u[i + 1]))

    def body(i: int, fss: jnp.ndarray) -> jnp.ndarray:
        return fss.at[i].set(2 * qr[0] * fs(u[i - 1], u[i]))

    return jax.lax.fori_loop(2, u.size - 1, body, fss)


def test_ss_weno_burgers_two_point_flux_first_order(n: int = 64) -> None:
    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=n, nghosts=0)

    from pyshocks import sbp

    q = sbp.make_sbp_21_first_derivative_q_stencil(dtype=grid.dtype)

    # check constant
    u = jnp.full_like(grid.x, 1.0)  # type: ignore

    fs_ref = (u[1:] * u[1:] + u[1:] * u[:-1] + u[:-1] * u[:-1]) / 6
    fs = two_point_entropy_flux_21(q.int, u)
    assert fs.shape == grid.f.shape

    # NOTE: constant solutions should just do nothing
    error = jnp.linalg.norm(fs[1:-1] - fs_ref)
    assert error < 1.0e-15

    # check non-constant
    u = jnp.sin(2.0 * jnp.pi * grid.x)

    fs_ref = (u[1:] * u[1:] + u[1:] * u[:-1] + u[:-1] * u[:-1]) / 6
    fs = two_point_entropy_flux_21(q.int, u)
    assert fs.shape == grid.f.shape

    error = jnp.linalg.norm(fs[1:-1] - fs_ref)
    assert error < 1.0e-15


@pytest.mark.parametrize("bc", [BoundaryType.Dirichlet])
def test_ss_weno_burgers_two_point_flux(bc: BoundaryType) -> None:
    grid = make_uniform_point_grid(a=-1.0, b=1.0, n=64, nghosts=0)

    from pyshocks import sbp

    q = sbp.make_sbp_42_first_derivative_q_stencil(dtype=grid.dtype)
    q = replace(q, left=None, right=None)
    Q = sbp.make_sbp_matrix_from_stencil(bc, grid.n, q)

    def fs(ul: jnp.ndarray, ur: jnp.ndarray) -> jnp.ndarray:
        return (ul * ul + ul * ur + ur * ur) / 6

    def two_point_flux_numpy_v0(u: jnp.ndarray) -> jnp.ndarray:
        q = jax.device_get(Q)
        w = jax.device_get(u)
        fss = np.zeros(w.size + 1, dtype=w.dtype)

        for i in range(1, w.size):
            for j in range(i):
                for k in range(i, u.size):
                    fss[i] += 2 * q[j, k] * fs(w[j], w[k])

        return jax.device_put(fss)

    from pyshocks import BlockTimer

    # {{{ reference numpy version

    u0 = jnp.sin(2.0 * jnp.pi * grid.x) ** 2
    with BlockTimer() as bt:
        fs_ref = two_point_flux_numpy_v0(u0)
    logger.info("%s", bt)

    # }}}

    # {{{ jax

    from pyshocks.burgers.ssweno import two_point_entropy_flux_42

    with BlockTimer() as bt:
        fs_jax = two_point_entropy_flux_42(q.int, u0)
    logger.info("%s", bt)

    # NOTE: repeat computation for the sake of the JIT, to see the speedup
    with BlockTimer() as bt:
        fs_jax = two_point_entropy_flux_42(q.int, u0)
    logger.info("%s", bt)

    error = jnp.linalg.norm(fs_ref - fs_jax) / jnp.linalg.norm(fs_ref)
    logger.info("error: %.12e", error)
    assert error < 1.0e-15

    error = jnp.linalg.norm(fs_ref - fs_jax) / jnp.linalg.norm(fs_ref)
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
