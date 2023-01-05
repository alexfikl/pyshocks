# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import jax
import jax.numpy as jnp
import jax.random

from pyshocks import EOCRecorder, get_logger, set_recommended_matplotlib
from pyshocks import advection, continuity
from pyshocks.finitedifference import Stencil

import pytest

logger = get_logger("test_finite_difference")
set_recommended_matplotlib()


# {{{ test advection vs continuity


@pytest.mark.parametrize("rec_name", ["constant", "wenojs32"])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet"])
def test_advection_vs_continuity(
    rec_name: str,
    bc_type: str,
    *,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 256,
    visualize: bool = False,
) -> None:
    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    # {{{ setup

    from pyshocks.reconstruction import make_reconstruction_from_name
    from pyshocks import make_uniform_cell_grid, Boundary

    rec = make_reconstruction_from_name(rec_name)
    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=rec.stencil_width)

    if bc_type == "periodic":
        from pyshocks.scalar import PeriodicBoundary

        boundary: Boundary = PeriodicBoundary()
    elif bc_type == "dirichlet":
        from pyshocks.scalar import make_dirichlet_boundary

        boundary = make_dirichlet_boundary(lambda t, x: jnp.zeros_like(x))
    else:
        raise ValueError(f"unknown 'bc_type': {bc_type}")

    # NOTE: the two schemes are only similar if the velocity is
    # divergence free; in 1d, that means it has to be constant
    velocity = jnp.ones_like(grid.x)
    ascheme = advection.Godunov(rec=rec, velocity=velocity)
    cscheme = continuity.Godunov(rec=rec, velocity=velocity)

    # }}}

    # {{{ check.. something?

    # NOTE: the advection and continuity operators are not the same in general
    # However, if we use the same input and a constant velocity, they should be

    i = grid.i_

    def dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return (x[i] * grid.dx[i]) @ y[i]

    from pyshocks import apply_operator

    aop = jax.jit(partial(apply_operator, ascheme, grid, boundary, 0.0))
    cop = jax.jit(partial(apply_operator, cscheme, grid, boundary, 0.0))

    u = jnp.sin(2.0 * jnp.pi * grid.x)
    v = jnp.sin(2.0 * jnp.pi * grid.x)

    error = abs(dot(u, aop(v)) - dot(cop(u), v))
    print(f"error: {error:.5e}")
    assert error < 1.0e-15

    # }}}

    if not visualize:
        return

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(grid.x[i], aop(v)[i], label="ADV")
    ax.plot(grid.x[i], cop(u)[i], label="CON")

    ax.set_xlabel("$x$")
    fig.savefig(f"finite_comparison_{type(ascheme).__name__}_{bc_type}")
    plt.close(fig)


# }}}


# {{{ test advection vs finite difference


@pytest.mark.parametrize("rec_name", ["constant"])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet"])
def test_advection_finite_difference_jacobian(
    rec_name: str,
    bc_type: str,
    *,
    a: float = -1.0,
    b: float = +1.0,
    n: int = 32,
    visualize: bool = False,
) -> None:
    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    # {{{ setup

    from pyshocks.reconstruction import make_reconstruction_from_name
    from pyshocks import make_uniform_cell_grid, Boundary

    rec = make_reconstruction_from_name(rec_name)
    grid = make_uniform_cell_grid(a=a, b=b, n=n, nghosts=rec.stencil_width)

    if bc_type == "periodic":
        from pyshocks.scalar import PeriodicBoundary

        boundary: Boundary = PeriodicBoundary()
    elif bc_type == "dirichlet":
        from pyshocks.scalar import make_dirichlet_boundary

        boundary = make_dirichlet_boundary(lambda t, x: jnp.zeros_like(x))
    else:
        raise ValueError(f"unknown 'bc_type': {bc_type}")

    key = jax.random.PRNGKey(42)
    velocity = jax.random.normal(key, grid.x.shape, dtype=jnp.float64)
    scheme = advection.Godunov(rec=rec, velocity=velocity)

    # }}}

    # {{{ construct finite difference jacobian

    # TODO: This would be more convincing as a convergence study

    from pyshocks import apply_operator

    op = jax.jit(partial(apply_operator, scheme, grid, boundary, 0.0))

    _, subkey = jax.random.split(key)
    u = jax.random.normal(subkey, grid.x.shape, dtype=jnp.float64)

    eps = 1.0e-3
    fddjac = []

    for i in range(u.size):
        up = u.at[i].add(+eps)
        um = u.at[i].add(-eps)
        fddjac.append((op(up) - op(um)) / (2.0 * eps))

    fddjac = jnp.stack(fddjac).T

    # }}}

    # {{{ compare

    jaxjac = jax.jacfwd(partial(apply_operator, scheme, grid, boundary, 0.0))(u)

    error = jnp.linalg.norm(jaxjac - fddjac) / jnp.linalg.norm(jaxjac)
    print(f"error: {error:.5e}")
    assert error < 1.0e-13

    # }}}

    if not visualize:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(fddjac)
    ax1.set_title("$Finite$")
    ax2.imshow(jaxjac)
    ax2.set_title("$JAX$")

    fig.savefig(f"finite_jacfwd_value_{type(scheme).__name__}_{bc_type}")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()

    p = ax.imshow(jnp.log10(jnp.abs(jaxjac - fddjac) + 1.0e-16))
    fig.colorbar(p, ax=ax)
    ax.set_title("$Error$")

    fig.savefig(f"finite_jacfwd_error_{type(scheme).__name__}_{bc_type}")
    plt.close(fig)


# }}}


# {{{ test_finite_difference_taylor


def finite_difference_convergence(d: Stencil) -> EOCRecorder:
    from pyshocks.finitedifference import apply_derivative

    eoc = EOCRecorder()

    s = jnp.s_[abs(d.indices[0]) + 1 : -abs(d.indices[-1]) - 1]
    for n in [32, 64, 128, 256, 512]:
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, dtype=d.coeffs.dtype)
        h = theta[1] - theta[0]

        f = jnp.sin(theta)
        num_df_dx = apply_derivative(d, f, h)

        df = jnp.cos(theta) if d.derivative % 2 == 1 else jnp.sin(theta)
        df_dx = (-1.0) ** ((d.derivative - 1) // 2 + 1) * df

        error = jnp.linalg.norm(df_dx[s] - num_df_dx[s]) / jnp.linalg.norm(df_dx[s])
        eoc.add_data_point(h, error)

    return eoc


def test_finite_difference_taylor_stencil(*, visualize: bool = False) -> None:
    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    from pyshocks.finitedifference import (
        make_taylor_approximation,
        make_fornberg_approximation,
    )

    stencils = [
        (
            make_fornberg_approximation(1, (-2, 2)),
            [1 / 12, -8 / 12, 0.0, 8 / 12, -1 / 12],
            4,
            -1 / 30,
        ),
        (
            make_taylor_approximation(1, (-2, 2)),
            [1 / 12, -8 / 12, 0.0, 8 / 12, -1 / 12],
            4,
            -1 / 30,
        ),
        (
            make_taylor_approximation(1, (-2, 1)),
            [1 / 6, -6 / 6, 3 / 6, 2 / 6],
            3,
            1 / 12,
        ),
        (
            make_taylor_approximation(1, (-1, 2)),
            [-2 / 6, -3 / 6, 6 / 6, -1 / 6],
            3,
            -1 / 12,
        ),
        (make_taylor_approximation(2, (-2, 1)), [0.0, 1.0, -2.0, 1.0], 2, 1 / 12),
        (
            make_taylor_approximation(2, (-2, 2)),
            [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            4,
            -1 / 90,
        ),
        (
            make_taylor_approximation(3, (-2, 2)),
            [-1 / 2, 2 / 2, 0.0, -2 / 2, 1 / 2],
            2,
            1 / 4,
        ),
        (make_taylor_approximation(4, (-2, 2)), [1.0, -4.0, 6.0, -4.0, 1.0], 2, 1 / 6),
    ]

    if visualize:
        fig = plt.figure()

    for s, a, order, coefficient in stencils:
        logger.info("stencil:\n%r", s)

        assert jnp.allclose(jnp.sum(s.coeffs), 0.0)
        assert jnp.allclose(s.coeffs, jnp.array(a, dtype=s.coeffs.dtype))
        assert jnp.allclose(s.trunc, coefficient)
        assert s.order == order

        eoc = finite_difference_convergence(s)
        logger.info("\n%s", eoc)
        assert eoc.estimated_order >= order - 0.25

        if visualize:
            part = jnp.real if s.derivative % 2 == 0 else jnp.imag

            from pyshocks.finitedifference import modified_wavenumber

            k = jnp.linspace(0.0, jnp.pi, 128)
            km = part(modified_wavenumber(s, k))
            sign = part(1.0j**s.derivative)

            ax = fig.gca()
            ax.plot(k, km)
            ax.plot(k, sign * k**s.derivative, "k--")

            ax.set_xlabel("$k h$")
            ax.set_ylabel(r"$\tilde{k} h$")
            ax.set_xlim([0.0, jnp.pi])
            ax.set_ylim([0.0, sign * jnp.pi**s.derivative])

            fig.savefig(f"finite_difference_wavenumber_{s.derivative}_{s.order}")
            fig.clf()

    if visualize:
        plt.close(fig)

    from pyshocks.finitedifference import determine_stencil_truncation_error

    a = jnp.array(
        [-0.02651995, 0.18941314, -0.79926643, 0.0, 0.79926643, -0.18941314, 0.02651995]
    )
    indices = jnp.arange(-3, 4)

    order, c = determine_stencil_truncation_error(1, a, indices, atol=1.0e-6)
    assert order == 4
    assert jnp.allclose(c, 0.01970656333333333)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
