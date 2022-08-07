# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import jax.random

from pyshocks import advection, continuity

import pytest


# {{{ test advection vs continuity


@pytest.mark.parametrize("rec_name", ["constant", "wenojs32"])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet"])
def test_advection_vs_continuity(
    rec_name: str,
    bc_type: str,
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

        boundary = make_dirichlet_boundary(
            lambda t, x: jnp.zeros_like(x)  # type: ignore[no-untyped-call]
        )
    else:
        raise ValueError(f"unknown 'bc_type': {bc_type}")

    # NOTE: the two schemes are only similar if the velocity is
    # divergence free; in 1d, that means it has to be constant
    velocity = jnp.ones_like(grid.x)  # type: ignore[no-untyped-call]
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

    u = jnp.sin(2.0 * np.pi * grid.x)
    v = jnp.sin(2.0 * np.pi * grid.x)

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
    ax.grid(True)
    fig.savefig(f"finite_comparison_{type(ascheme).__name__}_{bc_type}")
    plt.close(fig)


# }}}


# {{{ test advection vs finite difference


@pytest.mark.parametrize("rec_name", ["constant"])
@pytest.mark.parametrize("bc_type", ["periodic", "dirichlet"])
def test_advection_finite_difference_jacobian(
    rec_name: str,
    bc_type: str,
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

        boundary = make_dirichlet_boundary(
            lambda t, x: jnp.zeros_like(x)  # type: ignore[no-untyped-call]
        )
    else:
        raise ValueError(f"unknown 'bc_type': {bc_type}")

    key = jax.random.PRNGKey(42)
    velocity = jax.random.normal(key, grid.x.shape, dtype=np.float64)
    scheme = advection.Godunov(rec=rec, velocity=velocity)

    # }}}

    # {{{ construct finite difference jacobian

    # TODO: This would be more convincing as a convergence study

    from pyshocks import apply_operator

    op = jax.jit(partial(apply_operator, scheme, grid, boundary, 0.0))

    _, subkey = jax.random.split(key)
    u = jax.random.normal(subkey, grid.x.shape, dtype=np.float64)

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
