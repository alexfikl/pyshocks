# SPDX-FileCopyrightText: 2021 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from itertools import product
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from pyshocks import get_logger, set_recommended_matplotlib
from pyshocks import (
    SpatialFunction,
    BoundaryType,
    make_uniform_cell_grid,
)
from pyshocks.reconstruction import WENOJS, reconstruct, make_reconstruction_from_name

import pytest

logger = get_logger("test_weno")
set_recommended_matplotlib()


# {{{ test_weno_smoothness_indicator_vectorization


@pytest.mark.parametrize("rec_name", ["wenojs32", "wenojs53"])
def test_weno_smoothness_indicator_vectorization(
    rec_name: str, rtol: float = 2.0e-15, n: int = 64
) -> None:
    """Tests that the vectorized version of the smoothness indicator matches
    the explicitly looped version.
    """

    # {{{ setup

    rec = make_reconstruction_from_name(rec_name)
    assert isinstance(rec, WENOJS)

    a = rec.s.a
    b = rec.s.b
    c = rec.s.c

    nghosts = b.shape[-1] // 2
    nstencils = b.shape[0]
    stencil = jnp.arange(-nghosts, nghosts + 1)

    n = n + 2 * nghosts
    m = jnp.s_[nghosts:-nghosts]

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, dtype=jnp.float64)
    u = jnp.sin(theta)

    # }}}

    # {{{ compute smoothness indicator

    import numpy as np

    # loop-based
    beta0 = np.zeros((nstencils, n), dtype=jnp.float64)
    for j in range(*m.indices(n)):  # pylint: disable=no-member
        for i, k in product(range(nstencils), range(a.size)):
            beta0[i, j] += a[k] * jnp.sum(u[j + stencil] * b[i, k, ::-1]) ** 2

    # jnp.convolve-based
    from pyshocks.weno import weno_smoothness

    beta1 = weno_smoothness(rec.s, u)

    # }}}

    # {{{ compute stencils

    # loop-based
    uhat0 = np.zeros((nstencils, n), dtype=jnp.float64)
    for j in range(*m.indices(n)):  # pylint: disable=no-member
        for i in range(nstencils):
            uhat0[i, j] = jnp.sum(u[j + stencil] * c[i, ::-1])

    # jnp.convolve-based
    from pyshocks.weno import weno_reconstruct

    uhat1 = weno_reconstruct(rec.s, u)

    # }}}

    # {{{ check equality

    error = jla.norm(beta0[:, m] - beta1[:, m]) / jla.norm(beta0[:, m])
    logger.info("beta[%s]: %.8e", type(rec).__name__, error)
    assert error < rtol

    error = jla.norm(uhat0[:, m] - uhat1[:, m]) / jla.norm(uhat0[:, m])
    logger.info("uhat[%s]: %.8e", type(rec).__name__, error)
    assert error < rtol

    # }}}


# }}}


# {{{ test_weno_smoothness_indicator


@pytest.mark.parametrize(("rec_name", "n"), [("wenojs32", 512), ("wenojs53", 128)])
@pytest.mark.parametrize("is_smooth", [True, False])
def test_weno_smoothness_indicator(rec_name: str, n: int, is_smooth: bool) -> None:
    """Tests that the smoothness indicator actually works."""

    # {{{ setup

    rec = make_reconstruction_from_name(rec_name)
    assert isinstance(rec, WENOJS)

    b = rec.s.b

    nghosts = b.shape[-1] // 2
    n = n + 2 * nghosts
    m = jnp.s_[nghosts:-nghosts]

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, dtype=jnp.float64)
    if is_smooth:
        u = jnp.sin(theta)
    else:
        u = (theta < jnp.pi).astype(theta.dtype)

    # }}}

    # {{{ compute smoothness indicator

    from pyshocks.weno import weno_smoothness

    beta = weno_smoothness(rec.s, u)

    alpha = rec.s.d / (rec.eps + beta) ** 2
    omega = alpha / jnp.sum(alpha, axis=0, keepdims=True)

    # }}}

    # {{{ check smoothness

    # NOTE: rec.d are the "ideal" coefficients, so we're comparing how
    # close we are to those

    error = jnp.max(jnp.abs(omega[:, m] - rec.s.d))
    logger.info("error[%s, %s]: %.8e", type(rec).__name__, is_smooth, error)

    if is_smooth:
        assert error < 0.1
    else:
        assert error > 0.1

    # }}}


# }}}


# {{{ test_weno_vs_pyweno


def _pyweno_reconstruct(
    u: jnp.ndarray, order: int, side: str
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    import pyweno

    ul, sl = pyweno.weno.reconstruct(
        jax.device_get(u), order, side, return_smoothness=True
    )

    return jax.device_put(ul), jax.device_put(sl)


@pytest.mark.parametrize(
    ("rec_name", "order", "n"),
    [
        # NOTE: pyweno only seems to support order >= 5
        # ("wenojs32", 3, 256),
        ("wenojs53", 5, 512),
    ],
)
def test_weno_vs_pyweno(
    rec_name: str, order: int, n: int, visualize: bool = False
) -> None:
    """Compares our weno reconstruction to PyWENO"""
    pytest.importorskip("pyweno")

    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    # {{{ reference values

    rec = make_reconstruction_from_name(rec_name)
    assert isinstance(rec, WENOJS)

    grid = make_uniform_cell_grid(a=0, b=2.0 * jnp.pi, n=n, nghosts=rec.stencil_width)

    from pyshocks import make_leggauss_quadrature, cell_average

    quad = make_leggauss_quadrature(grid, order=order)
    u = cell_average(quad, lambda x: jnp.sin(x))  # pylint: disable=W0108

    uhost = u.copy()
    ul, sl = _pyweno_reconstruct(uhost, order, "left")
    ur, sr = _pyweno_reconstruct(uhost, order, "right")

    # }}}

    # {{{ compare

    from pyshocks import rnorm

    from pyshocks.weno import weno_smoothness

    betar = weno_smoothness(rec.s, u[::-1])[:, ::-1].T
    betal = weno_smoothness(rec.s, u).T

    error_l = rnorm(grid, sl, betal)
    error_r = rnorm(grid, sr, betar)
    logger.info("error smoothness: left %.5e right %.5e", error_l, error_r)
    assert error_l < 1.0e-5 and error_r < 1.0e-8

    ulhat, urhat = reconstruct(rec, grid, BoundaryType.Dirichlet, u, u, u)

    error_l = rnorm(grid, ul, ulhat)
    error_r = rnorm(grid, ur, urhat)
    logger.info("error reconstruct: left %.5e right %.5e", error_l, error_r)
    assert error_l < 1.0e-12 and error_r < 1.0e-12

    # }}}

    if not visualize:
        return

    s = grid.i_

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(grid.x[s], ul[s] - ulhat[s], label="left")
    ax.plot(grid.x[s], ur[s] - urhat[s], label="right")
    ax.grid()
    ax.legend()
    fig.savefig("test_weno_reference_reconstruct")
    fig.clf()

    ax = fig.gca()
    ax.plot(grid.x[s], sl[s] - betal[s], label="left")
    ax.plot(grid.x[s], sr[s] - betar[s], label="right")
    ax.grid()
    ax.legend()
    fig.savefig("test_weno_reference_smoothness")

    plt.close(fig)


# }}}


# {{{ test_weno_smooth_reconstruction_order


def get_function(name: str) -> SpatialFunction:
    if name == "sine":
        return lambda x: jnp.sin(2 * jnp.pi * x)
    if name == "linear":
        return lambda x: x
    if name == "quadratic":
        return lambda x: x**2 + 3 * x + 1
    if name == "cubic":
        return lambda x: x**3 - 1

    raise ValueError(f"unknown function name: '{name}'")


@pytest.mark.parametrize(
    ("name", "order", "resolutions"),
    [
        ("wenojs32", 3, list(range(192, 384 + 1, 32))),
        ("wenojs53", 5, list(range(32, 256 + 1, 32))),
        ("esweno32", 3, list(range(192, 384 + 1, 32))),
        ("ssweno242", 4, list(range(192, 384 + 1, 32))),
    ],
)
@pytest.mark.parametrize(
    "func_name",
    [
        "sine",
        # NOTE: mostly for debugging to see where the points fall
        # "linear",
        # "quadratic",
        # "cubic",
    ],
)
def test_weno_smooth_reconstruction_order_cell_values(
    name: str,
    order: int,
    resolutions: List[int],
    func_name: str,
    visualize: bool = False,
) -> None:
    from pyshocks import make_leggauss_quadrature, cell_average
    from pyshocks import EOCRecorder, rnorm

    eoc_l = EOCRecorder(name="ul")
    eoc_r = EOCRecorder(name="ur")
    func = get_function(func_name)

    for n in resolutions:
        rec = make_reconstruction_from_name(name)

        grid = make_uniform_cell_grid(-1.0, 1.0, n=n, nghosts=rec.stencil_width)
        quad = make_leggauss_quadrature(grid, order=order + 1)
        u0 = cell_average(quad, func)

        ul_ref = ur_ref = func(grid.f)
        ul, ur = reconstruct(rec, grid, BoundaryType.Dirichlet, u0, u0, u0)

        error_l = rnorm(grid, ul, ul_ref[:-1], p=jnp.inf)
        error_r = rnorm(grid, ur, ur_ref[1:], p=jnp.inf)
        logger.info("error: n %4d ul %.12e ur %.12e", n, error_l, error_r)

        eoc_l.add_data_point(grid.dx_max, error_l)
        eoc_r.add_data_point(grid.dx_max, error_r)

    logger.info("\n%s", eoc_l)
    logger.info("\n%s", eoc_r)

    assert eoc_l.satisfied(order - 0.5)
    assert eoc_r.satisfied(order - 0.5)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
