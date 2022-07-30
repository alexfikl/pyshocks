# SPDX-FileCopyrightText: 2021 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from itertools import product
from typing import Tuple

from pyshocks import get_logger
from pyshocks.burgers import WENOJS, WENOJS32, WENOJS53

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

import pytest

logger = get_logger("test_weno")


# {{{ test_weno_smoothness_indicator_vectorization


@pytest.mark.parametrize(
    "scheme",
    [
        WENOJS32(),
        WENOJS53(),
    ],
)
def test_weno_smoothness_indicator_vectorization(
    scheme: WENOJS, rtol: float = 2.0e-15, n: int = 64
) -> None:
    """Tests that the vectorized version of the smoothness indicator matches
    the explicitly looped version.
    """

    # {{{ setup

    a = scheme.a
    b = scheme.b
    c = scheme.c

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
    from pyshocks.weno import weno_js_smoothness

    beta1 = weno_js_smoothness(u, a, b)

    # }}}

    # {{{ compute stencils

    # loop-based
    uhat0 = np.zeros((nstencils, n), dtype=jnp.float64)
    for j in range(*m.indices(n)):  # pylint: disable=no-member
        for i in range(nstencils):
            uhat0[i, j] = jnp.sum(u[j + stencil] * c[i, ::-1])

    # jnp.convolve-based
    from pyshocks.weno import weno_js_reconstruct

    uhat1 = weno_js_reconstruct(u, c)

    # }}}

    # {{{ check equality

    error = jla.norm(beta0[:, m] - beta1[:, m]) / jla.norm(beta0[:, m])
    logger.info("beta[%s]: %.8e", type(scheme).__name__, error)
    assert error < rtol

    error = jla.norm(uhat0[:, m] - uhat1[:, m]) / jla.norm(uhat0[:, m])
    logger.info("uhat[%s]: %.8e", type(scheme).__name__, error)
    assert error < rtol

    # }}}


# }}}


# {{{ test_weno_smoothness_indicator


@pytest.mark.parametrize(
    ("scheme", "n"),
    [
        (WENOJS32(), 512),
        (WENOJS53(), 128),
    ],
)
@pytest.mark.parametrize("is_smooth", [True, False])
def test_weno_smoothness_indicator(scheme: WENOJS, n: int, is_smooth: bool) -> None:
    """Tests that the smoothness indicator actually works."""

    # {{{ setup

    a = scheme.a
    b = scheme.b

    nghosts = b.shape[-1] // 2
    n = n + 2 * nghosts
    m = jnp.s_[nghosts:-nghosts]

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, dtype=jnp.float64)
    if is_smooth:
        u = jnp.sin(theta)
    else:
        u = theta < jnp.pi

    # }}}

    # {{{ compute smoothness indicator

    from pyshocks.weno import weno_js_smoothness

    beta = weno_js_smoothness(u, a, b)

    alpha = scheme.d / (scheme.eps + beta) ** 2
    omega = alpha / jnp.sum(alpha, axis=0, keepdims=True)

    # }}}

    # {{{ check smoothness

    # NOTE: scheme.d are the "ideal" coefficients, so we're comparing how
    # close we are to those

    error = jnp.max(jnp.abs(omega[:, m] - scheme.d))
    logger.info("error[%s, %s]: %.8e", type(scheme).__name__, is_smooth, error)

    if is_smooth:
        assert error < 0.1
    else:
        assert error > 0.1

    # }}}


# }}}


# {{{


def _pyweno_reconstruct(
    u: jnp.ndarray, order: int, side: str
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    import pyweno

    ul, sl = pyweno.weno.reconstruct(
        jax.device_get(u), order, side, return_smoothness=True
    )

    return jax.device_put(ul), jax.device_put(sl)


@pytest.mark.parametrize(
    ("scheme", "order", "n"),
    [
        # NOTE: pyweno only seems to support order >= 5
        # (WENOJS32(), 3, 256),
        (WENOJS53(), 5, 512),
    ],
)
def test_weno_reference(
    scheme: WENOJS, order: int, n: int, visualize: bool = False
) -> None:
    """Compares our weno reconstruction to PyWENO"""
    pytest.importorskip("pyweno")

    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            visualize = False

    # {{{ reference values

    from pyshocks import make_uniform_grid

    grid = make_uniform_grid(a=0, b=2.0 * jnp.pi, n=n, nghosts=scheme.stencil_width)

    from pyshocks import Quadrature, cell_average

    quad = Quadrature(grid=grid, order=order)
    u = cell_average(quad, jnp.sin)

    uhost = u.copy()
    ul, sl = _pyweno_reconstruct(uhost, order, "left")
    ur, sr = _pyweno_reconstruct(uhost, order, "right")

    # }}}

    # {{{ compare

    from pyshocks import rnorm

    from pyshocks.weno import weno_js_smoothness

    betar = weno_js_smoothness(u[::-1], scheme.a, scheme.b)[:, ::-1].T
    betal = weno_js_smoothness(u, scheme.a, scheme.b).T

    errorl = rnorm(grid, sl, betal)
    errorr = rnorm(grid, sr, betar)
    logger.info("error smoothness: left %.5e right %.5e", errorl, errorr)
    assert errorl < 1.0e-5 and errorr < 1.0e-8

    from pyshocks.weno import reconstruct

    urhat = reconstruct(grid, scheme, u)
    ulhat = reconstruct(grid, scheme, u[::-1])[::-1]

    errorl = rnorm(grid, ul, ulhat)
    errorr = rnorm(grid, ur, urhat)
    logger.info("error reconstruct: left %.5e right %.5e", errorl, errorr)
    assert errorl < 1.0e-12 and errorr < 1.0e-12

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
