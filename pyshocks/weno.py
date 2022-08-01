# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autofunction:: weno_js_32_coefficients
.. autofunction:: weno_js_53_coefficients
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np


# {{{ WENOJS


def weno_js_32_coefficients() -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    # smoothness indicator coefficients
    a = jnp.array([1.0], dtype=jnp.float64)  # type: ignore[no-untyped-call]
    b = jnp.array(  # type: ignore[no-untyped-call]
        [
            [[0.0, -1.0, 1.0]],
            [[-1.0, 1.0, 0.0]],
        ],
        dtype=jnp.float64,
    )

    # stencil coefficients
    c = jnp.array(  # type: ignore[no-untyped-call]
        [[0.0, 3.0 / 2.0, -1.0 / 2.0], [1.0 / 2.0, 1.0 / 2.0, 0.0]], dtype=jnp.float64
    )

    # weights coefficients
    d = jnp.array(  # type: ignore[no-untyped-call]
        [[1.0 / 3.0, 2.0 / 3.0]], dtype=jnp.float64
    ).T

    return a, b, c, d


def weno_js_53_coefficients() -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    # NOTE: the arrays here are slightly modified from [Shu2009] by
    # * zero padding to the full stencil length
    # * flipping the stencils
    # so that they can be directly used with jnp.convolve for vectorization

    # equation 2.17 [Shu2009]
    a = jnp.array(  # type: ignore[no-untyped-call]
        [13.0 / 12.0, 1.0 / 4.0], dtype=np.float64
    )
    b = jnp.array(  # type: ignore[no-untyped-call]
        [
            [[0.0, 0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 3.0, -4.0, 1.0]],
            [[0.0, 1.0, -2.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0]],
            [[1.0, -2.0, 1.0, 0.0, 0.0], [1.0, -4.0, 3.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    # equation 2.11, 2.12, 2.13 [Shu2009]
    c = jnp.array(  # type: ignore[no-untyped-call]
        [
            [0.0, 0.0, 11.0 / 6.0, -7.0 / 6.0, 2.0 / 6.0],
            [0.0, 2.0 / 6.0, 5.0 / 6.0, -1.0 / 6.0, 0.0],
            [-1.0 / 6.0, 5.0 / 6.0, 2.0 / 6.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # equation 2.15 [Shu2009]
    d = jnp.array(  # type: ignore[no-untyped-call]
        [[1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0]], dtype=jnp.float64
    ).T

    return a, b, c, d


def weno_js_smoothness(u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            sum(
                a[j] * jnp.convolve(u, b[i, j, :], mode="same") ** 2
                for j in range(a.size)
            )
            for i in range(b.shape[0])
        ]
    )


def weno_js_reconstruct(u: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.convolve(u, c[i, :], mode="same") for i in range(c.shape[0])])


def weno_js_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    beta = weno_js_smoothness(u, a, b)
    alpha = d / (eps + beta) ** 2

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}


# {{{ ESWENO


def es_weno_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    beta = weno_js_smoothness(u, a, b)
    tau = jnp.pad((u[2:] - 2 * u[1:-1] + u[:-2]) ** 2, 1)  # type: ignore
    alpha = d * (1 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}
