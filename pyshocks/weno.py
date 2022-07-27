# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autoclass:: WENOMixin
    :no-show-inheritance:

.. autoclass:: WENOJSMixin
    :no-show-inheritance:

.. autoclass:: WENOJS32Mixin
    :no-show-inheritance:
.. autoclass:: WENOJS53Mixin
    :no-show-inheritance:

.. autofunction:: weno_js_32_coefficients
.. autofunction:: weno_js_53_coefficients

.. autofunction:: reconstruct
"""
from functools import singledispatch
from typing import ClassVar, Tuple

import jax.numpy as jnp
import numpy as np

from pyshocks import Grid, UniformGrid


# {{{ WENO


class WENOMixin:
    """
    .. attribute:: order

        Minimum order of the WENO scheme.

    .. automethod:: set_coefficients
    """

    def set_coefficients(self) -> None:
        """Initialize WENO coefficients."""
        raise NotImplementedError


# }}}


# {{{ coefficients for WENOJS


class WENOJSMixin(WENOMixin):  # pylint: disable=abstract-method
    """
    .. attribute:: a
    .. attribute:: b
    .. attribute:: c
    .. attribute:: d

    .. attribute:: eps
    """

    a: ClassVar[jnp.ndarray]
    b: ClassVar[jnp.ndarray]
    c: ClassVar[jnp.ndarray]
    d: ClassVar[jnp.ndarray]

    eps: float


class WENOJS32Mixin(WENOJSMixin):
    def set_coefficients(self) -> None:
        a, b, c, d = weno_js_32_coefficients()

        # NOTE: hack to keep the class frozen
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)

    @property
    def order(self) -> int:
        return 2

    @property
    def stencil_width(self) -> int:
        return 2


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


class WENOJS53Mixin(WENOJSMixin):
    def set_coefficients(self) -> None:
        a, b, c, d = weno_js_53_coefficients()

        # NOTE: hack to keep the class frozen
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)

    @property
    def order(self) -> int:
        return 3

    @property
    def stencil_width(self) -> int:
        return 3


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


# }}}


# {{{ helpers


@singledispatch
def reconstruct(grid: Grid, scheme: WENOMixin, u: jnp.ndarray) -> jnp.ndarray:
    """Generic WENO-like reconstruction.

    :param grid: grid on which to perform the reconstruction.
    :param scheme: WENO scheme used for the reconstruction.
    :param u: cell-centered value to reconstruct.

    :param: face-based values reconstructed from *u*.
    """
    raise NotImplementedError(type(grid).__name__)


# }}}


# {{{ uniform grid reconstruction


def _weno_js_smoothness(u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack(
        [
            sum(
                a[j] * jnp.convolve(u, b[i, j, :], mode="same") ** 2
                for j in range(a.size)
            )
            for i in range(b.shape[0])
        ]
    )


def _weno_js_reconstruct(u: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.convolve(u, c[i, :], mode="same") for i in range(c.shape[0])])


@reconstruct.register(UniformGrid)
def _reconstruct_uniform_wenojs(
    grid: Grid, scheme: WENOJSMixin, u: jnp.ndarray
) -> jnp.ndarray:
    """WENO-JS reconstruction from the [JiangShu1996]_.

    .. [JiangShu1996] G.-S. Jiang, C.-W. Shu, *Efficient Implementation of
        Weighted ENO Schemes*,
        Journal of Computational Physics, Vol. 126, pp. 202--228, 1996,
        `DOI <http://dx.doi.org/10.1006/jcph.1996.0130>`__.
    """
    beta = _weno_js_smoothness(u, scheme.a, scheme.b)
    uhat = _weno_js_reconstruct(u, scheme.c)

    alpha = scheme.d / (scheme.eps + beta) ** 2
    omega = alpha / jnp.sum(alpha, axis=0, keepdims=True)

    return jnp.sum(omega * uhat, axis=0)


# }}}
