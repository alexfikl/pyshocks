# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. autofunction:: weno_js_32_coefficients
.. autofunction:: weno_js_53_coefficients
.. autofunction:: weno_js_smoothness
.. autofunction:: weno_js_reconstruct
.. autofunction:: weno_js_weights

.. autofunction:: es_weno_weights
.. autofunction:: es_weno_parameters
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from pyshocks.grid import Grid


# {{{ WENOJS


def weno_js_32_coefficients() -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    r"""Initialize the coefficients of the third-order WENO-JS scheme.

    Let the coefficients be :math:`a, b, c` and `d`. Then, the smoothness
    indicators are given by

    .. math::

        \beta_{i, m} = \sum_j a_j \left(\sum_k u_{m, k} b_{i, j, k}\right)^2,

    where :math:`u_{m, k}` represents the stencil around :math:`m`. See
    Equations 3.2-3.4 in [JiangShu1996] for an example of the fifth-order
    scheme. The value of the function is reconstructed using the :math:`c`
    coefficients

    .. math::

        \hat{u}_{i, m} = \sum u_{m, j} c_{i, j}.

    Finally, the weights are given by::

    .. math::

        \alpha_{i, m} = \frac{d_i}{(\beta_{i, m} + \epsilon)^2}.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(1,)``, :math:`b`
        is of shape ``(2, 3, 1)``, :math:`c` is of shape ``(2, 3)`` and
        :math:`d` is of shape ``(1, 2)``.
    """

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
    r"""Initialize the coefficients of the fifth-order WENO-JS scheme.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(2,)``, :math:`b`
        is of shape ``(3, 4, 2)``, :math:`c` is of shape ``(3, 5)`` and
        :math:`d` is of shape ``(1, 3)``.
    """
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
    r"""Compute the smoothness coefficients for WENO-JS.

    :returns: the :math:`\beta_i` smoothness coefficient for each stencil around
        a cell :math:`m`. The shape of the returned array is
        ``(b.shape[0], u.size)``.
    """
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
    r"""Reconstruct the variable *u* at the cell faces for WENO-JS.

    :returns: the :math:`\hat{u}_i` reconstruction, for each stencil in the
        scheme. The returned array has a shape of ``(c.shape[0], u.size)``.
    """
    return jnp.stack([jnp.convolve(u, c[i, :], mode="same") for i in range(c.shape[0])])


def weno_js_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    r"""Compute the standard WENO-JS weights.

    :returns: the weights :math:`\omega_i` for each stencil reconstruction
        from :func:`weno_js_reconstruct`.
    """
    beta = weno_js_smoothness(u, a, b)
    alpha = d / (eps + beta) ** 2

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}


# {{{ ESWENO


def es_weno_parameters(grid: Grid, u0: jnp.ndarray) -> Tuple[float, float]:
    """Estimate the ESWENO32 parameters from the grid and initial condition.

    :returns: ``(eps, delta)``, where ``eps`` is given by Equation 65
        and ``delta`` is given by Equation 66 in [Yamaleev2009]_.
    """
    # FIXME: eps should also contain the derivative of u0 away from discontinuities,
    # but not sure how to compute that in a vaguely accurate way. Also, we mostly
    # look at piecewise constant shock solutions where du0/dx ~ 0, so this
    # should be sufficient for the moment.

    # NOTE: as in Yamaleev2009, we're using the L1 norm for u0
    i = grid.i_
    eps = jnp.sum(grid.dx[i] * jnp.abs(u0[i])) * grid.dx_min**2
    delta = grid.dx_min**2

    return eps, delta


def es_weno_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    r"""Compute the ESWENO weights.

    :returns: the weights :math:`\omega_i` for each stencil reconstruction
        from :func:`weno_js_reconstruct`.
    """
    beta = weno_js_smoothness(u, a, b)

    # NOTE: see Equations 21-22 in [Yamaleev2009]
    tau = jnp.pad((u[2:] - 2 * u[1:-1] + u[:-2]) ** 2, 1)  # type: ignore
    alpha = d * (1 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}
