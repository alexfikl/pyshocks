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

from typing import Any, Optional, Tuple

import jax.numpy as jnp

from pyshocks.grid import Grid


def weno_smoothness(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, *, mode: str = "same"
) -> jnp.ndarray:
    r"""Compute the smoothness coefficients for a WENO scheme.

    The coefficients must have the form

    .. math::

        \beta^r_m = \sum_j a_j \left(\sum u_{m - k} b_{r, j, k}\right)^2.

    :returns: the :math:`\beta_i` smoothness coefficient for each stencil around
        a cell :math:`m`. The shape of the returned array is
        ``(b.shape[0], u.size)``.
    """
    return jnp.stack(
        [
            sum(
                a[j] * jnp.convolve(u, b[i, j, :], mode=mode) ** 2
                for j in range(a.size)
            )
            for i in range(b.shape[0])
        ]
    )


def weno_reconstruct(
    u: jnp.ndarray, c: jnp.ndarray, *, mode: str = "same"
) -> jnp.ndarray:
    r"""Reconstruct the variable *u* at the cell faces for WENO-JS.

    The reconstruction has the form

    .. math::

        \hat{u}^r_m = \sum u_{m - k} c_{r, k}.

    :returns: the :math:`\hat{u}_i` reconstruction, for each stencil in the
        scheme. The returned array has a shape of ``(c.shape[0], u.size)``.
    """
    return jnp.stack([jnp.convolve(u, c[i, :], mode=mode) for i in range(c.shape[0])])


# {{{ WENOJS


def weno_js_32_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    THe general setup is more akin to the description provided in [Shu1998]_.

    .. [Shu1998] C.-W. Shu, *Essentially Non-Oscillatory and Weighted Essentially
        Non-Oscillatory Schemes for Hyperbolic Conservation Laws*,
        Lecture Notes in Mathematics, pp. 325--432, 1998,
        `DOI <http://dx.doi.org/10.1007/bfb0096355>`.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(1,)``, :math:`b`
        is of shape ``(2, 3, 1)``, :math:`c` is of shape ``(2, 3)`` and
        :math:`d` is of shape ``(1, 2)``.
    """

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # NOTE: the arrays here are slightly modified from [Shu2009] by
    # * zero padding to the full stencil length
    # * flipping the stencils
    # so that they can be directly used with jnp.convolve for vectorization

    # smoothness indicator coefficients ([Shu1998] Equation 2.62)
    a = jnp.array([1.0], dtype=dtype)  # type: ignore[no-untyped-call]
    b = jnp.array(  # type: ignore[no-untyped-call]
        [
            # i + 1, i, i - 1
            [[0.0, 1.0, -1.0]],
            [[1.0, -1.0, 0.0]],
        ],
        dtype=dtype,
    )

    # stencil coefficients ([Shu1998] Table 2.1)
    c = jnp.array(  # type: ignore[no-untyped-call]
        [[0.0, 3.0 / 2.0, -1.0 / 2.0], [1.0 / 2.0, 1.0 / 2.0, 0.0]], dtype=dtype
    )

    # weights coefficients ([Shu1998] Equation 2.54)
    d = jnp.array(  # type: ignore[no-untyped-call]
        [[1.0 / 3.0, 2.0 / 3.0]], dtype=dtype
    ).T

    return a, b, c, d


def weno_js_53_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Initialize the coefficients of the fifth-order WENO-JS scheme.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(2,)``, :math:`b`
        is of shape ``(3, 4, 2)``, :math:`c` is of shape ``(3, 5)`` and
        :math:`d` is of shape ``(1, 3)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # smoothness indicator (Equation 2.17 [Shu2009])
    a = jnp.array(  # type: ignore[no-untyped-call]
        [13.0 / 12.0, 1.0 / 4.0], dtype=dtype
    )
    b = jnp.array(  # type: ignore[no-untyped-call]
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [[0.0, 0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 3.0, -4.0, 1.0]],
            [[0.0, 1.0, -2.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0]],
            [[1.0, -2.0, 1.0, 0.0, 0.0], [1.0, -4.0, 3.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )

    # stencil coefficients (Equation 2.11, 2.12, 2.13 [Shu2009])
    c = jnp.array(  # type: ignore[no-untyped-call]
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [0.0, 0.0, 11.0 / 6.0, -7.0 / 6.0, 2.0 / 6.0],
            [0.0, 2.0 / 6.0, 5.0 / 6.0, -1.0 / 6.0, 0.0],
            [-1.0 / 6.0, 5.0 / 6.0, 2.0 / 6.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    # weights coefficients (Equation 2.15 [Shu2009])
    d = jnp.array(  # type: ignore[no-untyped-call]
        [[1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0]], dtype=dtype
    ).T

    return a, b, c, d


def weno_js_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    r"""Compute the standard WENO-JS weights.

    :returns: the weights :math:`\omega_i` for each stencil reconstruction
        from :func:`weno_js_reconstruct`.
    """
    beta = weno_smoothness(u, a, b)
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
    beta = weno_smoothness(u, a, b)

    # NOTE: see Equations 21-22 in [Yamaleev2009]
    tau = jnp.pad((u[2:] - 2 * u[1:-1] + u[:-2]) ** 2, 1)  # type: ignore
    alpha = d * (1 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}


# {{{ SSWENO


def ss_weno_parameters(grid: Grid, u0: jnp.ndarray) -> float:
    """Estimate the SSWENO parameters from the grid and initial condition.

    :arg u0: initial condition evaluated at given points.
    :returns: ``eps``, as given by Equation 75 in [Fisher2011]_.
    """

    # FIXME: eps should also contain the derivative of u0 away from discontinuities,
    # but not sure how to compute that in a vaguely accurate way. Also, we mostly
    # look at piecewise constant shock solutions where du0/dx ~ 0, so this
    # should be sufficient for the moment.

    i = grid.i_
    return float(jnp.sum(grid.dx[i] * jnp.abs(u0[i]))) * grid.dx_min**4


def ss_weno_242_weights(
    u: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, d: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    beta = weno_smoothness(u, a, b)
    tau = jnp.pad(  # type: ignore[no-untyped-call]
        (u[3:] - 3 * u[2:-1] + 3 * u[1:-2] - u[:-3]) ** 2, (1, 2)
    )
    alpha = d * (1.0 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# {{{ reconstruction


def ss_weno_242_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize coefficients for the fourth-order WENO scheme of [Fisher2013].

    The actual implementation details of the scheme are given in [Fisher2011]_.

    .. [Fisher2011] T. C. Fisher, M. H. Carpenter, N. K. Yamaleev, S. H. Frankel,
        *Boundary Closures for Fourth-Order Energy Stable Weighted Essentially
        Non-Oscillatory Finite-Difference Schemes*,
        Journal of Computational Physics, Vol. 230, pp. 3727-3752, 2011,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2011.01.043>`__.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(1,)``, :math:`b`
        is of shape ``(3, 4, 1)``, :math:`c` is of shape ``(3, 5)`` and
        :math:`d` is of shape ``(1, 3)``.
    """

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # smoothness indicator coefficients ([Fisher2011] Equation 71)
    a = jnp.array([1.0], dtype=dtype)  # type: ignore[no-untyped-call]
    b = jnp.array(  # type: ignore[no-untyped-call]
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [[0.0, 0.0, 1.0, -1.0, 0.0]],  # L
            [[0.0, 1.0, -1.0, 0.0, 0.0]],  # C
            [[1.0, -1.0, 0.0, 0.0, 0.0]],  # R
        ],
        dtype=dtype,
    )

    # stencil coefficients ([Fisher2011] Equation 77)
    c = jnp.array(  # type: ignore[no-untyped-call]
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [0.0, 0.0, -1.0 / 2.0, 3.0 / 2.0, 0.0],  # L
            [0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0],  # C
            [3.0 / 2.0, -1.0 / 2.0, 0.0, 0.0, 0.0],  # R
        ],
        dtype=dtype,
    )

    # weights coefficients ([Fisher2011] Equation 78)
    d = jnp.array(  # type: ignore[no-untyped-call]
        [[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]], dtype=dtype
    ).T

    return a, b, c, d


def ss_weno_242_bounary_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # boundary stencils ([Fisher2011] Equation 77)
    c = jnp.array(  # type: ignore[no-untyped-call]
        [
            [
                [0, 0],
                [0, 0],
                [-7 / 12, 19 / 12],
                [-23 / 48, 71 / 48],
                [-25 / 48, 73 / 48],
                [-5 / 12, 17 / 12],
                [-31 / 48, 79 / 48],
                [0, 0],
            ],
            [
                [1, 0],
                [31 / 48, 17 / 48],
                [5 / 12, 7 / 12],
                [25 / 48, 23 / 48],
                [23 / 48, 25 / 48],
                [7 / 12, 5 / 12],
                [17 / 48, 31 / 48],
                [0, 1],
            ],
            [
                [0, 0],
                [79 / 48, -31 / 48],
                [17 / 12, -5 / 12],
                [73 / 48, -25 / 48],
                [71 / 48, -23 / 48],
                [19 / 12, -7 / 12],
                [0, 0],
                [0, 0],
            ],
        ],
        dtype=dtype,
    )

    # weights coefficients ([Fisher2011] Equation 78)
    d = jnp.array(  # type: ignore[no-untyped-call]
        [
            [0, 0, 1, 0, 0],
            [0, 0, 24 / 31, 1013 / 4898, 3 / 158],
            [0, 11 / 56, 51 / 70, 3 / 40, 0],
            [3 / 142, 357 / 3266, 408 / 575, 4 / 25, 0],
        ]
    )

    return c, d


# }}}


# {{{ operators


def ss_weno_242_operator_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # [Fisher2013] Appendix A, Equation A.2 interior
    qi = jnp.array(  # type: ignore[no-untyped-call]
        [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]
    )

    # [Fisher2013] Appendix A, Equation A.5 interior
    hi = jnp.array(  # type: ignore[no-untyped-call]
        [-1 / 12, 7 / 12, 7 / 12, -1 / 12, 0],
        dtype=dtype,
    )

    return qi, hi


def ss_weno_242_operator_boundary_coefficients(
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Fisher2013] Appendix A, Equation A.1
    p = jnp.array(  # type: ignore[no-untyped-call]
        [17 / 48, 59 / 48, 43 / 48, 49 / 48],
        dtype=dtype,
    )

    # [Fisher2013] Appendix A, Equation A.2 boundary
    qb = jnp.array(  # type: ignore[no-untyped-call]
        [
            [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
            [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
            [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
            [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
        ],
        dtype=dtype,
    )

    # [Fisher2013] Appendix A, Equation A.5 boundary
    hb = jnp.array(  # type: ignore[no-untyped-call]
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
            [-11 / 96, 59 / 96, 17 / 32, -1 / 32, 0.0, 0.0],
            [-1 / 32, 0.0, 17 / 32, 7 / 12, -1 / 12, 0.0],
        ],
        dtype=dtype,
    )

    return p, qb, hb


def ss_weno_norm_matrix(p: jnp.ndarray, n: int) -> jnp.ndarray:
    assert n > 2 * p.size

    # [Fisher2013] Appendix A, Equation A.1
    P = jnp.concatenate(  # noqa: N806
        [
            p,
            jnp.ones(n - 2 * p.size, dtype=p.dtype),  # type: ignore[no-untyped-call]
            p[::-1],
        ]
    )
    assert P.shape == (n,)
    return P


def ss_weno_derivative_matrix(qi: jnp.ndarray, qb: jnp.ndarray, n: int) -> jnp.ndarray:
    # [Fisher2013] Appendix A.1, Equation A.4
    m = qi.size // 2
    Q: jnp.ndarray = sum(  # noqa: N806
        qi[k] * jnp.eye(n, n, k=k - m, dtype=qi.dtype)  # type: ignore[no-untyped-call]
        for k in range(qi.size)
    )

    # [Fisher2013] Appendix A.1, Equation A.2
    n, m = qb.shape
    Q = Q.at[:n, :m].set(qb)  # noqa: N806
    Q = Q.at[-n:, -m:].set(-qb[::-1, ::-1])  # noqa: N806

    return Q


def ss_weno_interpolation_matrix(
    hi: jnp.ndarray, hb: jnp.ndarray, n: int
) -> jnp.ndarray:
    # [Fisher2013] Appendix A.1, Equation A.4
    m = hi.size // 2
    H: jnp.ndarray = sum(  # noqa: N806
        hi[k] * jnp.eye(n, n, k=k - m, dtype=hi.dtype)  # type: ignore[no-untyped-call]
        for k in range(hi.size)
    )

    # [Fisher2013] Appendix A.1, Equation A.2
    m, p = hb.shape
    H = H.at[:m, :p].set(hb)  # noqa: N806
    H = H.at[-m:, -p:].set(hb[::-1, ::-1])  # noqa: N806

    return H


def ss_weno_circulant(q: jnp.ndarray, n: int) -> jnp.ndarray:
    m = q.size // 2
    return sum(
        jnp.roll(q[i] * jnp.eye(n, n), i - m, axis=1)  # type: ignore
        for i in range(q.size)
    )


# }}}

# }}}
