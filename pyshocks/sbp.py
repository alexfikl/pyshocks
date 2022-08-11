# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r""""
Summation-by-Parts (SBP) Operators
----------------------------------

These routines help construct several SBP operators with appropriate boundary
conditions (see [Mattsson2012]_). First, we have that the first derivative
operator is given by

.. math::

    D_1 = P^{-1} Q,

where :math:`P` is a symmetric positive-definite matrix used to approximate
the inner product (and the corresponding norm) of

.. math::

    \int_a^b u v \,\mathrm{d}x \approx u \cdot P v

and :math:`Q` satisfies

.. math::

    Q + Q^T = B,

where :math:`B = \mathrm{diag}(-1, 0, \dots, 0, 1)`. This construction ensures
that discrete integration by parts formulae are respected. Finally, the
second-order derivative is given by [Mattsson2012]_

.. math::

    D_2 = P^{-1} (-M + BS) = P^{-1} (D_1^T P D_1 + B S)

where :math:`S` includes an approximation of the first-order derivative at
the boundary. Second-, fourth- and sixth-order operators are provided in
[Mattsson2012]_ and the references therein.

.. [Mattsson2012] K. Mattsson, *Summation by Parts Operators for Finite
    Difference Approximations of Second-Derivatives With Variable Coefficients*,
    Journal of Scientific Computing, Vol. 51, pp. 650--682, 2012,
    `DOI <http://dx.doi.org/10.1007/s10915-011-9525-z>`__.

.. autofunction:: get_sbp_boundary_matrix
.. autofunction:: get_sbp_21_norm_matrix
.. autofunction:: get_sbp_21_first_derivative_matrix
.. autofunction:: get_sbp_21_second_derivative_matrix

.. autofunction:: get_sbp_42_second_derivative_matrix
"""

from typing import Any, Optional

import jax.numpy as jnp


# {{{ SBP helpers


def get_sbp_boundary_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the boundary :math:`B` operator for an SBP discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    b = jnp.zeros((n, n), dtype=dtype)  # type: ignore[no-untyped-call]
    b = b.at[0, 0].set(-1)
    b = b.at[-1, -1].set(1)

    return b


def get_sbp_norm_matrix(pb: jnp.ndarray, n: int) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for an SBP discretization.

    :arg pb: boundary stencil.
    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    p = jnp.ones(n, dtype=pb.dtype)  # type: ignore[no-untyped-call]
    p = p.at[: pb.size].set(pb)
    p = p.at[-pb.size :].set(pb[::-1])

    return p


def get_sbp_banded_matrix(
    qi: jnp.ndarray,
    qb: Optional[jnp.ndarray],
    n: int,
) -> jnp.ndarray:
    """Construct the derivative :math:`Q` operator for an SBP discretization.

    :arg qi: interior stencil of the operator.
    :arg qb: boundary stencil of the operator.
    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    o = qi.size // 2
    q: jnp.ndarray = sum(
        qi[k] * jnp.eye(n, n, k=k - o, dtype=qi.dtype)  # type: ignore[no-untyped-call]
        for k in range(qi.size)
    )

    if qb is not None:
        n, m = qb.shape
        q = q.at[:n, :m].set(qb)
        q = q.at[-n:, -m:].set(-qb[::-1, ::-1])

    return q


# }}}


# {{{ SBP21


def get_sbp_21_norm_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for the SBP 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    return get_sbp_norm_matrix(
        jnp.array([0.5], dtype=dtype), n  # type: ignore[no-untyped-call]
    )


def get_sbp_21_first_derivative_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the derivative :math:`Q` operator for the SBP 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    return get_sbp_banded_matrix(
        jnp.array([-0.5, 0.0, 0.5], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[-0.5, 0.5]], dtype=dtype),  # type: ignore[no-untyped-call]
        n,
    )


def get_sbp_21_second_derivative_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the derivative :math:`M` operator for the 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return get_sbp_banded_matrix(
        jnp.array([-1, 2, -1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([1, -1], dtype=dtype),  # type: ignore[no-untyped-call]
        n,
    )


def get_sbp_21_second_derivative_s_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return get_sbp_banded_matrix(
        jnp.array([1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[-3 / 2, 2, -1 / 2]], dtype=dtype),  # type: ignore[no-untyped-call]
        n,
    )


def get_sbp_21_second_derivative_d22_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return get_sbp_banded_matrix(
        jnp.array([1, -2, 1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[1, -2, 1]], dtype=dtype),  # type: ignore[no-untyped-call]
        n,
    )


def get_sbp_21_second_derivative_c22_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return get_sbp_banded_matrix(
        jnp.array([1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[0]], dtype=dtype),  # type: ignore[no-untyped-call]
        n,
    )


# }}}


# {{{ SBP42


def get_sbp_42_norm_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for the SBP 4-2 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    return get_sbp_norm_matrix(
        jnp.array(  # type: ignore[no-untyped-call]
            [17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype=dtype
        ),
        n,
    )


def get_sbp_42_second_derivative_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the derivative :math:`M` operator for the 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2

    m = (
        jnp.eye(n, n, k=1, dtype=dtype)  # type: ignore[no-untyped-call]
        - 2 * jnp.eye(n, n, k=0, dtype=dtype)  # type: ignore[no-untyped-call]
        + jnp.eye(n, n, k=-1, dtype=dtype)  # type: ignore[no-untyped-call]
    )

    return m


# }}}
