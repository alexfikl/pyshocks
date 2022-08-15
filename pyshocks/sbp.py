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


Interface
~~~~~~~~~

An SBP operator must implement the following interface, which relies on the
:func:`~functools.singledispatch` functionality.

.. autoclass:: SBPOperator

.. autofunction:: sbp_norm_matrix
.. autofunction:: sbp_first_derivative_matrix
.. autofunction:: sbp_second_derivative_matrix

The following helper functions are provided as well.

.. autofunction:: make_sbp_boundary_matrix
.. autofunction:: make_sbp_norm_matrix
.. autofunction:: make_sbp_banded_matrix

SBP 2-1
~~~~~~~

.. autoclass:: SBP21

.. autofunction:: make_sbp_21_norm_matrix
.. autofunction:: make_sbp_21_first_derivative_q_matrix

.. autofunction:: make_sbp_21_second_derivative_q_matrix
.. autofunction:: make_sbp_21_second_derivative_s_matrix
.. autofunction:: make_sbp_21_second_derivative_d22_matrix
.. autofunction:: make_sbp_21_second_derivative_c22_matrix

SBP 4-2
~~~~~~~

.. autoclass:: SBP42
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Dict, Optional, Tuple, Type

import jax.numpy as jnp

from pyshocks import UniformGrid


# {{{ SBP class


@dataclass(frozen=True)
class SBPOperator:
    """Generic family of SBP operators.

    .. attribute:: order

        Interior order of the SBP operator.

    .. attribute:: boundary_order

        Boundary order of the SBP operator. This is not valid for periodic
        boundaries.
    """

    @property
    def order(self) -> int:
        raise NotImplementedError

    @property
    def boundary_order(self) -> int:
        raise NotImplementedError


@singledispatch
def sbp_norm_matrix(op: SBPOperator, grid: UniformGrid) -> jnp.ndarray:
    """Construct the :math:`P` operator for and SBP approximation."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_first_derivative_matrix(op: SBPOperator, grid: UniformGrid) -> jnp.ndarray:
    """Construct a first derivative :math:`D` operator satisfying the SBP property."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_second_derivative_matrix(
    op: SBPOperator, grid: UniformGrid, b: jnp.ndarray
) -> jnp.ndarray:
    """Construct a second derivative :math:`D_2` operator satisfying the
    SBP properties.
    """
    raise NotImplementedError(type(op).__name__)


# }}}


# {{{ SBP helpers


def make_sbp_boundary_matrix(
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


def make_sbp_norm_matrix(n: int, pb: jnp.ndarray) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for an SBP discretization.

    :arg pb: boundary stencil.
    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    p = jnp.ones(n, dtype=pb.dtype)  # type: ignore[no-untyped-call]
    p = p.at[: pb.size].set(pb)
    p = p.at[-pb.size :].set(pb[::-1])

    return p


def make_sbp_banded_matrix(
    n: int,
    qi: jnp.ndarray,
    qb_l: Optional[jnp.ndarray] = None,
    qb_r: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Construct the derivative :math:`Q` operator for an SBP discretization.

    :arg qi: interior stencil of the operator. This should be an array of
        shape ``(n_i,)``.
    :arg qb: boundary stencil of the operator. This should be an array of
        shape ``(n_r, n_b)``, where :math:`n_r` denotes the number of boundary
        points and :math:`n_b` is the stencil width.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    o = qi.size // 2
    q: jnp.ndarray = sum(
        qi[k] * jnp.eye(n, n, k=k - o, dtype=qi.dtype)  # type: ignore[no-untyped-call]
        for k in range(qi.size)
    )

    if qb_l is not None:
        n, m = qb_l.shape
        q = q.at[:n, :m].set(qb_l)

        if qb_r is None:
            qb_r = -qb_l[::-1, ::-1]

    if qb_r is not None:
        n, m = qb_r.shape
        q = q.at[-n:, -m:].set(qb_r)

    return q


# }}}


# {{{ SBP21


@dataclass(frozen=True)
class SBP21(SBPOperator):
    """An SBP operator the is second-order accurate in the interior and
    first-order accurate at the boundary.

    For details, see Appendix A.1 in [Mattsson2012]_
    """

    @property
    def order(self) -> int:
        return 2

    @property
    def boundary_order(self) -> int:
        return 1


@sbp_norm_matrix.register(SBP21)
def _sbp_21_norm_matrix(op: SBP21, grid: UniformGrid) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)
    return grid.dx_min * make_sbp_21_norm_matrix(grid.x.size, dtype=grid.x.dtype)


@sbp_first_derivative_matrix.register(SBP21)
def _sbp_21_first_derivative_matrix(op: SBP21, grid: UniformGrid) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    Q = make_sbp_21_first_derivative_q_matrix(grid.x.size, dtype=grid.x.dtype)
    P = sbp_norm_matrix(op, grid)

    return jnp.diag(1.0 / P) @ Q  # type: ignore[no-untyped-call]


@sbp_second_derivative_matrix.register(SBP21)
def _sbp_21_second_derivative_matrix(
    op: SBP21, grid: UniformGrid, b: jnp.ndarray
) -> jnp.ndarray:
    from numbers import Number

    if isinstance(b, jnp.ndarray):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, b)  # type: ignore[no-untyped-call]
    else:
        raise TypeError(f"unknown diffusivity coefficient: '{type(b).__name__}'")

    assert b.shape == (grid.x.size,)
    assert isinstance(grid, UniformGrid)

    # NOTE: See [Mattsson2012] for details
    n = grid.x.size
    dx = grid.dx_min
    dtype = grid.x.dtype

    # get lower order operators
    P = sbp_norm_matrix(op, grid)
    D = sbp_first_derivative_matrix(op, grid)

    invP = jnp.diag(1 / P)  # type: ignore[no-untyped-call]
    P = jnp.diag(P)  # type: ignore[no-untyped-call]

    # get R matrix
    (B22,) = make_sbp_21_second_derivative_b_matrices(b, dtype=dtype)
    (D22,) = make_sbp_21_second_derivative_d_matrices(n, dtype=dtype)
    (C22,) = make_sbp_21_second_derivative_c_matrices(n, dtype=dtype)
    R = dx**3 / 4 * D22.T @ C22 @ B22 @ D22

    # get Bbar matrix
    B = B22
    Bbar = jnp.zeros_like(B)  # type: ignore[no-untyped-call]
    Bbar = Bbar.at[0, 0].set(-B[0, 0])
    Bbar = Bbar.at[-1, -1].set(B[-1, -1])

    # get S matrix
    S = make_sbp_21_second_derivative_s_matrix(n, dtype=dtype)

    # put it all together
    M = D.T @ P @ B @ D + R

    return invP @ (-M + Bbar @ S)


def make_sbp_21_norm_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for the SBP 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    return make_sbp_norm_matrix(
        n, jnp.array([0.5], dtype=dtype)  # type: ignore[no-untyped-call]
    )


def make_sbp_21_first_derivative_q_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the derivative :math:`Q` operator for the SBP 2-1 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    return make_sbp_banded_matrix(
        n,
        jnp.array([-0.5, 0.0, 0.5], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[-0.5, 0.5]], dtype=dtype),  # type: ignore[no-untyped-call]
    )


def make_sbp_21_second_derivative_s_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return make_sbp_banded_matrix(
        n,
        jnp.array([1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[-3 / 2, 2, -1 / 2]], dtype=dtype),  # type: ignore[no-untyped-call]
    )


def make_sbp_21_second_derivative_b_matrices(
    b: jnp.ndarray, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    return (jnp.diag(b),)  # type: ignore[no-untyped-call]


def make_sbp_21_second_derivative_d_matrices(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    D22 = make_sbp_banded_matrix(
        n,
        jnp.array([1, -2, 1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[1, -2, 1]], dtype=dtype),  # type: ignore[no-untyped-call]
    )

    return (D22,)


def make_sbp_21_second_derivative_c_matrices(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    C22 = make_sbp_banded_matrix(
        n,
        jnp.array([1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array([[0]], dtype=dtype),  # type: ignore[no-untyped-call]
    )

    return (C22,)


# }}}


# {{{ SBP42


@dataclass(frozen=True)
class SBP42(SBPOperator):
    """An SBP operator the is fourth-order accurate in the interior and
    second-order accurate at the boundary.

    For details, see Appendix A.2 in [Mattsson2012]_
    """

    @property
    def order(self) -> int:
        return 4

    @property
    def boundary_order(self) -> int:
        return 2


@sbp_norm_matrix.register(SBP42)
def _sbp_42_norm_matrix(op: SBP42, grid: UniformGrid) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)
    return grid.dx_min * make_sbp_42_norm_matrix(grid.x.size, dtype=grid.x.dtype)


@sbp_first_derivative_matrix.register(SBP42)
def _sbp_42_first_derivative_matrix(op: SBP42, grid: UniformGrid) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    Q = make_sbp_42_first_derivative_q_matrix(grid.x.size, dtype=grid.x.dtype)
    P = sbp_norm_matrix(op, grid)

    return jnp.diag(1.0 / P) @ Q  # type: ignore[no-untyped-call]


@sbp_second_derivative_matrix.register(SBP42)
def _sbp_42_second_derivative_matrix(
    op: SBP42, grid: UniformGrid, b: jnp.ndarray
) -> jnp.ndarray:
    from numbers import Number

    if isinstance(b, jnp.ndarray):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, b)  # type: ignore[no-untyped-call]
    else:
        raise TypeError(f"unknown diffusivity coefficient: '{type(b).__name__}'")

    assert b.shape == (grid.x.size,)
    assert isinstance(grid, UniformGrid)

    # NOTE: See [Mattsson2012] for details
    n = grid.x.size
    dx = grid.dx_min
    dtype = grid.x.dtype

    # get lower order operators
    P = sbp_norm_matrix(op, grid)
    D = sbp_first_derivative_matrix(op, grid)

    invP = jnp.diag(1 / P)  # type: ignore[no-untyped-call]
    P = jnp.diag(P)  # type: ignore[no-untyped-call]

    # get R matrix
    B34, B44 = make_sbp_42_second_derivative_b_matrices(b, dtype=dtype)
    D34, D44 = make_sbp_42_second_derivative_d_matrices(n, dtype=dtype)
    C34, C44 = make_sbp_42_second_derivative_c_matrices(n, dtype=dtype)
    R = dx**5 / 18 * D34.T @ C34 @ B34 @ D34 + dx**7 / 144 * D44.T @ C44 @ B44 @ D44

    # get Bbar matrix
    B = B44
    Bbar = jnp.zeros_like(B)  # type: ignore[no-untyped-call]
    Bbar = Bbar.at[0, 0].set(-B[0, 0])
    Bbar = Bbar.at[-1, -1].set(B[-1, -1])

    # get S matrix
    S = make_sbp_42_second_derivative_s_matrix(n, dtype=dtype)

    # put it all together
    M = D.T @ P @ B @ D + R

    return invP @ (-M + Bbar @ S)


def make_sbp_42_norm_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    """Construct the diagonal :math:`P` operator for the SBP 4-2 discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    return make_sbp_norm_matrix(
        n,
        jnp.array(  # type: ignore[no-untyped-call]
            [17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype=dtype
        ),
    )


def make_sbp_42_first_derivative_q_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Fisher2013] Appendix A, Equation A.1
    # [Fisher2013] Appendix A, Equation A.2 boundary
    return make_sbp_banded_matrix(
        n,
        jnp.array(  # type: ignore[no-untyped-call]
            [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype
        ),
        jnp.array(  # type: ignore[no-untyped-call]
            [
                [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
                [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
                [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
                [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
            ],
            dtype=dtype,
        ),
    )


def make_sbp_42_second_derivative_s_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    return make_sbp_banded_matrix(
        n,
        jnp.array([0], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array(  # type: ignore[no-untyped-call]
            [[-11 / 6, 3, -3 / 2, 1 / 3]], dtype=dtype
        ),
    )


def make_sbp_42_second_derivative_b_matrices(
    b: jnp.ndarray, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    B34 = jnp.pad((b[2:] + b[:-2]) / 2, 1)  # type: ignore[no-untyped-call]
    # TODO: [Mattsson2012] does not say what happens at the boundary points
    B34 = B34.at[0].set(b[0])
    B34 = B34.at[-1].set(b[-1])

    B34 = jnp.diag(B34)  # type: ignore[no-untyped-call]
    B44 = jnp.diag(b)  # type: ignore[no-untyped-call]

    return B34, B44


def make_sbp_42_second_derivative_d_matrices(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    D34 = make_sbp_banded_matrix(
        n,
        jnp.array([-1, 3, -3, 1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array(  # type: ignore[no-untyped-call]
            [
                [-1, 3, -3, 1, 0, 0],
                [-1, 3, -3, 1, 0, 0],
                [
                    -185893 / 301051,
                    79000249461 / 54642863857,
                    -33235054191 / 54642863857,
                    -36887526683 / 54642863857,
                    26183621850 / 54642863857,
                    -4386 / 181507,
                ],
            ],
            dtype=dtype,
        ),
    )

    d44 = jnp.array(  # type: ignore[no-untyped-call]
        [
            [-1, -4, 6, -4, 1],
            [-1, -4, 6, -4, 1],
            [-1, -4, 6, -4, 1],
        ],
        dtype=dtype,
    )
    D44 = make_sbp_banded_matrix(
        n,
        jnp.array([1, -4, 6, -4, 1], dtype=dtype),  # type: ignore[no-untyped-call]
        d44,
        d44,
    )

    return (D34, D44)


def make_sbp_42_second_derivative_c_matrices(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    C34 = make_sbp_banded_matrix(
        n,
        jnp.array([1], dtype=dtype),  # type: ignore[no-untyped-call]
        jnp.array(  # type: ignore[no-untyped-call]
            [[0, 0, 163928591571 / 53268010936, 189284 / 185893, 1]],
            dtype=dtype,
        ),
        jnp.array(  # type: ignore[no-untyped-call]
            [[1, 1189284 / 185893, 0, 63928591571 / 53268010936, 0, 0]],
            dtype=dtype,
        ),
    )

    c44 = jnp.array(  # type: ignore[no-untyped-call]
        [[0, 0, 1644330 / 301051, 156114 / 181507, 1]],
        dtype=dtype,
    )
    C44 = make_sbp_banded_matrix(
        n, jnp.array([1], dtype=dtype), c44, c44[::-1]  # type: ignore[no-untyped-call]
    )

    return C34, C44


# }}}


# {{{ make_operator_from_name

_OPERATORS: Dict[str, Type[SBPOperator]] = {
    "sbp21": SBP21,
    "sbp42": SBP42,
}


def operator_ids() -> Tuple[str, ...]:
    return tuple(_OPERATORS.keys())


def make_operator_from_name(name: str, **kwargs: Any) -> SBPOperator:
    """
    :arg name: name of the operator.
    :arg kwargs: additional arguments to pass to the operator. Any arguments
        that are not in the operator's fields are ignored.
    """

    cls = _OPERATORS.get(name)
    if cls is None:
        raise ValueError(f"scheme '{name}' not found; try one of {operator_ids()}")

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
