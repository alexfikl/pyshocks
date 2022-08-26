# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
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
^^^^^^^^^

An SBP operator must implement the following interface, which relies on the
:func:`~functools.singledispatch` functionality.

.. autoclass:: Stencil
.. autoclass:: SBPOperator

.. autofunction:: sbp_norm_matrix
.. autofunction:: sbp_first_derivative_matrix
.. autofunction:: sbp_second_derivative_matrix

.. autofunction:: apply_sbp_first_derivative
.. autofunction:: apply_sbp_second_derivative

The following helper functions are provided as well.

.. autofunction:: make_sbp_boundary_matrix
.. autofunction:: make_sbp_diagonal_matrix
.. autofunction:: make_sbp_circulant_matrix
.. autofunction:: make_sbp_banded_matrix

SBP 2-1
^^^^^^^

.. autoclass:: SBP21

.. autofunction:: make_sbp_21_norm_stencil
.. autofunction:: make_sbp_21_first_derivative_q_stencil

.. autofunction:: make_sbp_21_second_derivative_s_stencil
.. autofunction:: make_sbp_21_second_derivative_c_stencils
.. autofunction:: make_sbp_21_second_derivative_d_stencils

SBP 4-2
^^^^^^^

.. autoclass:: SBP42

.. autofunction:: make_sbp_42_norm_stencil
.. autofunction:: make_sbp_42_first_derivative_q_stencil

.. autofunction:: make_sbp_42_second_derivative_s_stencil
.. autofunction:: make_sbp_42_second_derivative_c_stencils
.. autofunction:: make_sbp_42_second_derivative_d_stencils

SBP 6-4
^^^^^^^

.. autoclass:: SBP64

.. autofunction:: make_sbp_64_norm_stencil
.. autofunction:: make_sbp_64_first_derivative_q_stencil
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Dict, Optional, Tuple, Type

import jax.numpy as jnp

from pyshocks.grid import UniformGrid
from pyshocks.schemes import BoundaryType


# {{{ SBP helpers


@dataclass(frozen=True)
class Stencil:
    """
    .. attribute:: int

        Interior stencil, as an array of shape ``(n_i,)``.

    .. attribute:: left

        Left boundary stencil, as an array of shape ``(n_l, m_l)``.

    .. attribute:: right

        Left boundary stencil, as an array of shape ``(n_r, m_r)``. If not
        provided and :attr:`left` is provided, the right boundary stencil
        is taken to be ``-left[::-1, ::-1]``.

    .. attribute:: is_diagonal

        If *True*, the resulting operator is assumed to be diagonal only.
        This is used to return an array of shape ``(n,)`` instead of a
        matrix of shape ``(n, m)`` in certain functions.

    .. attribute:: dtype

        The :class:`~numpy.dtype` of the stencils.
    """

    int: jnp.ndarray
    left: Optional[jnp.ndarray]
    right: Optional[jnp.ndarray]

    is_diagonal: bool = False

    def __post_init__(self) -> None:
        if self.left is not None and self.right is None:
            if self.left.ndim == 1:
                right = self.left[::-1]
            else:
                right = -self.left[::-1, ::-1]

            object.__setattr__(self, "right", right)

        if self.is_diagonal:
            assert self.int.shape == ()

    @property
    def dtype(self) -> "jnp.dtype[Any]":
        return jnp.dtype(self.int.dtype)


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


def make_sbp_diagonal_matrix(n: int, s: Stencil, *, weight: float = 1.0) -> jnp.ndarray:
    """Construct a diagonal matrix with a given boundary stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    assert s.is_diagonal
    mat = jnp.full(n, s.int, dtype=s.dtype)  # type: ignore[no-untyped-call]

    if s.left is not None:
        mat = mat.at[: s.left.size].set(s.left)

    if s.right is not None:
        mat = mat.at[-s.right.size :].set(s.right)

    return weight * mat


def make_sbp_circulant_matrix(
    n: int, s: Stencil, *, weight: float = 1.0
) -> jnp.ndarray:
    """Construct a circulat matrix with a given interior stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    m = s.int.size
    return sum(
        jnp.roll(
            weight * s.int[i] * jnp.eye(n, n, dtype=s.dtype),  # type: ignore
            i - m // 2,
            axis=1,
        )
        for i in range(m)
    )


def make_sbp_banded_matrix(n: int, s: Stencil, *, weight: float = 1.0) -> jnp.ndarray:
    """Construct a banded matrix with a given boundary stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    o = s.int.size // 2
    mat: jnp.ndarray = sum(
        weight * s.int[k] * jnp.eye(n, n, k=k - o, dtype=s.dtype)  # type: ignore
        for k in range(s.int.size)
    )

    if s.left is not None:
        n, m = s.left.shape
        mat = mat.at[:n, :m].set(weight * s.left)

    if s.right is not None:
        n, m = s.right.shape
        mat = mat.at[-n:, -m:].set(weight * s.right)

    return mat


def make_sbp_matrix_from_stencil(
    bc: BoundaryType, n: int, s: Stencil, *, weight: float = 1.0
) -> jnp.ndarray:
    if s.is_diagonal:
        mat = make_sbp_diagonal_matrix(n, s, weight=weight)
    else:
        if bc == BoundaryType.Periodic:
            mat = make_sbp_circulant_matrix(n, s, weight=weight)
        else:
            mat = make_sbp_banded_matrix(n, s, weight=weight)

    return mat


# }}}


# {{{ SBP class


@dataclass(frozen=True)
class SBPOperator:
    """Generic family of SBP operators.

    .. attribute:: ids

        Some form of unique string identifier for the operator.

    .. attribute:: order

        Interior order of the SBP operator.

    .. attribute:: boundary_order

        Boundary order of the SBP operator. This is not valid for periodic
        boundaries.
    """

    @property
    def ids(self) -> str:
        return f"{self.order}{self.boundary_order}"

    @property
    def order(self) -> int:
        raise NotImplementedError

    @property
    def boundary_order(self) -> int:
        raise NotImplementedError


def sbp_matrix_from_name(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, name: str
) -> jnp.ndarray:
    n = grid.n
    dtype = grid.dtype

    if name == "P":
        func = globals()[f"make_sbp_{op.ids}_norm_stencil"]
        return make_sbp_diagonal_matrix(n, func(bc, dtype=dtype))
    if name == "Q":
        func = globals()[f"make_sbp_{op.ids}_first_derivative_q_stencil"]
        return make_sbp_matrix_from_stencil(bc, n, func(bc, dtype=dtype))
    if name == "S":
        func = globals()[f"make_sbp_{op.ids}_second_derivative_s_stencil"]
        return make_sbp_matrix_from_stencil(
            bc, n, func(bc, dtype=dtype), weight=1.0 / grid.dx_min
        )
    if name == "R":
        func = globals()[f"make_sbp_{op.ids}_second_derivative_r_matrix"]
        return func(bc, jnp.ones_like(grid.x), dx=grid.dx_min)  # type: ignore

    raise ValueError(f"unknown SBP matrix name: '{name}'")


def make_sbp_mattsson2012_second_derivative(
    op: SBPOperator,
    grid: UniformGrid,
    bc: BoundaryType,
    b: jnp.ndarray,
    R: jnp.ndarray,
    S: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)
    n = grid.n
    dtype = grid.dtype

    # get lower order operators
    P = sbp_norm_matrix(op, grid, bc)
    D = sbp_first_derivative_matrix(op, grid, bc)

    invP = jnp.diag(1 / P)  # type: ignore[no-untyped-call]
    P = jnp.diag(P)  # type: ignore[no-untyped-call]

    # get Bbar matrix ([Mattsson2012] Definition 2.3)
    if bc != BoundaryType.Periodic:
        Bbar = jnp.zeros((n, n), dtype=dtype)  # type: ignore[no-untyped-call]
        Bbar = Bbar.at[0, 0].set(-b[0])
        Bbar = Bbar.at[-1, -1].set(b[-1])
        BS = Bbar @ S
    else:
        assert S is None
        BS = 0

    # put it all together ([Mattsson2012] Definition 2.4)
    B = jnp.diag(b)  # type: ignore[no-untyped-call]
    M = D.T @ P @ B @ D + R

    return invP @ (-M + BS)


@singledispatch
def sbp_norm_matrix(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType
) -> jnp.ndarray:
    """Construct the :math:`P` operator for and SBP approximation."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_first_derivative_matrix(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType
) -> jnp.ndarray:
    """Construct a first derivative :math:`D` operator satisfying the SBP property."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_second_derivative_matrix(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, b: jnp.ndarray
) -> jnp.ndarray:
    """Construct a second derivative :math:`D_2` operator satisfying the
    SBP properties.
    """
    raise NotImplementedError(type(op).__name__)


@singledispatch
def apply_sbp_first_derivative(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, u: jnp.ndarray
) -> jnp.ndarray:
    """A (potentially) matrix-free version of :func:`sbp_first_derivative_matrix`."""
    D1 = sbp_first_derivative_matrix(op, grid, bc)
    return D1 @ u


@singledispatch
def apply_sbp_second_derivative(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, b: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    """A (potentially) matrix-free version of :func:`sbp_second_derivative_matrix`."""
    D2 = sbp_second_derivative_matrix(op, grid, bc, b)
    return D2 @ u


# }}}


# {{{ SBP21


@dataclass(frozen=True)
class SBP21(SBPOperator):
    """An SBP operator the is second-order accurate in the interior and
    first-order accurate at the boundary.

    For details, see Appendix A.1 in [Mattsson2012]_.
    """

    @property
    def order(self) -> int:
        return 2

    @property
    def boundary_order(self) -> int:
        return 1


# {{{ interface


@sbp_norm_matrix.register(SBP21)
def _sbp_21_norm_matrix(op: SBP21, grid: UniformGrid, bc: BoundaryType) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_21_norm_stencil(bc, dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP21)
def _sbp_21_first_derivative_matrix(
    op: SBP21, grid: UniformGrid, bc: BoundaryType
) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_21_first_derivative_q_stencil(bc, dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q  # type: ignore[no-untyped-call]


@sbp_second_derivative_matrix.register(SBP21)
def _sbp_21_second_derivative_matrix(
    op: SBP21, grid: UniformGrid, bc: BoundaryType, b: jnp.ndarray
) -> jnp.ndarray:
    from numbers import Number

    assert isinstance(grid, UniformGrid)
    n = grid.n
    dtype = grid.dtype

    if isinstance(b, jnp.ndarray):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, b, dtype=dtype)  # type: ignore[no-untyped-call]
    else:
        raise TypeError(f"unknown coefficient type: '{type(b).__name__}'")

    assert b.shape == (grid.n,)

    # get R matrix ([Mattsson2012] Equation 8)
    R = make_sbp_21_second_derivative_r_matrix(bc, b, dx=grid.dx_min)

    # get S matrix
    if bc == BoundaryType.Periodic:
        S = None
    else:
        s = make_sbp_21_second_derivative_s_stencil(bc, dtype=dtype)
        S = make_sbp_matrix_from_stencil(bc, n, s, weight=1.0 / grid.dx_min)

    return make_sbp_mattsson2012_second_derivative(op, grid, bc, b, R, S)


# }}}


# {{{ stencils


def make_sbp_21_second_derivative_b_matrices(
    bc: BoundaryType, b: jnp.ndarray
) -> Tuple[jnp.ndarray]:
    # [Mattsson2012] Appendix A.1
    return (jnp.diag(b),)  # type: ignore[no-untyped-call]


def make_sbp_21_second_derivative_r_matrix(
    bc: BoundaryType, b: jnp.ndarray, dx: float
) -> jnp.ndarray:
    dtype = b.dtype
    (d22,) = make_sbp_21_second_derivative_d_stencils(bc, dtype=dtype)
    (c22,) = make_sbp_21_second_derivative_c_stencils(bc, dtype=dtype)

    n = b.size
    (B22,) = make_sbp_21_second_derivative_b_matrices(bc, b)
    D22 = make_sbp_matrix_from_stencil(bc, n, d22, weight=1.0 / dx**2)
    C22 = make_sbp_matrix_from_stencil(bc, n, c22)

    return dx**3 / 4 * D22.T @ C22 @ B22 @ D22


def make_sbp_21_norm_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    pi = jnp.array(dtype.type(1.0))  # type: ignore[no-untyped-call]
    pi_l = pi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        pi_l = jnp.array([0.5], dtype=dtype)  # type: ignore[no-untyped-call]

    return Stencil(int=pi, left=pi_l, right=pi_r, is_diagonal=True)


def make_sbp_21_first_derivative_q_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    qi = jnp.array([-0.5, 0.0, 0.5], dtype=dtype)  # type: ignore[no-untyped-call]
    qi_l = qi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        qi_l = jnp.array([[-0.5, 0.5]], dtype=dtype)  # type: ignore[no-untyped-call]

    return Stencil(int=qi, left=qi_l, right=qi_r)


def make_sbp_21_second_derivative_s_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    si = jnp.array([1], dtype=dtype)  # type: ignore[no-untyped-call]
    sb_l = sb_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        sb_l = jnp.array([[3 / 2, -2, 1 / 2]], dtype=dtype)  # type: ignore

    return Stencil(int=si, left=sb_l, right=sb_r)


def make_sbp_21_second_derivative_c_stencils(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    c22_i = jnp.array([1], dtype=dtype)  # type: ignore[no-untyped-call]
    c22_l = c22_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        c22_l = jnp.array([[0]], dtype=dtype)  # type: ignore[no-untyped-call]

    return (Stencil(int=c22_i, left=c22_l, right=c22_r),)


def make_sbp_21_second_derivative_d_stencils(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.1
    d22_i = jnp.array([1, -2, 1], dtype=dtype)  # type: ignore[no-untyped-call]
    d22_l = d22_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        d22_l = jnp.array(  # type: ignore[no-untyped-call]
            [[1, -2, 1], [1, -2, 1]], dtype=dtype
        )
        d22_r = jnp.array(  # type: ignore[no-untyped-call]
            [[1, -2, 1], [1, -2, 1]], dtype=dtype
        )

    return (Stencil(int=d22_i, left=d22_l, right=d22_r),)


# }}}

# }}}


# {{{ SBP42


@dataclass(frozen=True)
class SBP42(SBPOperator):
    """An SBP operator the is fourth-order accurate in the interior and
    second-order accurate at the boundary.

    For details, see Appendix A.2 in [Mattsson2012]_.
    """

    @property
    def order(self) -> int:
        return 4

    @property
    def boundary_order(self) -> int:
        return 2


# {{{ interface


@sbp_norm_matrix.register(SBP42)
def _sbp_42_norm_matrix(op: SBP42, grid: UniformGrid, bc: BoundaryType) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_42_norm_stencil(bc, dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP42)
def _sbp_42_first_derivative_matrix(
    op: SBP42, grid: UniformGrid, bc: BoundaryType
) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_42_first_derivative_q_stencil(bc, dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q  # type: ignore[no-untyped-call]


@sbp_second_derivative_matrix.register(SBP42)
def _sbp_42_second_derivative_matrix(
    op: SBP42, grid: UniformGrid, bc: BoundaryType, b: jnp.ndarray
) -> jnp.ndarray:
    from numbers import Number

    assert isinstance(grid, UniformGrid)
    n = grid.n
    dtype = grid.dtype

    if isinstance(b, jnp.ndarray):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, b)  # type: ignore[no-untyped-call]
    else:
        raise TypeError(f"unknown diffusivity coefficient: '{type(b).__name__}'")

    assert b.shape == (grid.n,)

    # get R matrix ([Mattsson2012] Equation 8)
    R = make_sbp_42_second_derivative_r_matrix(bc, b, dx=grid.dx_min)

    # get S matrix
    if bc == BoundaryType.Periodic:
        S = None
    else:
        s = make_sbp_42_second_derivative_s_stencil(bc, dtype=dtype)
        S = make_sbp_matrix_from_stencil(bc, n, s, weight=1.0 / grid.dx_min)

    return make_sbp_mattsson2012_second_derivative(op, grid, bc, b, R, S)


# }}}


# {{{ stencils


def make_sbp_42_second_derivative_b_matrices(
    bc: BoundaryType, b: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # [Mattsson2012] Appendix A.2
    B34 = jnp.pad((b[2:] + b[:-2]) / 2, 1)  # type: ignore[no-untyped-call]

    # TODO: [Mattsson2012] does not say what happens at the boundary points
    if bc == BoundaryType.Periodic:
        B34 = B34.at[0].set((b[0] + b[-1]) / 2)
        B34 = B34.at[-1].set((b[0] + b[-1]) / 2)
    else:
        B34 = B34.at[0].set(b[0])
        B34 = B34.at[-1].set(b[-1])

    B34 = jnp.diag(B34)  # type: ignore[no-untyped-call]
    B44 = jnp.diag(b)  # type: ignore[no-untyped-call]

    return B34, B44


def make_sbp_42_second_derivative_r_matrix(
    bc: BoundaryType, b: jnp.ndarray, dx: float
) -> jnp.ndarray:
    dtype = b.dtype
    (d34, d44) = make_sbp_42_second_derivative_d_stencils(bc, dtype=dtype)
    (c34, c44) = make_sbp_42_second_derivative_c_stencils(bc, dtype=dtype)

    n = b.size
    B34, B44 = make_sbp_42_second_derivative_b_matrices(bc, b)
    D34 = make_sbp_matrix_from_stencil(bc, n, d34, weight=1 / dx**3)
    D44 = make_sbp_matrix_from_stencil(bc, n, d44, weight=1 / dx**4)
    C34 = jnp.diag(make_sbp_matrix_from_stencil(bc, n, c34))  # type: ignore
    C44 = jnp.diag(make_sbp_matrix_from_stencil(bc, n, c44))  # type: ignore

    return (
        dx**5 / 18 * D34.T @ C34 @ B34 @ D34 + dx**7 / 144 * D44.T @ C44 @ B44 @ D44
    )


def make_sbp_42_norm_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    pi = jnp.array(dtype.type(1.0))  # type: ignore[no-untyped-call]
    pi_l = pi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        pi_l = jnp.array(  # type: ignore[no-untyped-call]
            [17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype=dtype
        )

    return Stencil(int=pi, left=pi_l, right=pi_r, is_diagonal=True)


def make_sbp_42_first_derivative_q_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Fisher2013] Appendix A, Equation A.1
    # [Fisher2013] Appendix A, Equation A.2 boundary

    qi = jnp.array(  # type: ignore[no-untyped-call]
        [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype
    )
    qi_l = qi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        qi_l = jnp.array(  # type: ignore[no-untyped-call]
            [
                [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
                [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
                [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
                [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
            ],
            dtype=dtype,
        )

    return Stencil(int=qi, left=qi_l, right=qi_r)


def make_sbp_42_second_derivative_s_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    si = jnp.array([0], dtype=dtype)  # type: ignore[no-untyped-call]
    sb_l = sb_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        sb_l = jnp.array(  # type: ignore[no-untyped-call]
            [[11 / 6, -3, 3 / 2, -1 / 3]], dtype=dtype
        )

    return Stencil(int=si, left=sb_l, right=sb_r)


def make_sbp_42_second_derivative_c_stencils(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    c34_i = jnp.array(1, dtype=dtype)  # type: ignore[no-untyped-call]
    c34_l = c34_r = None

    c44_i = jnp.array(1, dtype=dtype)  # type: ignore[no-untyped-call]
    c44_l = c44_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        c34_l = jnp.array(  # type: ignore[no-untyped-call]
            [0, 0, 163_928_591_571 / 53_268_010_936, 189_284 / 185_893, 1, 0],
            dtype=dtype,
        )
        c34_r = jnp.array(  # type: ignore[no-untyped-call]
            [1, 1_189_284 / 185_893, 0, 63_928_591_571 / 53_268_010_936, 0, 0],
            dtype=dtype,
        )

        c44_l = jnp.array(  # type: ignore[no-untyped-call]
            [0, 0, 1_644_330 / 301_051, 156_114 / 181_507, 1],
            dtype=dtype,
        )
        c44_r = c44_l[::-1]

    return (
        Stencil(int=c34_i, left=c34_l, right=c34_r, is_diagonal=True),
        Stencil(int=c44_i, left=c44_l, right=c44_r, is_diagonal=True),
    )


def make_sbp_42_second_derivative_d_stencils(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> Tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.2
    d34_i = jnp.array([-1, 3, -3, 1], dtype=dtype)  # type: ignore[no-untyped-call]
    d34_l = d34_r = None

    d44_i = jnp.array([1, -4, 6, -4, 1], dtype=dtype)  # type: ignore[no-untyped-call]
    d44_l = d44_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        d34_l = jnp.array(  # type: ignore[no-untyped-call]
            [
                [-1, 3, -3, 1, 0, 0],
                [-1, 3, -3, 1, 0, 0],
                [
                    -185_893 / 301_051,
                    +79_000_249_461 / 54_642_863_857,
                    -33_235_054_191 / 54_642_863_857,
                    -36_887_526_683 / 54_642_863_857,
                    +26_183_621_850 / 54_642_863_857,
                    -4_386 / 181_507,
                ],
            ],
            dtype=dtype,
        )

        d44_l = jnp.array(  # type: ignore[no-untyped-call]
            [
                [1, -4, 6, -4, 1],
                [1, -4, 6, -4, 1],
                [1, -4, 6, -4, 1],
            ],
            dtype=dtype,
        )
        d44_r = d44_l

    return (
        Stencil(int=d34_i, left=d34_l, right=d34_r),
        Stencil(int=d44_i, left=d44_l, right=d44_r),
    )


# }}}

# }}}


# {{{ SBP63


@dataclass(frozen=True)
class SBP64(SBPOperator):
    """An SBP operator the is sixth-order accurate in the interior and
    fourth-order accurate at the boundary.

    For details, see Appendix A.3 in [Mattsson2012]_.
    """

    @property
    def order(self) -> int:
        return 6

    @property
    def boundary_order(self) -> int:
        return 4


# {{{ interface


@sbp_norm_matrix.register(SBP64)
def _sbp_64_norm_matrix(op: SBP64, grid: UniformGrid, bc: BoundaryType) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_64_norm_stencil(bc, dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP64)
def _sbp_64_first_derivative_matrix(
    op: SBP64, grid: UniformGrid, bc: BoundaryType
) -> jnp.ndarray:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_64_first_derivative_q_stencil(bc, dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q  # type: ignore[no-untyped-call]


# }}}


# {{{ stencils


def make_sbp_64_norm_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # [Mattsson2012] Appendix A.3
    pi = jnp.array(dtype.type(1.0))  # type: ignore[no-untyped-call]
    pi_l = pi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        pi_l = jnp.array(  # type: ignore[no-untyped-call]
            [
                13649 / 43200,
                12013 / 8640,
                2711 / 4320,
                5359 / 4320,
                7877 / 8640,
                43801 / 43200,
            ],
            dtype=dtype,
        )

    return Stencil(int=pi, left=pi_l, right=pi_r, is_diagonal=True)


def make_sbp_64_first_derivative_q_stencil(
    bc: BoundaryType, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    qi = jnp.array(  # type: ignore[no-untyped-call]
        [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype
    )
    qi_l = qi_r = None

    if bc == BoundaryType.Periodic:
        pass
    else:
        qi_l = jnp.array(  # type: ignore[no-untyped-call]
            [
                [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
                [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
                [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
                [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
            ],
            dtype=dtype,
        )

    return Stencil(int=qi, left=qi_l, right=qi_r)


# }}}

# }}}


# {{{ make_operator_from_name

_OPERATORS: Dict[str, Type[SBPOperator]] = {
    "default": SBP42,
    "sbp21": SBP21,
    "sbp42": SBP42,
    "sbp64": SBP64,
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
