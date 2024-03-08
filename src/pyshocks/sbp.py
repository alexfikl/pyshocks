# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Summation-by-Parts (SBP) Operators
----------------------------------

These routines help construct several SBP operators with appropriate boundary
conditions (see [Mattsson2012]_ and [Parisi2010]_). First, we have that the
first derivative operator is given by

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

Interface
^^^^^^^^^

An SBP operator must implement the following interface, which relies on the
:func:`~functools.singledispatch` functionality.

.. autoclass:: Stencil
    :no-show-inheritance:
    :members:
.. autoclass:: SecondDerivativeType
    :members:
.. autoclass:: SBPOperator
    :no-show-inheritance:
    :members:

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

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any

import jax
import jax.numpy as jnp

from pyshocks.grid import UniformGrid
from pyshocks.schemes import BoundaryType
from pyshocks.tools import Array, ScalarLike

# {{{ SBP helpers


@dataclass(frozen=True)
class Stencil:
    int: Array
    """Interior stencil, as an array of shape ``(n_i,)``."""
    left: Array | None
    """Left boundary stencil, as an array of shape ``(n_l, m_l)``."""
    right: Array | None
    """Left boundary stencil, as an array of shape ``(n_r, m_r)``. If not
    provided and :attr:`left` is provided, the right boundary stencil
    is taken to be ``-left[::-1, ::-1]``.
    """

    is_diagonal: bool = False
    """If *True*, the resulting operator is assumed to be diagonal only.
    This is used to return an array of shape ``(n,)`` instead of a
    matrix of shape ``(n, m)`` in certain functions.
    """

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
        """The :class:`~numpy.dtype` of the stencils."""
        return jnp.dtype(self.int.dtype)


def make_sbp_boundary_matrix(n: int, *, dtype: Any = None) -> Array:
    """Construct the boundary :math:`B` operator for an SBP discretization.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    b = jnp.zeros((n, n), dtype=dtype)
    b = b.at[0, 0].set(-1)
    b = b.at[-1, -1].set(1)

    return b


def make_sbp_diagonal_matrix(n: int, s: Stencil, *, weight: ScalarLike = 1.0) -> Array:
    """Construct a diagonal matrix with a given boundary stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n,)`` representing the diagonal.
    """
    assert s.is_diagonal
    mat = jnp.full(n, s.int, dtype=s.dtype)

    if s.left is not None:
        mat = mat.at[: s.left.size].set(s.left)

    if s.right is not None:
        mat = mat.at[-s.right.size :].set(s.right)

    return weight * mat


def make_sbp_circulant_matrix(n: int, s: Stencil, *, weight: ScalarLike = 1.0) -> Array:
    """Construct a circulat matrix with a given interior stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    m = s.int.size
    w = weight * jnp.eye(n, n, dtype=s.dtype)
    return sum(
        [jnp.roll(s.int[i] * w, i - m // 2, axis=1) for i in range(m)], jnp.array(0.0)
    )


def make_sbp_banded_matrix(n: int, s: Stencil, *, weight: ScalarLike = 1.0) -> Array:
    """Construct a banded matrix with a given boundary stencil.

    :arg n: size of the matrix.
    :returns: an array of shape ``(n, n)``.
    """
    o = s.int.size // 2
    mat: Array = sum(
        [
            weight * s.int[k] * jnp.eye(n, n, k=k - o, dtype=s.dtype)
            for k in range(s.int.size)
        ],
        jnp.array(0.0),
    )

    if s.left is not None:
        n, m = s.left.shape
        mat = mat.at[:n, :m].set(weight * s.left)

    if s.right is not None:
        n, m = s.right.shape
        mat = mat.at[-n:, -m:].set(weight * s.right)

    return mat


def make_sbp_matrix_from_stencil(
    bc: BoundaryType, n: int, s: Stencil, *, weight: ScalarLike = 1.0
) -> Array:
    if s.is_diagonal:
        if bc == BoundaryType.Periodic:
            # NOTE: this is just here so we can remove some if statements below
            # when constructing the stencils
            s = replace(s, left=None, right=None)

        mat = make_sbp_diagonal_matrix(n, s, weight=weight)
    elif bc == BoundaryType.Periodic:
        mat = make_sbp_circulant_matrix(n, s, weight=weight)
    else:
        mat = make_sbp_banded_matrix(n, s, weight=weight)

    return mat


# }}}


# {{{ SBP class


@enum.unique
class SecondDerivativeType(enum.Enum):
    Compatible = enum.auto()
    """A second derivative that is compatible with the first derivative in
    the sense described by [Mattsson2012]_.
    """

    FullyCompatible = enum.auto()
    """A second derivative that is compatible with the first derivative
    in the sense described by [Parisi2010]_.
    """

    Narrow = enum.auto()
    """A narrow-stencil second-order derivative."""


@dataclass(frozen=True)
class SBPOperator:
    """Generic family of SBP operators."""

    second_derivative: SecondDerivativeType
    """A :class:`SecondDerivativeType` describing the construction of the
    second derivative.
    """

    @property
    def ids(self) -> str:
        """An identifier for the operator."""
        return f"{self.order}{self.boundary_order}"

    @property
    def order(self) -> int:
        """Interior order of the SBP operator."""
        raise NotImplementedError

    @property
    def boundary_order(self) -> int:
        """Boundary order of the SBP operator. This is not valid for periodic
        boundaries."""
        raise NotImplementedError

    def make_second_derivative_r_matrix(
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array:
        raise NotImplementedError

    def make_second_derivative_s_matrix(
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array | None:
        raise NotImplementedError


def sbp_matrix_from_name(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, name: str
) -> Array:
    n = grid.n
    dtype = grid.dtype

    if name == "P":
        func = globals()[f"make_sbp_{op.ids}_norm_stencil"]
        result = make_sbp_diagonal_matrix(n, func(dtype=dtype))
    elif name == "Q":
        func = globals()[f"make_sbp_{op.ids}_first_derivative_q_stencil"]
        result = make_sbp_matrix_from_stencil(bc, n, func(dtype=dtype))
    elif name == "S":
        func = globals()[f"make_sbp_{op.ids}_second_derivative_s_stencil"]
        result = make_sbp_matrix_from_stencil(
            bc, n, func(op.second_derivative, dtype=dtype), weight=1.0 / grid.dx_min
        )
    elif name == "R":
        func = globals()[f"make_sbp_{op.ids}_second_derivative_r_matrix"]
        result = func(bc, jnp.ones_like(grid.x), dx=grid.dx_min)
    else:
        raise ValueError(f"Unknown SBP matrix name: {name!r}.")

    return jnp.array(result, dtype=grid.x.dtype)


def make_sbp_mattsson2012_second_derivative(
    op: SBPOperator,
    grid: UniformGrid,
    bc: BoundaryType,
    b: Array,
    M: Array | None = None,
) -> Array:
    assert isinstance(grid, UniformGrid)
    # get mass matrix
    P = sbp_norm_matrix(op, grid, bc)
    invP = jnp.diag(1 / P)

    # put it all together ([Mattsson2012] Definition 2.4)
    S = op.make_second_derivative_s_matrix(grid, bc, b)

    # get Bbar matrix ([Mattsson2012] Definition 2.3)
    if bc == BoundaryType.Periodic:
        assert S is None
        BS = jnp.array(0.0, dtype=P.dtype)
    else:
        assert S is not None
        Bbar = jnp.zeros(S.shape, dtype=S.dtype)
        Bbar = Bbar.at[0, 0].set(-b[0])
        Bbar = Bbar.at[-1, -1].set(b[-1])
        BS = Bbar @ S

    if M is None:
        P = jnp.diag(P)
        D = sbp_first_derivative_matrix(op, grid, bc)
        B = jnp.diag(b)
        R = op.make_second_derivative_r_matrix(grid, bc, b)

        M = D.T @ P @ B @ D + R

    # NOTE: these are mostly here to catch obvious mistakes, so the tolerances
    # are set quite higher than they should be
    assert jnp.linalg.norm(M - M.T) < 1.0e-8
    assert jnp.linalg.norm(jnp.sum(M, axis=1)) < 1.0e-8

    return invP @ (-M + BS)


@singledispatch
def sbp_norm_matrix(op: SBPOperator, grid: UniformGrid, bc: BoundaryType) -> Array:
    """Construct the :math:`P` operator for and SBP approximation."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_first_derivative_matrix(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType
) -> Array:
    """Construct a first derivative :math:`D` operator satisfying the SBP property."""
    raise NotImplementedError(type(op).__name__)


@singledispatch
def sbp_second_derivative_matrix(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, b: float | Array
) -> Array:
    """Construct a wide-stencil second derivative :math:`D_2` operator
    that is compatible with :func:`sbp_first_derivative_matrix`.
    """
    raise NotImplementedError(type(op).__name__)


@singledispatch
def apply_sbp_first_derivative(
    op: SBPOperator, grid: UniformGrid, bc: BoundaryType, u: Array
) -> Array:
    """A (potentially) matrix-free version of :func:`sbp_first_derivative_matrix`."""
    D1 = sbp_first_derivative_matrix(op, grid, bc)
    return D1 @ u


@singledispatch
def apply_sbp_second_derivative(
    op: SBPOperator,
    grid: UniformGrid,
    bc: BoundaryType,
    b: float | Array,
    u: Array,
) -> Array:
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

    def make_second_derivative_r_matrix(  # noqa: PLR6301
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array:
        return make_sbp_21_second_derivative_r_matrix(bc, b, dx=grid.dx_min)

    def make_second_derivative_s_matrix(
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array | None:
        if bc == BoundaryType.Periodic:
            return None

        s = make_sbp_21_second_derivative_s_stencil(
            self.second_derivative, dtype=grid.dtype
        )

        return make_sbp_matrix_from_stencil(bc, grid.n, s, weight=1.0 / grid.dx_min)


# {{{ interface


@sbp_norm_matrix.register(SBP21)
def _sbp_21_norm_matrix(op: SBP21, grid: UniformGrid, bc: BoundaryType) -> Array:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_21_norm_stencil(dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP21)
def _sbp_21_first_derivative_matrix(
    op: SBP21, grid: UniformGrid, bc: BoundaryType
) -> Array:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_21_first_derivative_q_stencil(dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q


@sbp_second_derivative_matrix.register(SBP21)
def _sbp_21_second_derivative_matrix(
    op: SBP21, grid: UniformGrid, bc: BoundaryType, b: float | Array
) -> Array:
    from numbers import Number

    if isinstance(b, jax.Array):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, jnp.array(b), dtype=grid.dtype)
    else:
        raise TypeError(f"Unknown coefficient type: {type(b).__name__!r}.")

    assert isinstance(b, jax.Array)
    assert isinstance(grid, UniformGrid)
    assert b.shape == (grid.n,)

    if op.second_derivative == SecondDerivativeType.Narrow:
        M = make_sbp_21_second_derivative_m_matrix(bc, b, grid.dx_min)
    else:
        M = None

    return make_sbp_mattsson2012_second_derivative(op, grid, bc, b, M=M)


# }}}


# {{{ stencils


def make_sbp_21_second_derivative_b_matrices(
    bc: BoundaryType, b: Array
) -> tuple[Array]:
    # [Mattsson2012] Appendix A.1
    return (jnp.diag(b),)


def make_sbp_21_second_derivative_r_matrix(
    bc: BoundaryType, b: Array, dx: ScalarLike
) -> Array:
    dtype = b.dtype
    (d22,) = make_sbp_21_second_derivative_d_stencils(dtype=dtype)
    (c22,) = make_sbp_21_second_derivative_c_stencils(dtype=dtype)

    n = b.size
    (B22,) = make_sbp_21_second_derivative_b_matrices(bc, b)
    D22 = make_sbp_matrix_from_stencil(bc, n, d22)
    C22 = make_sbp_matrix_from_stencil(bc, n, c22)

    return dx**3 / 4 * D22.T @ C22 @ B22 @ D22


def make_sbp_21_second_derivative_m_matrix(
    bc: BoundaryType, b: Array, dx: ScalarLike
) -> Array:
    def make_boundary(b1: Array, b2: Array, b3: Array, b4: Array) -> Array:
        return jnp.array(
            [
                [(b1 + b2) / 2, -(b1 + b2) / 2, 0, 0],
                [-(b1 + b2) / 2, (b1 + 2 * b2 + b3) / 2, -(b2 + b3) / 2, 0],
                [0, -(b2 + b3) / 2, (b2 + 2 * b3 + b4) / 2, -(b3 + b4) / 2],
            ],
            dtype=b.dtype,
        )

    M = (
        jnp.diag(-b[:-1] / 2 - b[1:] / 2, k=-1)
        + jnp.diag(jnp.pad(b[:-2] / 2 + b[1:-1] + b[2:] / 2, 1), k=0)
        + jnp.diag(-b[:-1] / 2 - b[1:] / 2, k=+1)
    )

    if bc == BoundaryType.Periodic:
        M = M.at[0, 0].set(b[-1] / 2 + b[0] + b[1] / 2)
        M = M.at[0, -1].set(-b[-1] / 2 - b[0] / 2)
        M = M.at[-1, 0].set(-b[-1] / 2 - b[0] / 2)
        M = M.at[-1, -1].set(b[-2] / 2 + b[-1] + b[0] / 2)
    else:
        mb_l = make_boundary(*b[:4])
        mb_r = make_boundary(*b[-4:][::-1])[::-1, ::-1]

        n, m = mb_l.shape
        M = M.at[:n, :m].set(mb_l)
        n, m = mb_r.shape
        M = M.at[-n:, -m:].set(mb_r)

    return M / dx


def make_sbp_21_norm_stencil(dtype: Any = None) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.1
    pi = jnp.array(1, dtype=dtype)

    pb_l = jnp.array([0.5], dtype=dtype)
    pb_r = pb_l

    return Stencil(int=pi, left=pb_l, right=pb_r, is_diagonal=True)


def make_sbp_21_first_derivative_q_stencil(
    dtype: Any = None,
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.1
    qi = jnp.array([-0.5, 0.0, 0.5], dtype=dtype)

    qb_l = jnp.array([[-0.5, 0.5]], dtype=dtype)
    qb_r = -qb_l[::-1, ::-1]

    return Stencil(int=qi, left=qb_l, right=qb_r)


def make_sbp_21_second_derivative_s_stencil(
    sd: SecondDerivativeType,
    dtype: Any = None,
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.1
    si = jnp.array([1], dtype=dtype)

    if sd == SecondDerivativeType.FullyCompatible:
        # NOTE: this matches the stencil of H^{-1} Q at the boundary
        sb_l = jnp.array([[-1, 1]], dtype=dtype)
        sb_r = -sb_l[::-1, ::-1]
    else:
        sb_l = jnp.array([[-3 / 2, 2, -1 / 2]], dtype=dtype)
        sb_r = -sb_l[::-1, ::-1]

    return Stencil(int=si, left=sb_l, right=sb_r)


def make_sbp_21_second_derivative_c_stencils(
    dtype: Any = None,
) -> tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.1
    c22_i = jnp.array([1], dtype=dtype)

    c22_l = jnp.array([[0]], dtype=dtype)
    c22_r = -c22_l[::-1, ::-1]

    return (Stencil(int=c22_i, left=c22_l, right=c22_r),)


def make_sbp_21_second_derivative_d_stencils(
    dtype: Any = None,
) -> tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.1
    d22_i = jnp.array([1, -2, 1], dtype=dtype)

    d22_l = jnp.array([[1, -2, 1], [1, -2, 1]], dtype=dtype)
    d22_r = jnp.array([[1, -2, 1], [1, -2, 1]], dtype=dtype)

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

    def make_second_derivative_r_matrix(  # noqa: PLR6301
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array:
        return make_sbp_42_second_derivative_r_matrix(bc, b, dx=grid.dx_min)

    def make_second_derivative_s_matrix(
        self, grid: UniformGrid, bc: BoundaryType, b: Array
    ) -> Array | None:
        if bc == BoundaryType.Periodic:
            return None

        s = make_sbp_42_second_derivative_s_stencil(
            self.second_derivative, dtype=grid.dtype
        )

        return make_sbp_matrix_from_stencil(bc, grid.n, s, weight=1.0 / grid.dx_min)


# {{{ interface


@sbp_norm_matrix.register(SBP42)
def _sbp_42_norm_matrix(op: SBP42, grid: UniformGrid, bc: BoundaryType) -> Array:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_42_norm_stencil(dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP42)
def _sbp_42_first_derivative_matrix(
    op: SBP42, grid: UniformGrid, bc: BoundaryType
) -> Array:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_42_first_derivative_q_stencil(dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q


@sbp_second_derivative_matrix.register(SBP42)
def _sbp_42_second_derivative_matrix(
    op: SBP42, grid: UniformGrid, bc: BoundaryType, b: float | Array
) -> Array:
    from numbers import Number

    if isinstance(b, jax.Array):
        pass
    elif isinstance(b, Number):
        b = jnp.full_like(grid.x, jnp.array(b), dtype=grid.dtype)
    else:
        raise TypeError(f"Unknown diffusivity coefficient: {type(b).__name__!r}.")

    assert isinstance(b, jax.Array)
    assert isinstance(grid, UniformGrid)
    assert b.shape == (grid.n,)

    if op.second_derivative == SecondDerivativeType.Narrow:
        M = make_sbp_42_second_derivative_m_matrix(bc, b, grid.dx_min)
    else:
        M = None

    return make_sbp_mattsson2012_second_derivative(op, grid, bc, b, M=M)


# }}}


# {{{ stencils


def make_sbp_42_second_derivative_b_matrices(
    bc: BoundaryType, b: Array
) -> tuple[Array, Array]:
    # [Mattsson2012] Appendix A.2
    B34 = jnp.pad((b[2:] + b[:-2]) / 2, 1)

    # TODO: [Mattsson2012] does not say what happens at the boundary points
    if bc == BoundaryType.Periodic:
        B34 = B34.at[0].set((b[0] + b[-1]) / 2)
        B34 = B34.at[-1].set((b[0] + b[-1]) / 2)
    else:
        B34 = B34.at[0].set(b[0])
        B34 = B34.at[-1].set(b[-1])

    B34 = jnp.diag(B34)
    B44 = jnp.diag(b)

    return B34, B44


def make_sbp_42_second_derivative_r_matrix(
    bc: BoundaryType, b: Array, dx: ScalarLike
) -> Array:
    dtype = b.dtype
    (d34, d44) = make_sbp_42_second_derivative_d_stencils(dtype=dtype)
    (c34, c44) = make_sbp_42_second_derivative_c_stencils(dtype=dtype)

    n = b.size
    B34, B44 = make_sbp_42_second_derivative_b_matrices(bc, b)
    D34 = make_sbp_matrix_from_stencil(bc, n, d34)
    D44 = make_sbp_matrix_from_stencil(bc, n, d44)
    C34 = jnp.diag(make_sbp_matrix_from_stencil(bc, n, c34))
    C44 = jnp.diag(make_sbp_matrix_from_stencil(bc, n, c44))

    return dx**5 / 18 * D34.T @ C34 @ B34 @ D34 + dx**7 / 144 * D44.T @ C44 @ B44 @ D44


def make_sbp_42_second_derivative_m_matrix(
    bc: BoundaryType, b: Array, dx: ScalarLike
) -> Array:
    def make_boundary(
        b1: Array,
        b2: Array,
        b3: Array,
        b4: Array,
        b5: Array,
        b6: Array,
        b7: Array,
        b8: Array,
    ) -> Array:
        # NOTE: M is 0-indexed to write out the transpose easier
        # NOTE: b is 1-indexed to match the notation in [Mattsson2012]
        m0 = jnp.array(
            [
                # M_00
                12 / 17 * b1
                + 59 / 192 * b2
                + 27_010_400_129 / 345_067_064_608 * b3
                + 69_462_376_031 / 2_070_402_387_648 * b4,
                # M_01
                -59 / 68 * b1
                - 6_025_413_881 / 21_126_554_976 * b3
                - 537_416_663 / 7_042_184_992 * b4,
                # M_02
                2 / 17 * b1
                - 59 / 192 * b2
                + 213_318_005 / 16_049_630_912 * b4
                + 2_083_938_599 / 8_024_815_456 * b3,
                # M_03
                3 / 68 * b1
                - 1_244_724_001 / 21_126_554_976 * b3
                + 752_806_667 / 21_126_554_976 * b4,
                # M_04
                49_579_087 / 10_149_031_312 * b3 - 49_579_087 / 10_149_031_312 * b4,
                # M_05
                -1 / 784 * b4 + 1 / 784 * b3,
                # M_06
                0,
                # M_07
                0,
            ],
            dtype=b.dtype,
        )
        m1 = jnp.array(
            [
                # M_10
                m0[1],
                # M_11
                3_481 / 3_264 * b1
                + 9_258_282_831_623_875 / 7_669_235_228_057_664 * b3
                + 236_024_329_996_203 / 1_278_205_871_342_944 * b4,
                # M_12
                -59 / 408 * b1
                - 29_294_615_794_607 / 29_725_717_938_208 * b3
                - 2_944_673_881_023 / 29_725_717_938_208 * b4,
                # M_13
                -59 / 1088 * b1
                + 260_297_319_232_891 / 2_556_411_742_685_888 * b3
                - 60_834_186_813_841 / 1_278_205_871_342_944 * b4,
                # M_14
                -1_328_188_692_663 / 37_594_290_333_616 * b3
                + 1_328_188_692_663 / 37_594_290_333_616 * b4,
                # M_15
                -8_673 / 2_904_112 * b3 + 8_673 / 2_904_112 * b4,
                # M_16
                0,
                # M_17
                0,
            ],
            dtype=b.dtype,
        )
        m2 = jnp.array(
            [
                # M_20
                m0[2],
                # M_21
                m1[2],
                # M_22
                1 / 51 * b1
                + 59 / 192 * b2
                + 13_777_050_223_300_597 / 26_218_083_221_499_456 * b4
                + 564_461 / 13_384_296 * b5
                + 378_288_882_302_546_512_209 / 270_764_341_349_677_687_456 * b3,
                # M_23
                1 / 136 * b1
                - 125_059 / 743_572 * b5
                - 4_836_340_090_442_187_227 / 5_525_802_884_687_299_744 * b3
                - 17_220_493_277_981 / 89_177_153_814_624 * b4,
                # M_24
                -10_532_412_077_335 / 42_840_005_263_888 * b4
                + 1_613_976_761_032_884_305 / 7_963_657_098_519_931_984 * b3
                + 564_461 / 4_461_432 * b5,
                # M_25
                -960_119 / 1_280_713_392 * b4
                - 3_391 / 6_692_148 * b5
                + 33_235_054_191 / 26_452_850_508_784 * b3,
                # M_26
                0,
                # M_27
                0,
            ],
            dtype=b.dtype,
        )
        m3 = jnp.array(
            [
                # M_30
                m0[3],
                # M_31
                m1[3],
                # M_32
                m2[3],
                # M_33
                3 / 1_088 * b1
                + 507_284_006_600_757_858_213 / 475_219_048_083_107_777_984 * b3
                + 1_869_103 / 2_230_716 * b5
                + 1 / 24 * b6
                + 1_950_062_198_436_997 / 3_834_617_614_028_832 * b4,
                # M_34
                -4_959_271_814_984_644_613 / 20_965_546_238_960_637_264 * b3
                - 1 / 6 * b6
                - 15_998_714_909_649 / 37_594_290_333_616 * b4
                - 375_177 / 743_572 * b5,
                # M_35
                -368_395 / 2_230_716 * b5
                + 752_806_667 / 539_854_092_016 * b3
                + 1_063_649 / 8_712_336 * b4
                + 1 / 8 * b6,
                # M_36
                0,
                # M_37
                0,
            ],
            dtype=b.dtype,
        )
        m4 = jnp.array(
            [
                # M_40
                m0[4],
                # M_41
                m1[4],
                # M_42
                m2[4],
                # M_43
                m3[4],
                # M_44
                8_386_761_355_510_099_813 / 128_413_970_713_633_903_242 * b3
                + 2_224_717_261_773_437 / 2_763_180_339_520_776 * b4
                + 5 / 6 * b6
                + 1 / 24 * b7
                + 280_535 / 371_786 * b5,
                # M_45
                -35_039_615 / 213_452_232 * b4
                - 1 / 6 * b7
                - 13_091_810_925 / 13_226_425_254_392 * b3
                - 1_118_749 / 2_230_716 * b5
                - 1 / 2 * b6,
                # M_46
                -1 / 6 * b6 + 1 / 8 * b5 + 1 / 8 * b7,
                # M_47
                0,
            ],
            dtype=b.dtype,
        )
        m5 = jnp.array(
            [
                # M_50
                m0[5],
                # M_51
                m1[5],
                # M_52
                m2[5],
                # M_53
                m3[5],
                # M_54
                m4[5],
                # M_55
                3_290_636 / 80_044_587 * b4
                + 5_580_181 / 6_692_148 * b5
                + 5 / 6 * b7
                + 1 / 24 * b8
                + 660_204_843 / 13_226_425_254_392 * b3
                + 3 / 4 * b6,
                # M_56
                -1 / 6 * b5 - 1 / 6 * b8 - 1 / 2 * b6 - 1 / 2 * b7,
                # M_57
                -1 / 6 * b7 + 1 / 8 * b6 + 1 / 8 * b8,
            ],
            dtype=b.dtype,
        )

        return -jnp.stack([m0, m1, m2, m3, m4, m5])

    M = (
        jnp.diag(b[1:-1] / 6 - b[:-2] / 8 - b[2:] / 8, k=-2)
        + jnp.diag(
            jnp.pad(b[:-3] / 6 + b[3:] / 6 + b[1:-2] / 2 + b[2:-1] / 2, 1),
            k=-1,
        )
        + jnp.diag(
            jnp.pad(
                -b[:-4] / 24
                - 5 * b[1:-3] / 6
                - 5 * b[3:-1] / 6
                - b[4:] / 24
                - 3 * b[2:-2] / 4,
                2,
            ),
            k=0,
        )
        + jnp.diag(
            jnp.pad(b[:-3] / 6 + b[3:] / 6 + b[1:-2] / 2 + b[2:-1] / 2, 1),
            k=+1,
        )
        + jnp.diag(b[1:-1] / 6 - b[:-2] / 8 - b[2:] / 8, k=+2)
    )

    if bc == BoundaryType.Periodic:
        # row 0
        M = M.at[0, -2].set(b[-1] / 6 - b[-2] / 8 - b[0] / 8)
        M = M.at[0, -1].set(b[-2] / 6 + b[1] / 6 + b[-1] / 2 + b[0] / 2)
        M = M.at[0, 0].set(
            -b[-2] / 24 - 5 / 6 * b[-1] - 5 / 6 * b[1] - b[2] / 24 - 3 * b[0] / 4
        )
        M = M.at[0, 1].set(b[-1] / 6 + b[2] / 6 + b[0] / 2 + b[1] / 2)
        # row 1
        M = M.at[1, -1].set(b[0] / 6 - b[-1] / 8 - b[1] / 8)
        M = M.at[1, 0].set(b[-1] / 6 + b[2] / 6 + b[0] / 2 + b[1] / 2)
        M = M.at[1, 1].set(
            -b[-1] / 24 - 5 * b[0] / 6 - 5 * b[2] / 6 - b[3] / 24 - 3 * b[1] / 4
        )
        # row N - 2
        M = M.at[-2, -2].set(
            -b[-4] / 24 - 5 * b[-3] / 6 - 5 * b[-1] / 6 - b[0] / 24 - 3 * b[-2] / 4
        )
        M = M.at[-2, -1].set(b[-3] / 6 + b[0] / 6 + b[-2] / 2 + b[-1] / 2)
        M = M.at[-2, 0].set(b[-1] / 6 - b[-2] / 8 - b[0] / 8)
        # row N - 1
        M = M.at[-1, -2].set(b[-3] / 6 + b[0] / 6 + b[-2] / 2 + b[-1] / 2)
        M = M.at[-1, -1].set(
            -b[-3] / 24 - 5 * b[-2] / 6 - 5 * b[0] / 6 - b[1] / 24 - 3 * b[-1] / 4
        )
        M = M.at[-1, 0].set(b[-2] / 6 + b[1] / 6 + b[-1] / 2 + b[0] / 2)
        M = M.at[-1, 1].set(b[0] / 6 - b[-1] / 8 - b[1] / 8)
    else:
        mb_l = make_boundary(*b[:8])
        mb_r = make_boundary(*b[-8:][::-1])[::-1, ::-1]

        n, m = mb_l.shape
        M = M.at[:n, :m].set(mb_l)
        n, m = mb_r.shape
        M = M.at[-n:, -m:].set(mb_r)

    return -M / dx


def make_sbp_42_norm_stencil(dtype: Any = None) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.2
    pi = jnp.array(1, dtype=dtype)

    pb_l = jnp.array([17 / 48, 59 / 48, 43 / 48, 49 / 48], dtype=dtype)
    pb_r = pb_l[::-1]

    return Stencil(int=pi, left=pb_l, right=pb_r, is_diagonal=True)


def make_sbp_42_first_derivative_q_stencil(
    dtype: Any = None,
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Fisher2013] Appendix A, Equation A.1
    # [Fisher2013] Appendix A, Equation A.2 boundary

    qi = jnp.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype)

    qb_l = jnp.array(
        [
            [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
            [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
            [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
            [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
        ],
        dtype=dtype,
    )
    qb_r = -qb_l[::-1, ::-1]

    return Stencil(int=qi, left=qb_l, right=qb_r)


def make_sbp_42_second_derivative_s_stencil(
    sd: SecondDerivativeType,
    dtype: Any = None,
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.2
    si = jnp.array([0], dtype=dtype)

    if sd == SecondDerivativeType.FullyCompatible:
        # NOTE: this matches the stencil of H^{-1} Q at the boundary
        sb_l = jnp.array([[-1 / 2, 59 / 96, -1 / 12, -1 / 32]], dtype=dtype)
        sb_r = -sb_l[::-1, ::-1]
    else:
        sb_l = jnp.array([[11 / 6, -3, 3 / 2, -1 / 3]], dtype=dtype)

        if sd == SecondDerivativeType.Narrow:
            # NOTE: this is a copy pasting difference between the various
            # Mattsson papers, should probably clean it up sometime
            sb_l = -sb_l

        sb_r = -sb_l[::-1, ::-1]

    return Stencil(int=si, left=sb_l, right=sb_r)


def make_sbp_42_second_derivative_c_stencils(
    dtype: Any = None,
) -> tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.2
    c34_i = jnp.array(1, dtype=dtype)

    c34_l = jnp.array(
        [0, 0, 163_928_591_571 / 53_268_010_936, 189_284 / 185_893, 1, 0],
        dtype=dtype,
    )
    c34_r = jnp.array(
        [1, 1_189_284 / 185_893, 0, 63_928_591_571 / 53_268_010_936, 0, 0],
        dtype=dtype,
    )

    # [Mattsson2012] Appendix A.2
    c44_i = jnp.array(1, dtype=dtype)

    c44_l = jnp.array(
        [0, 0, 1_644_330 / 301_051, 156_114 / 181_507, 1],
        dtype=dtype,
    )
    c44_r = c44_l[::-1]

    return (
        Stencil(int=c34_i, left=c34_l, right=c34_r, is_diagonal=True),
        Stencil(int=c44_i, left=c44_l, right=c44_r, is_diagonal=True),
    )


def make_sbp_42_second_derivative_d_stencils(
    dtype: Any = None,
) -> tuple[Stencil, ...]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.2

    d34_i = jnp.array([-1, 3, -3, 1], dtype=dtype)

    d34_l = jnp.array(
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
    d34_r = -d34_l[::-1, ::-1]

    d44_i = jnp.array([1, -4, 6, -4, 1], dtype=dtype)
    d44_l = jnp.array(
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
def _sbp_64_norm_matrix(op: SBP64, grid: UniformGrid, bc: BoundaryType) -> Array:
    assert isinstance(grid, UniformGrid)

    p = make_sbp_64_norm_stencil(dtype=grid.dtype)
    return make_sbp_matrix_from_stencil(bc, grid.n, p, weight=grid.dx_min)


@sbp_first_derivative_matrix.register(SBP64)
def _sbp_64_first_derivative_matrix(
    op: SBP64, grid: UniformGrid, bc: BoundaryType
) -> Array:
    assert isinstance(grid, UniformGrid)

    q = make_sbp_64_first_derivative_q_stencil(dtype=grid.dtype)
    Q = make_sbp_matrix_from_stencil(bc, grid.n, q)
    P = sbp_norm_matrix(op, grid, bc)

    return jnp.diag(1.0 / P) @ Q


# }}}


# {{{ stencils


def make_sbp_64_norm_stencil(dtype: Any = None) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # [Mattsson2012] Appendix A.3
    pi = jnp.array(1, dtype=dtype)

    pb_l = jnp.array(
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
    pb_r = pb_l[::-1]

    return Stencil(int=pi, left=pb_l, right=pb_r, is_diagonal=True)


def make_sbp_64_first_derivative_q_stencil(
    dtype: Any = None,
) -> Stencil:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    qi = jnp.array([1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12], dtype=dtype)

    qb_l = jnp.array(
        [
            [-1 / 2, 59 / 96, -1 / 12, -1 / 32, 0.0, 0.0],
            [-59 / 96, 0.0, 59 / 96, 0.0, 0.0, 0.0],
            [1 / 12, -59 / 96, 0.0, 59 / 96, -1 / 12, 0.0],
            [1 / 32, 0.0, -59 / 96, 0.0, 2 / 3, -1 / 12],
        ],
        dtype=dtype,
    )
    qb_r = -qb_l[::-1, ::-1]

    return Stencil(int=qi, left=qb_l, right=qb_r)


# }}}

# }}}


# {{{ make_operator_from_name

_OPERATORS: dict[str, type[SBPOperator]] = {
    "default": SBP42,
    "sbp21": SBP21,
    "sbp42": SBP42,
    "sbp64": SBP64,
}


def operator_ids() -> tuple[str, ...]:
    return tuple(_OPERATORS.keys())


def make_operator_from_name(name: str, **kwargs: Any) -> SBPOperator:
    """
    :arg name: name of the operator.
    :arg kwargs: additional arguments to pass to the operator. Any arguments
        that are not in the operator's fields are ignored.
    """

    cls = _OPERATORS.get(name)
    if cls is None:
        from pyshocks.tools import join_or

        raise ValueError(
            f"Scheme {name!r} not found. Try one of {join_or(operator_ids())}."
        )

    from dataclasses import fields

    if "second_derivative" not in kwargs:
        kwargs["second_derivative"] = SecondDerivativeType.Narrow

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
