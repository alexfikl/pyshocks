# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Summation-by-Parts (SBP) Operators
----------------------------------

.. autofunction:: make_sbp_operator_for_order
"""

import enum
from typing import Tuple

import jax.numpy as jnp

from pyshocks.grid import Grid


@enum.unique
class BoundaryType(enum.Enum):
    Periodic = enum.auto()
    Dirichlet = enum.auto()


def make_sbp_operator_for_order(
    grid: Grid,
    order: int,
    *,
    bctype: BoundaryType = BoundaryType.Dirichlet,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if order == 2:
        H = second_order_sbp_h(grid, bctype)  # noqa: N806
        D = second_order_sbp_dx(grid, bctype)  # noqa: N806
    elif order == 4:
        H = fourth_order_sbp_h(grid, bctype)  # noqa: N806
        D = fourth_order_sbp_dx(grid, bctype)  # noqa: N806
    else:
        raise ValueError(f"unsupported order: {order}")

    return H, D


# {{{ second-order operators


def second_order_sbp_h(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    mat = jnp.ones(grid.n, dtype=grid.x.dtype)  # type: ignore[no-untyped-call]

    if bctype == BoundaryType.Periodic:
        pass
    elif bctype == BoundaryType.Dirichlet:
        mat = mat.at[jnp.array([0, -1])].set(0.5)  # type: ignore[no-untyped-call]
    else:
        raise ValueError(f"unknown boundary type: {bctype}")

    return mat


def second_order_sbp_dx(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    d = jnp.full(grid.n, 0.5, dtype=grid.x.dtype)  # type: ignore[no-untyped-call]
    mat = jnp.diag(d, k=1) - jnp.diag(d, k=-1)  # type: ignore[no-untyped-call]

    if bctype == BoundaryType.Periodic:
        pass
    elif bctype == BoundaryType.Dirichlet:
        irow = jnp.array([0, 0, -1, -1])  # type: ignore[no-untyped-call]
        jcol = jnp.array([0, 1, -2, -1])  # type: ignore[no-untyped-call]
        c = jnp.array([-1.0, 1.0, -1.0, 1.0])  # type: ignore[no-untyped-call]
        mat = mat.at[irow, jcol].set(c)
    else:
        raise ValueError(f"unknown boundary type: {bctype}")

    return mat


def second_order_sbp_ddx(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    raise NotImplementedError


# }}}


# {{{ fourth-order operators


def fourth_order_sbp_h(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    mat = jnp.ones(grid.n, dtype=grid.x.dtype)  # type: ignore[no-untyped-call]

    if bctype == BoundaryType.Periodic:
        pass
    elif bctype == BoundaryType.Dirichlet:
        i = jnp.array([0, 1, 2, 3])  # type: ignore[no-untyped-call]
        c = jnp.array(  # type: ignore[no-untyped-call]
            [17 / 48, 59 / 48, 43 / 48, 49 / 48],
            dtype=mat.dtype,
        )
        mat = mat.at[i].set(c).at[-1 - i].set(c)
    else:
        raise ValueError(f"unknown boundary type: {bctype}")

    return mat


def fourth_order_sbp_dx(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    # interior nodes
    d = jnp.ones(grid.n, dtype=grid.x.dtype)  # type: ignore[no-untyped-call]
    mat = (  # noqa: N806
        jnp.diag(1 / 12 * d, k=-2)  # type: ignore[no-untyped-call]
        - jnp.diag(2 / 3 * d, k=-1)  # type: ignore[no-untyped-call]
        + jnp.diag(2 / 3 * d, k=1)  # type: ignore[no-untyped-call]
        - jnp.diag(1 / 12 * d, k=2)  # type: ignore[no-untyped-call]
    )

    # boundary nodes
    if bctype == BoundaryType.Periodic:
        pass
    elif bctype == BoundaryType.Dirichlet:
        irow = jnp.tile(  # type: ignore[no-untyped-call]
            jnp.arange(4).reshape(-1, 1), 6
        ).flatten()
        jcol = jnp.tile(  # type: ignore[no-untyped-call]
            jnp.arange(6).reshape(-1, 1), 4
        ).T.flatten()
        c = jnp.array(  # type: ignore[no-untyped-call]
            [
                [-24 / 17, 59 / 34, -4 / 17, -3 / 34, 0.0, 0.0],
                [-1 / 2, 0.0, 1 / 2, 0.0, 0.0, 0.0],
                [4 / 43, -59 / 86, 0.0, 59 / 86, -4 / 43, 0.0],
                [3 / 98, 0.0, -59 / 98, 0.0, 32 / 49, -4 / 49],
            ]
        ).flatten()

        mat = mat.at[irow, jcol].set(c).at[-1 - irow, -1 - jcol].set(c)
    else:
        raise ValueError(f"unknown boundary type: {bctype}")

    return mat


def fourth_order_sbp_ddx(grid: Grid, bctype: BoundaryType) -> jnp.ndarray:
    raise NotImplementedError


# }}}
