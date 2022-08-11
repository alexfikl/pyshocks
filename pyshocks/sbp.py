# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Any, Optional

import jax.numpy as jnp


# {{{ SBP21


def get_sbp_21_norm_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    h = jnp.ones(n, dtype=dtype)  # type: ignore[no-untyped-call]
    h = h.at[0].set(0.5)
    h = h.at[-1].set(0.5)

    return h


def get_sbp_21_first_derivative_matrix(
    n: int, *, dtype: Optional["jnp.dtype[Any]"] = None
) -> jnp.ndarray:
    q = jnp.eye(n, n, k=1) - jnp.eye(n, n, k=-1)  # type: ignore[no-untyped-call]
    q = q.at[0, 0].set(-1)
    q = q.at[-1, -1].set(1)

    return 0.5 * q


# }}}
