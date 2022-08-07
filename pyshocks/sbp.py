# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Tuple

import jax.numpy as jnp


def get_sbp_21_matrices(
    n: int,
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # construct SBP norm matrix
    h = jnp.ones(n, dtype=dtype)  # type: ignore[no-untyped-call]
    h = h.at[0].set(0.5)
    h = h.at[-1].set(0.5)

    # construct SBP derivative matrix
    q = jnp.eye(n, n, k=1) - jnp.eye(n, n, k=-1)  # type: ignore[no-untyped-call]
    q = q.at[0, 0].set(-1)
    q = q.at[-1, -1].set(1)

    return h, 0.5 * q
