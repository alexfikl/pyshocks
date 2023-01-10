# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Optional

import jax.experimental.sparse as js
import jax.numpy as jnp


def sparse_banded(qi: jnp.ndarray) -> js.BCOO:
    raise NotImplementedError


def sparse_banded_boundary(
    qi: jnp.ndarray,
    qb_l: Optional[jnp.ndarray],
    qb_r: Optional[jnp.ndarray],
) -> js.BCOO:
    raise NotImplementedError


def sparse_circulant(qi: jnp.ndarray) -> js.BCOO:
    raise NotImplementedError
