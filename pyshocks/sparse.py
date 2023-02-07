# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Optional

import jax.experimental.sparse as js

from pyshocks.tools import Array


def sparse_banded(qi: Array) -> js.BCOO:
    raise NotImplementedError


def sparse_banded_boundary(
    qi: Array,
    qb_l: Optional[Array],
    qb_r: Optional[Array],
) -> js.BCOO:
    raise NotImplementedError


def sparse_circulant(qi: Array) -> js.BCOO:
    raise NotImplementedError
