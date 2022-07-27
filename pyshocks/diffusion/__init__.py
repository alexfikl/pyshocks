# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Linear Diffusion Equation
-------------------------

This module implements schemes for the linear diffusion equation

.. math::

    \frac{\partial u}{\partial t}
    - \frac{\partial}{\partial x} \left(d(x) \frac{\partial u}{\partial x}\right)
    = 0

where :math:`d(x) > 0` is the diffusivity.

Schemes
^^^^^^^

.. autoclass:: Scheme
.. autoclass:: CenteredScheme
"""

from typing import Tuple

import jax.numpy as jnp

from pyshocks import Grid
from pyshocks.diffusion.schemes import Scheme, CenteredScheme


# {{{ exact solutions


def ex_expansion(
    grid: Grid,
    t: float,
    x: jnp.ndarray,
    *,
    modes: Tuple[float, ...] = (1.0,),
    amplitudes: Tuple[float, ...] = (1.0,),
    diffusivity: float = 1.0,
) -> jnp.ndarray:
    assert len(modes) > 0
    assert len(modes) == len(amplitudes)
    assert diffusivity > 0

    L = grid.b - grid.a  # noqa: N806
    return sum(
        [
            a
            * jnp.sin(jnp.pi * n * x / L)
            * jnp.exp(-diffusivity * (n * jnp.pi / L) ** 2 * t)
            for a, n in zip(amplitudes, modes)
        ],
        jnp.zeros_like(x),  # type: ignore[no-untyped-call]
    )


# }}}


__all__ = (
    "Scheme",
    "CenteredScheme",
    "ex_expansion",
)
