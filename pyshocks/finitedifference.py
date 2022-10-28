# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Finite Difference Approximations
--------------------------------

.. autoclass:: Stencil
.. autofunction:: determine_stencil_truncation_error
.. autofunction:: make_taylor_approximation
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp


# {{{ stencil


@dataclass(frozen=True)
class Stencil:
    r"""Approximation of a derivative by finite difference on a uniform grid.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}
        \approx \sum_{i \in \text{indices}} \frac{a_i}{h^n} f_i

    where :math:`a_i` are the given coefficients :attr:`coeffs` and :math:`f_i`
    are the point function evaluations. The approximation is to an order of
    :attr:`order` with a truncation order coefficient of :attr:`trunc`.

    .. attribute:: derivative
    .. attribute:: order
    .. attribute:: coeffs
    .. attribute:: indices
    .. attribute:: trunc

    .. attribute:: padded_coeffs
    """

    derivative: int
    order: int
    coeffs: jnp.ndarray
    indices: jnp.ndarray
    trunc: float

    @property
    def padded_coeffs(self) -> jnp.ndarray:
        n = jnp.max(jnp.abs(self.indices))

        c = jnp.zeros(2 * n + 1, dtype=self.coeffs.dtype)  # type: ignore
        c = c.at[n + self.indices].set(self.coeffs)

        return c

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        import jax
        from fractions import Fraction

        coeffs = jax.device_get(self.coeffs)
        indices = jax.device_get(self.indices)

        a = ", ".join(
            [f"{i}: {Fraction(a).limit_denominator()}" for a, i in zip(coeffs, indices)]
        )

        return (
            f"{type(self).__name__}("
            f"derivative={self.derivative}, "
            f"order={self.order}, "
            f"coeffs={{{a}}})"
        )


def determine_stencil_truncation_error(
    derivative: int,
    a: jnp.ndarray,
    indices: jnp.ndarray,
    *,
    atol: float = 1.0e-6,
) -> Tuple[int, float]:
    r"""Determine the order and truncation error for the stencil *a* and *indices*.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}
        - \sum_{i \in \text{indices}} \frac{a_i}{h^n} f_i
        = c \frac{\mathrm{d}^p\, f}{\mathrm{d}\, x^p},

    where :math:`c` is the expected truncation error coefficient and :math:`p`
    is the order of the approximation.
    """

    c = 0.0
    i = derivative
    while i < 64 and jnp.allclose(c, 0.0, atol=atol, rtol=0.0):
        i += 1
        c = a @ indices**i / math.factorial(i)

    return i - derivative, c


# }}}


# {{{ Taylor finite difference approximation


def make_taylor_approximation(
    derivative: int,
    stencil: Tuple[int, int],
    *,
    atol: float = 1.0e-6,
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Stencil:
    r"""
    :arg derivative: integer order of the approximated derivative, e.g. ``1`` for
        the first derivative.
    :arg stencil: left and right bounds on the stencil around a point :math:`x_i`.
        For example, ``(-1, 2)`` defines the 4 point stencil
        :math:`\{x_{i - 1}, x_i, x_{i + 1}, x_{i + 2}\}`.

    :arg atol: tolerance used in determining the order of the approximation.
    """
    assert len(stencil) == 2
    assert stencil[0] < stencil[1]
    assert derivative > 0

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # set up
    indices = jnp.arange(stencil[0], stencil[1] + 1)
    A = jnp.array(
        [indices**i / math.factorial(i) for i in range(indices.size)], dtype=dtype
    )
    b = jnp.zeros(indices.shape, dtype=dtype)  # type: ignore[no-untyped-call]
    b = b.at[derivative].set(1)

    # determine coefficients
    x = jnp.linalg.solve(A, b)
    assert jnp.allclose(jnp.sum(x), 0.0)

    # determine order
    # FIXME: we can probably figure this out without a while loop
    order, c = determine_stencil_truncation_error(derivative, x, indices)

    return Stencil(
        derivative=derivative, order=order, coeffs=x, indices=indices, trunc=c
    )


# }}}
