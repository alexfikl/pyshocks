# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Finite Difference Approximations
--------------------------------

.. autoclass:: Stencil
.. autofunction:: determine_stencil_truncation_error
.. autofunction:: make_taylor_approximation
.. autofunction:: make_fornberg_approximation
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


def apply_stencil(d: Stencil, f: jnp.ndarray) -> jnp.ndarray:
    pass


# }}}


# {{{ Taylor finite difference approximation


def make_taylor_approximation(
    derivative: int,
    stencil: Tuple[int, int],
    *,
    atol: float = 1.0e-6,
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Stencil:
    r"""Determine a finite difference stencil by solving a linear system from the
    Taylor expansion.

    :arg derivative: integer order of the approximated derivative, e.g. ``1`` for
        the first derivative.
    :arg stencil: left and right bounds on the stencil around a point :math:`x_i`.
        For example, ``(-1, 2)`` defines the 4 point stencil
        :math:`\{x_{i - 1}, x_i, x_{i + 1}, x_{i + 2}\}`.

    :arg atol: tolerance used in determining the order of the approximation.
    """
    assert len(stencil) == 2
    assert stencil[0] < stencil[1]
    assert stencil[0] < 0 or stencil[1] > 0
    assert derivative > 0

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # setup
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


# {{{ Fornberg approximation


def make_fornberg_approximation(
    derivative: int,
    stencil: Tuple[int, int],
    *,
    atol: float = 1.0e-6,
    dtype: Optional["jnp.dtype[Any]"] = None,
) -> Stencil:
    r"""Determine a finite difference stencil by Fornberg's method [Fornberg1998]_.

    This method gives essentially the same results as :func:`make_taylor_approximation`
    but uses an alternative method. This function is less vectorized and likely
    to perform worse than the Taylor method if used in a loop.

    .. [Fornberg1998] B. Fornberg, *Calculation of Weights in Finite
        Difference Formulas*, SIAM Review, Vol. 40, pp. 685--691, 1998,
        `DOI <http://dx.doi.org/10.1137/s0036144596322507>`__.

    :arg derivative: integer order of the approximated derivative, e.g. ``1`` for
        the first derivative.
    :arg stencil: left and right bounds on the stencil around a point :math:`x_i`.
        For example, ``(-1, 2)`` defines the 4 point stencil
        :math:`\{x_{i - 1}, x_i, x_{i + 1}, x_{i + 2}\}`.

    :arg atol: tolerance used in determining the order of the approximation.
    """

    assert len(stencil) == 2
    assert stencil[0] < stencil[1]
    assert stencil[0] < 0 or stencil[1] > 0
    assert derivative > 0

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)

    # setup
    x = jnp.arange(stencil[0], stencil[1] + 1, dtype=dtype)
    xd = 0

    # {{{ determine coefficients

    c = jnp.zeros((x.size, derivative + 1), dtype=dtype)
    c = c.at[0, 0].set(1.0)

    c1, c4 = 1, x[0] - xd
    for i in range(1, x.size):
        j = jnp.arange(0, min(i, derivative) + 1)
        c2, c5, c4 = 1, c4, x[i] - xd

        for k in range(i):
            c3 = x[i] - x[k]
            c2, c6, c7 = c2 * c3, j * c[k, j - 1], c[k, j]
            c = c.at[k, j].set((c4 * c7 - c6) / c3)

        c = c.at[i, j].set(c1 * (c6 - c5 * c7) / c2)
        c1 = c2

    # only need the last derivative
    c = c[:, -1]

    # }}}

    # determine order
    # FIXME: we can probably figure this out without a while loop
    order, trunc = determine_stencil_truncation_error(derivative, c, x)

    return Stencil(
        derivative=derivative,
        order=order,
        coeffs=c,
        indices=x.astype(jnp.int64),
        trunc=trunc,
    )


# }}}
