# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Finite Difference Approximations
--------------------------------

.. autoclass:: Stencil
    :members:
.. autofunction:: determine_stencil_truncation_error
.. autofunction:: make_taylor_approximation
.. autofunction:: make_fornberg_approximation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from pyshocks.tools import Array, Scalar, ScalarLike

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
    """

    derivative: int
    """Order of the derivative approximated by the stencil."""
    order: int
    """Order of the approximation."""
    coeffs: Array
    """Coefficients used in the stencil."""
    indices: Array
    """Indices around the centered :math:`0` used in the stencil."""
    trunc: Scalar
    r"""Estimated truncation error, which assumes the next derivative in the
    expansion is :math:`\mathcal{O}(1)`.
    """

    padded_coeffs: Array
    """A version of :attr:`coeffs` that is symmetric around 0."""

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        from fractions import Fraction

        import jax

        coeffs = jax.device_get(self.coeffs)
        indices = jax.device_get(self.indices)

        a = ", ".join([
            f"{i}: {Fraction(a).limit_denominator()}" for a, i in zip(coeffs, indices)
        ])

        return (
            f"{type(self).__name__}("
            f"derivative={self.derivative}, "
            f"order={self.order}, "
            f"coeffs={{{a}}})"
        )


def determine_stencil_truncation_error(
    derivative: int,
    a: Array,
    indices: Array,
    *,
    atol: float = 1.0e-6,
) -> tuple[int, Scalar]:
    r"""Determine the order and truncation error for the stencil *a* and *indices*.

    .. math::

        \frac{\mathrm{d}^n\, f}{\mathrm{d}\, x^n}
        - \sum_{i \in \text{indices}} \frac{a_i}{h^n} f_i
        = c \frac{\mathrm{d}^p\, f}{\mathrm{d}\, x^p},

    where :math:`c` is the expected truncation error coefficient and :math:`p`
    is the order of the approximation.
    """

    c = jnp.array(0.0, dtype=a.dtype)
    i = derivative
    indices = indices.astype(a.dtype)
    while i < 64 and jnp.allclose(c, 0.0, atol=atol, rtol=0.0):
        i += 1
        c = a @ indices**i / math.factorial(i)

    return i - derivative, c


def apply_derivative(d: Stencil, f: Array, h: ScalarLike) -> Array:
    return jnp.array(
        jnp.convolve(f, d.padded_coeffs, mode="same") / h**d.derivative, dtype=f.dtype
    )


# }}}


# {{{ Taylor finite difference approximation


def make_taylor_approximation(
    derivative: int,
    stencil: tuple[int, int],
    *,
    atol: float = 1.0e-6,
    dtype: Any = None,
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
    dtype = jnp.dtype(dtype)

    # setup
    indices = jnp.arange(stencil[0], stencil[1] + 1)
    A = jnp.array(
        [indices**i / math.factorial(i) for i in range(indices.size)], dtype=dtype
    )
    b = jnp.zeros(indices.shape, dtype=dtype)
    b = b.at[derivative].set(1)

    # determine coefficients
    x = jnp.linalg.solve(A, b)
    assert jnp.allclose(jnp.sum(x), 0.0)

    n = max(abs(stencil[0]), abs(stencil[1]))
    padded_x = jnp.zeros(2 * n + 1, dtype=x.dtype)
    padded_x = padded_x.at[n + indices].set(x)

    # determine order
    # FIXME: we can probably figure this out without a while loop
    order, c = determine_stencil_truncation_error(derivative, x, indices)

    return Stencil(
        derivative=derivative,
        order=order,
        coeffs=x,
        indices=indices,
        trunc=c,
        padded_coeffs=padded_x,
    )


# }}}


# {{{ Fornberg approximation


def make_fornberg_approximation(
    derivative: int,
    stencil: tuple[int, int],
    *,
    atol: float = 1.0e-6,
    dtype: Any = None,
) -> Stencil:
    r"""Determine a finite difference stencil by Fornberg's method [Fornberg1998]_.

    This method gives essentially the same results as :func:`make_taylor_approximation`
    but uses an alternative method. This function is less vectorized and likely
    to perform worse than the Taylor method if used in a loop.

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
    dtype = jnp.dtype(dtype)

    # setup
    x = jnp.arange(stencil[0], stencil[1] + 1, dtype=dtype)
    xd = 0

    # {{{ determine coefficients

    c = jnp.zeros((x.size, derivative + 1), dtype=dtype)
    c = c.at[0, 0].set(1.0)

    c1, c4 = jnp.array(1, dtype=c.dtype), x[0] - xd
    for i in range(1, x.size):
        j = jnp.arange(0, min(i, derivative) + 1)

        # NOTE: avoids implicit conversion to float
        cj = j.astype(dtype)
        c2, c5, c4 = jnp.array(1, dtype=c.dtype), c4, x[i] - xd

        for k in range(i):
            c3 = x[i] - x[k]
            c2, c6, c7 = c2 * c3, cj * c[k, j - 1], c[k, j]
            c = c.at[k, j].set((c4 * c7 - c6) / c3)

        c = c.at[i, j].set(c1 * (c6 - c5 * c7) / c2)
        c1 = c2

    # only need the last derivative
    c = c[:, -1]
    x = x.astype(jnp.int64)

    n = max(abs(stencil[0]), abs(stencil[1]))
    padded_c = jnp.zeros(2 * n + 1, dtype=c.dtype)
    padded_c = padded_c.at[n + x].set(c)

    # }}}

    # determine order
    # FIXME: we can probably figure this out without a while loop
    order, trunc = determine_stencil_truncation_error(derivative, c, x)

    return Stencil(
        derivative=derivative,
        order=order,
        coeffs=c,
        indices=x,
        trunc=trunc,
        padded_coeffs=padded_c,
    )


# }}}


# {{{ wavenumber


def modified_wavenumber(st: Stencil, k: Array) -> Array:
    km = jnp.empty(k.shape, dtype=jnp.complex128)

    for n in range(k.shape[0]):
        exp = jnp.exp(1.0j * st.indices * k[n])
        km = km.at[n].set(jnp.sum(st.coeffs * exp))

    return km


# }}}
