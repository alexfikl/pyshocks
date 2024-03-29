# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Weighted Essentially Non-Oscillatory (WENO) Reconstruction
----------------------------------------------------------

.. autoclass:: Stencil
    :members:
.. autoclass:: BoundaryStencil
    :members:
.. autofunction:: weno_smoothness
.. autofunction:: weno_interp

WENO-JS
^^^^^^^

.. autofunction:: weno_js_32_coefficients
.. autofunction:: weno_js_53_coefficients
.. autofunction:: weno_js_weights

ES-WENO
^^^^^^^

.. autofunction:: es_weno_weights
.. autofunction:: es_weno_parameters

SS-WENO
^^^^^^^

.. autofunction:: ss_weno_242_coefficients
.. autofunction:: ss_weno_242_boundary_coefficients

.. autofunction:: ss_weno_242_parameters
.. autofunction:: ss_weno_242_weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from pyshocks.grid import Grid
from pyshocks.tools import Array, Scalar, ScalarLike

if TYPE_CHECKING:
    from pyshocks.convolve import ConvolutionType
    from pyshocks.schemes import BoundaryType

# {{{ weno


@dataclass(frozen=True)
class Stencil:
    r"""Stencil coefficients for a WENO reconstruction.

    Let the coefficients be :math:`a, b, c` and `d`. Then, the smoothness
    indicators are given by

    .. math::

        \beta_{i, m} = \sum_j a_j \left(\sum_k u_{m, k} b_{i, j, k}\right)^2,

    where :math:`u_{m, k}` represents the stencil around :math:`m`. See
    Equations 3.2-3.4 in [JiangShu1996] for an example of the fifth-order
    scheme. The value of the function is interpolated using the :math:`c`
    coefficients as

    .. math::

        \hat{u}_{i, m} = \sum u_{m, j} c_{i, j}.

    Finally, the weights are constructed from

    .. math::

        \alpha_{i, m} = \frac{d_i}{(\beta_{i, m} + \epsilon)^2}

    as

    .. math::

        \omega_{i, m} = \frac{\alpha_{i, m}}{\sum_n \alpha_{i, n}}.

    The general setup is more akin to the description provided in [Shu1998]_.
    """

    a: Array | None
    """Coefficients that are part of the smoothness coefficient expressions."""
    b: Array | None
    """Coefficients that define the smoothness stencils."""
    c: Array
    """Coefficients that define the solution interpolation stencils."""
    d: Array
    """Ideal weights for the WENO scheme, that define a high-order
    reconstruction using the stencils :attr:`c`.
    """


@dataclass(frozen=True)
class BoundaryStencil:
    si: Stencil
    """Interior stencil coefficients."""

    bc: BoundaryType
    """The type of the boundary, as given by :class:`~pyshocks.BoundaryType`."""
    sl: Stencil | None
    """If not *None*, the stencil for the left boundary."""
    sr: Stencil | None
    """If not *None*, the stencil for the right boundary."""


def weno_smoothness(
    s: Stencil, u: Array, *, mode: ConvolutionType | None = None
) -> Array:
    r"""Compute the smoothness coefficients for a WENO scheme.

    The coefficients must have the form

    .. math::

        \beta^r_m = \sum_j a_j \left(\sum u_{m - k} b_{r, j, k}\right)^2.

    :returns: the :math:`\beta_i` smoothness coefficient for each stencil around
        a cell :math:`m`. The shape of the returned array is
        ``(b.shape[0], u.size)``.
    """
    from pyshocks.convolve import convolve1d

    assert s.a is not None
    assert s.b is not None

    return jnp.stack([
        sum(
            s.a[j] * convolve1d(u, s.b[i, j, :], mode=mode) ** 2
            for j in range(s.a.size)
        )
        for i in range(s.b.shape[0])
    ])


def weno_interp(s: Stencil, u: Array, *, mode: ConvolutionType | None = None) -> Array:
    r"""Interpolate the variable *u* at the cell faces for WENO-JS.

    The interpolation has the form

    .. math::

        \hat{u}^r_m = \sum c_{r, k} u_{m - k}.

    :returns: the interpolated :math:`\hat{u}_i`, for each stencil in the
        scheme. The returned array has a shape of ``(c.shape[0], u.size)``.
    """
    from pyshocks.convolve import convolve1d

    return jnp.stack([convolve1d(u, s.c[i, :], mode=mode) for i in range(s.c.shape[0])])


# }}}


# {{{ WENOJS


def weno_js_32_coefficients(dtype: Any = None) -> Stencil:
    r"""Initialize the coefficients of the third-order WENO-JS scheme.

    :returns: a :class:`Stencil` containing the coefficients :math:`a, b, c`
        and :math:`d`. For a third-order scheme, :math:`a` is of shape ``(1,)``,
        :math:`b` is of shape ``(2, 3, 1)``, :math:`c` is of shape ``(2, 3)`` and
        :math:`d` is of shape ``(1, 2)``.
    """

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # NOTE: the arrays here are slightly modified from [Shu2009] by
    # * zero padding to the full stencil length
    # * flipping the stencils
    # so that they can be directly used with jnp.convolve for vectorization

    # smoothness indicator coefficients ([Shu1998] Equation 2.62)
    a = jnp.array([1.0], dtype=dtype)
    b = jnp.array(
        [
            # i + 1, i, i - 1
            [[0.0, 1.0, -1.0]],
            [[1.0, -1.0, 0.0]],
        ],
        dtype=dtype,
    )

    # stencil coefficients ([Shu1998] Table 2.1)
    c = jnp.array(
        [[0.0, 3.0 / 2.0, -1.0 / 2.0], [1.0 / 2.0, 1.0 / 2.0, 0.0]], dtype=dtype
    )

    # weights coefficients ([Shu1998] Equation 2.54)
    d = jnp.array([[1.0 / 3.0, 2.0 / 3.0]], dtype=dtype).T

    return Stencil(a=a, b=b, c=c, d=d)


def weno_js_53_coefficients(dtype: Any = None) -> Stencil:
    r"""Initialize the coefficients of the fifth-order WENO-JS scheme.

    :returns: a :class:`Stencil` containing the coefficients :math:`a, b, c`
        and :math:`d`. For a fifth-order scheme, :math:`a` is of shape ``(2,)``,
        :math:`b` is of shape ``(3, 4, 2)``, :math:`c` is of shape ``(3, 5)`` and
        :math:`d` is of shape ``(1, 3)``.
    """
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # smoothness indicator (Equation 2.17 [Shu2009])
    a = jnp.array([13.0 / 12.0, 1.0 / 4.0], dtype=dtype)
    b = jnp.array(
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [[0.0, 0.0, 1.0, -2.0, 1.0], [0.0, 0.0, 3.0, -4.0, 1.0]],
            [[0.0, 1.0, -2.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0]],
            [[1.0, -2.0, 1.0, 0.0, 0.0], [1.0, -4.0, 3.0, 0.0, 0.0]],
        ],
        dtype=dtype,
    )

    # stencil coefficients (Equation 2.11, 2.12, 2.13 [Shu2009])
    c = jnp.array(
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [0.0, 0.0, 11.0 / 6.0, -7.0 / 6.0, 2.0 / 6.0],
            [0.0, 2.0 / 6.0, 5.0 / 6.0, -1.0 / 6.0, 0.0],
            [-1.0 / 6.0, 5.0 / 6.0, 2.0 / 6.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    # weights coefficients (Equation 2.15 [Shu2009])
    d = jnp.array([[1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0]], dtype=dtype).T

    return Stencil(a=a, b=b, c=c, d=d)


def weno_js_weights(s: Stencil, u: Array, *, eps: ScalarLike) -> Array:
    r"""Compute the standard WENO-JS weights.

    :returns: the weights :math:`\omega_i` for each stencil interpolation
        from :func:`weno_interp`.
    """
    beta = weno_smoothness(s, u)
    alpha = s.d / (eps + beta) ** 2

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}


# {{{ ESWENO


def es_weno_parameters(grid: Grid, u0: Array) -> tuple[Scalar, Scalar]:
    """Estimate the ESWENO32 parameters from the grid and initial condition.

    :returns: ``(eps, delta)``, where ``eps`` is given by Equation 65
        and ``delta`` is given by Equation 66 in [Yamaleev2009]_.
    """
    # FIXME: eps should also contain the derivative of u0 away from discontinuities,
    # but not sure how to compute that in a vaguely accurate way. Also, we mostly
    # look at piecewise constant shock solutions where du0/dx ~ 0, so this
    # should be sufficient for the moment.

    # NOTE: as in Yamaleev2009, we're using the L1 norm for u0
    i = grid.i_
    eps = jnp.sum(grid.dx[i] * jnp.abs(u0[i])) * grid.dx_min**2
    delta = grid.dx_min**2

    return eps, delta


def es_weno_weights(s: Stencil, u: Array, *, eps: ScalarLike) -> Array:
    r"""Compute the ESWENO weights.

    :returns: the weights :math:`\omega_i` for each stencil interpolation
        from :func:`weno_interp`.
    """
    beta = weno_smoothness(s, u)

    # NOTE: see Equations 21-22 in [Yamaleev2009]
    tau = jnp.pad((u[2:] - 2 * u[1:-1] + u[:-2]) ** 2, 1)
    alpha = s.d * (1 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# }}}


# {{{ SSWENO


def ss_weno_242_parameters(grid: Grid, u0: Array) -> Scalar:
    """Estimate the SSWENO parameters from the grid and initial condition.

    :arg u0: initial condition evaluated at given points.
    :returns: ``eps``, as given by Equation 75 in [Fisher2011]_.
    """

    # FIXME: eps should also contain the derivative of u0 away from discontinuities,
    # but not sure how to compute that in a vaguely accurate way. Also, we mostly
    # look at piecewise constant shock solutions where du0/dx ~ 0, so this
    # should be sufficient for the moment.

    i = grid.i_
    return jnp.sum(grid.dx[i] * jnp.abs(u0[i])) * grid.dx_min**4


def ss_weno_242_weights(s: Stencil, u: Array, *, eps: ScalarLike) -> Array:
    beta = weno_smoothness(s, u)
    tau = jnp.pad((u[3:] - 3 * u[2:-1] + 3 * u[1:-2] - u[:-3]) ** 2, (1, 2))
    alpha = s.d * (1.0 + tau / (eps + beta))

    return alpha / jnp.sum(alpha, axis=0, keepdims=True)


# {{{ interpolation


def ss_weno_242_coefficients(bc: BoundaryType, dtype: Any = None) -> BoundaryStencil:
    from pyshocks.schemes import BoundaryType

    si = ss_weno_242_interior_coefficients(dtype)

    if bc == BoundaryType.Periodic:
        sl = sr = None
    else:
        sl, sr = ss_weno_242_boundary_coefficients(dtype)

    return BoundaryStencil(bc=bc, si=si, sl=sl, sr=sr)


def ss_weno_242_mask(sb: BoundaryStencil, u: Array) -> Array:
    from pyshocks.schemes import BoundaryType

    if sb.bc == BoundaryType.Periodic:
        mask = jnp.ones((3, u.size), dtype=u.dtype)
    else:
        assert sb.sl is not None
        assert sb.sr is not None
        mask = jnp.ones((5, u.size + 1), dtype=u.dtype)
        mask = mask.at[0, :].set(0)
        mask = mask.at[-1, :].set(0)

        d = sb.sl.d
        mask = mask.at[:, : d.shape[1]].set((d > 0).astype(u.dtype))
        d = sb.sr.d
        mask = mask.at[:, -d.shape[1] :].set((d > 0).astype(u.dtype))

    return mask


def ss_weno_242_interior_coefficients(dtype: Any = None) -> Stencil:
    """Initialize coefficients for the fourth-order WENO scheme of [Fisher2013].

    The actual implementation details of the scheme are given in [Fisher2011]_.

    :returns: a 4-tuple containing the coefficients :math:`a, b, c` and :math:`d`.
        For a third-order scheme, :math:`a` is of shape ``(1,)``, :math:`b`
        is of shape ``(3, 4, 1)``, :math:`c` is of shape ``(3, 5)`` and
        :math:`d` is of shape ``(1, 3)``.
    """

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # smoothness indicator coefficients ([Fisher2011] Equation 71)
    a = jnp.array([1.0], dtype=dtype)
    b = jnp.array(
        [
            # i + 3, i + 2, i + 1, i, i - 1, i - 2
            [[0.0, 0.0, 1.0, -1.0, 0.0]],  # L
            [[0.0, 1.0, -1.0, 0.0, 0.0]],  # C
            [[1.0, -1.0, 0.0, 0.0, 0.0]],  # R
        ],
        dtype=dtype,
    )

    # swatencil coefficients ([Fisher2011] Equation 77)
    c = jnp.array(
        [
            # i + 2, i + 1, i, i - 1, i - 2
            [0.0, 0.0, 3.0 / 2.0, -1.0 / 2.0, 0.0],  # L
            [0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0],  # C
            [-1.0 / 2.0, 3.0 / 2.0, 0.0, 0.0, 0.0],  # R
        ],
        dtype=dtype,
    )

    # weights coefficients ([Fisher2011] Equation 78)
    d = jnp.array([[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]], dtype=dtype).T

    return Stencil(a=a, b=b, c=c, d=d)


def ss_weno_242_boundary_coefficients(
    dtype: Any = None,
) -> tuple[Stencil, Stencil]:
    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    # boundary stencils ([Fisher2011] Equation 77)
    cl = jnp.array(
        [
            # I_LL
            [
                # i = 0, 1, 2, 3, 4, 5
                [0, 0, 0, 0, 0, 0],  # j = 1
                [0, 0, 0, 0, 0, 0],  # j = 2
                [0, 0, 0, 0, 0, 0],  # j = 3
                [-71 / 48, 119 / 48, 0, 0, 0, 0],  # j = 4
            ],
            # I_L
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-7 / 12, 19 / 12, 0, 0, 0, 0],
                [0, -23 / 48, 71 / 48, 0, 0, 0],
            ],
            # I_C
            [
                [1, 0, 0, 0, 0, 0],
                [31 / 48, 17 / 48, 0, 0, 0, 0],
                [0, 5 / 12, 7 / 12, 0, 0, 0],
                [0, 0, 25 / 48, 23 / 48, 0, 0],
            ],
            # I_R
            [
                [0, 0, 0, 0, 0, 0],
                [0, 79 / 48, -31 / 48, 0, 0, 0],
                [0, 0, 17 / 12, -5 / 12, 0, 0],
                [0, 0, 0, 73 / 48, -25 / 48, 0],
            ],
            # I_RR
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 127 / 48, -79 / 48, 0, 0],
                [0, 0, 0, 29 / 12, -17 / 12, 0],
                [0, 0, 0, 0, 121 / 48, -73 / 48],
            ],
        ],
        dtype=dtype,
    )
    cr = cl[::-1, ::-1, ::-1]

    # weights coefficients ([Fisher2011] Equation 78)
    dl = jnp.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 24 / 31, 1013 / 4898, 3 / 158],
            [0, 11 / 56, 51 / 70, 3 / 40, 0],
            [3 / 142, 357 / 3266, 408 / 575, 4 / 25, 0],
        ],
        dtype=dtype,
    ).T
    dr = dl[::-1, ::-1]

    return Stencil(a=None, b=None, c=cl, d=dl), Stencil(a=None, b=None, c=cr, d=dr)


def ss_weno_242_interp(sb: BoundaryStencil, u: Array) -> Array:
    from pyshocks.convolve import ConvolutionType
    from pyshocks.schemes import BoundaryType

    assert isinstance(sb.bc, BoundaryType)
    if sb.bc == BoundaryType.Periodic:
        assert sb.sl is None
        assert sb.sr is None
        uhat = weno_interp(sb.si, u, mode=ConvolutionType.Wrap)
    else:
        assert sb.sl is not None
        assert sb.sr is not None

        uhat = jnp.empty((5, u.size + 1), dtype=u.dtype)
        uhat = uhat.at[1:-1, 1:].set(weno_interp(sb.si, u))

        n, m = sb.sl.c.shape[1:]
        uhat = uhat.at[:, :n].set(sb.sl.c @ u[:m])
        n, m = sb.sr.c.shape[1:]
        uhat = uhat.at[:, -n:].set(sb.sr.c @ u[-m:])

    return uhat


# }}}

# }}}
