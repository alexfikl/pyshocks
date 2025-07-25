# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Grid
^^^^

.. autoclass:: Grid
    :no-show-inheritance:
    :members:
.. autoclass:: UniformGrid

.. autofunction:: make_uniform_cell_grid
.. autofunction:: make_uniform_point_grid
.. autofunction:: make_uniform_ssweno_grid

Quadrature
^^^^^^^^^^
.. autoclass:: Quadrature
    :no-show-inheritance:
    :members:

.. autofunction:: make_leggauss_quadrature
.. autofunction:: cell_average

.. autofunction:: norm
.. autofunction:: rnorm
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from pyshocks.tools import Array, Scalar, ScalarLike, SpatialFunction

# {{{ grids


@dataclass(frozen=True)
class Grid:
    a: ScalarLike
    """Left domain bound for :math:`[a, b]`."""
    b: ScalarLike
    """Right domain bound for :math:`[a, b]`."""
    nghosts: int
    """Number of ghost cells."""

    x: Array
    """Array of solution point coordinates of shape ``(n + 2 g,)``."""
    dx: Array
    """Array of cell sizes (dependent on the representation)."""

    f: Array
    """Midpoint between the solution points."""
    df: Array
    """Array of cell sizes between the staggered points :attr:`f`."""

    dx_min: Scalar
    """Smallest cell size in the domain."""
    dx_max: Scalar
    """Largest cell size in the domain."""

    is_periodic: bool
    """If *True*, the grid is assumed to be periodic."""

    @property
    def dtype(self) -> jnp.dtype[Any]:
        return jnp.dtype(self.x.dtype)

    @property
    def n(self) -> int:
        """Number of (interior) cells."""
        return int(self.x.size)  # - 2 * self.nghosts

    @property
    def i_(self) -> slice:
        """A :class:`slice` of interior indices, i.e. without the ghost cell layer."""
        return jnp.s_[self.nghosts : self.x.size - self.nghosts]

    @property
    def f_(self) -> slice:
        """A :class:`slice` of interior staggered indices, i.e. without the
        ghost cell layer."""
        return jnp.s_[self.nghosts : self.f.size - self.nghosts]

    @property
    def b_(self) -> tuple[int, int, int]:
        """Gives the index of the first and last point :attr:`x` that is not a
        ghost point. The indices are obtained from ``Grid.b_[-1]`` and
        ``Grid.b_[+1]``, respectively. Note that these can be the same as
        :attr:`a` and :attr:`b`, but not necessarily.
        """
        # NOTE: -1 should be unused, but we set it to -1 for type consistency
        return (-1, self.x.size - self.nghosts - 1, self.nghosts)

    @property
    def g_(self) -> tuple[None, slice, slice]:
        """A :class:`tuple` of ghost indices. ``Grid.g_[+1]`` and ``Grid.g_[-1]``
        give a :class:`slice` for the ghost indices on the right and left sides,
        respectively.
        """
        return (None, jnp.s_[self.x.size - self.nghosts :], jnp.s_[: self.nghosts])

    @property
    def gi_(self) -> tuple[None, slice, slice]:
        """Similar to :attr:`g_`, but gives the first and last ``nghosts`` interior
        points."""
        g = self.nghosts
        return (None, jnp.s_[self.x.size - 2 * g : self.x.size - g], jnp.s_[g : 2 * g])


@dataclass(frozen=True)
class UniformGrid(Grid):
    pass


# }}}


# {{{ finite volume / cell-focused


def make_uniform_cell_grid(
    a: ScalarLike,
    b: ScalarLike,
    n: int,
    *,
    nghosts: int = 1,
    dtype: Any = None,
) -> Grid:
    """
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of cells that discretize the domain.
    :arg nghosts: number of ghost cells on each side of the domain.
    """
    if b < a:
        raise ValueError(f"Incorrect interval a > b: {a!r} > {b!r}.")

    if n <= 0:
        raise ValueError(f"Number of cells should be > 0: {n!r} <= 0.")

    assert n >= 0
    assert nghosts >= 0

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    h = (b - a) / n

    f = jnp.linspace(a - nghosts * h, b + nghosts * h, n + 2 * nghosts + 1, dtype=dtype)
    x = (f[1:] + f[:-1]) / 2

    df = jnp.diff(x)
    dx = jnp.full_like(x, h)

    assert jnp.linalg.norm(jnp.diff(f) - h) < 1.0e-8 * h

    return UniformGrid(
        a=a,
        b=b,
        x=x,
        f=f,
        nghosts=nghosts,
        dx=dx,
        df=df,
        dx_min=jnp.array(h, dtype=x.dtype),
        dx_max=jnp.array(h, dtype=x.dtype),
        is_periodic=False,
    )


# }}}


# {{{ finite difference / point-centered


def make_uniform_point_grid(
    a: ScalarLike,
    b: ScalarLike,
    n: int,
    *,
    nghosts: int = 0,
    is_periodic: bool = False,
    dtype: Any = None,
) -> UniformGrid:
    """
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of points that discretize the domain.
    :arg nghosts: number of ghost points on each side of the domain.
    """
    if b < a:
        raise ValueError(f"Incorrect interval a > b: {a!r} > {b!r}.")

    if n <= 0:
        raise ValueError(f"Number of cells should be > 0: {n!r} <= 0.")

    assert n >= 0
    assert nghosts >= 0

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    h = (b - a) / (n - 1)
    df = jnp.full(n + 2 * nghosts - 1, h, dtype=dtype)

    if is_periodic:
        if nghosts == 0:
            x = jnp.linspace(a, b, n - 1, endpoint=False, dtype=dtype)
            f = jnp.hstack([(x[1:] + x[:-1]) / 2, x[-1] + h / 2])
        else:
            x = jnp.linspace(
                a - nghosts * h,
                b + nghosts * h,
                n + 2 * nghosts - 1,
                endpoint=False,
                dtype=dtype,
            )
            f = jnp.hstack([(x[1:] + x[:-1]) / 2, x[-1] + h / 2])
        dx = df
    else:
        x = jnp.linspace(a - nghosts * h, b + nghosts * h, n + 2 * nghosts, dtype=dtype)
        f = jnp.hstack([x[0], (x[1:] + x[:-1]) / 2, x[-1]])
        dx = jnp.diff(f)

    assert jnp.linalg.norm(jnp.diff(x) - h) < 1.0e-8 * h
    assert dx.shape == x.shape

    return UniformGrid(
        a=a,
        b=b,
        nghosts=nghosts,
        x=x,
        dx=dx,
        f=f,
        df=df,
        dx_min=jnp.array(h, dtype=x.dtype),
        dx_max=jnp.array(h, dtype=x.dtype),
        is_periodic=is_periodic,
    )


# }}}


# {{{ SS-WENO grid


def make_uniform_ssweno_grid(
    a: ScalarLike,
    b: ScalarLike,
    n: int,
    *,
    is_periodic: bool = False,
    dtype: Any = None,
) -> UniformGrid:
    """Construct the complementary grid described in [Fisher2011]_.

    In the periodic case, this is the standard grid, so it matches the one
    produced by :func:`make_uniform_point_grid`.

    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of points that discretize the domain.
    """
    if b < a:
        raise ValueError(f"Incorrect interval a > b: {a!r} > {b!r}.")

    if n <= 0:
        raise ValueError(f"Number of cells should be > 0: {n!r} <= 0.")

    if dtype is None:
        dtype = jnp.dtype(jnp.float64)
    dtype = jnp.dtype(dtype)

    assert n >= 0

    h = (b - a) / (n - 1)
    df = jnp.full(n, h, dtype=dtype)

    if is_periodic:
        x = jnp.linspace(a, b, n - 1, endpoint=False, dtype=dtype)
        f = x + h / 2
        dx = df
    else:
        x = jnp.linspace(a, b, n, dtype=dtype)

        from pyshocks.sbp import make_sbp_42_norm_stencil, make_sbp_matrix_from_stencil
        from pyshocks.schemes import BoundaryType

        p = make_sbp_matrix_from_stencil(
            BoundaryType.Dirichlet,
            n,
            make_sbp_42_norm_stencil(dtype=dtype),
            weight=1,
        )

        f = jnp.zeros(n + 1, dtype=dtype)
        f = f.at[0].set(x[0])
        f = f.at[1:].set(h * p)
        f = jnp.cumsum(f)

        dx = jnp.diff(f)

    assert jnp.linalg.norm(jnp.diff(x) - h) < 1.0e-8 * h
    assert jnp.linalg.norm(df - h) < 1.0e-8 * h

    return UniformGrid(
        a=a,
        b=b,
        nghosts=0,
        x=x,
        dx=dx,
        f=f,
        df=df,
        dx_min=jnp.array(h, dtype=x.dtype),
        dx_max=jnp.array(h, dtype=x.dtype),
        is_periodic=is_periodic,
    )


# }}}


# }}}


# {{{ cell averaging


@dataclass(frozen=True)
class Quadrature:
    """Compositve quadrature of given :attr:`order`.

    .. automethod:: __call__
    """

    order: int
    """Expected order of convergence of the quadrature"""
    x: Array
    """Quadrature points. The array has a shape of ``(nnodes, ncells)``."""
    w: Array
    """Quadrature weights of the same size as :attr:`x`."""

    dx: Array
    """Cell sizes, as given by :attr:`Grid.dx`. These are only used when
    computing cell averages.
    """

    def __post_init__(self) -> None:
        if self.x.shape != self.w.shape:
            raise ValueError(
                "'x' and 'w' should have the same shape: "
                f"got {self.x.shape} and {self.w.shape}."
            )

    @property
    def nnodes(self) -> int:
        """Number of quadrature nodes in each cell."""
        return int(self.x.shape[0])

    @property
    def ncells(self) -> int:
        """Number of cells in the domain."""
        return int(self.x.shape[1])

    def __call__(self, fn: SpatialFunction, axis: int | None = None) -> Array:
        """Integral over the grid of the function.

        :arg axis: If *None*, this computes the integral over the full grid.
            If *0*, a cellwise integral is computed and the resulting array
            has size :attr:`ncells`.
        """

        if axis not in {0, None}:
            raise ValueError(f"Unsupported axis value: {axis!r}.")

        return jnp.sum(fn(self.x) * self.w, axis=axis)


def make_leggauss_quadrature(grid: Grid, order: int) -> Quadrature:
    """Construct a Gauss-Legendre quadrature"""
    if order < 1:
        raise ValueError(f"Invalid order: {order!r}.")

    if grid.is_periodic and grid.nghosts == 0:
        f = jnp.pad(grid.f, (1, 0), constant_values=grid.x[0] - grid.dx[0] / 2)
    else:
        f = grid.f

    return make_leggauss_quadrature_from_points(f, order)


def make_leggauss_quadrature_from_points(x: Array, order: int) -> Quadrature:
    # get Gauss-Legendre quadrature nodes and weights
    from numpy.polynomial.legendre import leggauss

    xi, wi = leggauss(order)
    xi = jax.device_put(xi).reshape(-1, 1)
    wi = jax.device_put(wi).reshape(-1, 1)

    # get grid sizes
    dx = x[1:] - x[:-1]
    dxm = 0.5 * dx.reshape(1, -1)
    xm = 0.5 * (x[1:] + x[:-1]).reshape(1, -1)

    # translate
    xim = xm + dxm * xi
    wim = dxm * wi

    return Quadrature(order=order, x=xim, w=wim, dx=dx)


def cell_average(quad: Quadrature, fn: SpatialFunction) -> Array:
    r"""Computes the cell average of the callable *fn* as

    .. math::

        f_i = \frac{1}{|\Omega_i|} \int_{\Omega_i} f(x) \,\mathrm{d}x.

    :arg fn: a callable taking an array of size ``(npoints, ncells)``.
    """
    return quad(fn, axis=0) / quad.dx


# }}}


# {{{ norms


@partial(jax.jit, static_argnums=(2,))
def _norm(u: Array, dx: Array, p: str | ScalarLike) -> Scalar:
    u = jnp.abs(u)

    if p == 1:
        return jnp.sum(u * dx)

    if p == 2:
        return jnp.sqrt(jnp.sum(u**2 * dx))

    if p in {jnp.inf, "inf"}:
        return jnp.max(u)

    if p in {-jnp.inf, "-inf"}:
        return jnp.min(u)

    if p == "tvd":
        return jnp.sum(jnp.abs(jnp.diff(u)))

    if isinstance(p, (int, float, jax.Array)):
        p = float(p)
        if p <= 0:
            raise ValueError(f"'p' must be a positive float: {p!r} <= 0.")

        return jnp.sum(u**p * dx) ** (-1.0 / p)

    raise ValueError(f"Unrecognized norm order 'p': {p!r}.")


def norm(
    grid: Grid,
    u: float | Array,
    *,
    p: str | ScalarLike = 1,
    weighted: bool = False,
) -> Scalar:
    r"""Computes the interior :math:`\ell^p` norm of *u*.

    Note that the weighted :math:`\ell^p` norm actually results in the
    :math:`L^p` norm, as *u* are cell averages. This is not the case of the
    remaining norms.

    :arg p: can be a (positive) floating point value or a string. The order can
        also be ``"inf"`` or ``"-inf"`` or ``"tvd"`` for the maximum, minimum
        or the total variation.
    :arg weighted: if *True*, the standard :math:`p` are weighted by the cell
        sizes.
    """
    from numbers import Number

    if isinstance(u, (float, Number)) or u.shape == ():
        return jnp.array(jnp.abs(u), dtype=grid.x.dtype)

    if u.shape[:1] == grid.x.shape:
        dx = grid.dx[grid.i_] if weighted else 1.0
        result = jnp.array(_norm(u[grid.i_], dx, p), dtype=u.dtype)
    elif u.shape[:1] == grid.f.shape:
        df = grid.df[grid.f_] if weighted else 1.0
        result = jnp.array(_norm(u[grid.f_], df, p), dtype=u.dtype)
    else:
        raise ValueError(f"Array has unexpected shape: {u.shape[:1]}")

    return result


def rnorm(
    grid: Grid,
    u: float | Array,
    v: float | Array,
    *,
    p: str | ScalarLike = 1,
    weighted: bool = False,
    atol: ScalarLike = 1.0e-14,
) -> Scalar:
    r"""Computes the interior :math:`\ell^p` relative error norm of *u* and *v*
    as

    .. math::

        \frac{\|x - y\|_p}{\|y\|_p},

    where the numerator is ignored if it is close to zero.
    """
    vnorm = norm(grid, v, p=p, weighted=weighted)
    if vnorm < atol:
        vnorm = jnp.array(1.0, dtype=grid.x.dtype)

    return norm(grid, u - v, p=p, weighted=weighted) / vnorm


# }}}
