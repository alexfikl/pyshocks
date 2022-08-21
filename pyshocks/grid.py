# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Grid
^^^^

.. autoclass:: Grid
    :no-show-inheritance:

.. autoclass:: UniformGrid

.. autofunction:: make_uniform_cell_grid
.. autofunction:: make_uniform_point_grid

.. autoclass:: Quadrature
    :no-show-inheritance:

.. autofunction:: make_leggauss_quadrature
.. autofunction:: cell_average

.. autofunction:: norm
.. autofunction:: rnorm
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp

from pyshocks.tools import SpatialFunction

# {{{ grids


@dataclass(frozen=True)
class Grid:
    """
    .. attribute:: a
    .. attribute:: b

        Domain bounds for :math:`[a, b]`.

    .. attribute:: n

        Number of (interior) cells.

    .. attribute:: nghosts

        Number of ghost cells

    .. attribute:: x

        Array of solution point coordinates of shape ``(n + 2 g,)``.

    .. attribute:: dx

        Array of cell sizes (dependent on the representation).

    .. attribute:: f

        Midpoint between the solution points.

    .. attribute:: df

        Array of cell sizes between the staggered points :attr:`f`.

    .. attribute:: dx_min
    .. attribute:: dx_max

        Smallest (and largest) element size in the domain.

    .. attribute:: b_

        Gives the index of the first and last point :attr:`x` that is not a
        ghost point. The indices are obtained from ``Grid.b_[-1]`` and
        ``Grid.b_[+1]``, respectively. Note that these can be the same as
        :attr:`a` and :attr:`b`, but not necessarily.

    .. attribute:: i_

        A :class:`slice` of interior indices, i.e. without the ghost cell layer.

    .. attribute:: g_

        A :class:`tuple` of ghost indices. ``Grid.g_[+1]`` and ``Grid.g_[-1]``
        give a :class:`slice` for the ghost indices on the right and left sides,
        respectively.

    .. attribute:: gi_

        Similar to :attr:`g_`, but gives the first and last ``nghosts`` interior
        points.
    """

    a: float
    b: float
    nghosts: int

    x: jnp.ndarray
    dx: jnp.ndarray

    f: jnp.ndarray
    df: jnp.ndarray

    dx_min: float
    dx_max: float

    @property
    def n(self) -> int:
        return int(self.x.size) - 2 * self.nghosts

    @property
    def i_(self) -> slice:
        return jnp.s_[self.nghosts : self.x.size - self.nghosts]

    @property
    def b_(self) -> Tuple[None, int, int]:
        return (None, self.x.size - self.nghosts - 1, self.nghosts)

    @property
    def g_(self) -> Tuple[None, slice, slice]:
        return (None, jnp.s_[self.x.size - self.nghosts :], jnp.s_[: self.nghosts])

    @property
    def gi_(self) -> Tuple[None, slice, slice]:
        g = self.nghosts
        return (None, jnp.s_[self.x.size - 2 * g : self.x.size - g], jnp.s_[g : 2 * g])


@dataclass(frozen=True)
class UniformGrid(Grid):
    pass


# }}}


# {{{ finite volume / cell-focused


def make_uniform_cell_grid(a: float, b: float, n: int, *, nghosts: int = 1) -> Grid:
    """
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of cells that discretize the domain.
    :arg nghosts: number of ghost cells on each side of the domain.
    """
    if b < a:
        raise ValueError(f"incorrect interval a > b: '{a}' > '{b}'")

    if n <= 0:
        raise ValueError(f"number of cells should be > 0: '{n}'")

    dx0 = (b - a) / n

    f = jnp.linspace(
        a - nghosts * dx0, b + nghosts * dx0, n + 2 * nghosts + 1, dtype=jnp.float64
    )
    x = (f[1:] + f[:-1]) / 2

    df = jnp.diff(x)
    dx = jnp.diff(f)

    return UniformGrid(
        a=a,
        b=b,
        x=x,
        f=f,
        nghosts=nghosts,
        dx=dx,
        df=df,
        dx_min=jnp.min(dx),
        dx_max=jnp.max(dx),
    )


# }}}


# {{{ finite difference / point-centered


def make_uniform_point_grid(
    a: float,
    b: float,
    n: int,
    *,
    nghosts: int = 0,
    is_periodic: bool = False,
) -> UniformGrid:
    """
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of points that discretize the domain.
    :arg nghosts: number of ghost points on each side of the domain.
    """
    if b < a:
        raise ValueError(f"incorrect interval a > b: '{a}' > '{b}'")

    if n <= 0:
        raise ValueError(f"number of cells should be > 0: '{n}'")

    assert n >= 0
    assert nghosts >= 0

    dx0 = (b - a) / (n - 1)

    if is_periodic:
        if nghosts != 0:
            raise ValueError("ghost cells are not supported with periodicity")

        x = jnp.linspace(a, b, n, endpoint=False)
        f = jnp.hstack([(x[1:] + x[:-1]) / 2, x[-1] + dx0 / 2])  # type: ignore
    else:
        x = jnp.linspace(
            a - nghosts * dx0, b + nghosts * dx0, n + 2 * nghosts, dtype=jnp.float64
        )
        f = jnp.hstack([x[0], (x[1:] + x[:-1]) / 2, x[-1]])  # type: ignore

    dx = jnp.full_like(x, dx0)  # type: ignore[no-untyped-call]
    df = jnp.diff(f)

    assert jnp.linalg.norm(dx - dx0) / dx0 < 1.0e-8

    return UniformGrid(
        a=a,
        b=b,
        nghosts=nghosts,
        x=x,
        dx=dx,
        f=f,
        df=df,
        dx_min=jnp.min(dx),
        dx_max=jnp.max(dx),
    )


# }}}


# }}}


# {{{ cell averaging


@dataclass(frozen=True)
class Quadrature:
    """Compositve quadrature of given :attr:`order`.

    .. attribute:: order
    .. attribute:: nnodes

        Number of quadrature nodes in each cell.

    .. attribute:: ncells

        Number of cells in the domain.

    .. attribute:: x

        Quadrature points. The array has a shape of ``(nnodes, ncells)``.

    .. attribute:: w

        Quadrature weights of the same size as :attr:`x`.

    .. attribute:: dx

        Cell sizes, as given by :attr:`Grid.dx`. These are only used when
        computing cell averages.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    order: int
    x: jnp.ndarray
    w: jnp.ndarray

    dx: jnp.ndarray

    def __post_init__(self) -> None:
        if self.x.shape != self.w.shape:
            raise ValueError(
                "'x' and 'w' should have the same shape: "
                f"got {self.x.shape} and {self.w.shape}"
            )

    @property
    def nnodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def ncells(self) -> int:
        return int(self.x.shape[1])

    def __call__(self, fn: SpatialFunction, axis: Optional[int] = None) -> jnp.ndarray:
        """Integral over the grid of the function.

        :arg axis: If *None*, this computes the integral over the full grid.
            If *0*, a cellwise integral is computed and the resulting array
            has size :attr:`ncells`.
        """

        if axis not in (0, None):
            raise ValueError(f"unsupported axis value: '{axis}'")

        return jnp.sum(fn(self.x) * self.w, axis=axis)


def make_leggauss_quadrature(grid: Grid, order: int) -> Quadrature:
    """Construct a Gauss-Legendre quadrature"""
    if order < 1:
        raise ValueError(f"invalid order: '{order}'")

    # get Gauss-Legendre quadrature nodes and weights
    from numpy.polynomial.legendre import leggauss

    xi, wi = leggauss(order)  # type: ignore[no-untyped-call]
    xi = jax.device_put(xi).reshape(-1, 1)
    wi = jax.device_put(wi).reshape(-1, 1)

    # get grid sizes
    dx = 0.5 * grid.dx.reshape(1, -1)
    xm = grid.x.reshape(1, -1)

    # translate
    xi = xm + dx * xi
    wi = dx * wi

    return Quadrature(order=order, x=xi, w=wi, dx=grid.dx)


def cell_average(quad: Quadrature, fn: SpatialFunction) -> jnp.ndarray:
    r"""Computes the cell average of the callable *fn* as

    .. math::

        f_i = \frac{1}{|\Omega_i|} \int_{\Omega_i} f(x) \,\mathrm{d}x.

    :param fn: a callable taking an array of size ``(npoints, ncells)``.
    """
    return quad(fn, axis=0) / quad.dx


# }}}


# {{{ norms


@partial(jax.jit, static_argnums=(2,))
def _norm(u: jnp.ndarray, dx: jnp.ndarray, p: Union[str, float]) -> jnp.ndarray:
    if p == 1:
        return jnp.sum(u * dx)

    if p == 2:
        return jnp.sqrt(jnp.sum(u**2 * dx))

    if p in (jnp.inf, "inf"):
        return jnp.max(u)

    if p in (-jnp.inf, "-inf"):
        return jnp.min(u)

    if p == "tvd":
        return jnp.sum(jnp.abs(jnp.diff(u)))

    if isinstance(p, (int, float)):
        if p <= 0:
            raise ValueError(f"'p' must be a positive float: {p}")

        return jnp.sum(u**p * dx) ** (-1.0 / p)

    raise ValueError(f"unrecognized 'p': {p}")


def norm(
    grid: Grid, u: jnp.ndarray, *, p: Union[str, float] = 1, weighted: bool = False
) -> jnp.ndarray:
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
    dx = grid.dx[grid.i_] if weighted else 1.0
    u = jnp.abs(u[grid.i_])

    return _norm(u, dx, p)


def rnorm(
    grid: Grid,
    u: jnp.ndarray,
    v: jnp.ndarray,
    *,
    p: Union[str, float] = 1,
    weighted: bool = False,
    atol: float = 1.0e-14,
) -> jnp.ndarray:
    r"""Computes the interior :math:`\ell^p` relative error norm of *u* and *v*
    as

    .. math::

        \frac{\|x - y\|_p}{\|y\|_p},

    where the numerator is ignored if it is close to zero.
    """
    vnorm = norm(grid, v, p=p)
    if vnorm < atol:
        vnorm = 1.0

    return norm(grid, u - v, p=p) / vnorm


# }}}
