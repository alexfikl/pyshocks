# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Grid
^^^^

.. autoclass:: Grid
    :no-show-inheritance:
.. autoclass:: UniformGrid

.. autofunction:: make_uniform_grid

.. autoclass:: Quadrature
    :no-show-inheritance:

.. autofunction:: cell_average

.. autofunction:: norm
.. autofunction:: rnorm
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp


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

    .. attribute:: ncells

        Number of total cells (ghost and interior).

    .. attribute:: nfaces

        Number of total faces (ghost and interior).

    .. attribute:: x

        Array of cell center coordinates of shape ``(n + 2 g,)``.

    .. attribute:: f

        Array of face coordinates of shape ``(n + 2 g + 1,)``.

    .. attribute:: dx

        Array of cell sizes (same shape as :attr:`x`).

    .. attribute:: df

        Array of staggered cell sizes (centered at the face coordinates)
        of size ``(n + 2 g - 1,)``.

    .. attribute:: dx_min
    .. attribute:: dx_max

        Smallest (and largest) :attr:`dx` in the domain.

    .. attribute:: i_

        A :class:`slice` of interior indices, i.e. without the ghost cell layer.

    .. attribute:: g_

        A :class:`slice` of ghost indices. ``Grid.g_[+1]`` and ``Grid.g_[-1]``
        give the ghost indices on the right and left sides, respectively.
    """

    a: float
    b: float

    x: jnp.ndarray
    f: jnp.ndarray
    nghosts: int

    # NOTE: these are here explicitly because jax.jit does not work well with
    # caching attributes (they leak outside on the first call to a jitted function)

    dx: jnp.ndarray
    df: jnp.ndarray
    dx_min: float
    dx_max: float

    @property
    def n(self):
        return self.ncells - 2 * self.nghosts

    @property
    def ncells(self):
        return self.x.size

    @property
    def nfaces(self):
        return self.f.size

    @property
    def g_(self):
        return (None, jnp.s_[: +self.nghosts], jnp.s_[-self.nghosts :])

    @property
    def i_(self):
        return jnp.s_[self.nghosts : -self.nghosts]


@dataclass(frozen=True)
class UniformGrid(Grid):
    pass


def make_uniform_grid(a: float, b: float, n: int, *, nghosts: int = 1) -> UniformGrid:
    """
    :arg a: left boundary of the domain :math:`[a, b]`.
    :arg b: right boundary of the domain :math:`[a, b]`.
    :arg n: number of cells the discretize the domain.
    :arg nghosts: number of ghost cells on each side of the domain.
    """
    if b < a:
        raise ValueError(f"incorrect interval a > b: '{a}' > '{b}'")

    if n <= 0:
        raise ValueError(f"number of cells should be > 0: '{n}'")

    if nghosts <= 0:
        raise ValueError(f"number of ghost cells should be > 0: '{nghosts}'")

    dx0 = (b - a) / n

    f = jnp.linspace(
        a - nghosts * dx0, b + nghosts * dx0, n + 2 * nghosts + 1,
        dtype=jnp.float64)
    x = (f[1:] + f[:-1]) / 2

    df = jnp.diff(x)
    dx = jnp.diff(f)

    # NOTE: this seems to fail quite generously; why is that?
    # assert jnp.linalg.norm(dx - dx0, ord=jnp.inf) < 1.0e-12 * abs(dx0)

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


# {{{ cell averaging


@dataclass(frozen=True)
class Quadrature:
    """Gauss-Legendre quadrature of given :attr:`order`.

    .. attribute:: grid
    .. attribute:: order
    .. attribute:: x

        Quadrature points on the grid :attr:`grid`. The array has a shape of
        ``(npoints, ncells)``.

    .. attribute:: w

        Quadrature weights of the same size as :attr:`x`.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    grid: Grid
    order: int

    x: jnp.ndarray = field(init=False, repr=False)
    w: jnp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.order < 1:
            raise ValueError(f"invalid order: '{self.order}'")

        # get Gauss-Legendre quadrature nodes and weights
        from numpy.polynomial.legendre import leggauss

        xi, wi = leggauss(self.order)
        xi = jax.device_put(xi).reshape(-1, 1)
        wi = jax.device_put(wi).reshape(-1, 1)

        # get grid sizes
        dx = 0.5 * self.grid.dx.reshape(1, -1)
        xm = self.grid.x.reshape(1, -1)

        # translate
        xi = xm + dx * xi
        wi = dx * wi

        object.__setattr__(self, "x", xi)
        object.__setattr__(self, "w", wi)

    def __call__(self, fn):
        """Integral over the grid of the function."""
        return jnp.sum(fn(self.x) * self.w)


def cell_average(quad: Quadrature, fn, staggered=False) -> jnp.ndarray:
    r"""Computes the cell average of the callable *fn* as

    .. math::

        f_i = \frac{1}{|\Omega_i|} \int_{\Omega_i} f(x) \,\mathrm{d}x.

    :param fn: a callable taking an array of size ``(npoints, ncells)``.
    """
    if quad.order == 1:
        return fn(quad.grid.x)

    return jnp.sum(fn(quad.x) * quad.w, axis=0) / jnp.sum(quad.w, axis=0)


# }}}


# {{{ norms


@partial(jax.jit, static_argnums=(2,))
def _norm(x, dx, p):
    if p == 1:
        return jnp.sum(x * dx)

    if p == 2:
        return jnp.sqrt(jnp.sum(x**2 * dx))

    if p == jnp.inf:
        return jnp.max(x)

    if p == -jnp.inf:
        return jnp.min(x)

    from numbers import Number

    if isinstance(p, Number):
        return jnp.sum(x**p * dx) ** (-1.0 / p)

    raise ValueError(f"unrecognized 'p': {p}")


def norm(grid: Grid, x: jnp.ndarray, *, p: float = 2, weighted: bool = False) -> float:
    r"""Computes the interior :math:`\ell^p` norm of *x*."""
    dx = grid.dx[grid.i_] if weighted else 1.0
    x = jnp.abs(x[grid.i_])

    return _norm(x, dx, p)


def rnorm(
    grid: Grid,
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    p: float = 2,
    weighted: bool = False,
    atol: float = 1.0e-14,
) -> float:
    r"""Computes the interior :math:`\ell^p` relative error norm of *x* and *y*
    as

    .. math::

        \frac{\|x - y\|_p}{\|y\|_p},

    where the numerator is ignored if it is close to zero.
    """
    ynorm = norm(grid, y, p=p)
    if ynorm < atol:
        ynorm = 1.0

    return norm(grid, x - y, p=p) / ynorm


# }}}
