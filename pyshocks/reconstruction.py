# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Reconstruction
--------------

.. autoclass:: Reconstruction
    :no-show-inheritance:
.. autofunction:: reconstruct

.. autoclass:: ConstantReconstruction
.. autoclass:: MUSCL
.. autoclass:: WENOJS
.. autoclass:: WENOJS32
.. autoclass:: WENOJS53

.. autofunction:: reconstruction_ids
.. autofunction:: make_reconstruction_from_name
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, ClassVar, Dict, Tuple, Type

import jax.numpy as jnp

from pyshocks import Grid
from pyshocks.limiters import Limiter, limit

# {{{


@dataclass(frozen=True)
class Reconstruction:
    """Describes a reconstruction algorithm for finite volume type methods.

    In finite volume methods, we have access to averaged cell values. A
    recontruction algorithm will use these values to obtain high-order
    expressions for the values at cell faces, which are then used in flux
    computations.

    The reconstruct is performed using :func:`reconstruct`.

    .. attribute:: order

        Order of the reconstruction. Most schemes are based on a form of
        weighted interpolation for which this value represents the order of the
        interpolation.

    .. attribute:: stencil_width

        The stencil width required for the reconstruction.
    """

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @property
    def order(self) -> int:
        raise NotImplementedError(type(self).__name__)

    @property
    def stencil_width(self) -> int:
        raise NotImplementedError(type(self).__name__)


@singledispatch
def reconstruct(
    rec: Reconstruction, grid: Grid, u: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Reconstruct face values from the cell-averaged values *u*.

    In this implementation, we use the convention that::

            i - 1             i           i + 1
        --------------|--------------|--------------
                u^R_{i - 1}      u^R_i
                        u^L_i         u^L_{i + 1}

    i.e. :math:`u^R_i` referres to the right reconstructed value in cell :math:`i`
    and represents the value at :math:`x_{i + \frac{1}{2}}` and :math:`u^L_i`
    referres the left reconstructed value in the cell :math:`i` and represents
    the value at :math:`x_{i - \frac{1}{2}}`.

    :arg rec: a reconstruction algorithm.
    :arg grid: the computational grid representation.
    :arg u: variable to reconstruct at the left and right cell faces.

    :returns: a :class:`tuple` of ``(ul, ur)`` containing a reconstructed
        state on the left and right side of each cell face. The dimension
        of the returned arrays matches :attr:`pyshocks.Grid.ncells`.
    """
    raise NotImplementedError(type(rec).__name__)


# }}}


# {{{ first-order


@dataclass(frozen=True)
class ConstantReconstruction(Reconstruction):
    r"""A standard first-order finite volume reconstruction.

    For this reconstruction, we have that

    .. math::

        (u^R_i, u^L_i) = (u_i, u_i).

    which results in a first-order scheme.
    """

    @property
    def name(self) -> str:
        return "constant"

    @property
    def order(self) -> int:
        return 1

    @property
    def stencil_width(self) -> int:
        return 1


@reconstruct.register(ConstantReconstruction)
def _reconstruct_first_order(
    rec: ConstantReconstruction, grid: Grid, u: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert grid.nghosts >= rec.stencil_width
    return u, u


# }}}


# {{{ MUSCL


@dataclass(frozen=True)
class MUSCL(Reconstruction):
    r"""A second-order MUSCL (Monotonic Upstream-centered Scheme for
    Conservation Laws) reconstruction.

    The MUSCL reconstruction uses a limited linear interpolation to obtain
    the values at the cell faces by

    .. math::

        \begin{aligned}
        u^R_i =\,\, &
            u_i + \frac{\phi(r_i)}{2} (u_{i + 1} - u_i), \\
        u^L_i =\,\, &
            u_i - \frac{\phi(r_i)}{2} (u_{i + 1} - u_i), \\
        \end{aligned}

    where :math:`\phi` is the limiter given by :attr:`lm`. This method requires
    that the limiter takes values in :math:`\phi(r) \in [0, 1]`, which not all
    limiters do.

    .. attribute:: lm

        A :class:`~pyshocks.limiters.Limiter` used in the MUSCL reconstruction.
    """

    lm: Limiter

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{type(self.lm).__name__[:-7]}".lower()

    @property
    def order(self) -> int:
        return 2

    @property
    def stencil_width(self) -> int:
        return 2


@reconstruct.register(MUSCL)
def _reconstruct_muscl(
    rec: MUSCL, grid: Grid, u: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    from pyshocks import UniformGrid

    # FIXME: would this work on non-uniform grids? may need to change the
    # limiters a bit to also contain the dx slopes?
    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("MUSCL is only implemented for uniform grids")

    assert grid.nghosts >= rec.stencil_width
    phi = limit(rec.lm, grid, u)

    ur = u[:-1] + 0.5 * phi[:-1] * (u[1:] - u[:-1])
    ul = u[1:] - 0.5 * phi[1:] * (u[1:] - u[:-1])

    return jnp.pad(ul, (1, 0)), jnp.pad(ur, (0, 1))  # type: ignore[no-untyped-call]


# }}}


# {{{ WENOJS


@dataclass(frozen=True)
class WENOJS(Reconstruction):  # pylint: disable=abstract-method
    """A WENO-JS reconstruction from [JiangShu1996]_.

    .. [JiangShu1996] G.-S. Jiang, C.-W. Shu, *Efficient Implementation of
        Weighted ENO Schemes*,
        Journal of Computational Physics, Vol. 126, pp. 202--228, 1996,
        `DOI <http://dx.doi.org/10.1006/jcph.1996.0130>`__.

    .. attribute:: eps

        Small fudge factor used in smoothness indicators.
    """

    eps: float

    # coefficients
    a: ClassVar[jnp.ndarray]
    b: ClassVar[jnp.ndarray]
    c: ClassVar[jnp.ndarray]
    d: ClassVar[jnp.ndarray]

    @property
    def stencil_width(self) -> int:
        return self.order


@dataclass(frozen=True)
class WENOJS32(WENOJS):
    """A third-order WENO-JS scheme that decays to second-order in non-smooth
    regions.
    """

    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        from pyshocks.weno import weno_js_32_coefficients

        a, b, c, d = weno_js_32_coefficients()

        # NOTE: hack to keep the class frozen
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)

    @property
    def order(self) -> int:
        return 2


@dataclass(frozen=True)
class WENOJS53(WENOJS):
    """A fifth-order WENO-JS scheme that decays to third-order in non-smooth
    regions.
    """

    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        from pyshocks.weno import weno_js_53_coefficients

        a, b, c, d = weno_js_53_coefficients()

        # NOTE: hack to keep the class frozen
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "d", d)

    @property
    def order(self) -> int:
        return 3


def _reconstruct_weno_js_side(rec: WENOJS, u: jnp.ndarray) -> jnp.ndarray:
    from pyshocks.weno import weno_js_smoothness, weno_js_reconstruct

    beta = weno_js_smoothness(u, rec.a, rec.b)
    uhat = weno_js_reconstruct(u, rec.c)

    alpha = rec.d / (rec.eps + beta) ** 2
    omega = alpha / jnp.sum(alpha, axis=0, keepdims=True)

    return jnp.sum(omega * uhat, axis=0)


@reconstruct.register(WENOJS)
def _reconstruct_wenojs(
    rec: WENOJS, grid: Grid, u: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    from pyshocks import UniformGrid

    assert grid.nghosts >= rec.stencil_width

    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("WENO-JS is only implemented for uniform grids")

    ur = _reconstruct_weno_js_side(rec, u)
    ul = _reconstruct_weno_js_side(rec, u[::-1])[::-1]

    return ul, ur


# }}}


# {{{ make_reconstruction_from_name

_RECONSTRUCTION: Dict[str, Type[Reconstruction]] = {
    "default": ConstantReconstruction,
    "constant": ConstantReconstruction,
    "muscl": MUSCL,
    "wenojs32": WENOJS32,
    "wenojs53": WENOJS53,
}


def reconstruction_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available reconstruction algorithms.
    """
    return tuple(_RECONSTRUCTION.keys())


def make_reconstruction_from_name(name: str, **kwargs: Any) -> Reconstruction:
    """
    :arg name: name of the reconstruction algorithm.
    :arg kwargs: additional arguments to pass to the algorithm. Any arguments
        that are not in the algorithm's fields are ignored.
    """
    cls = _RECONSTRUCTION.get(name)
    if cls is None:
        raise ValueError(
            f"flux limiter '{name}' not found; try one of {reconstruction_ids()}"
        )

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
