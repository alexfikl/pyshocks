# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Reconstruction
--------------

.. autoclass:: Reconstruction
    :no-show-inheritance:
    :members:
.. autofunction:: reconstruct

.. autoclass:: ConstantReconstruction
.. autoclass:: MUSCL
    :members:
.. autoclass:: MUSCLS
.. autoclass:: WENOJS
    :members:
.. autoclass:: WENOJS32
.. autoclass:: WENOJS53
.. autoclass:: ESWENO32
.. autoclass:: SSWENO242

.. autofunction:: reconstruction_ids
.. autofunction:: make_reconstruction_from_name
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple, Type

import jax.numpy as jnp

from pyshocks import weno
from pyshocks.grid import Grid
from pyshocks.limiters import Limiter
from pyshocks.tools import Array, ScalarLike

if TYPE_CHECKING:
    from pyshocks.schemes import BoundaryType


# {{{ interface


@dataclass(frozen=True)
class Reconstruction:
    """Describes a reconstruction algorithm.

    In finite volume methods, we have access to averaged cell values. A
    reconstruction algorithm will use these values to obtain high-order
    expressions for the values at cell faces, which are then used in flux
    computations. Alternatively, in finite difference methods, we have access
    to point (or node) values that are used to reconstruct in cell values.

    The reconstruct is performed using :func:`reconstruct`.
    """

    @property
    def name(self) -> str:
        """An identifier for the reconstruction algorithm."""
        return type(self).__name__.lower()

    @property
    def order(self) -> int:
        """Order of the reconstruction. Most schemes are based on a form of
        weighted interpolation for which this value represents the order of the
        interpolation."""
        raise NotImplementedError(type(self).__name__)

    @property
    def stencil_width(self) -> int:
        """The stencil width required for the reconstruction."""
        raise NotImplementedError(type(self).__name__)


@singledispatch
def reconstruct(
    rec: Reconstruction,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    r"""Reconstruct *f* as a function of *u*.

    In this implementation, we use the convention that::

            i - 1             i           i + 1
        --------------|--------------|--------------
                f^R_{i - 1}      f^R_i
                        f^L_i         f^L_{i + 1}

    i.e. :math:`f^R_i` refers to the right reconstructed value in cell :math:`i`
    and represents the value at :math:`x_{i + \frac{1}{2}}` and :math:`f^L_i`
    refers the left reconstructed value in the cell :math:`i` and represents
    the value at :math:`x_{i - \frac{1}{2}}`.

    Note that this notation can be directly interpreted in a finite difference
    setting, where the :math:`u_i, f_i` and :math:`w_i` are point values at
    cell faces used to reconstruct an in-cell quantity.

    :arg rec: a reconstruction algorithm.
    :arg grid: the computational grid representation.
    :arg bc: generic type of the boundary required to build the reconstruction
    :arg f: variable to reconstruct as a function of *u*.
    :arg u: base variable used in the reconstruction.
    :arg wavespeed: wave speed with which *u* is transported, that can be used
        to additionally upwind the reconstruction of *f*.

    :returns: a :class:`tuple` of ``(fl, fr)`` containing a reconstructed
        state on the left and right side of each cell face. The dimension
        of the returned arrays matches :attr:`pyshocks.Grid.x`.
    """
    raise NotImplementedError(type(rec).__name__)


# }}}


# {{{ first-order: constant reconstruction


@dataclass(frozen=True)
class ConstantReconstruction(Reconstruction):
    r"""A standard first-order constant reconstruction.

    For this reconstruction, we have that

    .. math::

        (f^R_i, f^L_i) = (f_i, f_i).

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
    rec: ConstantReconstruction,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    assert grid.nghosts >= rec.stencil_width
    return f, f


# }}}


# {{{ MUSCL: Flux Limiter


@dataclass(frozen=True)
class MUSCL(Reconstruction):
    r"""A second-order MUSCL (Monotonic Upstream-centered Scheme for
    Conservation Laws) reconstruction.

    The MUSCL reconstruction uses a limited linear interpolation to obtain
    the values at the cell faces by

    .. math::

        \begin{aligned}
        f^R_i =\,\, &
            f_i + \frac{\phi(r_i)}{2} (f_{i + 1} - f_i), \\
        f^L_i =\,\, &
            f_i - \frac{\phi(r_i^{-1})}{2} (f_i - f_{i - 1}), \\
        \end{aligned}

    where :math:`\phi` is a limiter given by :attr:`lm`. Note that here
    :math:`f \equiv f(u)` and the limiter :math:`\phi` is computed based on
    the underlying variable :math:`u`.
    """

    #: A :class:`~pyshocks.limiters.Limiter` used in the MUSCL reconstruction.
    lm: Limiter
    #: Absolute tolerance in cutting of the slope in constant regions.
    atol: float = 1.0e-12

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
    rec: MUSCL,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    from pyshocks import UniformGrid

    # FIXME: would this work on non-uniform grids? may need to change the
    # limiters a bit to also contain the dx slopes?
    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("MUSCL is only implemented for uniform grids.")

    assert grid.nghosts >= rec.stencil_width

    from pyshocks.limiters import evaluate, local_slope_ratio

    # FIXME: should this be f?
    r = jnp.pad(local_slope_ratio(u, atol=rec.atol), 1)
    phi_r = evaluate(rec.lm, r)

    # FIXME: the symmetric case has some more simplifications; worth it?
    if rec.lm.is_symmetric:
        phi_inv_r = jnp.where(jnp.abs(r) < rec.atol, 0.0, phi_r / r)
    else:
        inv_r = jnp.where(jnp.abs(r) < rec.atol, rec.atol, 1 / r)
        phi_inv_r = evaluate(rec.lm, inv_r)

    ur = jnp.pad(f[:-1] + 0.5 * phi_r[:-1] * (f[1:] - f[:-1]), (0, 1))
    ul = jnp.pad(f[1:] - 0.5 * phi_inv_r[1:] * (f[1:] - f[:-1]), (1, 0))

    return ul, ur


# }}}


# {{{ MUSCL: Slope Limiter


@dataclass(frozen=True)
class MUSCLS(MUSCL):
    r"""A second-order MUSCL (Monotonic Upstream-centered Scheme for
    Conservation Laws) reconstruction.

    Unlike :class:`MUSCL`, this class uses slope limiters to construct an
    approximation in each cell. The results are similar in most cases, but
    it is a bit simpler to construct. Note that not all limiters support
    :func:`~pyshocks.limiters.slope_limit`.

    Also worth noting is that, unlike the :class:`MUSCL` reconstruction, the
    limiter uses the function value *f* to construct the slope, not the
    underlying variable *u*. This is imposed by the formulation.
    """


@reconstruct.register(MUSCLS)
def _reconstruct_muscls(
    rec: MUSCLS,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    from pyshocks import UniformGrid

    # FIXME: would this work on non-uniform grids? may need to change the
    # limiters a bit to also contain the dx slopes?
    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("MUSCL is only implemented for uniform grids.")

    assert grid.nghosts >= rec.stencil_width

    from pyshocks.limiters import slope_limit

    du = slope_limit(rec.lm, grid, f)

    ur = f + 0.5 * grid.dx * du
    ul = f - 0.5 * grid.dx * du

    return ul, ur


# }}}


# {{{ WENOJS


@dataclass(frozen=True)
class WENOJS(Reconstruction):  # pylint: disable=abstract-method
    """A WENO-JS reconstruction from [JiangShu1996]_."""

    #: Small fudge factor used in smoothness indicators.
    eps: float
    #: A :class:`~pyshocks.weno.Stencil` for the reconstruction.
    s: ClassVar[weno.Stencil]

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
        object.__setattr__(self, "s", weno.weno_js_32_coefficients())

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
        object.__setattr__(self, "s", weno.weno_js_53_coefficients())

    @property
    def order(self) -> int:
        return 3


def _reconstruct_weno_js_side(rec: WENOJS, f: Array) -> Array:
    omega = weno.weno_js_weights(rec.s, f, eps=rec.eps)
    uhat = weno.weno_interp(rec.s, f)

    return jnp.sum(omega * uhat, axis=0)


@reconstruct.register(WENOJS)
def _reconstruct_wenojs(
    rec: WENOJS,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    from pyshocks import UniformGrid

    assert grid.nghosts >= rec.stencil_width

    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("WENO-JS is only implemented for uniform grids.")

    ur = _reconstruct_weno_js_side(rec, f)
    ul = _reconstruct_weno_js_side(rec, f[::-1])[::-1]

    return ul, ur


# }}}


# {{{ ESWENO


@dataclass(frozen=True)
class ESWENO32(Reconstruction):
    """Third-order WENO reconstruction for the Energy Stable WENO (ESWENO)
    scheme of [Yamaleev2009]_.

    Note that this scheme only implements the modified weights of [Yamaleev2009]_,
    not the entire energy stable WENO scheme.
    """

    eps: ScalarLike = 1.0e-6
    delta: ScalarLike = 1.0e-6

    # coefficients
    s: ClassVar[weno.Stencil]

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", weno.weno_js_32_coefficients())

    @property
    def order(self) -> int:
        return 2

    @property
    def stencil_width(self) -> int:
        return 2


def _reconstruct_es_weno_side(rec: ESWENO32, f: Array) -> Array:
    omega = weno.es_weno_weights(rec.s, f, eps=rec.eps)
    uhat = weno.weno_interp(rec.s, f)

    return jnp.sum(omega * uhat, axis=0)


@reconstruct.register(ESWENO32)
def _reconstruct_esweno32(
    rec: ESWENO32,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    from pyshocks import UniformGrid

    assert grid.nghosts >= rec.stencil_width

    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("ES-WENO is only implemented for uniform grids.")

    ur = _reconstruct_es_weno_side(rec, f)
    ul = _reconstruct_es_weno_side(rec, f[::-1])[::-1]

    return ul, ur


# }}}


# {{{ SSWENO


@dataclass(frozen=True)
class SSWENO242(Reconstruction):
    """Fourth-order WENO reconstruction for the Entropy Stable WENO (SSWENO)
    scheme of [Fisher2013]_.

    Note that the scheme deteriorates to second-order in regions of strong
    gradients (and shocks) and boundaries. Boundaries are expected to be
    implemented as described in [Fisher2013]_ to achieve an entropy-stable
    scheme.
    """

    eps: float = 1.0e-6

    # coefficients
    si: ClassVar[weno.Stencil]
    sb: ClassVar[weno.Stencil]

    def __post_init__(self) -> None:
        object.__setattr__(self, "si", weno.ss_weno_242_interior_coefficients())
        object.__setattr__(self, "sb", weno.ss_weno_242_boundary_coefficients())

    @property
    def order(self) -> int:
        return 2

    @property
    def stencil_width(self) -> int:
        return 3


def _reconstruct_ss_weno_side(
    rec: SSWENO242,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
) -> Array:
    if grid.nghosts >= rec.stencil_width:
        w = f
    else:
        assert grid.nghosts == 0

        # FIXME: put this in the weno code to "prepare" for WENO
        from pyshocks.schemes import BoundaryType

        if bc == BoundaryType.Periodic:
            w = jnp.pad(f, rec.stencil_width, mode="wrap")
        else:
            w = jnp.pad(f, rec.stencil_width, constant_values=jnp.inf)

    omega = weno.ss_weno_242_weights(rec.si, w, eps=rec.eps)
    what = weno.weno_interp(rec.si, w)

    return jnp.sum(omega * what[1:-1], axis=0)


@reconstruct.register(SSWENO242)
def _reconstruct_ssweno242(
    rec: SSWENO242,
    grid: Grid,
    bc: "BoundaryType",
    f: Array,
    u: Array,
    wavespeed: Array,
) -> Tuple[Array, Array]:
    from pyshocks import UniformGrid

    if not isinstance(grid, UniformGrid):
        raise NotImplementedError("SS-WENO is only implemented for uniform grids.")

    ur = _reconstruct_ss_weno_side(rec, grid, bc, f)
    ul = _reconstruct_ss_weno_side(rec, grid, bc, f[::-1])[::-1]

    return ul, ur


# }}}


# {{{ make_reconstruction_from_name

_RECONSTRUCTION: Dict[str, Type[Reconstruction]] = {
    "default": ConstantReconstruction,
    "constant": ConstantReconstruction,
    "muscl": MUSCL,
    "muscls": MUSCLS,
    "wenojs32": WENOJS32,
    "wenojs53": WENOJS53,
    "esweno32": ESWENO32,
    "ssweno242": SSWENO242,
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
        from pyshocks.tools import join_or

        raise ValueError(
            f"Flux limiter {name!r} not found. Try one of "
            + f"{join_or(reconstruction_ids())}."
        )

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
