# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Schemes
^^^^^^^

.. autoclass:: SchemeBase
    :no-show-inheritance:

.. autoclass:: SchemeBaseV2

.. autoclass:: ConservationLawScheme
.. autoclass:: ConservationLawSchemeV2
.. autoclass:: CombineConservationLawScheme

.. autofunction:: flux
.. autofunction:: numerical_flux
.. autofunction:: apply_operator
.. autofunction:: predict_timestep

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: Boundary
    :no-show-inheritance:
.. autoclass:: OneSidedBoundary
.. autoclass:: TwoSidedBoundary

.. autofunction:: apply_boundary
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Optional, Tuple

import jax.numpy as jnp

from pyshocks.grid import Grid
from pyshocks.reconstruction import Reconstruction


# {{{ schemes


@dataclass(frozen=True)
class SchemeBase:
    r"""Describes numerical schemes for a type of PDE.

    The general form of the equations we will be looking at is

    .. math::

        \frac{\partial \mathbf{u}}{\partial t} =
            \mathbf{A}(t, \mathbf{x}, \mathbf{u}, \nabla \mathbf{u}).

    .. attribute:: order
    .. attribute:: stencil_width
    """

    @property
    def order(self) -> int:
        raise NotImplementedError

    @property
    def stencil_width(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class SchemeBaseV2(SchemeBase):
    """
    .. attribute:: rec

        A :class:`~pyshocks.reconstruction.Reconstruction` object that is used
        to reconstruct high-order face-based values when needed by the numerical
        scheme.
    """

    rec: Reconstruction

    @property
    def order(self) -> int:
        return self.rec.order

    @property
    def stencil_width(self) -> int:
        return self.rec.stencil_width


@singledispatch
def flux(scheme: SchemeBase, t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the physical flux at the given parameters.

    :param scheme: scheme for which to compute the (physical) flux. The
        scheme also identifies the type of equations we are solving.
    :param t: time at which to evaluate the flux.
    :param x: coordinates at which to evaluate the flux.
    :param u: variable values at which to evaluate the flux, whose size must
        match that of *x*.

    :returns: an array the size of *u* with the flux.
    """
    raise NotImplementedError(type(scheme).__name__)


@singledispatch
def numerical_flux(
    scheme: SchemeBase, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    """Approximate the flux at each cell-cell interface.

    :param scheme: scheme for which to compute the (numerical) flux.
    :param grid: grid on which to evaluate the flux.
    :param t: time at which to evaluate the flux.
    :param u: variable values at which to evaluate the flux, whose size must
        match the number of cells in the *grid* (including ghost cells).

    :returns: the numerical flux at each face in the mesh, i.e. matches the
        size of :attr:`Grid.f`.
    """
    raise NotImplementedError(type(scheme).__name__)


@singledispatch
def apply_operator(
    scheme: SchemeBase, grid: Grid, bc: "Boundary", t: float, u: jnp.ndarray
) -> jnp.ndarray:
    r"""Applies right-hand side operator for a "Method of Lines" approach.
    For any PDE, we have that

    .. math::

        \frac{\partial u}{\partial t} = A(t, u),

    where :math:`A` is the nonlinear differential operator (in general)
    computed by this function.

    :param scheme: scheme used to approximated the operator, effectively
        describes the :func:`numerical_flux` used.
    :param grid: grid on which to evaluate the operator.
    :param bc: boundary conditions for *u*.
    :param t: time at which to evaluate the operator.
    :param u: variable value at *t* used to evaluate the operator, whose size
        must match the number of cells in *grid* (including ghost cells).

    :returns: the numerical operator approximation at cell centers, i.e.
        matching the size of :attr:`Grid.x`.
    """
    raise NotImplementedError(type(scheme).__name__)


@singledispatch
def predict_timestep(
    scheme: SchemeBase, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    r"""Estimate the time step based on the current solution. This time step
    prediction can then be used together with a Courant number to give
    the final time step

    .. math::

        \Delta t = \theta \Delta \tilde{t},

    where the maximum allowable value of :math:`\theta` for stability very
    much depends on the time integrator that is used.

    :param scheme: scheme for which to compute the (numerical) flux.
    :param grid: grid on which to evaluate the flux.
    :param t: time at which to evaluate the flux.
    :param u: variable values at which to evaluate the flux, whose size must
        match the number of cells in the *grid* (including ghost cells).
    """
    raise NotImplementedError(type(scheme).__name__)


# }}}


# {{{ conservation law schemes


@dataclass(frozen=True)
class ConservationLawScheme(SchemeBase):  # pylint: disable=abstract-method
    r"""Describes numerical schemes for conservation laws.

    We consider conservation laws of the form

    .. math::

        \frac{\partial \mathbf{u}}{\partial t}
        + \nabla \cdot \mathbf{f}(\mathbf{u}) = 0,

    where :math:`\mathbf{f}` is the (possibly nonlinear) flux.
    """


@apply_operator.register(ConservationLawScheme)
def _apply_operator_conservation_law(
    scheme: ConservationLawScheme, grid: Grid, bc: "Boundary", t: float, u: jnp.ndarray
) -> jnp.ndarray:
    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -(f[1:] - f[:-1]) / grid.dx


@dataclass(frozen=True)
class ConservationLawSchemeV2(SchemeBaseV2):
    pass


@apply_operator.register(ConservationLawSchemeV2)
def _apply_operator_conservation_law_v2(
    scheme: ConservationLawSchemeV2,
    grid: Grid,
    bc: "Boundary",
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -(f[1:] - f[:-1]) / grid.dx


@dataclass(frozen=True)
class CombineConservationLawScheme(ConservationLawScheme):  # pylint: disable=W0223
    r"""Implements a combined operator of conservation laws.

    In this case, we consider a conservation law in the form

    .. math::

        \frac{\partial \mathbf{u}}{\partial t}
        + \sum_{k = 0}^M \nabla \cdot \mathbf{f}_k(\mathbf{u}) = 0,

    where each flux :math:`\mathbf{f}_k` is defined by an element of
    :attr:`schemes`. The main benefit of using this class is that it avoids
    applying the boundary conditions on every call to :func:`apply_operator`
    (and the additional convenience).

    .. attribute:: schemes

        A tuple of :class:`ConservationLawScheme` objects.
    """
    schemes: Tuple[ConservationLawScheme, ...]

    def __post_init__(self) -> None:
        assert len(self.schemes) > 1
        assert all(isinstance(s, ConservationLawScheme) for s in self.schemes)


@predict_timestep.register(CombineConservationLawScheme)
def _predict_time_combine_conservation_law(
    scheme: CombineConservationLawScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    dt = jnp.inf
    for s in scheme.schemes:
        dt = jnp.minimum(dt, predict_timestep(s, grid, t, u))

    return dt


@numerical_flux.register(CombineConservationLawScheme)
def _numerical_flux_combine_conversation_law(
    scheme: CombineConservationLawScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    f = numerical_flux(scheme.schemes[0], grid, t, u)
    for s in scheme.schemes[1:]:
        f = f + numerical_flux(s, grid, t, u)

    return f


@apply_operator.register(CombineConservationLawScheme)
def _apply_operator_combine_conservation_law(
    scheme: CombineConservationLawScheme,
    grid: Grid,
    bc: "Boundary",
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -(f[1:] - f[:-1]) / grid.dx


# }}}


# {{{ boundary conditions


@dataclass(frozen=True)
class Boundary:
    """Boundary conditions for one-dimensional domains."""


@singledispatch
def apply_boundary(bc: Boundary, grid: Grid, t: float, u: jnp.ndarray) -> jnp.ndarray:
    """Apply boundary conditions in place in the solution *u*.

    :param bc: boundary condition description.
    :param grid:
    :param t: time at which to evaluate the boundary condition.
    :param u: solution at the given time *t* of a size that matches
        :attr:`Grid.x`.
    :return: a copy of *u* with boundary conditions set in the ghost layer.
    """
    raise NotImplementedError(type(bc).__name__)


@dataclass(frozen=True)
class OneSidedBoundary(Boundary):
    """
    .. attribute:: side

        Integer ``+1`` or ``-1`` indicating the side on which this boundary
        condition applies.

    .. automethod:: __init__
    """

    side: int


@dataclass(frozen=True)
class TwoSidedBoundary(Boundary):
    """
    .. attribute:: left
    .. attribute:: right

    .. automethod:: __init__
    """

    left: Optional[OneSidedBoundary] = None
    right: Optional[OneSidedBoundary] = None

    def __post_init__(self) -> None:
        if isinstance(self.left, OneSidedBoundary) and self.left.side != -1:
            raise ValueError("left boundary has incorrect side")

        if isinstance(self.right, OneSidedBoundary) and self.right.side != +1:
            raise ValueError("right boundary has incorrect side")


@apply_boundary.register(TwoSidedBoundary)
def _apply_boundary_two_sided(
    bc: TwoSidedBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    if bc.left is not None:
        u = apply_boundary(bc.left, grid, t, u)

    if bc.right is not None:
        u = apply_boundary(bc.right, grid, t, u)

    return u


# }}}
