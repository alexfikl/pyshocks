# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

.. autoclass:: ScalarFunction
.. autoclass:: VectorFunction
.. autoclass:: SpatialFunction

.. autoclass:: SchemeBase
    :no-show-inheritance:
.. autoclass:: ConservationLawScheme

.. autofunction:: flux
.. autofunction:: numerical_flux
.. autofunction:: apply_operator
.. autofunction:: predict_timestep

.. autoclass:: Boundary
    :no-show-inheritance:
.. autoclass:: OneSidedBoundary
.. autoclass:: TwoSidedBoundary
.. autofunction:: apply_boundary

"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Optional, Protocol

import jax.numpy as jnp

from pyshocks.grid import Grid


# {{{


class ScalarFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, x: jnp.ndarray) -> float:
        ...


class VectorFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        ...


class SpatialFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`\mathbf{x}`.

    .. automethod:: __call__
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ...


# }}}


# {{{ schemes


@dataclass(frozen=True)
class SchemeBase:
    r"""Describes numerical scheme for a type of PDE.

    The general form of the equations we will be looking at is

    .. math::

        \frac{\partial \mathbf{u}}{\partial t} =
            \mathbf{A}(t, \mathbf{x}, \mathbf{u}, \nabla \mathbf{u}).
    """


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
):
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
def predict_timestep(scheme: SchemeBase, grid: Grid, t: float, u: jnp.ndarray) -> float:
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
class ConservationLawScheme(SchemeBase):
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
):
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

    def __post_init__(self):
        if self.left is not None and self.left.side != -1:
            raise ValueError("left boundary has incorrect side")

        if self.right is not None and self.right.side != +1:
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
