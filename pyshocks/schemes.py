# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Schemes
^^^^^^^

.. autoclass:: SchemeBase
    :no-show-inheritance:
.. autoclass:: CombineScheme

.. autofunction:: bind
.. autofunction:: apply_operator
.. autofunction:: predict_timestep

.. autoclass:: FiniteVolumeSchemeBase
.. autoclass:: FiniteDifferenceSchemeBase

Finite Volume Schemes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConservationLawScheme
.. autoclass:: CombineConservationLawScheme

.. autofunction:: flux
.. autofunction:: numerical_flux

Finite Difference Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: Boundary
    :no-show-inheritance:
.. autoclass:: OneSidedBoundary
.. autoclass:: TwoSidedBoundary

.. autofunction:: evaluate_boundary
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

    To define a scheme for such an equation, a new implementation of
    :func:`apply_operator` needs to be provided. Optionally, :func:`predict_timestep`
    can also be implemented to take advantage of the knowledge of the
    :math:`\mathbf{A}` operator.

    .. attribute:: name

        A string identifier for the scheme in question (these are not expected
        to be unique).

    .. attribute:: order

        Expected order of the scheme. This is the minimum convergence order
        attainable for the scheme (e.g. WENO schemes can attain higher orders
        for smooth solutions).

    .. attribute:: stencil_width

        The required stencil width for the scheme and reconstruction.

    .. attribute:: rec

        A :class:`~pyshocks.reconstruction.Reconstruction` object that is used
        to reconstruct high-order face-based values when needed by the numerical
        scheme.
    """

    rec: Optional[Reconstruction]

    @property
    def name(self) -> str:
        if self.rec is not None:
            return f"{type(self).__name__}_{self.rec.name}".lower()
        return type(self).__name__.lower()

    @property
    def order(self) -> int:
        if self.rec is not None:
            return self.rec.order

        raise NotImplementedError

    @property
    def stencil_width(self) -> int:
        if self.rec is not None:
            return self.rec.stencil_width

        raise NotImplementedError


@singledispatch
def bind(scheme: SchemeBase, grid: Grid, bc: "Boundary") -> SchemeBase:
    """Binds the scheme to the given grid and boundary conditions.

    This method is mean to allow initialization of a numerical scheme based on
    the grid, e.g. for caching data that is reused at every call to
    :func:`apply_operator`. In general, caching cannot be done on-demand as
    :func:`jax.jit` does not allow leaking values.

    :returns: a new :class:`SchemeBase` with the same parameters. If no changes
        are required, the same scheme is returned.
    """
    if isinstance(scheme, SchemeBase):
        return scheme

    raise NotImplementedError(type(scheme).__name__)


@singledispatch
def apply_operator(
    scheme: SchemeBase, grid: Grid, bc: "Boundary", t: float, u: jnp.ndarray
) -> jnp.ndarray:
    r"""Applies right-hand side operator for a "Method of Lines" approach.
    For any PDE, we have that

    .. math::

        \frac{\partial \mathbf{u}}{\partial t} = \mathbf{A}(t, u),

    where :math:`\mathbf{A}` is the nonlinear differential operator (in general)
    computed by this function.

    :param scheme: scheme used to approximated the operator.
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


@dataclass(frozen=True)
class FiniteVolumeSchemeBase(SchemeBase):
    """Describes a finite volume-type scheme for PDEs.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceSchemeBase(SchemeBase):
    """Describes a finite difference-type scheme for PDEs.

    .. automethod:: __init__
    """


# }}}


# {{{ scheme linear combination


@dataclass(frozen=True)
class CombineScheme(SchemeBase):
    r"""Implements a combined operator for multiple schemes.

    In this case, we consider an equation of the form

    .. math::

        \frac{\partial \mathbf{u}}{\partial t}
        + \sum_{k = 0}^M \mathbf{A}_k(t, \mathbf{x}, \mathbf{u}, \nabla \mathbf{u})
        = 0,

    where each :math:`\mathbf{A}_k` is defined by an element of
    :attr:`schemes`. This class makes no attempt at checking if the different
    schemes are consistent in any way.

    The expectation is that each one implements a different operator and they
    are combined as is. For example, one can combine a scheme implementing the
    advective operator, one for the diffusive operator and one for the
    reactive operator in a advection-diffusion-reaction equation.

    .. attribute:: schemes

        A tuple of :class:`SchemeBase` objects.
    """
    schemes: Tuple[SchemeBase, ...]

    def __post_init__(self) -> None:
        assert len(self.schemes) > 1
        assert all(isinstance(s, SchemeBase) for s in self.schemes)

    @property
    def order(self) -> int:
        return min(s.order for s in self.schemes)

    @property
    def stencil_width(self) -> int:
        return max(s.stencil_width for s in self.schemes)


@predict_timestep.register(CombineScheme)
def _predict_time_combine(
    scheme: CombineScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    from functools import reduce

    return reduce(
        jnp.minimum, [predict_timestep(s, grid, t, u) for s in scheme.schemes], jnp.inf
    )


@apply_operator.register(CombineScheme)
def _apply_operator_combine(
    scheme: CombineScheme,
    grid: Grid,
    bc: "Boundary",
    t: float,
    u: jnp.ndarray,
) -> jnp.ndarray:
    return sum(apply_operator(s, grid, bc, t, u) for s in scheme.schemes)


# }}}


# {{{ conservation law schemes


@dataclass(frozen=True)
class ConservationLawScheme(FiniteVolumeSchemeBase):
    r"""Describes numerical schemes for conservation laws.

    We consider conservation laws of the form

    .. math::

        \frac{\partial \mathbf{u}}{\partial t}
        = -\nabla \cdot \mathbf{f}(\mathbf{u}),

    where :math:`\mathbf{f}` is the (possibly nonlinear) flux. To define
    a scheme for a conservation law of this form, one needs to implement
    :func:`flux` and :func:`numerical_flux`. These functions are used in a
    :func:`apply_operator` to define the approximation.
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


@apply_operator.register(ConservationLawScheme)
def _apply_operator_conservation_law(
    scheme: ConservationLawScheme, grid: Grid, bc: "Boundary", t: float, u: jnp.ndarray
) -> jnp.ndarray:
    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -(f[1:] - f[:-1]) / grid.dx


@dataclass(frozen=True)
class CombineConservationLawScheme(CombineScheme, ConservationLawScheme):
    schemes: Tuple[ConservationLawScheme, ...]

    def __post_init__(self) -> None:
        assert len(self.schemes) > 1
        assert all(isinstance(s, ConservationLawScheme) for s in self.schemes)


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


@singledispatch
def evaluate_boundary(
    bc: Boundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    """Evaluates the boundary conditions out of place.

    Unlike :func:`apply_boundary`, this function simply returns a set of
    boundary values at the required boundary points and zero elsewhere.

    :returns: an array of the same shape as *u* containing the boundary
        conditions.
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


@evaluate_boundary.register(TwoSidedBoundary)
def _evaluate_boundary_two_sided(
    bc: TwoSidedBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    ub = 0
    if bc.left is not None:
        ub = ub + evaluate_boundary(bc.left, grid, t, u)

    if bc.right is not None:
        ub = ub + evaluate_boundary(bc.right, grid, t, u)

    return ub


# }}}
