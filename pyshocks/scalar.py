# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Scalar Conservation Laws
------------------------

Fluxes
^^^^^^

These fluxes are based on the seminal work of [LeVeque2002]_.

.. [LeVeque2002] R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
    Cambridge University Press, 2002.

.. autofunction:: scalar_flux_upwind
.. autofunction:: scalar_flux_rusanov
.. autofunction:: scalar_flux_lax_friedrichs
.. autofunction:: scalar_flux_engquist_osher

.. autofunction:: lax_friedrichs_initial_condition_correction

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: OneSidedBoundary
.. autoclass:: TwoSidedBoundary

Ghost Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DirichletBoundary
.. autoclass:: NeumannBoundary
.. autoclass:: PeriodicBoundary

.. autofunction:: make_dirichlet_boundary
.. autofunction:: make_neumann_boundary

Simultaneous-Approximation-Term (SAT) Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OneSidedSATBoundary
.. autoclass:: OneSidedAdvectionSATBoundary
.. autoclass:: OneSidedDiffusionSATBoundary
.. autoclass:: OneSidedBurgersBoundary
.. autoclass:: SATBoundary

.. autofunction:: make_advection_sat_boundary
.. autofunction:: make_diffusion_sat_boundary
.. autofunction:: make_burgers_sat_boundary
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

import jax.numpy as jnp

from pyshocks.grid import Grid, UniformGrid
from pyshocks.schemes import (
    ConservationLawScheme,
    flux,
    Boundary,
    BoundaryType,
    apply_boundary,
    evaluate_boundary,
)
from pyshocks.tools import TemporalFunction, VectorFunction, SpatialFunction


# {{{ fluxes

# {{{ upwind


def scalar_flux_upwind(
    scheme: ConservationLawScheme,
    grid: Grid,
    bc: BoundaryType,
    t: float,
    a: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    r"""Implements the classic upwind flux (see [LeVeque2002]_).

    The flux is given by

    .. math::

        f(u_L, u_R) =
        \begin{cases}
        f(u_L), & \quad S(a_L, a_R) > 0, \\
        f(u_R), & \quad \text{otherwise},
        \end{cases}

    where the speed :math:`S` is taken as a simple average, i.e.

    .. math::

        S(a_L, a_R) = \frac{1}{2} (a_L + a_R).
    """
    assert u.shape[0] == grid.x.size
    assert a.shape == u.shape
    assert scheme.rec is not None

    from pyshocks.reconstruction import reconstruct

    ul, ur = reconstruct(scheme.rec, grid, bc, u)
    al, ar = reconstruct(scheme.rec, grid, bc, a)

    fl = flux(scheme, t, grid.f, ul)
    fr = flux(scheme, t, grid.f, ur)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, fr[:-1], fl[1:])

    return jnp.pad(fnum, 1)


# }}}


# {{{ Rusanov (aka Local Lax-Friedrichs)


def lax_friedrichs_initial_condition_correction(
    grid: UniformGrid,
    func: SpatialFunction,
    *,
    order: Optional[int] = None,
) -> jnp.ndarray:
    """Implements a correction to the initial condition that ensures local
    total variation preservation with the Lax-Friedrichs scheme.

    This correction is described in Algorithm 3.1 from [Breuss2004]. Note that
    in [Breuss2004]_ the authors also recommend modified boundary conditions,
    which are not implemented here.

    .. [Breuss2004] M. Breuß, *The Correct Use of the Lax–Friedrichs Method*,
        ESAIM: Mathematical Modelling and Numerical Analysis, Vol. 38,
        pp. 519--540, 2004,
        `DOI <http://dx.doi.org/10.1051/m2an:2004027>`__.

    :arg order: if not *None*, cell averages are computed, otherwise point
        values are used.
    """
    if grid.x.size % 2 != 0 and grid.nghosts % 2 == 0:
        raise ValueError("only grids with even number of cells are supported")

    half_grid = type(grid)(
        a=grid.a,
        b=grid.b,
        nghosts=grid.nghosts // 2,
        x=grid.x[::2],
        dx=grid.dx[::2],
        # NOTE: these are not actually used anywhere below
        f=grid.f,
        df=grid.df,
        dx_min=grid.dx_min,
        dx_max=grid.dx_max,
    )

    if order is None:
        half_u0 = func(half_grid.x)
    else:
        from pyshocks.grid import make_leggauss_quadrature, cell_average

        # evaluate cell averages on the half-grid
        quad = make_leggauss_quadrature(half_grid, order=order)
        half_u0 = cell_average(quad, func)

    u0 = jnp.tile(half_u0.reshape(-1, 1), 2).reshape(-1)
    assert u0.shape == grid.x.shape

    return u0


def scalar_flux_rusanov(
    scheme: ConservationLawScheme,
    grid: Grid,
    bc: BoundaryType,
    t: float,
    a: jnp.ndarray,
    u: jnp.ndarray,
    alpha: float = 1.0,
) -> jnp.ndarray:
    r"""Implements the Rusanov flux (also referred to as the *local* Lax-Friedrichs)
    flux for scalar conservation laws (see Section 12.5 in [LeVeque2002]_).

    The flux is given by

    .. math::

        f(u_L, u_R) = \frac{1}{2} (f(u_L) + f(u_R))
            - \frac{1}{2} s(a_L, a_R) \nu (u_R - u_L),

    where the choice of speed :math:`s` gives rise to different versions of
    this scheme. Here, we choose

    .. math::

        s(a_L, a_R) = \max(|a_L|, |a_R|).

    Finally, the artificial dissipation is given by

    .. math::

        \nu = \Delta x^{\alpha - 1},

    where values of :math:`\alpha < 1` are more dissipative than the standard
    version.
    """
    assert u.shape[0] == grid.x.size
    assert scheme.rec is not None

    # artificial viscosity
    if abs(alpha - 1.0) > 1.0e-8:
        nu = grid.df ** (alpha - 1)
    else:
        nu = 1.0

    from pyshocks.reconstruction import reconstruct

    ul, ur = reconstruct(scheme.rec, grid, bc, u)
    fl = flux(scheme, t, grid.f, ul)
    fr = flux(scheme, t, grid.f, ur)

    # largest local wave speed
    a = jnp.abs(a)
    if isinstance(a, jnp.ndarray) and a.size == u.size:
        # FIXME: should the reconstruct a and use al/ar to get a local speed?
        a = jnp.maximum(a[1:], a[:-1])

    fnum = 0.5 * (fl[1:] + fr[:-1]) - 0.5 * a * nu * (ul[1:] - ur[:-1])
    return jnp.pad(fnum, 1)


# }}}


# {{{ Lax-Friedrichs (aka Global Lax-Friedrichs)


def scalar_flux_lax_friedrichs(
    scheme: ConservationLawScheme,
    grid: Grid,
    bc: BoundaryType,
    t: float,
    a: jnp.ndarray,
    u: jnp.ndarray,
    alpha: float = 1.0,
) -> jnp.ndarray:
    r"""Implements the *global* Lax-Friedrichs flux (see Section 12.5 from
    [LeVeque2002]_).

    This flux is the same as :func:`scalar_flux_rusanov`, where the speed
    is taken as the maximum over the entire domain, i.e.

    .. math::

        S(a_L, a_R) = \max_i a_i.
    """
    amax = jnp.max(jnp.abs(a))
    return scalar_flux_rusanov(scheme, grid, bc, t, amax, u, alpha=alpha)


# }}}


# {{{ Engquist-Osher


def scalar_flux_engquist_osher(
    scheme: ConservationLawScheme,
    grid: Grid,
    bc: BoundaryType,
    t: float,
    u: jnp.ndarray,
    omega: float = 0.0,
) -> jnp.ndarray:
    r"""Implements the Engquist-Osher flux (see Section 12.6 in [LeVeque2002]_)
    for **convex** physical fluxes.

    The flux is given by

    .. math::

        f(u_L, u_R) = \frac{1}{2} (f(u_L) + f(u_R))
            - \frac{1}{2} \int_{u_L}^{u_R} |f'(u)| \,\mathrm{d}u.
    """
    assert u.shape[0] == grid.x.size
    assert scheme.rec is not None

    from pyshocks.reconstruction import reconstruct

    ul, ur = reconstruct(scheme.rec, grid, bc, u)
    fr = flux(scheme, t, grid.f, jnp.maximum(ur, omega))
    fl = flux(scheme, t, grid.f, jnp.minimum(ul, omega))
    fo = flux(
        scheme,
        t,
        grid.x,
        jnp.full_like(grid.df, omega),
    )

    fnum = fr[:-1] + fl[1:] - fo
    return jnp.pad(fnum, 1)


# }}}

# }}}


# {{{ boundary conditions


@dataclass(frozen=True)
class OneSidedBoundary(Boundary):  # pylint: disable=abstract-method
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

    left: OneSidedBoundary
    right: OneSidedBoundary

    def __post_init__(self) -> None:
        assert isinstance(self.left, OneSidedBoundary)
        assert isinstance(self.right, OneSidedBoundary)

        if self.left.side != -1:
            raise ValueError("left boundary has incorrect side")

        if self.right.side != +1:
            raise ValueError("right boundary has incorrect side")

    @property
    def boundary_type(self) -> BoundaryType:
        if self.left.boundary_type != self.right.boundary_type:
            raise NotImplementedError()

        return self.left.boundary_type


@apply_boundary.register(TwoSidedBoundary)
def _apply_boundary_two_sided(
    bc: TwoSidedBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    u = apply_boundary(bc.left, grid, t, u)
    u = apply_boundary(bc.right, grid, t, u)

    return u


@evaluate_boundary.register(TwoSidedBoundary)
def _evaluate_boundary_two_sided(
    bc: TwoSidedBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return evaluate_boundary(bc.left, grid, t, u) + evaluate_boundary(
        bc.right, grid, t, u
    )


# {{{ Dirichlet


@dataclass(frozen=True)
class DirichletBoundary(OneSidedBoundary):
    """Imposes Dirichlet-type boundary conditions of the form

    .. math::

        u(t, a) = g_d(t).

    .. attribute:: f

        Callable that can be used to evaluate the boundary condition in the
        ghost cells on the given :attr:`~OneSidedBoundary.side`.

    .. automethod:: __init__
    """

    g: VectorFunction

    @property
    def boundary_type(self) -> BoundaryType:
        return BoundaryType.Dirichlet


@apply_boundary.register(DirichletBoundary)
def _apply_boundary_scalar_dirichlet(
    bc: DirichletBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    ito = grid.g_[bc.side]
    return u.at[ito].set(bc.g(t, grid.x[ito]), unique_indices=True)


def make_dirichlet_boundary(
    ga: VectorFunction, gb: Optional[VectorFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = DirichletBoundary(side=-1, g=ga)
    bb = DirichletBoundary(side=+1, g=gb)
    return TwoSidedBoundary(left=ba, right=bb)


# }}}


# {{{ Neumann boundary conditions


@dataclass(frozen=True)
class NeumannBoundary(OneSidedBoundary):
    r"""Imposes Neumann-type boundary conditions of the form

    .. math::

        \pm \frac{\partial u}{\partial x}(t, a) = g_n(t).

    using a second-order approximation.

    .. attribute:: f

        Callable that can be used to evaluate the boundary condition in the
        ghost cells on the given :attr:`~OneSidedBoundary.side`.

    .. automethod:: __init__
    """

    g: TemporalFunction

    @property
    def boundary_type(self) -> BoundaryType:
        return BoundaryType.Neumann


@apply_boundary.register(NeumannBoundary)
def _apply_boundary_scalar_neumann(
    bc: NeumannBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    # NOTE: the indexing here is computed as follows (for nghosts = 3)
    #                        3     4     5
    #   |-----|-----|-----|-----|-----|-----|-----
    #      0     1     2  a
    #
    # where we need to compute
    #
    #   (u_{i + k} - u_{i - k}) / (x_{i + k} - x_{i - k}) = \pm g_n
    #
    # for k in {0, 1, 2} and i = 3. So basically the first batch {0, 1, 2}
    # gets reversed.

    g = grid.nghosts
    ifrom = grid.gi_[bc.side]
    if bc.side == -1:
        ito = jnp.arange(g - 1, -1, -1)
    else:
        ito = jnp.arange(u.size - 1, u.size - g - 1, -1)

    ub = u[ifrom] + bc.side * (grid.x[ifrom] - grid.x[ito]) * bc.g(t)

    return u.at[ito].set(ub, unique_indices=True)


def make_neumann_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = NeumannBoundary(side=-1, g=ga)
    bb = NeumannBoundary(side=+1, g=gb)
    return TwoSidedBoundary(left=ba, right=bb)


# }}}


# {{{ periodic


@dataclass(frozen=True)
class PeriodicBoundary(Boundary):
    """Periodic boundary conditions for one dimensional domains."""

    @property
    def boundary_type(self) -> BoundaryType:
        return BoundaryType.Periodic


@apply_boundary.register(PeriodicBoundary)
def _apply_boundary_scalar_periodic(
    bc: PeriodicBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    for side in [+1, -1]:
        ito = grid.g_[side]
        ifrom = grid.gi_[-side]
        u = u.at[ito].set(u[ifrom], unique_indices=True)

    return u


@evaluate_boundary.register(PeriodicBoundary)
def _evaluate_boundary_scalar_periodic(
    bc: PeriodicBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    return jnp.zeros_like(u)


# }}}


# {{{ SAT


@dataclass(frozen=True)
class OneSidedSATBoundary(OneSidedBoundary):
    r"""Simultaneous Approximation Term (SAT) Dirichlet-type boundary conditions.

    Returns the penalty term that needs to be added to weakly impose the
    required boundary condition. The form of this penalty term is given by

    .. math::

        \tau (u_i - g) \mathbf{e}_i.

     where :math:`i` is either the first or last point in the grid.

    .. attribute:: g
    .. attribute:: tau

        Weight used in the SAT penalty term.
    """

    g: TemporalFunction
    tau: float

    def __post_init__(self) -> None:
        assert self.tau >= 0.5

    @property
    def boundary_type(self) -> BoundaryType:
        return BoundaryType.Dirichlet


@evaluate_boundary.register(OneSidedSATBoundary)
def _evaluate_boundary_sat(
    bc: OneSidedSATBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()

    return bc.tau * (u[i] - bc.g(t)) * e_i


@dataclass(frozen=True)
class SATBoundary(TwoSidedBoundary):
    left: OneSidedSATBoundary
    right: OneSidedSATBoundary


def make_sat_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> SATBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedSATBoundary(side=-1, g=ga, tau=1.0)
    bb = OneSidedSATBoundary(side=+1, g=gb, tau=1.0)
    return SATBoundary(left=ba, right=bb)


# }}}


# {{{ advection SAT boundary conditions


@dataclass(frozen=True)
class OneSidedAdvectionSATBoundary(OneSidedSATBoundary):
    """Implements the SAT boundary condition for the advection equation.

    .. attribute:: velocity

        Velocity at the boundary (not dotted with the normal), which is used
        to determine if the boundary conditions is necessary.
    """

    velocity: float


@evaluate_boundary.register(OneSidedAdvectionSATBoundary)
def _evaluate_boundary_advection_sat(
    bc: OneSidedAdvectionSATBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()

    return e_i * jnp.where(bc.side * bc.velocity > 0, 0.0, bc.tau * (u[i] - bc.g(t)))


def make_advection_sat_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> SATBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedAdvectionSATBoundary(side=-1, g=ga, tau=1.0, velocity=1.0)
    bb = OneSidedAdvectionSATBoundary(side=+1, g=gb, tau=1.0, velocity=1.0)
    return SATBoundary(left=ba, right=bb)


# }}}


# {{{ diffusion SAT boundary conditions


@dataclass(frozen=True)
class OneSidedDiffusionSATBoundary(OneSidedSATBoundary):
    """Implements the SAT boundary condition for the diffusion equation.

    The boundary condition is described in [Mattsson2004]_ Equation 16. Note
    that :class:`OneSidedSATBoundary` can also be used, but is only
    energy stable if the derivative at the boundary vanishes.

    .. [Mattsson2004] K. Mattsson, J. Nordström, *Summation by Parts Operators
        for Finite Difference Approximations of Second Derivatives*,
        Journal of Computational Physics, Vol. 199, pp. 503--540, 2004,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2004.03.001>`__.
    """

    S: ClassVar[jnp.ndarray]


@evaluate_boundary.register(OneSidedDiffusionSATBoundary)
def _evaluate_boundary_diffusion_sat(
    bc: OneSidedDiffusionSATBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()

    Su = bc.S[i, :] @ u
    return bc.tau * ((u[i] + Su) - bc.g(t)) * e_i


def make_diffusion_sat_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> SATBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedDiffusionSATBoundary(side=-1, g=ga, tau=1.0)
    bb = OneSidedDiffusionSATBoundary(side=+1, g=gb, tau=1.0)
    return SATBoundary(left=ba, right=bb)


# }}}


# {{{ Burgers SAT boundary conditions


@dataclass(frozen=True)
class OneSidedBurgersBoundary(OneSidedSATBoundary):
    """SSWENO boundary conditions for Burgers' equation.

    The boundary conditions implemented here only consider the inviscid problem,
    as given by :class:`~pyshocks.burgers.SSWENO242`. They are described in
    [Fisher2013]_ Equation 4.8.
    """


@evaluate_boundary.register(OneSidedBurgersBoundary)
def _evaluate_boundary_ssweno_burgers(
    bc: OneSidedBurgersBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()

    # NOTE: [Fisher2013] Equation 4.8
    s = bc.side
    return s * ((u[i] + s * abs(u[i])) * u[i] / 3 + s * bc.g(t)) * e_i


def make_burgers_sat_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> SATBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedBurgersBoundary(side=-1, g=ga, tau=1.0)
    bb = OneSidedBurgersBoundary(side=+1, g=gb, tau=1.0)
    return SATBoundary(left=ba, right=bb)


# }}}

# }}}
