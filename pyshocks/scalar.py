# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Scalar Equation Helpers
^^^^^^^^^^^^^^^^^^^^^^^

These fluxes are based on the seminal work of [LeVeque2002]_.

.. [LeVeque2002] R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
    Cambridge University Press, 2002.

.. autofunction:: scalar_flux_upwind
.. autofunction:: scalar_flux_rusanov
.. autofunction:: scalar_flux_lax_friedrichs
.. autofunction:: scalar_flux_engquist_osher

.. autoclass:: DirichletBoundary
.. autoclass:: NeumannBoundary
.. autoclass:: PeriodicBoundary
.. autoclass:: SATBoundary
.. autoclass:: SSWENOBurgersBoundary

.. autofunction:: make_dirichlet_boundary
.. autofunction:: make_neumann_boundary
.. autofunction:: make_sat_boundary
.. autofunction:: make_ssweno_boundary
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks.grid import Grid
from pyshocks.schemes import (
    ConservationLawScheme,
    flux,
    Boundary,
    OneSidedBoundary,
    TwoSidedBoundary,
    apply_boundary,
    evaluate_boundary,
)
from pyshocks.reconstruction import reconstruct
from pyshocks.tools import TemporalFunction, VectorFunction


# {{{ fluxes

# {{{ upwind


def scalar_flux_upwind(
    scheme: ConservationLawScheme,
    grid: Grid,
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

    ul, ur = reconstruct(scheme.rec, grid, u)
    al, ar = reconstruct(scheme.rec, grid, a)

    fl = flux(scheme, t, grid.f, ul)
    fr = flux(scheme, t, grid.f, ur)

    aavg = (ar[:-1] + al[1:]) / 2
    fnum = jnp.where(aavg > 0, fr[:-1], fl[1:])  # type: ignore[no-untyped-call]

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ Rusanov (aka Local Lax-Friedrichs)


def scalar_flux_rusanov(
    scheme: ConservationLawScheme,
    grid: Grid,
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

    # artificial viscosity
    if abs(alpha - 1.0) > 1.0e-8:
        nu = grid.df ** (alpha - 1)
    else:
        nu = 1.0

    ul, ur = reconstruct(scheme.rec, grid, u)
    fl = flux(scheme, t, grid.f, ul)
    fr = flux(scheme, t, grid.f, ur)

    # largest local wave speed
    a = jnp.abs(a)
    if isinstance(a, jnp.ndarray) and a.size == u.size:
        # FIXME: should the reconstruct a and use al/ar to get a local speed?
        a = jnp.maximum(a[1:], a[:-1])

    fnum = 0.5 * (fl[1:] + fr[:-1]) - 0.5 * a * nu * (ul[1:] - ur[:-1])
    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ Lax-Friedrichs (aka Global Lax-Friedrichs)


def scalar_flux_lax_friedrichs(
    scheme: ConservationLawScheme,
    grid: Grid,
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
    return scalar_flux_rusanov(scheme, grid, t, amax, u, alpha=alpha)


# }}}


# {{{ Engquist-Osher


def scalar_flux_engquist_osher(
    scheme: ConservationLawScheme,
    grid: Grid,
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

    ul, ur = reconstruct(scheme.rec, grid, u)
    fr = flux(scheme, t, grid.f, jnp.maximum(ur, omega))
    fl = flux(scheme, t, grid.f, jnp.minimum(ul, omega))
    fo = flux(
        scheme,
        t,
        grid.x,
        jnp.full_like(grid.df, omega),  # type: ignore[no-untyped-call]
    )

    fnum = fr[:-1] + fl[1:] - fo
    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}

# }}}


# {{{ boundary conditions


def make_dirichlet_boundary(
    ga: VectorFunction, gb: Optional[VectorFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = DirichletBoundary(side=-1, g=ga)
    bb = DirichletBoundary(side=+1, g=gb)
    return TwoSidedBoundary(left=ba, right=bb)


# {{{ Dirichlet


@dataclass(frozen=True)
class DirichletBoundary(OneSidedBoundary):
    """Imposes Dirichlet-type boundary conditions of the form

    .. math::

        u(t, a) = g_d(t).

    .. attribute:: f

        Callable that can be used to evaluate the boundary condition in the
        ghost cells on the given :attr:`~pyshocks.OneSidedBoundary.side`.

    .. automethod:: __init__
    """

    g: VectorFunction


@apply_boundary.register(DirichletBoundary)
def _apply_boundary_scalar_dirichlet(
    bc: DirichletBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    ito = grid.g_[bc.side]
    return u.at[ito].set(bc.g(t, grid.x[ito]), unique_indices=True)


# }}}


# {{{ Neumann boundary conditions


def make_neumann_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = NeumannBoundary(side=-1, g=ga)
    bb = NeumannBoundary(side=+1, g=gb)
    return TwoSidedBoundary(left=ba, right=bb)


@dataclass(frozen=True)
class NeumannBoundary(OneSidedBoundary):
    r"""Imposes Neumann-type boundary conditions of the form

    .. math::

        \pm \frac{\partial u}{\partial x}(t, a) = g_n(t).

    using a second-order approximation.

    .. attribute:: f

        Callable that can be used to evaluate the boundary condition in the
        ghost cells on the given :attr:`~pyshocks.OneSidedBoundary.side`.

    .. automethod:: __init__
    """

    g: TemporalFunction


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


# }}}


# {{{ periodic


@dataclass(frozen=True)
class PeriodicBoundary(Boundary):
    """Periodic boundary conditions for one dimensional domains."""


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


# }}}


# {{{ SAT


def make_sat_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedSATBoundary(side=-1, g=ga)
    bb = OneSidedSATBoundary(side=+1, g=gb)
    return SATBoundary(left=ba, right=bb)


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
    tau: float = 1.0

    def __post_init__(self) -> None:
        assert self.tau >= 0.5


@dataclass(frozen=True)
class SATBoundary(TwoSidedBoundary):
    left: Optional[OneSidedSATBoundary]
    right: Optional[OneSidedSATBoundary]


@evaluate_boundary.register(OneSidedSATBoundary)
def _evaluate_boundary_sat(
    bc: OneSidedSATBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()  # type: ignore[no-untyped-call]

    return bc.tau * (u[i] - bc.g(t)) * e_i


# }}}


# {{{ SSWENO


def make_ss_weno_boundary(
    ga: TemporalFunction, gb: Optional[TemporalFunction] = None
) -> TwoSidedBoundary:
    if gb is None:
        gb = ga

    ba = OneSidedSSWENOBurgersBoundary(side=-1, g=ga)
    bb = OneSidedSSWENOBurgersBoundary(side=+1, g=gb)
    return SSWENOBurgersBoundary(left=ba, right=bb)


@dataclass(frozen=True)
class OneSidedSSWENOBurgersBoundary(OneSidedSATBoundary):
    """SSWENO boundary conditions for Burgers' equation.

    The boundary conditions implemented here only consider the inviscid problem,
    as given by :class:`~pyshocks.burgers.SSWENO242`. They are described in
    [Fisher2013]_ Equation 4.8.
    """


@dataclass(frozen=True)
class SSWENOBurgersBoundary(TwoSidedBoundary):
    left: Optional[OneSidedSSWENOBurgersBoundary]
    right: Optional[OneSidedSSWENOBurgersBoundary]


@evaluate_boundary.register(OneSidedSSWENOBurgersBoundary)
def _evaluate_boundary_ssweno_burgers(
    bc: OneSidedSSWENOBurgersBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert grid.nghosts == 0
    assert grid.x.shape == u.shape

    i = grid.b_[bc.side]
    e_i = jnp.eye(1, u.size, i).squeeze()  # type: ignore[no-untyped-call]

    # NOTE: [Fisher2013] Equation 4.8
    s = bc.side
    return s * ((u[i] + s * abs(u[i])) * u[i] / 3 + s * bc.g(t)) * e_i


# }}}

# }}}
