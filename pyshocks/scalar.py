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
.. autoclass:: PeriodicBoundary
.. autofunction:: dirichlet_boundary
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks.grid import Grid
from pyshocks.schemes import (
    FiniteVolumeScheme,
    flux,
    Boundary,
    OneSidedBoundary,
    TwoSidedBoundary,
    apply_boundary,
)
from pyshocks.reconstruction import reconstruct
from pyshocks.tools import VectorFunction


# {{{ fluxes

# {{{ upwind


def scalar_flux_upwind(
    scheme: FiniteVolumeScheme, grid: Grid, t: float, a: jnp.ndarray, u: jnp.ndarray
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
    scheme: FiniteVolumeScheme,
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
    scheme: FiniteVolumeScheme,
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
    scheme: FiniteVolumeScheme, grid: Grid, t: float, u: jnp.ndarray, omega: float = 0.0
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


def dirichlet_boundary(
    fa: VectorFunction, fb: Optional[VectorFunction] = None
) -> TwoSidedBoundary:
    if fb is None:
        fb = fa

    ba = DirichletBoundary(side=-1, f=fa)
    bb = DirichletBoundary(side=+1, f=fb)
    return TwoSidedBoundary(left=ba, right=bb)


# {{{ Dirichlet


@dataclass(frozen=True)
class DirichletBoundary(OneSidedBoundary):
    """
    .. attribute:: f

        Callable that can be used to evaluate the boundary condition in the
        ghost cells on the given :attr:`~pyshocks.OneSidedBoundary.side`.

    .. automethod:: __init__
    """

    f: VectorFunction


@apply_boundary.register(DirichletBoundary)
def _apply_boundary_scalar_dirichlet(
    bc: DirichletBoundary, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert u.size == grid.x.size

    ito = grid.g_[bc.side]
    return u.at[ito].set(bc.f(t, grid.x[ito]), unique_indices=True)


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
    g = grid.nghosts

    # left
    ito = grid.g_[+1]
    u = u.at[ito].set(u[-2 * g : -g], unique_indices=True)

    # right
    ito = grid.g_[-1]
    u = u.at[ito].set(u[g : 2 * g], unique_indices=True)

    return u


# }}}

# }}}
