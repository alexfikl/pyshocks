# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Fluxes
^^^^^^

.. autofunction:: scalar_flux_upwind
.. autofunction:: scalar_flux_rusanov
.. autofunction:: scalar_flux_lax_friedrichs
.. autofunction:: scalar_flux_engquist_osher

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DirichletBoundary
.. autoclass:: PeriodicBoundary
.. autofunction:: dirichlet_boundary
"""

from dataclasses import dataclass

import jax.numpy as jnp

from pyshocks.grid import Grid
from pyshocks.schemes import (
        VectorFunction,
        SchemeBase, flux,
        Boundary, OneSidedBoundary, TwoSidedBoundary, apply_boundary,
        )


# {{{ fluxes

# {{{ upwind

def scalar_flux_upwind(
        scheme: SchemeBase,
        grid: Grid,
        t: float,
        a: "jnp.ndarray",
        u: "jnp.ndarray") -> jnp.ndarray:
    assert u.shape[0] == grid.x.size

    am = 0.5 * (a[1:] + a[:-1])
    um = jnp.where(am > 0, u[:-1], u[1:])

    fnum = flux(scheme, t, grid.f, um)
    return jnp.pad(fnum, 1)

# }}}


# {{{ Rusanov (aka Local Lax-Friedrichs)

def scalar_flux_rusanov(
        scheme: SchemeBase,
        grid: Grid,
        t: float,
        a: jnp.ndarray,
        u: jnp.ndarray,
        alpha: float = 1.0):
    assert u.shape[0] == grid.x.size
    f = flux(scheme, 0.0, grid.x, u)

    # artificial viscosity
    if abs(alpha - 1.0) > 1.0e-8:
        nu = grid.df**(alpha - 1)
    else:
        nu = 1

    # largest local wave speed
    a = jnp.abs(a)
    if isinstance(a, jnp.ndarray) and a.size == u.size:
        a = jnp.maximum(a[1:], a[:-1])

    fnum = 0.5 * (f[1:] + f[:-1]) - 0.5 * a * nu * (u[1:] - u[:-1])
    return jnp.pad(fnum, 1)

# }}}


# {{{ Lax-Friedrichs (aka Global Lax-Friedrichs)

def scalar_flux_lax_friedrichs(
        scheme: SchemeBase,
        grid: Grid,
        t: float,
        a: jnp.ndarray,
        u: jnp.ndarray,
        alpha: float = 1.0) -> jnp.ndarray:
    amax = jnp.max(jnp.abs(a))
    return scalar_flux_rusanov(scheme, grid, t, amax, u, alpha=alpha)

# }}}


# {{{ Engquist-Osher

def scalar_flux_engquist_osher(
        scheme: SchemeBase,
        grid: Grid,
        t: float,
        u: jnp.ndarray,
        omega: float = 0.0) -> jnp.ndarray:
    assert u.shape[0] == grid.x.size
    fp = flux(scheme, t, grid.x, jnp.maximum(u, omega))
    fm = flux(scheme, t, grid.x, jnp.minimum(u, omega))
    fo = flux(scheme, t, grid.x, jnp.full_like(grid.df, omega))

    fnum = (fp[:-1] + fm[1:] - fo)
    return jnp.pad(fnum, 1)

# }}}

# }}}


# {{{ boundary conditions

def dirichlet_boundary(fa, fb=None) -> TwoSidedBoundary:
    if fb is None:
        fb = fa

    fa = DirichletBoundary(side=-1, f=fa)
    fb = DirichletBoundary(side=+1, f=fb)
    return TwoSidedBoundary(left=fa, right=fb)


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
        bc: DirichletBoundary, grid: Grid, t: float, u: jnp.ndarray) -> jnp.ndarray:
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
        bc: PeriodicBoundary, grid: Grid, t: float, u: jnp.ndarray):
    assert u.size == grid.x.size
    g = grid.nghosts

    # left
    ito = grid.g_[+1]
    u = u.at[ito].set(u[-2*g:-g], unique_indices=True)

    # right
    ito = grid.g_[-1]
    u = u.at[ito].set(u[g:2*g], unique_indices=True)

    return u

# }}}

# }}}
