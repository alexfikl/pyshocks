from dataclasses import dataclass

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase, Boundary
from pyshocks import numerical_flux, apply_operator, predict_timestep
from pyshocks.weno import WENOJS32Mixin, WENOJS53Mixin


# {{{ base

@dataclass(frozen=True)
class Scheme(SchemeBase):
    """Base class for numerical schemes for the linear advection equation.

    .. attribute:: a

        Advection velocity at cell centers.
    """

    velocity: jnp.array


@predict_timestep.register(Scheme)
def _(scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray) -> float:
    # NOTE: keep in sync with pyshocks.continuity.schemes.predict_timestep
    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax


@apply_operator.register(Scheme)
def _(
        scheme: SchemeBase,
        grid: Grid,
        bc: Boundary,
        t: float,
        u: jnp.ndarray):
    from pyshocks import apply_boundary
    u = apply_boundary(bc, grid, t, u)
    f = numerical_flux(scheme, grid, t, u)

    return -scheme.velocity * (f[1:] - f[:-1]) / grid.dx

# }}}


# {{{ upwind

@dataclass(frozen=True)
class Godunov(Scheme):
    """First-order Godunov (upwind) scheme for the advection equation.

    .. attribute:: order
    .. automethod:: __init__
    """

    @property
    def order(self):
        return 1


@numerical_flux.register(Godunov)
def _(scheme: Godunov,
        grid: Grid, t: float,
        u: jnp.ndarray) -> jnp.ndarray:
    assert u.shape[0] == grid.x.size

    a = scheme.velocity
    aavg = (a[1:] + a[:-1]) / 2
    fnum = jnp.where(aavg > 0, u[:-1], u[1:])

    return jnp.pad(fnum, 1)

# }}}


# {{{ WENO

# FIXME: a bit copy-pasty?

@dataclass(frozen=True)
class WENOJS(Scheme):
    """See :class:`pyshocks.burgers.WENOJS`."""
    def __post_init__(self):
        # pylint: disable=no-member
        self.set_coefficients()


@dataclass(frozen=True)
class WENOJS32(WENOJS32Mixin, WENOJS):
    """See :class:`pyshocks.burgers.WENOJS32`."""
    eps: float = 1.0e-6


@dataclass(frozen=True)
class WENOJS53(WENOJS53Mixin, WENOJS):
    """See :class:`pyshocks.burgers.WENOJS53`."""
    eps: float = 1.0e-12


@numerical_flux.register(WENOJS)
def _(scheme: WENOJS,
        grid: Grid, t: float,
        u: jnp.ndarray) -> jnp.ndarray:
    assert u.size == grid.x.size

    from pyshocks.weno import reconstruct
    up = reconstruct(grid, scheme, u)
    ap = reconstruct(grid, scheme, scheme.velocity)

    um = reconstruct(grid, scheme, u[::-1])[::-1]
    am = reconstruct(grid, scheme, scheme.velocity[::-1])[::-1]

    # NOTE: this uses an upwind flux
    aavg = (ap[:-1] + am[1:]) / 2
    fnum = jnp.where(aavg > 0, up[:-1], um[1:])

    return jnp.pad(fnum, 1)

# }}}
