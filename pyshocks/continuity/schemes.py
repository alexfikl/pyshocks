from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from pyshocks import Grid, ConservationLawScheme
from pyshocks import flux, numerical_flux, predict_timestep
from pyshocks.weno import WENOJSMixin, WENOJS32Mixin, WENOJS53Mixin


# {{{ base

@dataclass(frozen=True)
class Scheme(ConservationLawScheme):
    """Base class for numerical schemes for the continuity equation.

    .. attribute:: a

        Advection velocity at cell centers.
    """
    velocity: Optional[jnp.ndarray]


@flux.register(Scheme)
def _flux_continuity(
            scheme: Scheme,
            t: float,
            x: jnp.ndarray,
            u: jnp.ndarray) -> jnp.ndarray:
    assert scheme.velocity is not None

    return scheme.velocity * u


@predict_timestep.register(Scheme)
def _predict_timestep_continuity(
            scheme: Scheme,
            grid: Grid,
            t: float,
            u: jnp.ndarray) -> float:
    assert scheme.velocity is not None

    amax = jnp.max(jnp.abs(scheme.velocity[grid.i_]))
    return grid.dx_min / amax

# }}}


# {{{ upwind

@dataclass(frozen=True)
class Godunov(Scheme):
    """First-order Godunov (upwind) scheme for the continuity equation.

    The flux of the Godunov scheme is given by
    :func:`~pyshocks.scalar.scalar_flux_upwind`.

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
    assert scheme.velocity is not None
    assert u.shape[0] == grid.x.size

    am = jnp.maximum(-scheme.velocity, 0.0)
    ap = jnp.maximum(+scheme.velocity, 0.0)

    fnum = (ap[:-1] * u[:-1] - am[1:] * u[1:])
    return jnp.pad(fnum, 1)

# }}}


# {{{ WENO

@dataclass(frozen=True)
class WENOJS(Scheme, WENOJSMixin):          # pylint: disable=abstract-method
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
    assert scheme.velocity is not None
    assert u.size == grid.x.size

    from pyshocks.weno import reconstruct
    up = reconstruct(grid, scheme, u)
    fp = flux(scheme, t, grid.f[1:], up)

    um = reconstruct(grid, scheme, u[::-1])[::-1]
    fm = flux(scheme, t, grid.f[:-1], um)

    # NOTE: using the *global* Lax-Friedrichs flux
    a = scheme.velocity[grid.i_]
    amax = jnp.max(jnp.abs(a))

    fnum = (fp[:-1] + fm[1:]) / 2 + amax * (up[:-1] - um[1:]) / 2
    return jnp.pad(fnum, 1)

# }}}
