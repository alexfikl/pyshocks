# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Limiters
--------

.. autoclass:: Limiter
    :no-show-inheritance:
.. autofunction:: limit

.. autoclass:: MINMODLimiter
.. autoclass:: MonotonizedCentralLimiter
.. autoclass:: SUPERBEELimiter
.. autoclass:: KorenLimiter

.. autofunction:: limiter_ids
.. autofunction:: make_limiter_from_name
"""

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Dict, Tuple, Type

import jax.numpy as jnp

from pyshocks import Grid


# {{{ limiter interface


@dataclass(frozen=True)
class Limiter:
    r"""Describes a flux limiter for high-order finite volume schemes.

    A limiter is a function of the form

    .. math::

        \phi(r) = \phi\left(\frac{u_i - u_{i - 1}}{u_{i + 1} - u_i}\right) \ge 0

    which becomes zero when there is no need for limiting in the variable *u*.
    The limiter is applied by calling :func:`limit`. Note that the limiter is
    generally not guaranteed to be :math:`\phi(r) \le 1`.
    """


@singledispatch
def limit(lm: Limiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    """
    :arg lm: an object that describes how to limit the variable.
    :arg u: variable with cell-centered values
    """
    raise NotImplementedError(type(lm).__name__)


# }}}


# {{{ minmod


@dataclass(frozen=True)
class MINMODLimiter(Limiter):
    r"""The parametrized ``minmod`` limiter given by

    .. math::

        \phi(r) = \max\left(0, \min\left(
            \theta, \theta r, \frac{1 + r}{2}
            \right)\right),

    where :math:`\theta \in [1, 2]`. The value of :math:`\theta = 1` recovers
    the classic ``minmod`` limiter, which is very dissipative. On the other
    hand, :math:`\theta = 2` gives a less dissipative limiter with similar
    properties.
    """

    theta: float


@limit.register(MINMODLimiter)
def _limit_minmod(lm: MINMODLimiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    r = (u[1:] - u[:-1]) / (u[2:] - u[1:-1])
    phi = jnp.maximum(
        0.0, jnp.minimum(jnp.minimum(lm.theta, lm.theta * r)), (1 + r) / 2
    )

    return jnp.pad(phi, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ monotonized central


@dataclass(frozen=True)
class MonotonizedCentralLimiter(Limiter):
    r"""The monotonized central limiter is given by

    .. math::

        \phi(r) = max\left(0, \min\left(4, 2 r, \frac{1 + 3 r}{4}\right)\right)
    """


@limit.register(MonotonizedCentralLimiter)
def _limit_monotonized_central(
    lm: MonotonizedCentralLimiter, grid: Grid, u: jnp.ndarray
) -> jnp.ndarray:
    r = (u[1:] - u[:-1]) / (u[2:] - u[1:-1])
    phi = jnp.maximum(0.0, jnp.minimum(jnp.minimum(4, 2 * r)), (1 + 3 * r) / 4)

    return jnp.pad(phi, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ superbee


@dataclass(frozen=True)
class SUPERBEELimiter(Limiter):
    pass


@limit.register(SUPERBEELimiter)
def _limit_superbee(lm: SUPERBEELimiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    r = (u[1:] - u[:-1]) / (u[2:] - u[1:-1])
    phi = jnp.maximum(0.0, jnp.maximum(jnp.minimum(1, 2 * r), jnp.minimum(2, r)))

    return jnp.pad(phi, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ Koren


@dataclass(frozen=True)
class KorenLimiter(Limiter):
    pass


@limit.register(KorenLimiter)
def _limit_koren(lm: KorenLimiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    r = (u[1:] - u[:-1]) / (u[2:] - u[1:-1])
    phi = jnp.maximum(0.0, jnp.minimum(jnp.minimum(2, 2 * r)), (1 + 2 * r) / 3)

    return jnp.pad(phi, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ make_flux_limiter_from_name


_LIMITERS: Dict[str, Type[Limiter]] = {
    "default": KorenLimiter,
    "minmod": MINMODLimiter,
    "mc": MonotonizedCentralLimiter,
    "superbee": SUPERBEELimiter,
    "koren": KorenLimiter,
}


def limiter_ids() -> Tuple[str, ...]:
    return tuple(_LIMITERS.keys())


def make_limiter_from_name(name: str, **kwargs: Any) -> Limiter:
    cls = _LIMITERS.get(name)
    if cls is None:
        raise ValueError(f"flux limiter '{name}' not found; try one of {limiter_ids()}")

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
