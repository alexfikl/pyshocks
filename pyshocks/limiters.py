# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Limiters
--------

.. autoclass:: Limiter
    :no-show-inheritance:
.. autofunction:: evaluate
.. autofunction:: limit

.. autoclass:: UnlimitedLimiter
.. autoclass:: MINMODLimiter
.. autoclass:: MonotonizedCentralLimiter
.. autoclass:: SUPERBEELimiter
.. autoclass:: VanAlbadaLimiter
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
def evaluate(lm: Limiter, r: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the limiter at a given slope ratio.

    This is a separate function from :func:`limit` mostly for testing purposes.

    :arg lm: an object that describes how to limit the variable.
    :arg r: value at which to evaluate the limiter. In practice, this is
        computed from the function value slopes as described by
        :class:`Limiter`.
    """
    raise NotImplementedError(type(lm).__name__)


@singledispatch
def limit(lm: Limiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    """
    :arg lm: an object that describes how to limit the variable.
    :arg u: variable with cell-centered values
    """
    return jnp.pad(evaluate(lm, slope(u)), 1)  # type: ignore[no-untyped-call]


def slope(u: jnp.ndarray) -> jnp.ndarray:
    # NOTE: the slope ratio is computed from the following cells
    #       i - 1          i         i + 1
    #   ------------|------------|------------
    #           --------      --------

    sl = u[1:-1] - u[:-2]
    sr = u[2:] - u[1:-1]

    return sl / (sr + 1.0e-12)


# }}}

# {{{ unlimited


@dataclass(frozen=True)
class UnlimitedLimiter(Limiter):
    """This limiter performs no limiting and is mostly meant for testing."""


@evaluate.register(UnlimitedLimiter)
def _evaluate_unlimited(lm: UnlimitedLimiter, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(r)  # type: ignore[no-untyped-call]


@limit.register(UnlimitedLimiter)
def _limit_unlimited(lm: UnlimitedLimiter, grid: Grid, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones_like(u)  # type: ignore[no-untyped-call]


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

    def __post_init__(self) -> None:
        if self.theta < 1.0 or self.theta > 2.0:
            raise ValueError(f"'theta' must be in [1, 2]: {self.theta}")


@evaluate.register(MINMODLimiter)
def _evaluate_minmod(lm: MINMODLimiter, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(
        0.0, jnp.minimum(jnp.minimum(lm.theta, lm.theta * r), (1 + r) / 2)
    )


# }}}


# {{{ monotonized central


@dataclass(frozen=True)
class MonotonizedCentralLimiter(Limiter):
    r"""The monotonized central limiter is given by

    .. math::

        \phi(r) = max\left(0, \min\left(4, 2 r, \frac{1 + 3 r}{4}\right)\right).
    """


@evaluate.register(MonotonizedCentralLimiter)
def _evaluate_monotonized_central(
    lm: MonotonizedCentralLimiter, r: jnp.ndarray
) -> jnp.ndarray:
    return jnp.maximum(0.0, jnp.minimum(jnp.minimum(2, 2 * r), (1 + r) / 2))


# }}}


# {{{ superbee


@dataclass(frozen=True)
class SUPERBEELimiter(Limiter):
    r"""The classic SUPERBEE limiter that is given by

    .. math::

        \phi(r) = \max(0, \min(1, 2 r), \min(2, r)).
    """


@evaluate.register(SUPERBEELimiter)
def _evaluate_superbee(lm: SUPERBEELimiter, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0.0, jnp.maximum(jnp.minimum(1, 2 * r), jnp.minimum(2, r)))


# }}}


# {{{ van Albada


@dataclass(frozen=True)
class VanAlbadaLimiter(Limiter):
    r"""The van Albada limiter that is given by

    .. math::

        \phi_v(r) =
        \begin{cases}
        \dfrac{r^2 + r}{r^2 + 1}, & \quad v = 1, \\
        \dfrac{2 r}{r^2 + 1}, & \quad v = 2, \\
        \end{cases}

    where :math:`v` denotes the :attr:`variant` of the limiter.

    .. attribute:: variant

        Choses one of the two variants of the van Albada limiter.
    """

    variant: int

    def __post_init__(self) -> None:
        assert self.variant in (1, 2)


@evaluate.register(VanAlbadaLimiter)
def _evaluate_van_albada_1(lm: VanAlbadaLimiter, r: jnp.ndarray) -> jnp.ndarray:
    if lm.variant == 1:
        phi = (r**2 + r) / (r**2 + 1)
    else:
        phi = 2 * r / (r**2 + 1)

    return jnp.maximum(phi, 0.0)


# }}}


# {{{ Koren


@dataclass(frozen=True)
class KorenLimiter(Limiter):
    """A third-order TVD limiter described in [Kuzmin2006]_ due to B. Koren.

    .. [Kuzmin2006] D. Kuzmin, *On the Design of General-Purpose Flux Limiters
        for Finite Element Schemes. I. Scalar Convection*,
        Journal of Computational Physics, Vol. 219, pp. 513--531, 2006,
        `DOI <http://dx.doi.org/10.1016/j.jcp.2006.03.034>`__.
    """


@evaluate.register(KorenLimiter)
def _evaluate_koren(lm: KorenLimiter, r: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0.0, jnp.minimum(jnp.minimum(2, 2 * r), (1 + 2 * r) / 3))


# }}}


# {{{ make_flux_limiter_from_name


_LIMITERS: Dict[str, Type[Limiter]] = {
    # NOTE: this is the default for now because it's popular
    "default": MINMODLimiter,
    "none": UnlimitedLimiter,
    "minmod": MINMODLimiter,
    "mc": MonotonizedCentralLimiter,
    "superbee": SUPERBEELimiter,
    "vanalbada": VanAlbadaLimiter,
    "koren": KorenLimiter,
}


def limiter_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available limiters.
    """
    return tuple(_LIMITERS.keys())


def make_limiter_from_name(name: str, **kwargs: Any) -> Limiter:
    """
    :arg name: name of the limiter.
    :arg kwargs: additional arguments to pass to the limiter. Any arguments
        that are not in the limiter's fields are ignored.
    """
    cls = _LIMITERS.get(name)
    if cls is None:
        raise ValueError(f"flux limiter '{name}' not found; try one of {limiter_ids()}")

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}
