# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Linear Advection Equation
-------------------------

This module implements schemes for the linear advection equation

.. math::

    \frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = 0,

where :math:`a(x)` is a known velocity field. For a conservative version
of the same equation see :mod:`pyshocks.continuity`, which also provides
initial conditions.

Schemes
^^^^^^^

.. autoclass:: Scheme
.. autoclass:: FiniteVolumeScheme
.. autoclass:: FiniteDifferenceScheme

.. autoclass:: Godunov
.. autoclass:: ESWENO32

.. autoclass:: FluxSplitGodunov
.. autoclass:: SBPSAT

.. class:: Upwind

    An alias for :class:`Godunov`.

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name
"""

from typing import Any, Dict, Tuple, Type

from pyshocks.advection.schemes import (
    ESWENO32,
    SBPSAT,
    FiniteDifferenceScheme,
    FiniteVolumeScheme,
    FluxSplitGodunov,
    Godunov,
    Scheme,
)

# NOTE: just providing an alias for common usage
Upwind = Godunov

# {{{ make_scheme_from_name

_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": Godunov,
    "godunov": Godunov,
    "upwind": Godunov,
    "esweno32": ESWENO32,
    "sbp": SBPSAT,
    "splitgodunov": FluxSplitGodunov,
}


def scheme_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available schemes.
    """
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve the linear advection equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """

    cls = _SCHEMES.get(name)
    if cls is None:
        raise ValueError(f"scheme '{name}' not found; try one of {scheme_ids()}")

    from dataclasses import fields

    if "velocity" not in kwargs:
        kwargs["velocity"] = None

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}

__all__ = (
    "Scheme",
    "FiniteVolumeScheme",
    "FiniteDifferenceScheme",
    "Godunov",
    "Upwind",
    "ESWENO32",
    "SBPSAT",
    "FluxSplitGodunov",
    "scheme_ids",
    "make_scheme_from_name",
)
