# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Linear Diffusion Equation
-------------------------

This module implements schemes for the linear diffusion equation

.. math::

    \frac{\partial u}{\partial t}
    - \frac{\partial}{\partial x} \left(d(x) \frac{\partial u}{\partial x}\right)
    = 0

where :math:`d(x) > 0` is the diffusivity.

Schemes
^^^^^^^

.. autoclass:: Scheme
.. autoclass:: FiniteVolumeScheme
.. autoclass:: FiniteDifferenceScheme
.. autoclass:: CenteredScheme
.. autoclass:: SBPSAT

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name
"""

from typing import Any, Dict, Tuple, Type

from pyshocks.diffusion.schemes import (
    Scheme,
    FiniteVolumeScheme,
    FiniteDifferenceScheme,
    CenteredScheme,
    SBPSAT,
)


# {{{ make_scheme_from_name

_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": CenteredScheme,
    "centered": CenteredScheme,
    "sbp": SBPSAT,
}


def scheme_ids() -> Tuple[str, ...]:
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve the linear diffusion equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """

    cls = _SCHEMES.get(name)
    if cls is None:
        raise ValueError(f"scheme '{name}' not found; try one of {scheme_ids()}")

    from dataclasses import fields

    if "diffusivity" not in kwargs:
        kwargs["diffusivity"] = None

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}

__all__ = (
    "Scheme",
    "FiniteVolumeScheme",
    "FiniteDifferenceScheme",
    "CenteredScheme",
    "SBPSAT",
    "scheme_ids",
    "make_scheme_from_name",
)
