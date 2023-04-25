# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

r"""
Burgers' Equation
-----------------

This module implements schemes for the inviscid Burgers' equation

.. math::

    \frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial}{\partial x} (u^2)  = 0.

Schemes
^^^^^^^

.. autoclass:: Scheme
.. autoclass:: FiniteVolumeScheme
.. autoclass:: FiniteDifferenceScheme

.. autoclass:: Godunov
.. autoclass:: Rusanov
    :members:
.. autoclass:: LaxFriedrichs
    :members:
.. autoclass:: EngquistOsher
    :members:

.. autoclass:: ESWENO32
.. autoclass:: SSMUSCL
    :members:

.. autoclass:: FluxSplitRusanov
    :members:
.. autoclass:: SSWENO242
    :members:

.. autofunction:: scheme_ids
.. autofunction:: make_scheme_from_name
"""

from typing import Any, Dict, Tuple, Type

from pyshocks.burgers.schemes import (
    ESWENO32,
    SSMUSCL,
    EngquistOsher,
    FiniteDifferenceScheme,
    FiniteVolumeScheme,
    FluxSplitRusanov,
    Godunov,
    LaxFriedrichs,
    Rusanov,
    Scheme,
)
from pyshocks.burgers.ssweno import SSWENO242

# {{{ make_scheme_from_name

_SCHEMES: Dict[str, Type[Scheme]] = {
    "default": LaxFriedrichs,
    "godunov": Godunov,
    "rusanov": Rusanov,
    "lf": LaxFriedrichs,
    "eo": EngquistOsher,
    "esweno32": ESWENO32,
    "ssmuscl": SSMUSCL,
    "ssweno242": SSWENO242,
    "splitrusanov": FluxSplitRusanov,
}


def scheme_ids() -> Tuple[str, ...]:
    """
    :returns: a :class:`tuple` of available schemes.
    """
    return tuple(_SCHEMES.keys())


def make_scheme_from_name(name: str, **kwargs: Any) -> Scheme:
    """
    :arg name: name of the scheme used to solve Burgers' equation.
    :arg kwargs: additional arguments to pass to the scheme. Any arguments
        that are not in the scheme's fields are ignored.
    """
    cls = _SCHEMES.get(name)
    if cls is None:
        from pyshocks.tools import join_or

        raise ValueError(
            f"Scheme {name!r} not found. Try one of {join_or(scheme_ids())}."
        )

    from dataclasses import fields

    return cls(**{f.name: kwargs[f.name] for f in fields(cls) if f.name in kwargs})


# }}}


__all__ = (
    "Scheme",
    "FiniteVolumeScheme",
    "FiniteDifferenceScheme",
    "Godunov",
    "Rusanov",
    "LaxFriedrichs",
    "EngquistOsher",
    "ESWENO32",
    "SSMUSCL",
    "SSWENO242",
    "FluxSplitRusanov",
    "scheme_ids",
    "make_scheme_from_name",
)
