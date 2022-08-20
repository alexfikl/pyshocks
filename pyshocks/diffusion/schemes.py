# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from pyshocks import (
    Grid,
    UniformGrid,
    Boundary,
    SchemeBase,
    FiniteDifferenceSchemeBase,
    ConservationLawScheme,
)
from pyshocks import (
    bind,
    apply_operator,
    numerical_flux,
    predict_timestep,
    evaluate_boundary,
)
from pyshocks import sbp


# {{{ base


@dataclass(frozen=True)
class Scheme(SchemeBase):
    """
    .. attribute:: diffusivity
    """

    diffusivity: jnp.ndarray


@predict_timestep.register(Scheme)
def _predict_timestep_diffusion(
    scheme: Scheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.diffusivity is not None

    dmax = jnp.max(jnp.abs(scheme.diffusivity[grid.i_]))
    return 0.5 * grid.dx_min**2 / dmax


@dataclass(frozen=True)
class FiniteVolumeScheme(Scheme, ConservationLawScheme):
    """Base class for finite volume-based numerical schemes the heat equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(Scheme, FiniteDifferenceSchemeBase):
    """Base class for finite difference-based numerical schemes for the heat equation.

    .. automethod:: __init__
    """


# }}}


# {{{ centered


@dataclass(frozen=True)
class CenteredScheme(FiniteVolumeScheme):
    pass


@numerical_flux.register(CenteredScheme)
def _numerical_flux_diffusion_centered_scheme(
    scheme: CenteredScheme, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    assert scheme.diffusivity is not None

    # FIXME: higher order?
    d = scheme.diffusivity
    davg = (d[1:] + d[:-1]) / 2

    fnum = -davg * (u[1:] - u[:-1]) / grid.df

    return jnp.pad(fnum, 1)  # type: ignore[no-untyped-call]


# }}}


# {{{ SBP-SAT


@dataclass(frozen=True)
class SBPSAT(FiniteDifferenceScheme):
    """
    .. attribute:: op

        A :class:`~pyshocks.sbp.SBP` operator that is used to construct the
        required second-order derivative.
    """

    op: sbp.SBPOperator

    P: ClassVar[jnp.ndarray]
    D2: ClassVar[jnp.ndarray]

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{type(self.op).__name__}".lower()

    @property
    def order(self) -> int:
        return self.op.boundary_order

    @property
    def stencil_width(self) -> int:
        return 0


@bind.register(SBPSAT)
def _bind_diffusion_sbp(  # type: ignore[misc]
    scheme: SBPSAT, grid: UniformGrid, bc: Boundary
) -> SBPSAT:
    # {{{ scheme

    assert isinstance(grid, UniformGrid)
    assert scheme.diffusivity is not None

    P = sbp.sbp_norm_matrix(scheme.op, grid, bc.boundary_type)
    D2 = sbp.sbp_second_derivative_matrix(
        scheme.op, grid, bc.boundary_type, scheme.diffusivity
    )

    # FIXME: make these into sparse matrices
    object.__setattr__(scheme, "P", P)
    object.__setattr__(scheme, "D2", D2)

    breakpoint()

    # }}}

    # {{{ boundary

    from pyshocks.scalar import SATBoundary, PeriodicBoundary

    if isinstance(bc, SATBoundary):
        from pyshocks.scalar import OneSidedDiffusionSATBoundary

        # NOTE: this fixes the boundary conditions for the energy conserving
        # version, see implementation and [Mattsson2004] for details
        if isinstance(bc.left, OneSidedDiffusionSATBoundary):
            assert isinstance(bc.right, OneSidedDiffusionSATBoundary)
            S = sbp.sbp_matrix_from_name(scheme.op, grid, bc.boundary_type, "S")

            object.__setattr__(bc.left, "S", S)
            object.__setattr__(bc.left, "tau", -scheme.diffusivity[0])

            object.__setattr__(bc.right, "S", S)
            object.__setattr__(bc.right, "tau", +scheme.diffusivity[-1])
        else:
            assert not isinstance(bc.right, OneSidedDiffusionSATBoundary)
    elif isinstance(bc, PeriodicBoundary):
        pass
    else:
        raise TypeError(f"unsupported boundary type: '{type(bc).__name__}'")

    # }}}

    return scheme


@apply_operator.register(SBPSAT)
def _apply_operator_diffusion_sbp(
    scheme: SBPSAT, grid: Grid, bc: Boundary, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    gb = evaluate_boundary(bc, grid, t, u)
    return scheme.D2 @ u - gb / scheme.P


# }}}
