# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp

from pyshocks import (
    Boundary,
    ConservationLawScheme,
    FiniteDifferenceSchemeBase,
    Grid,
    SchemeBase,
    UniformGrid,
    apply_operator,
    bind,
    evaluate_boundary,
    numerical_flux,
    predict_timestep,
    sbp,
)
from pyshocks.tools import Array, Scalar, ScalarLike

# {{{ base


@dataclass(frozen=True)
class DiffusionScheme(SchemeBase):
    diffusivity: Array
    """Diffusivity coefficient."""


@predict_timestep.register(DiffusionScheme)
def _predict_timestep_diffusion(
    scheme: DiffusionScheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Scalar:
    assert scheme.diffusivity is not None

    dmax = jnp.max(jnp.abs(scheme.diffusivity[grid.i_]))
    return 0.5 * grid.dx_min**2 / dmax


@dataclass(frozen=True)
class FiniteVolumeScheme(DiffusionScheme, ConservationLawScheme):
    """Base class for finite volume-based numerical schemes the heat equation.

    .. automethod:: __init__
    """


@dataclass(frozen=True)
class FiniteDifferenceScheme(DiffusionScheme, FiniteDifferenceSchemeBase):
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
    scheme: CenteredScheme, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    assert scheme.diffusivity is not None

    # FIXME: higher order?
    d = scheme.diffusivity
    davg = (d[1:] + d[:-1]) / 2

    fnum = -davg * (u[1:] - u[:-1]) / grid.df

    return jnp.pad(fnum, 1)


# }}}


# {{{ SBP-SAT


@dataclass(frozen=True)
class SBPSAT(FiniteDifferenceScheme):
    op: sbp.SBPOperator
    """A :class:`~pyshocks.sbp.SBPOperator` operator that is used to construct the
    required second-order derivative.
    """

    P: ClassVar[Array]
    """Diagonal norm operator used by the scheme."""
    D2: ClassVar[Array]
    """Second-order derivative operator used by the scheme."""

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
def _bind_diffusion_sbp(scheme: SBPSAT, grid: UniformGrid, bc: Boundary) -> SBPSAT:
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

    # }}}

    # {{{ boundary

    from pyshocks.scalar import PeriodicBoundary, SATBoundary

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
        raise TypeError(f"Unsupported boundary type: {type(bc).__name__!r}.")

    # }}}

    return scheme


@apply_operator.register(SBPSAT)
def _apply_operator_diffusion_sbp(
    scheme: SBPSAT, grid: Grid, bc: Boundary, t: ScalarLike, u: Array
) -> Array:
    gb = evaluate_boundary(bc, grid, t, u)
    return scheme.D2 @ u - gb / scheme.P


# }}}
