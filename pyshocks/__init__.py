# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from importlib import metadata
import jax

from pyshocks.grid import (
    Grid,
    Quadrature,
    UniformGrid,
    cell_average,
    make_uniform_grid,
    norm,
    rnorm,
)
from pyshocks.schemes import (
    SchemeBase,
    FiniteVolumeScheme,
    ConservationLawScheme,
    CombineConservationLawScheme,
    flux,
    numerical_flux,
    apply_operator,
    predict_timestep,
    Boundary,
    OneSidedBoundary,
    TwoSidedBoundary,
    apply_boundary,
)
from pyshocks.tools import (
    T,
    P,
    ScalarFunction,
    VectorFunction,
    SpatialFunction,
    EOCRecorder,
    estimate_order_of_convergence,
    BlockTimer,
    IterationTimer,
    timeme,
    get_logger,
    set_recommended_matplotlib,
)

# {{{ config

__version__ = metadata.version("pyshocks")

# NOTE: without setting this flag, jax forcefully makes everything float32, even
# if the user requested float64, so it needs to stay on until this is fixed
#       https://github.com/google/jax/issues/8178
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

# NOTE: forcing to "cpu" until anyone bothers even trying to test on GPUs
jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]

# }}}

__all__ = (
    "T",
    "P",
    "ScalarFunction",
    "VectorFunction",
    "SpatialFunction",
    "Grid",
    "UniformGrid",
    "make_uniform_grid",
    "SchemeBase",
    "FiniteVolumeScheme",
    "ConservationLawScheme",
    "CombineConservationLawScheme",
    "flux",
    "numerical_flux",
    "apply_operator",
    "predict_timestep",
    "Boundary",
    "OneSidedBoundary",
    "TwoSidedBoundary",
    "apply_boundary",
    "Quadrature",
    "cell_average",
    "norm",
    "rnorm",
    "EOCRecorder",
    "estimate_order_of_convergence",
    "BlockTimer",
    "IterationTimer",
    "timeme",
    "get_logger",
    "set_recommended_matplotlib",
)
