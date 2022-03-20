# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import jax

from pyshocks.grid import (
    Grid,
    UniformGrid,
    Quadrature,
    cell_average,
    norm,
    rnorm,
)
from pyshocks.schemes import (
    ScalarFunction, VectorFunction, SpatialFunction,
    SchemeBase,
    ConservationLawScheme,
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
    EOCRecorder,
    estimate_order_of_convergence,
    BlockTimer,
    IterationTimer,
    timeme,
    get_logger,
)

# {{{ version

try:
    # python >=3.8 only
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore[no-redef]

__version__ = metadata.version("pyshocks")

# NOTE: without setting this flag, jax forcefully makes everything float32, even
# if the user requested float64, so it needs to stay on until this is fixed
#       https://github.com/google/jax/issues/8178
jax.config.update("jax_enable_x64", True)

# NOTE: forcing to "cpu" until anyone bothers even trying to test on GPUs
jax.config.update("jax_platform_name", "cpu")

# }}}

__all__ = (
    "ScalarFunction", "VectorFunction", "SpatialFunction",
    "Grid",
    "UniformGrid",
    "SchemeBase",
    "ConservationLawScheme",
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
)
