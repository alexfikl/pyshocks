# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from importlib import metadata
import jax

from pyshocks.tools import (
    T,
    P,
    ScalarFunction,
    VectorFunction,
    SpatialFunction,
    TemporalFunction,
    EOCRecorder,
    estimate_order_of_convergence,
    TimeResult,
    BlockTimer,
    IterationTimer,
    timeit,
    repeatit,
    profileit,
    get_logger,
    set_recommended_matplotlib,
)
from pyshocks.grid import (
    Grid,
    Quadrature,
    UniformGrid,
    make_leggauss_quadrature,
    cell_average,
    make_uniform_cell_grid,
    make_uniform_point_grid,
    make_uniform_ssweno_grid,
    norm,
    rnorm,
)
from pyshocks.schemes import (
    SchemeT,
    SchemeBase,
    FiniteVolumeSchemeBase,
    FiniteDifferenceSchemeBase,
    bind,
    apply_operator,
    predict_timestep,
    CombineScheme,
    ConservationLawScheme,
    CombineConservationLawScheme,
    flux,
    numerical_flux,
    Boundary,
    BoundaryType,
    apply_boundary,
    evaluate_boundary,
)

# {{{ config

__version__ = metadata.version("pyshocks")

# NOTE: without setting this flag, jax forcefully makes everything float32, even
# if the user requested float64, so it needs to stay on until this is fixed
#       https://github.com/google/jax/issues/8178
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

# NOTE: forcing to "cpu" until anyone bothers even trying to test on GPUs
jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]

# NOTE: enabled on more recent versions of jax
jax.config.update("jax_array", True)  # type: ignore[no-untyped-call]

# NOTE: useful options while debugging
if __debug__:
    jax.config.update("jax_debug_infs", True)  # type: ignore[no-untyped-call]
    jax.config.update("jax_debug_nans", True)  # type: ignore[no-untyped-call]
    jax.config.update("jax_enable_checks", True)  # type: ignore[no-untyped-call]

# }}}

__all__ = (
    "T",
    "P",
    "ScalarFunction",
    "VectorFunction",
    "SpatialFunction",
    "TemporalFunction",
    "Grid",
    "UniformGrid",
    "make_uniform_cell_grid",
    "make_uniform_point_grid",
    "make_uniform_ssweno_grid",
    "SchemeT",
    "SchemeBase",
    "CombineScheme",
    "FiniteVolumeSchemeBase",
    "FiniteDifferenceSchemeBase",
    "bind",
    "apply_operator",
    "predict_timestep",
    "ConservationLawScheme",
    "CombineConservationLawScheme",
    "flux",
    "numerical_flux",
    "Boundary",
    "BoundaryType",
    "apply_boundary",
    "evaluate_boundary",
    "Quadrature",
    "make_leggauss_quadrature",
    "cell_average",
    "norm",
    "rnorm",
    "EOCRecorder",
    "estimate_order_of_convergence",
    "TimeResult",
    "BlockTimer",
    "IterationTimer",
    "timeit",
    "repeatit",
    "profileit",
    "get_logger",
    "set_recommended_matplotlib",
)
