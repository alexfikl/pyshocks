# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from importlib import metadata

import jax

from pyshocks.grid import (
    Grid,
    Quadrature,
    UniformGrid,
    cell_average,
    make_leggauss_quadrature,
    make_uniform_cell_grid,
    make_uniform_point_grid,
    make_uniform_ssweno_grid,
    norm,
    rnorm,
)
from pyshocks.logging import get_logger
from pyshocks.schemes import (
    Boundary,
    BoundaryType,
    CombineConservationLawScheme,
    CombineScheme,
    ConservationLawScheme,
    FiniteDifferenceSchemeBase,
    FiniteVolumeSchemeBase,
    SchemeBase,
    SchemeT,
    apply_boundary,
    apply_operator,
    bind,
    evaluate_boundary,
    flux,
    numerical_flux,
    predict_timestep,
)
from pyshocks.tools import (
    BlockTimer,
    EOCRecorder,
    IterationTimer,
    P,
    ScalarFunction,
    SpatialFunction,
    T,
    TemporalFunction,
    TimeResult,
    VectorFunction,
    estimate_order_of_convergence,
    profileit,
    repeatit,
    set_recommended_matplotlib,
    timeit,
)

# {{{ config

__version__ = metadata.version("pyshocks")

# NOTE: without setting this flag, jax forcefully makes everything float32, even
# if the user requested float64, so it needs to stay on until this is fixed
#       https://github.com/google/jax/issues/8178
jax.config.update("jax_enable_x64", val=True)  # type: ignore[no-untyped-call]

# NOTE: forcing to "cpu" until anyone bothers even trying to test on GPUs
jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]

# NOTE: enabled on more recent versions of jax
try:  # noqa: SIM105
    # NOTE: this option is removed in newer version
    jax.config.update("jax_array", val=True)  # type: ignore[no-untyped-call]
except AttributeError:
    pass

# NOTE: useful options while debugging
if __debug__:
    jax.config.update("jax_debug_infs", val=True)  # type: ignore[no-untyped-call]
    jax.config.update("jax_debug_nans", val=True)  # type: ignore[no-untyped-call]
    jax.config.update("jax_enable_checks", val=True)  # type: ignore[no-untyped-call]
    jax.config.update("jax_numpy_dtype_promotion", "strict")  # type: ignore


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
