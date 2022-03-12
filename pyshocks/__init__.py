# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import jax

from pyshocks.grid import (
        Grid, UniformGrid,
        Quadrature, cell_average, norm, rnorm,
        )
from pyshocks.schemes import (
        SchemeBase, ConservationLawScheme,
        flux, numerical_flux, apply_operator, predict_timestep,

        Boundary, OneSidedBoundary, TwoSidedBoundary, apply_boundary,
        )
from pyshocks.tools import (
        EOCRecorder, estimate_order_of_convergence,
        BlockTimer, IterationTimer, timeme,
        get_logger,
        )

# {{{ version

try:
    # python >=3.8 only
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata       # type: ignore[no-redef]

__version__ = metadata.version("pyshocks")

# NOTE: jax defaults to 32bit because neural networks
jax.config.update("jax_enable_x64", 1)

# }}}

__all__ = (
        "Grid", "UniformGrid",

        "SchemeBase", "ConservationLawScheme",
        "flux", "numerical_flux", "apply_operator", "predict_timestep",
        "Boundary", "OneSidedBoundary", "TwoSidedBoundary", "apply_boundary",

        "Quadrature", "cell_average",

        "norm", "rnorm",
        "EOCRecorder", "estimate_order_of_convergence",

        "BlockTimer", "IterationTimer", "timeme",

        "get_logger",
)
