# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp

from pyshocks import get_logger, set_recommended_matplotlib
from pyshocks.convolve import ConvolutionType

import pytest

logger = get_logger("test_utils")
set_recommended_matplotlib()


# {{{ test_convolve_vs_scipy


@pytest.mark.parametrize(
    "cv",
    [
        # ConvolutionType.Reflect,
        ConvolutionType.Constant,
        # ConvolutionType.Nearest,
        # ConvolutionType.Wrap,
    ],
)
@pytest.mark.parametrize(
    ("n", "m"),
    [
        (128, 4),
        (128, 5),
        (128, 10),
        (128, 11),
        (257, 4),
        (257, 5),
        (257, 10),
        (257, 11),
    ],
)
def test_convolve_vs_scipy(
    cv: ConvolutionType, n: int, m: int, visualize: bool = True
) -> None:
    # {{{ compute scipy

    import numpy as np
    import scipy.ndimage as snd

    rng = np.random.default_rng(seed=42)

    a = rng.random(n, dtype=np.float64)
    w = rng.random(m, dtype=np.float64)

    result_sp = snd.convolve1d(a, w, mode=cv.name.lower())

    # }}}

    # {{{ compare

    from pyshocks.convolve import convolve1d

    a = jax.device_put(a)
    w = jax.device_put(w)

    result_ps = convolve1d(a, w, mode=cv)

    error = jnp.linalg.norm(result_sp - result_ps) / jnp.linalg.norm(result_sp)
    logger.info("")
    logger.info("error: %s %.12e", cv, error)
    # assert error < 1.0e-12

    # }}}

    if visualize:
        from pyshocks.tools import figure

        filename = f"convolve_vs_scipy_{cv.name.lower()}_{n}_{m}"
        with figure(filename) as fig:
            ax = fig.gca()

            ax.plot(result_ps)
            ax.plot(result_sp, "k--")
            ax.set_title(cv.name)


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
