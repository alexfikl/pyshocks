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
        ConvolutionType.Reflect,
        ConvolutionType.Mirror,
        ConvolutionType.Constant,
        ConvolutionType.Nearest,
        ConvolutionType.Wrap,
    ],
)
def test_convolve_vs_scipy(cv: ConvolutionType, *, visualize: bool = False) -> None:
    import numpy as np
    from pyshocks.tools import BlockTimer

    rng = np.random.default_rng(seed=42)

    for _ in range(16):
        # NOTE: ideally we want a combination of odd and even values for both
        n = rng.integers(126, 384)
        m = rng.integers(10, 27)

        # {{{ compute scipy

        import scipy.ndimage as snd

        a = rng.random(n, dtype=np.float64)
        w = rng.random(m, dtype=np.float64)

        with BlockTimer() as bt:
            result_sp = snd.convolve1d(a, w, mode=cv.name.lower())
        logger.info("scipy:     %s", bt)

        # }}}

        # {{{ compare

        from pyshocks.convolve import convolve1d

        a = jax.device_put(a)
        w = jax.device_put(w)

        convolve1d_jitted = jax.jit(convolve1d, static_argnames=("mode",))
        _ = convolve1d_jitted(a, w, mode=cv)

        with BlockTimer() as bt:
            result_ps = convolve1d_jitted(a, w, mode=cv)
        logger.info("jax:       %s", bt)

        error = jnp.linalg.norm(result_sp - result_ps) / jnp.linalg.norm(result_sp)
        logger.info("error: %s n %3d m %3d %.12e", cv, n, m, error)

        # }}}

        if visualize:
            from pyshocks.tools import figure

            filename = f"convolve_vs_scipy_{cv.name.lower()}_{n}_{m}"
            with figure(filename) as fig:
                ax = fig.gca()

                ax.plot(result_ps)
                ax.plot(result_sp, "k--")
                ax.set_title(cv.name)

        assert error < 1.0e-12


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
