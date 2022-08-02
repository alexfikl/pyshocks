# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Any, Dict

import pytest

import jax.numpy as jnp

from pyshocks import get_logger, set_recommended_matplotlib

logger = get_logger("test_muscl")
set_recommended_matplotlib()


# {{{ test_limiters


def func_sine(x: jnp.ndarray, k: int = 2) -> jnp.ndarray:
    return jnp.sin(2.0 * jnp.pi * k * x)


def func_step(x: jnp.ndarray, w: float = 1.0) -> jnp.ndarray:
    mid = (x[0] + x[-1]) / 2

    return w * (x < mid).astype(x.dtype)


@pytest.mark.parametrize(
    ("lm_name", "lm_kwargs"),
    [
        ("minmod", {"theta": 1.0}),
        ("minmod", {"theta": 1.5}),
        ("minmod", {"theta": 2.0}),
        ("mc", {}),
        ("superbee", {}),
        ("vanalbada", {"variant": 1}),
        ("vanalbada", {"variant": 2}),
        ("koren", {}),
    ],
)
@pytest.mark.parametrize(
    "smooth",
    [
        True,
        False,
    ],
)
def test_limiters(
    lm_name: str, lm_kwargs: Dict[str, Any], smooth: bool, visualize: bool = True
) -> None:
    from pyshocks.limiters import make_limiter_from_name, limit, evaluate

    lm = make_limiter_from_name(lm_name, **lm_kwargs)

    from pyshocks.grid import make_uniform_grid

    grid = make_uniform_grid(-1.0, 1.0, n=256, nghosts=1)

    if smooth:
        u = func_sine(grid.x)
    else:
        u = func_step(grid.x)

    phi = limit(lm, grid, u)
    assert phi.shape == u.shape

    i = grid.i_
    logger.info("phi: %.5e %.5e", jnp.min(phi[i]), jnp.max(phi[i]))

    if not visualize:
        return

    sm_name = "sine" if smooth else "step"
    lm_args = "_".join(f"{k}_{v}" for k, v in lm_kwargs.items())
    filename = f"muscl_limiter_{sm_name}_{lm_name}_{lm_args}".replace(".", "_")

    import matplotlib.pyplot as mp

    fig = mp.figure()

    # {{{ plot solution

    ax = fig.gca()
    ax.plot(grid.x[i], u[i], label="$u$")
    ax.plot(grid.x[i], phi[i], "k--", label=r"$\phi(r)$")

    ax.set_xlabel("$x$")
    ax.legend()

    fig.savefig(filename)
    fig.clf()

    # }}}

    # {{{ plot limiter

    if smooth:
        # https://en.wikipedia.org/wiki/Flux_limiter
        r = jnp.linspace(0.0, 3.0, 256, dtype=grid.x.dtype)
        phi = evaluate(lm, r)

        ax = fig.gca()
        ax.plot(r, phi)
        ax.fill_between(
            [0.0, 0.5, 1.0, 2.0, 3.0],
            [0.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            color="k",
            alpha=0.1,
        )
        ax.set_xlabel("$r$")
        ax.set_ylabel(r"$\phi(r)$")

        fig.savefig(f"{filename}_tvd")
        fig.clf()

    # }}}

    mp.close(fig)


# }}}


# {{{ check tvd

# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
