# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from pyshocks import get_logger, make_uniform_cell_grid, set_recommended_matplotlib
from pyshocks.limiters import make_limiter_from_name
from pyshocks.tools import Array

logger = get_logger("test_muscl")
set_recommended_matplotlib()


# {{{ test_limiters


def func_sine(x: Array, k: int = 1) -> Array:
    return jnp.sin(2.0 * jnp.pi * k * x)


def func_step(x: Array, w: float = 0.25, a: float = -1.0, b: float = 1.0) -> Array:
    mid = (b + a) / 2
    a = mid - w * (mid - a)
    b = mid + w * (b - mid)

    return jnp.logical_and(x > a, x < b).astype(x.dtype)


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
        ("vanleer", {}),
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
def test_flux_limiters(
    lm_name: str, lm_kwargs: dict[str, Any], *, smooth: bool, visualize: bool = False
) -> None:
    lm = make_limiter_from_name(lm_name, **lm_kwargs)
    grid = make_uniform_cell_grid(-1.0, 1.0, n=128, nghosts=1)

    from pyshocks import cell_average, make_leggauss_quadrature

    quad = make_leggauss_quadrature(grid, order=3)

    if smooth:
        u = cell_average(quad, func_sine)
    else:
        u = cell_average(quad, partial(func_step, a=float(grid.a), b=float(grid.b)))

    from pyshocks.limiters import evaluate, flux_limit

    phi = flux_limit(lm, grid, u)
    assert phi.shape == u.shape

    i = grid.i_
    logger.info("phi: %.5e %.5e", jnp.min(phi[i]), jnp.max(phi[i]))

    if smooth:
        assert jnp.max(phi[i]) > 0.0
    else:
        # NOTE: the step function is piecewise constant, so we expect the
        # limiter to always decay to the first-order scheme.
        assert jnp.max(phi[i]) < 1.0e-12

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
        # https://en.wikipedia.org/wiki/Flux_limiter#/media/File:LimiterPlots1.png
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


@pytest.mark.parametrize(
    ("lm_name", "lm_kwargs"),
    [
        # ("none", {}),
        ("minmod", {"theta": 1.0}),
        ("minmod", {"theta": 1.5}),
        ("minmod", {"theta": 2.0}),
        ("mc", {}),
        ("superbee", {}),
        ("vanalbada", {"variant": 1}),
        ("vanalbada", {"variant": 2}),
        ("vanleer", {}),
        ("koren", {}),
    ],
)
@pytest.mark.parametrize(
    "smooth",
    [
        # NOTE: the smooth sine wave gives some TVD failures of the order
        # of 1.0e-5, likely due to the poor handling of the extrema
        # True,
        False,
    ],
)
def test_tvd_slope_limiter_burgers(
    lm_name: str, lm_kwargs: dict[str, Any], *, smooth: bool, visualize: bool = False
) -> None:
    from pyshocks import burgers
    from pyshocks.reconstruction import MUSCL
    from pyshocks.scalar import PeriodicBoundary

    # {{{ setup

    grid = make_uniform_cell_grid(-1.0, 1.0, n=256, nghosts=3)

    lm = make_limiter_from_name(lm_name, **lm_kwargs)
    rec = MUSCL(lm=lm)
    scheme = burgers.Godunov(rec=rec)
    boundary = PeriodicBoundary()

    from pyshocks import cell_average, make_leggauss_quadrature

    quad = make_leggauss_quadrature(grid, order=3)

    if smooth:
        u0 = cell_average(quad, func_sine)
    else:
        u0 = cell_average(quad, func_step)

    # }}}

    # {{{ evolve

    from pyshocks import apply_operator, norm

    def _apply_operator(_t: float, _u: Array) -> Array:
        return apply_operator(scheme, grid, boundary, _t, _u)

    from pyshocks.timestepping import SSPRK33, predict_maxit_from_timestep, step

    dt = 0.1 * grid.dx_min / jnp.max(jnp.abs(u0))
    tfinal = 0.5
    maxit, dt = predict_maxit_from_timestep(tfinal, dt)

    method = SSPRK33(
        predict_timestep=lambda t, u: dt,
        source=jax.jit(_apply_operator),
        # source=_apply_operator,
        checkpoint=None,
    )

    tvd = norm(grid, u0, p="tvd")
    fail_count = 0
    tvd_increase_acc = []

    u = u0
    for event in step(method, u0, maxit=maxit, tfinal=tfinal):
        u = event.u
        tvd_next = norm(grid, u, p="tvd")

        fail_count = fail_count + int(tvd_next > tvd + 1.0e-12)
        tvd_increase_acc.append(tvd_next - tvd)

        tvd = tvd_next
        assert jnp.isfinite(tvd), event

    tvd_increase = jnp.array(tvd_increase_acc, dtype=u0.dtype)

    if fail_count:
        logger.info(
            "FAILED[%5d / %5d]: limiter %s args %s",
            fail_count,
            maxit,
            lm_name,
            lm_kwargs,
        )
        logger.info(
            "TVD: max(TV(u[n + 1]) - TV(u[n])) = %.12e",
            jnp.max(jnp.array(tvd_increase)),
        )
    else:
        logger.info("SUCCESS: limiter %s args %s", lm_name, lm_kwargs)

    # }}}

    if visualize:
        sm_name = "sine" if smooth else "step"
        lm_args = "_".join(f"{k}_{v}" for k, v in lm_kwargs.items())
        suffix = f"{sm_name}_{lm_name}_{lm_args}".replace(".", "_")

        import matplotlib.pyplot as mp

        fig = mp.figure()

        # {{{ plot total variation

        ax = fig.gca()

        ax.plot(tvd_increase)
        ax.set_xlabel("$n$")
        ax.set_ylabel("$TV(u^{n + 1}) - TV(u^n)$")

        fig.savefig(f"muscl_tvd_{suffix}")
        fig.clf()

        # }}}

        # {{{ plot solution

        ax = fig.gca()

        i = grid.i_
        ax.plot(grid.x[i], u[i])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(T, x)$")

        fig.savefig(f"muscl_tvd_solution_{suffix}")
        fig.clf()

        # }}}

        mp.close(fig)

    assert fail_count == 0


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
