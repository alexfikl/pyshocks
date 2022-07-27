# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from pyshocks import get_logger

logger = get_logger("example-autodiff")


def scalar_fn(
    A: jnp.ndarray, x: jnp.ndarray, b: jnp.ndarray  # noqa: N803
) -> jnp.ndarray:
    y = A @ x + b
    return (y @ y) / 2.0


def vector_fn(
    A: jnp.ndarray, x: jnp.ndarray, b: jnp.ndarray  # noqa: N803
) -> jnp.ndarray:
    return A @ x + b


def main(n: int = 128) -> None:
    key = jax.random.PRNGKey(42)
    A = jax.random.uniform(key, shape=(n, n), dtype=np.float64)  # noqa: N806
    b = jax.random.uniform(key, shape=(n,), dtype=np.float64)

    x0 = jax.random.uniform(key, shape=(n,), dtype=np.float64)
    y0 = jax.random.uniform(key, shape=(n,), dtype=np.float64)

    # {{{ test gradient: df/dx_i

    jax_grad = jax.grad(lambda x: scalar_fn(A, x, b))(x0)
    our_grad = A.T @ (A @ x0 + b)

    error = jla.norm(jax_grad - our_grad) / jla.norm(our_grad)
    logger.info("error[grad]: %.15e", error)

    # }}}

    # {{{ test jacobian: df_i/dx_j

    # NOTE: From the documentation at
    #   https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    #
    # These two functions compute the same values (up to machine numerics),
    # but differ in their implementation: jacfwd uses forward-mode automatic
    # differentiation, which is more efficient for “tall” Jacobian matrices,
    # while jacrev uses reverse-mode, which is more efficient for “wide”
    # Jacobian matrices. For matrices that are near-square, jacfwd probably
    # has an edge over jacrev.

    jax_jac = jax.jacfwd(lambda x: vector_fn(A, x, b))(x0)
    our_jac = A

    error = jla.norm(jax_jac - our_jac) / jla.norm(our_jac)
    logger.info("error[jacfwd]: %.15e", error)

    jax_jac = jax.jacrev(lambda x: vector_fn(A, x, b))(x0)
    our_jac = A

    error = jla.norm(jax_jac - our_jac) / jla.norm(our_jac)
    logger.info("error[jacrev]: %.15e", error)

    # }}}

    # {{{ test jacobian product: df_i/dx_j(x0) y0_j

    _, jax_jac = jax.jvp(lambda x: vector_fn(A, x, b), (x0,), (y0,))
    our_jac = A @ y0

    error = jla.norm(jax_jac - our_jac) / jla.norm(our_jac)
    logger.info("error[jvp]: %.15e", error)

    # }}}

    # {{{ test reverse jacobian product: df_i/dx_j y0_i

    _, jax_jac = jax.vjp(
        lambda x: vector_fn(A, x, b), x0  # type: ignore[no-any-return]
    )
    jax_jac = jax_jac(y0)[0]
    our_jac = A.T @ y0

    error = jla.norm(jax_jac - our_jac) / jla.norm(our_jac)
    logger.info("error[vjp]: %.15e", error)

    # }}}


if __name__ == "__main__":
    main()
