# SPDX-FileCopyrightText: 2022 Daniel J. Bodony <bodony@illinois.edu>
# SPDX-License-Identifier: MIT

from typing import Any, Tuple

import numpy as np

# FIXME: should probably use np.ndarray[Any, np.float64] or something
Array = Any
Matrix = Any


# {{{ initial conditions


def ic_riemann(x: Array, ul: float, ur: float) -> Array:
    L = x[-1] - x[0]
    x0 = x[0] + 0.25 * L

    H = np.array(x < x0, dtype=x.dtype)
    return ur + (ul - ur) * H


def ic_tanh(x: Array, ul: float, ur: float) -> Array:
    L = x[-1] - x[0]

    x0 = x[0] + 0.25 * L
    delta = L / 20

    return ul + 0.5 * (ur - ul) * (1.0 + np.tanh((x - x0) / delta))


# }}}

# {{{ SBP operators


def norm_matrix(dx: Array, nx: int) -> Matrix:
    mat = dx * np.eye(nx + 1)
    mat[0, 0] = 0.5 * dx
    mat[-1, -1] = 0.5 * dx

    return mat


def first_derivative_matrix(invH: Matrix) -> Matrix:
    nx = invH.shape[0] - 1

    diagv = np.zeros(nx + 1)
    diagv[0] = -0.5
    diagv[-1] = 0.5

    diagvp1 = 0.5 * np.ones(nx)
    diagvp1[0] = 0.5

    diagvm1 = -np.flip(diagvp1)

    mat = np.diag(diagv) + np.diag(diagvp1, 1) + np.diag(diagvm1, -1)
    return invH @ mat


def second_derivative_matrix(
    dx: Array,
    H: Matrix,
    invH: Matrix,
    D1: Matrix,
    nu: float,
) -> Matrix:
    nx = H.shape[0] - 1

    # S-matrix
    S = np.eye(nx + 1)
    S[0, 0:3] = [-1.5, 2, -0.5]
    S[-1, :] = -np.flip(S[0])

    # B-matrix = B22 for 2nd order
    B = nu * np.eye(nx + 1)

    # Bbar-matrix
    Bbar = np.zeros((nx + 1, nx + 1))
    Bbar[0, 0] = -B[0, 0]
    Bbar[-1, -1] = B[-1, -1]

    # D_2^(2)-matrix
    diagv = -2 * np.ones(nx + 1)
    diagv[0] = 1
    diagv[-1] = 1

    D22 = np.diag(diagv) + np.diag(np.ones(nx), -1) + np.diag(np.ones(nx), +1)
    D22[0, 0:3] = [1, -2, 1]
    D22[-1, :] = np.flip(D22[0])

    # C_2^(2)-matrix
    C22 = np.eye(nx + 1)
    C22[0, 0] = 0.0
    C22[-1, -1] = 0.0

    # R^(b) matrix
    R = 0.25 * dx**3 * (D22.T @ C22 @ B @ D22)

    # M^b matrix
    M = D1.T @ H @ B @ D1 + R

    # D2 operator
    return invH @ (-M + Bbar @ S)


# }}}


# {{{ MUSCL


def slope_limiter(r: Array) -> Array:
    return np.maximum(0.0, np.minimum(1, r))


def artificial_dissipation_muscl(invH: Matrix, u: Array) -> Array:
    nx = invH.shape[0] - 1

    # D1hat-matrix
    D1hat = -np.identity(nx + 1) + np.diag(np.ones(nx), 1)
    D1hat[-1, -1] = 1
    D1hat[-1, -2] = -1

    # BM-matrix for MUSCL coefficients
    bv = np.zeros(nx + 1)

    for i in range(2, nx - 1):
        uiph = 0.5 * (u[i] + u[i + 1])
        duip1 = u[i + 2] - u[i + 1]
        dui = u[i + 1] - u[i]
        duim1 = u[i] - u[i - 1]

        ri = duim1 / (dui + 1e-14)
        rip1 = dui / (duip1 + 1e-14)

        phi = slope_limiter(ri)
        phip1 = slope_limiter(rip1)
        psip1 = phip1 / (rip1 + 1e-14)

        uliph = u[i] + 0.5 * phi * dui
        uriph = u[i + 1] - 0.5 * phip1 * duip1

        ar = 0.5 * (uriph + u[i + 1])
        al = 0.5 * (uliph + u[i])

        bv[i] = 0.5 * (
            np.abs(uiph) * (1.0 - 0.5 * phi - 0.5 * psip1)
            + 0.5 * (ar * psip1 - al * phi)
        )

    bv[0] = bv[2]
    bv[1] = bv[2]
    bv[nx - 1] = bv[nx - 2]
    bv[nx] = bv[nx - 2]
    BM = np.diag(bv)

    # -P^{-1} D1Hat^T BM D1Hat u

    return -(invH @ (D1hat.T @ (BM @ (D1hat @ u))))


# }}}


# {{{ Roe


def artificial_dissipation_roe(invH: Matrix, u: Array) -> Array:
    nx = invH.shape[0] - 1

    # D1hat-matrix

    D1hat = -np.eye(nx + 1) + np.diag(np.ones(nx), 1)
    D1hat[-1, -1] = 1
    D1hat[-1, -2] = -1

    # BM-matrix for Roe coefficients
    bv = np.zeros(nx + 1)

    for i in range(2, nx - 1):
        bv[i] = 0.5 * (u[i] + u[i + 1])

    bv[0] = bv[2]
    bv[1] = bv[2]
    bv[nx - 1] = bv[nx - 2]
    bv[nx] = bv[nx - 2]
    BM = np.diag(bv)

    # -P^{-1} D1Hat^T BM D1Hat u
    return -(invH @ (D1hat.T @ (BM @ (D1hat @ u))))


def artificial_dissipation_roe_logarithmic_entropy_variables(
    invH: Matrix, u: Array
) -> Array:
    nx = invH.shape[0] - 1
    w = 1 / u

    # D1hat-matrix
    D1hat = -np.eye(nx + 1) + np.diag(np.ones(nx), 1)
    D1hat[-1, -1] = 1
    D1hat[-1, -2] = -1

    # BM-matrix for Roe coefficients
    bv = np.zeros(nx + 1)

    for i in range(2, nx - 1):
        bv[i] = (-0.5 * (w[i] + w[i + 1])) ** (-3.0)

    bv[0] = bv[2]
    bv[1] = bv[2]
    bv[nx - 1] = bv[nx - 2]
    bv[nx] = bv[nx - 2]
    BM = np.diag(bv)

    # -P^{-1} D1Hat^T BM D1Hat u
    return -(invH @ (D1hat.T @ (BM @ (D1hat @ w))))


# }}}


def burgers_rhs(
    t: float,
    u: Array,
    e0: Array,
    invH: Matrix,
    D1: Matrix,
    D2: Matrix,
    ul: float,
    tau: float = 1.0 / 3.0,
) -> Array:
    A = np.diag(u)
    E = invH @ (np.outer(e0, e0) @ A)

    f = u**2 / 2
    return (
        -(D1 @ f)
        - 2.0 * tau * (E @ (A @ (u - ul * e0)))
        + (D2 @ u)
        + artificial_dissipation_muscl(invH, u)
        # + artificial_dissipation_roe(invH, u)
        # + artificial_dissipation_roe_logarithmic_entropy_variables(invH, u)
    )


def main(
    nx: int = 100,  # number of spatial grid cells
    nt: int = 400,  # number of timesteps
    a: float = -3.0,  # left-most grid point
    b: float = 3.0,  # right-most grid point
    t0: float = 0.0,  # initial time
    tf: float = 1.0,  # final time
    ul: float = 1.5,  # initial left state
    ur: float = 0.5,  # initial right state
    nu: float = 0.0e-2,  # "laminar" viscosity
) -> None:
    # derived constants
    L = b - a
    T = tf - t0

    # set time array
    # dt = T / (nt - 1)
    tspan = np.linspace(0, T, nt)

    # set x arrays
    dx = L / nx
    x = np.linspace(a, b, nx + 1)

    # initialize the state
    u0 = ic_riemann(x, ul, ur)
    # u0 = ic_tanh(x, ul, ur)

    # return static matrices
    H = norm_matrix(dx, nx)
    invH = np.linalg.inv(H)

    e0 = np.zeros(nx + 1)
    e0[0] = 1

    D1 = first_derivative_matrix(invH)
    D2 = second_derivative_matrix(dx, H, invH, D1, nu)

    print("Starting integration ...", end=" ")

    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: burgers_rhs(t, y, e0, invH, D1, D2, ul),
        [tspan[0], tspan[-1]],
        u0,
        t_eval=tspan,
        rtol=1e-5,
    )

    print("Done.")

    # animate the solution
    # code originally lifted and modified from
    # https://matplotlib.org/examples/animation/simple_anim.html

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")

    # plot the initial condition
    ax.plot(x, u0, color="r", marker="o", linewidth=2)

    # prepare for animated lines
    (line,) = ax.plot(x, x, color="b", linestyle="-", linewidth=2, marker="*")

    def animate(n: int) -> Tuple[Any, ...]:
        line.set_ydata(sol.y[:, n])
        return (line,)

    # Init only required for blitting to give a clean slate.
    def init() -> Tuple[Any, ...]:
        line.set_ydata(x)
        return (line,)

    from matplotlib import animation

    anim = animation.FuncAnimation(  # noqa: F841
        fig, animate, np.arange(1, nt), init_func=init, interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    main()
