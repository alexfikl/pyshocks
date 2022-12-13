# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import enum

import jax.numpy as jnp

from pyshocks.schemes import BoundaryType


@enum.unique
class ConvolutionType:
    """Convolution types supported by :func:`scipy.ndimage.convolve1d`."""

    #: Reflected around the edge of the last pixel, e.g.
    #: ``(d c b a | a b c d | d c b a)``.
    Reflect: enum.auto()
    #: Filled with a constant beyond the edge of the last pixel, e.g.
    #: ``(k k k k | a b c d | k k k k)``.
    Constant: enum.auto()
    #: Filled with the value of the last pixel beyond the boundary, e.g.
    #: ``(a a a a | a b c d | d d d d)``.
    Nearest: enum.auto()
    #: Reflected around the center of the last pixel, i.e.
    #: ``(d c b | a b c d | c b a)``.
    Mirror: enum.auto()
    #: Wrapping around the opposite edge, i.e.
    #: ``(a b c d | a b c d | a b c d)``.
    Wrap = enum.auto()


_BOUNDARY_TYPE_TO_CONVOLUTION_TYPE = {
    BoundaryType.Periodic: ConvolutionType.Wrap,
    BoundaryType.Dirichlet: ConvolutionType.Nearest,
    BoundaryType.Neumann: ConvolutionType.Nearest,
    BoundaryType.HomogeneousNeumann: ConvolutionType.Reflect,
}

_CONVOLUTION_TYPE_TO_PAD_MODE = {
    ConvolutionType.Reflect: "reflect",
    ConvolutionType.Constant: "constant",
    ConvolutionType.Nearest: "edge",
    # ConvolutionType.Mirror: "reflect", ??
    ConvolutionType.Wrap: "wrap",
}


def boundary_type_to_mode(bc: BoundaryType) -> ConvolutionType:
    if bc not in _BOUNDARY_TYPE_TO_CONVOLUTION_TYPE:
        raise ValueError(f"unknown boundary type: {bc}")

    return _BOUNDARY_TYPE_TO_CONVOLUTION_TYPE[bc]


def convolution_type_to_pad_mode(cv: ConvolutionType) -> str:
    if cv not in _CONVOLUTION_TYPE_TO_PAD_MODE:
        raise ValueError(f"unsupported convolution type: {cv}")

    return _CONVOLUTION_TYPE_TO_PAD_MODE[cv]


def convolve1d(
    input: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    mode: ConvolutionType = ConvolutionType.Reflect,
) -> jnp.ndarray:
    """Perform a convolution of one-dimensional arrays.

    Should perform identically to :func:`scipy.ndimage.convolve1d`. Performance
    is not guaranteed.
    """
    n = weights.size // 2 + 1

    u = jnp.pad(input, n, mode=convolution_type_to_pad_mode(mode))
    u = jnp.convolve(u, weights, mode="same")

    result = u[n:-n]
    assert result.shape == u.shape

    return result
