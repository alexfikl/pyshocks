# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Convolutions
------------

.. autoclass:: ConvolutionType
   :undoc-members:
   :inherited-members:

.. autofunction:: boundary_type_to_mode
.. autofunction:: convolution_type_to_pad_mode

.. autofunction:: convolve1d
"""
import enum
from typing import Optional

import jax.numpy as jnp

from pyshocks.schemes import BoundaryType


@enum.unique
class ConvolutionType(enum.Enum):
    """Convolution types supported by :func:`scipy.ndimage.convolve1d`."""

    #: Matching :func:`numpy.convolve`.
    Same = enum.auto()

    #: Reflected around the edge of the last pixel, e.g.
    #: ``(d c b a | a b c d | d c b a)``.
    Reflect = enum.auto()
    #: Filled with a constant beyond the edge of the last pixel, e.g.
    #: ``(k k k k | a b c d | k k k k)``.
    Constant = enum.auto()
    #: Filled with the value of the last pixel beyond the boundary, e.g.
    #: ``(a a a a | a b c d | d d d d)``.
    Nearest = enum.auto()
    #: Reflected around the center of the last pixel, i.e.
    #: ``(d c b | a b c d | c b a)``.
    Mirror = enum.auto()
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
    ConvolutionType.Reflect: "symmetric",
    ConvolutionType.Constant: "constant",
    ConvolutionType.Nearest: "edge",
    ConvolutionType.Mirror: "reflect",
    ConvolutionType.Wrap: "wrap",
}


def boundary_type_to_mode(bc: BoundaryType) -> ConvolutionType:
    """Naive matching of boundary types to convolution types.

    :returns: a best-effort guess at the :class:`ConvolutionType` for *bc*, e.g.
        for periodic boundary conditions :attr:`~ConvolutionType.Wrap` is most
        appropriate.
    """
    if bc not in _BOUNDARY_TYPE_TO_CONVOLUTION_TYPE:
        raise ValueError(f"unknown boundary type: {bc}")

    return _BOUNDARY_TYPE_TO_CONVOLUTION_TYPE[bc]


def convolution_type_to_pad_mode(cv: ConvolutionType) -> str:
    """Convert from :class:`ConvolutionType` to the mode of :func:`jax.numpy.pad`.

    :returns: a string that corresponds to the padding mode.
    """
    if cv not in _CONVOLUTION_TYPE_TO_PAD_MODE:
        raise ValueError(f"unsupported convolution type: {cv}")

    return _CONVOLUTION_TYPE_TO_PAD_MODE[cv]


def convolve1d(
    ary: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    mode: Optional[ConvolutionType] = None,
) -> jnp.ndarray:
    """Perform a convolution of one-dimensional arrays.

    Should perform identically to :func:`scipy.ndimage.convolve1d`. Performance
    is not guaranteed.
    """
    if mode is None:
        mode = ConvolutionType.Same

    if mode == ConvolutionType.Same:
        result = jnp.convolve(ary, weights, mode="same")
    else:
        n = weights.size // 2
        if weights.size % 2 == 0:
            # FIXME: better way to fix this? needed to match scipy.ndimage...
            weights = jnp.pad(weights, (0, 1))

        # TODO: this seems to be about an order of magnitude slower than the scipy
        # equivalent when jitted, so not great. This is mostly true for small
        # weight sizes.

        # TODO: would be nice to have jax do this for us properly :(
        u = jnp.pad(ary, n, mode=convolution_type_to_pad_mode(mode))
        u = jnp.convolve(u, weights, mode="same")

        result = u[n:-n]

    assert result.shape == ary.shape
    return result
