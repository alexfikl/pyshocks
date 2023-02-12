# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
Checkpointing
-------------

.. autoclass:: Checkpoint
    :no-show-inheritance:
.. autofunction:: save
.. autofunction:: load

.. autoclass:: InMemoryCheckpoint
.. autoclass:: PickleCheckpoint
.. autoclass:: NumpyCheckpoint
"""

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Dict, Hashable, Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class Checkpoint(ABC):
    """A generic checkpointing mechanism.

    The basic idea is that a user can call :func:`save` with a given index and
    a dictionary of values. This function uses :meth:`index_to_key` and
    :attr:`basename` to construct a key for the respective checkpoint and
    save it to a storage backend. For example, a roundtrip can read like

    .. code:: python

        chk = InMemoryCheckpoint(basename="Iteration")
        for i in range(42):
            save(chk, i, {"i": i})

        for i in range(42):
            data = load(chk, i)
            assert data["i"] == i

    .. attribute:: basename

        A basename for the checkpointing keys. A specific checkpoint implementation
        can use this as a key in a dictionary, as a filename or something else
        entirely.

    .. automethod:: index_to_key

        Transforms an integer index into a key used to denote the checkpoint.

    .. automethod:: __contains__
    """

    basename: str

    @abstractmethod
    def index_to_key(self, i: int) -> Hashable:
        """
        :arg i: an index used to denote the checkpoint.
        """

    @abstractmethod
    def __contains__(self, i: int) -> bool:
        pass


@singledispatch
def save(chk: Checkpoint, idx: int, values: Dict[str, Any]) -> None:
    """
    :arg idx: an integer index denoting the checkpoint.
    :arg values: a :class:`dict` of values to be checkpointed.
    """
    raise NotImplementedError(type(chk).__name__)


@singledispatch
def load(
    chk: Checkpoint, idx: int, *, include: Optional[Tuple[str, ...]] = None
) -> Dict[str, Any]:
    """
    :arg idx: an integer index denoting the checkpot.
    :arg include: if provided, only the keys in this
    """
    raise NotImplementedError(type(chk).__name__)


# {{{ in memory checkpointing


@dataclass(frozen=True)
class InMemoryCheckpoint(Checkpoint):
    """A class that stores all the checkpoints in a dictionary in memory.

    As expected, this is not suitable for larger simulations.
    """

    storage: Dict[Hashable, Dict[str, Any]] = field(default_factory=dict)

    def index_to_key(self, i: int) -> Hashable:
        return (self.basename, i)

    def __contains__(self, i: int) -> bool:
        return self.index_to_key(i) in self.storage


@save.register(InMemoryCheckpoint)
def _save_in_memory(chk: InMemoryCheckpoint, idx: int, values: Dict[str, Any]) -> None:
    key = chk.index_to_key(idx)
    if key in chk.storage:
        raise KeyError(f"Cannot set existing checkpoint at {idx!r}.")

    chk.storage[key] = values


@load.register(InMemoryCheckpoint)
def _load_in_memory(
    chk: InMemoryCheckpoint, idx: int, *, include: Optional[Tuple[str, ...]] = None
) -> Dict[str, Any]:
    key = chk.index_to_key(idx)
    if key not in chk.storage:
        raise KeyError(f"Cannot find checkpoint at index {idx!r}.")

    if include is None:
        return chk.storage[key]

    return {k: v for k, v in chk.storage[key].items() if k in include}


# }}}


# {{{ pickle checkpoint


@dataclass(frozen=True)
class PickleCheckpoint(Checkpoint):
    """A class that stores checkpoints in a compressed format with the
    :mod:`pickle` module. The data is compressed using :mod:`lzma`.

    .. attribute:: dirname

        The directory in which to store the checkpoints.
    """

    dirname: pathlib.Path

    def index_to_key(self, i: int) -> pathlib.Path:
        return self.dirname / f"{self.basename}_{i:09d}.npz"

    def __contains__(self, i: int) -> bool:
        return self.index_to_key(i).exists()


@save.register(PickleCheckpoint)
def _save_pickle(chk: PickleCheckpoint, idx: int, values: Dict[str, Any]) -> None:
    filename = chk.index_to_key(idx)
    if not filename.exists():
        raise KeyError(f"Cannot set existing checkpoint at {idx!r}.")

    try:
        import cloudpickle as pickle
    except ImportError:
        import pickle

    import lzma

    with lzma.open(filename, "wb") as outfile:
        pickle.dump(values, outfile)


@load.register(PickleCheckpoint)
def _load_pickle(
    chk: PickleCheckpoint,
    idx: int,
    *,
    include: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    filename = chk.index_to_key(idx)
    if not filename.exists():
        raise KeyError(f"Cannot find checkpoint at index {idx!r}.")

    try:
        import cloudpickle as pickle
    except ImportError:
        import pickle

    import lzma

    with open(filename, "rb") as infile:
        values = pickle.load(lzma.decompress(infile.read()))

    if include is None:
        return dict(values)

    return {k: v for k, v in values.items() if k in include}


# }}}


# {{{ npz checkpoint


@dataclass(frozen=True)
class NumpyCheckpoint(Checkpoint):
    """A class that stores all the checkpoints in an uncompressed ``npz`` file.

    See :func:`numpy.savez` for details on the mechanism. Note that, at this
    time, :mod:`jax.numpy` does not support :func:`numpy.savez_compressed`,
    so compression is not possible.

    .. attribute:: dirname

        The directory in which to store the checkpoints.
    """

    dirname: pathlib.Path

    def index_to_key(self, i: int) -> pathlib.Path:
        return self.dirname / f"{self.basename}_{i:09d}.npz"

    def __contains__(self, i: int) -> bool:
        return self.index_to_key(i).exists()


@save.register(NumpyCheckpoint)
def _save_npz(chk: NumpyCheckpoint, idx: int, values: Dict[str, Any]) -> None:
    filename = chk.index_to_key(idx)
    if not filename.exists():
        raise KeyError(f"Cannot set existing checkpoint at {idx!r}.")

    jnp.savez(filename, **values)


@load.register(NumpyCheckpoint)
def _load_npz(
    chk: NumpyCheckpoint,
    idx: int,
    *,
    include: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    filename = chk.index_to_key(idx)
    if not filename.exists():
        raise KeyError(f"Cannot find checkpoint at index {idx!r}.")

    values = dict(jnp.load(filename, allow_pickle=False))
    if include is None:
        return values

    return {k: v for k, v in values.items() if k in include}


# }}}
