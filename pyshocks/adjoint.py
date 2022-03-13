# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Dict, List, Optional

import jax.numpy as jnp

from pyshocks import Grid, SchemeBase
from pyshocks import numerical_flux, predict_timestep


# {{{ checkpointing


@dataclass
class Checkpoint:
    basename: str = "Iteration"
    count: int = field(default=0, init=False, repr=False)

    def index_to_key(self, i):
        return f"{self.basename}_{i:09d}"

    def __len__(self):
        return self.count


@singledispatch
def save(chk: Checkpoint, idx: int, values: Dict[str, Any]):
    raise NotImplementedError(type(chk).__name__)


@singledispatch
def load(
    chk: Checkpoint, idx: int, *, include: Optional[List[str]] = None
) -> Dict[str, Any]:
    raise NotImplementedError(type(chk).__name__)


# {{{ in memory checkpointing


@dataclass
class InMemoryCheckpoint(Checkpoint):
    storage: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@save.register(InMemoryCheckpoint)
def _save_in_memory(chk: InMemoryCheckpoint, idx: int, values: Dict[str, Any]):
    key = chk.index_to_key(idx)
    if key in chk.storage:
        raise KeyError(f"cannot set existing checkpoint at '{idx}'")

    chk.storage[key] = values
    chk.count += 1


@load.register(InMemoryCheckpoint)
def _load_in_memory(
    chk: InMemoryCheckpoint, idx: int, *, include: Optional[List[str]] = None
) -> Dict[str, Any]:
    key = chk.index_to_key(idx)
    if key not in chk.storage:
        raise KeyError(f"cannot find checkpoint at index '{idx}'")

    if include is None:
        return chk.storage[key]

    return {k: v for k, v in chk.storage[key].items() if k in include}


# }}}

# }}}


# {{{ adjoint scheme


@dataclass(frozen=True)
class AutoAdjoint(SchemeBase):
    flux: object
    cost: object
    checkpoints: Checkpoint

    index: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "index", self.checkpoints.count - 1)


@numerical_flux.register(AutoAdjoint)
def _numerical_flux_adjoint(
    scheme: AutoAdjoint, grid: Grid, t: float, u: jnp.ndarray
) -> jnp.ndarray:
    pass


@predict_timestep.register(AutoAdjoint)
def _predict_time_step_adjoint(
    scheme: AutoAdjoint, grid: Grid, t: float, u: jnp.ndarray
) -> float:
    dt = load(scheme.checkpoints, scheme.index, include=["dt"])["dt"]
    object.__setattr__(scheme, "index", scheme.index - 1)

    return dt


# }}}
