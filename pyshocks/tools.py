# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

Typing
------

.. class:: T
    :canonical: pyshocks.tools.T

    Generic unbound invariant :class:`typing.TypeVar`.

.. class:: P
    :canonical: pyshocks.tools.P

    Generic unbound invariant :class:`typing.ParamSpec`.

.. autoclass:: ScalarFunction
.. autoclass:: VectorFunction
.. autoclass:: SpatialFunction
.. autoclass:: TemporalFunction

Convergence
-----------

.. autoclass:: EOCRecorder
    :no-show-inheritance:
.. autofunction:: estimate_order_of_convergence

Timing and Profiling
--------------------

.. autoclass:: TimeResult
    :no-show-inheritance:
.. autoclass:: BlockTimer
    :no-show-inheritance:
.. autoclass:: IterationTimer
    :no-show-inheritance:

.. autofunction:: timeit
.. autofunction:: repeatit
.. autofunction:: profileit
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterator,
    Iterable,
    List,
    Optional,
    Protocol,
    Union,
    Tuple,
    Type,
    TypeVar,
)

try:
    # NOTE: needs python 3.10
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec  # type: ignore[assignment]

from types import TracebackType
import pathlib

import jax.numpy as jnp

from pyshocks.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

PathLike = Union[pathlib.Path, str]


# {{{ callable protocols


class ScalarFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, x: jnp.ndarray) -> float:
        ...


class VectorFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        ...


class SpatialFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`\mathbf{x}`.

    .. automethod:: __call__
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ...


class TemporalFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`t`.

    .. automethod:: __call__
    """

    def __call__(self, t: float) -> float:
        ...


# }}}


# {{{ eoc


def estimate_order_of_convergence(
    x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("need at least two values to estimate order")

    eps = jnp.finfo(x.dtype).eps  # type: ignore[no-untyped-call]
    c = jnp.polyfit(jnp.log10(x + eps), jnp.log10(y + eps), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(
    x: jnp.ndarray, y: jnp.ndarray, *, gliding_mean: Optional[int] = None
) -> jnp.ndarray:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("need at least two values to estimate order")

    if gliding_mean is None:
        gliding_mean = x.size

    npoints = x.size - gliding_mean + 1
    return jnp.array(
        [
            estimate_order_of_convergence(
                x[i : i + gliding_mean], y[i : i + gliding_mean] + 1.0e-16
            )
            for i in range(npoints)
        ]
    )


class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs.

    .. attribute:: estimated_order

        Estimated order of convergence for currently available data. The
        order is estimated by least squares through the given data
        (see :func:`estimate_order_of_convergence`).

    .. attribute:: max_error

        Largest error (in absolute value) in current data.

    .. automethod:: __init__
    .. automethod:: add_data_point
    .. automethod:: satisfied
    """

    def __init__(self, *, name: str = "Error", dtype: Any = None) -> None:
        if dtype is None:
            dtype = jnp.float64
        dtype = jnp.dtype(dtype)

        self.name = name
        self.dtype = dtype
        self.history: List[Tuple[jnp.ndarray, jnp.ndarray]] = []

    @property
    def _history(self) -> jnp.ndarray:
        return jnp.array(self.history, dtype=self.dtype).T

    def add_data_point(self, h: jnp.ndarray, error: jnp.ndarray) -> None:
        """
        :arg h: abscissa, a value representative of the "grid size".
        :arg error: error at given *h*.
        """
        self.history.append((h, error))

    @property
    def estimated_order(self) -> jnp.ndarray:
        import numpy as np

        if not self.history:
            return np.nan

        h, error = self._history
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> jnp.ndarray:
        return max(error for _, error in self.history)

    def satisfied(
        self, order: float, atol: Optional[float] = None, *, slack: float = 0
    ) -> bool:
        """
        :arg order: expected order of convergence of the data.
        :arg atol: expected maximum error.
        :arg slack: additional allowable slack in the order of convergence.

        :returns: *True* if the expected order or the maximum order are hit
            and *False* otherwise.
        """

        if not self.history:
            return True

        _, error = self._history
        if atol is None:
            atol = 1.0e2 * jnp.finfo(error.dtype).eps  # type: ignore[no-untyped-call]

        return bool(self.estimated_order >= (order - slack) or jnp.max(error) < atol)

    def __str__(self) -> str:
        return stringify_eoc(self)


def flatten(iterable: Iterable[Iterable[T]]) -> Tuple[T, ...]:
    from itertools import chain

    return tuple(chain.from_iterable(iterable))


def stringify_eoc(*eocs: EOCRecorder) -> str:
    r"""
    :arg eocs: an iterable of :class:`EOCRecorder`\ s that are assumed to have
        the same number of entries in their histories.
    :returns: a string representing the results in *eocs* in the
        GitHub Markdown format.
    """
    histories = [eoc._history for eoc in eocs]  # pylint: disable=protected-access
    orders = [
        estimate_gliding_order_of_convergence(h, error, gliding_mean=2)
        for h, error in histories
    ]

    h = histories[0][0]
    ncolumns = 1 + 2 * len(eocs)
    nrows = h.size

    lines = []
    lines.append(("h",) + flatten([(eoc.name, "EOC") for eoc in eocs]))

    lines.append((":-:",) * ncolumns)

    for i in range(nrows):
        lines.append(
            (f"{h[i]:.3e}",)
            + flatten(
                [
                    (f"{error[i]:.6e}", "---" if i == 0 else f"{order[i - 1, i]:.3f}")
                    for (_, error), order in zip(histories, orders)
                ]
            )
        )

    lines.append(
        ("Overall",) + flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs])
    )

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]  # pylint: disable=C0209

    return "\n".join(
        [
            " | ".join(fmt.format(value) for fmt, value in zip(formats, line))
            for line in lines
        ]
    )


def visualize_eoc(
    filename: PathLike,
    eoc: EOCRecorder,
    order: float,
    *,
    abscissa: str = "h",
    ylabel: str = "Error",
    enable_legend: bool = True,
    overwrite: bool = False,
) -> None:
    """
    :arg fig_or_filename: output file name or an existing figure.
    :arg order: expected order for all the errors recorded in *eocs*.
    :arg abscissa: name for the abscissa.
    """
    import matplotlib.pyplot as mp

    fig = mp.figure()
    ax = fig.gca()

    # {{{ plot eoc

    h, error = eoc._history  # pylint: disable=protected-access
    ax.loglog(h, error, "o-", label=ylabel)

    # }}}

    # {{{ plot order

    max_h = jnp.max(h)
    min_e = jnp.min(error)
    max_e = jnp.max(error)
    min_h = jnp.exp(jnp.log(max_h) + jnp.log(min_e / max_e) / order)

    ax.loglog(
        [max_h, min_h], [max_e, min_e], "k-", label=rf"$\mathcal{{O}}(h^{order})$"
    )

    # }}}

    ax.grid(visible=True, which="major", linestyle="-", alpha=0.75)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.5)

    ax.set_xlabel(abscissa)

    if enable_legend:
        ax.legend()

    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    fig.savefig(filename)
    mp.close(fig)


# }}}


# {{{ timer


@dataclass
class BlockTimer:
    """A context manager for timing blocks of code.

    .. code::

        with BlockTimer("my-code-block") as bt:
            # do some code

        print(bt)

    .. attribute:: name
    .. attribute:: t_wall
    .. attribute:: t_proc

    .. automethod:: finalize
    """

    name: str = "block"

    t_wall: float = field(default=-1, init=False)
    t_wall_start: float = field(default=-1, init=False)

    t_proc: float = field(default=-1, init=False)
    t_proc_start: float = field(default=-1, init=False)

    @property
    def t_cpu(self) -> float:
        return self.t_proc / self.t_wall

    def __enter__(self) -> "BlockTimer":
        import time

        self.t_wall_start = time.perf_counter()
        self.t_proc_start = time.process_time()
        return self

    def finalize(self) -> None:
        """Perform additional processing on ``__exit__``.

        This functions is meant to be modified by subclasses to add behavior.
        """

        import time

        self.t_wall = time.perf_counter() - self.t_wall_start
        self.t_proc = time.process_time() - self.t_proc_start

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.finalize()

    def __str__(self) -> str:
        return (
            f"{self.name}: completed "
            f"({self.t_wall:.3e}s wall, {self.t_cpu:.3f}x cpu)"
        )


@dataclass(frozen=True)
class TimeResult:
    """
    .. attribute:: walltime

        Smallest value of the walltime for all the runs.

    .. attribute:: mean

       Mean value for the walltime.

    .. attribute:: std

        Standard deviation for the walltime.
    """

    __slots__ = {"walltime", "mean", "std"}

    walltime: float
    mean: float
    std: float

    def __str__(self) -> str:
        return f"wall {self.walltime:.3e}s mean {self.mean:.3e}s ± {self.std:.3e}"

    @classmethod
    def from_measurements(cls, deltas: jnp.ndarray, *, skip: int = 5) -> "TimeResult":
        # NOTE: skipping the first few iterations because they mostly measure
        # the jit warming up, so they'll skew the standard deviation
        return TimeResult(
            walltime=jnp.sum(deltas),
            mean=jnp.mean(deltas[skip:]),
            std=jnp.std(deltas[skip:], ddof=1),
        )


@dataclass
class IterationTimer:
    """A manager for timing blocks of code in an iterative algorithm.

    .. code::

        timer = IterationTimer("my-algorithm-timer")
        for iteration in range(maxit):
            with timer.tick() as bt:
                # do something for this iteration

            print(bt)

        print(timer.stats())

    .. attribute:: name
    .. attribute:: total

        Total time elapsed in the :meth:`tick` calls.

    .. automethod:: tick
    .. automethod:: stats
    """

    name: str = "iteration"
    t_deltas: List[jnp.ndarray] = field(default_factory=list, init=False, repr=False)

    def tick(self) -> BlockTimer:
        """
        :returns: a :class:`BlockTimer` that can be used to time a single
            iteration.
        """

        @dataclass
        class _BlockTimer(BlockTimer):
            t_deltas: Optional[jnp.ndarray] = None

            def finalize(self) -> None:
                super().finalize()

                assert self.t_deltas is not None
                self.t_deltas.append(self.t_wall)

        return _BlockTimer(name="inner", t_deltas=self.t_deltas)

    @property
    def total(self) -> jnp.ndarray:
        return jnp.sum(jnp.array(self.t_deltas))

    def stats(self) -> TimeResult:
        """Compute statistics across all the iterations."""
        return TimeResult.from_measurements(jnp.array(self.t_deltas), skip=5)


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that applies :class:`BlockTimer`."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with BlockTimer(func.__qualname__) as t:
            ret = func(*args, **kwargs)

        print(t)
        return ret

    return wrapper


def profileit(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that runs :mod:`cProfile`."""
    import cProfile
    import datetime

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        today = datetime.datetime.utcnow()
        filename = f"{func.__name__}-{today}.cProfile".replace(" ", "-")

        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)

        prof.dump_stats(filename)
        return retval

    return wrapper


def repeatit(
    stmt: Union[str, Callable[[], Any]],
    *,
    setup: Union[str, Callable[[], Any]] = "pass",
    repeat: int = 16,
    number: int = 1,
) -> TimeResult:
    """Run *stmt* using :func:`timeit.repeat`.

    :returns: a :class:`TimeResult` with statistics about the runs.
    """

    import timeit as _timeit

    r = _timeit.repeat(stmt=stmt, setup=setup, repeat=repeat + 1, number=number)
    return TimeResult.from_measurements(jnp.array(r), skip=3)


# }}}


# {{{ colors


@dataclass(frozen=True)
class Color:
    Black: str = "\033[0;30m"
    Red: str = "\033[0;31m"
    Green: str = "\033[0;32m"
    Brown: str = "\033[0;33m"
    Blue: str = "\033[0;34m"
    Purple: str = "\033[0;35m"
    Cyan: str = "\033[0;36m"
    LightGray: str = "\033[0;37m"
    DarkGray: str = "\033[1;30m"
    LightRed: str = "\033[1;31m"
    LightGreen: str = "\033[1;32m"
    Yellow: str = "\033[1;33m"
    LightBlue: str = "\033[1;34m"
    LightPurple: str = "\033[1;35m"
    LightCyan: str = "\033[1;36m"
    White: str = "\033[1;37m"
    Normal: str = "\033[0m"

    @staticmethod
    def warn(s: Any) -> str:
        return f"{Color.Yellow}{s}{Color.Normal}"

    @staticmethod
    def info(s: Any) -> str:
        return f"{Color.DarkGray}{s}{Color.Normal}"

    @staticmethod
    def message(s: Any, *, success: bool = True) -> str:
        return Color.info(s) if success else Color.warn(s)

    @staticmethod
    def wrap(s: Any, color: str) -> str:
        return f"{color}{s}{Color.Normal}"


# }}}


# {{{ matplotlib


def check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    if matplotlib.__version__ < "3.6.0":
        return bool(matplotlib.checkdep_usetex(s))

    # NOTE: simplified version from matplotlib
    # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

    import shutil

    if not shutil.which("tex"):
        return False

    if not shutil.which("dvipng"):
        return False

    if not shutil.which("gs"):
        return False

    return True


def set_recommended_matplotlib(use_tex: Optional[bool] = None) -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    defaults = {
        "figure": {"figsize": (8, 8), "dpi": 300, "constrained_layout": {"use": True}},
        "text": {"usetex": use_tex},
        "legend": {"fontsize": 32},
        "lines": {"linewidth": 2, "markersize": 10},
        "axes": {
            "labelsize": 32,
            "titlesize": 32,
            "grid": True,
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 24, "direction": "inout"},
        "ytick": {"labelsize": 24, "direction": "inout"},
        "axes.grid": {"axis": "both", "which": "both"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    try:
        # NOTE: since v1.1.0 an import is required to import the styles
        import SciencePlots  # noqa: F401
    except ImportError:
        try:
            import scienceplots  # noqa: F401
        except ImportError:
            pass

    if "science" in mp.style.library:
        mp.style.use(["science", "ieee"])
    else:
        # NOTE: try to use the upstream seaborn styles and fallback to matplotlib

        try:
            import seaborn

            mp.style.use(seaborn.axes_style("whitegrid"))
        except ImportError:
            if "seaborn-v0_8" in mp.style.library:
                # NOTE: matplotlib v3.6 deprecated all the seaborn styles
                mp.style.use("seaborn-v0_8-white")
            elif "seaborn" in mp.style.library:
                # NOTE: for older versions of matplotlib
                mp.style.use("seaborn-white")

    for group, params in defaults.items():
        try:
            mp.rc(group, **params)
        except KeyError:
            pass


@contextmanager
def figure(filename: Optional[PathLike] = None, **kwargs: Any) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    fig = mp.figure()
    try:
        yield fig
    finally:
        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show()

        mp.close(fig)


@contextmanager
def gca(
    fig: Any,
    filename: Optional[PathLike] = None,
    *,
    clear: bool = True,
    **kwargs: Any,
) -> Iterator[Any]:
    try:
        yield fig.gca()
    finally:
        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            import matplotlib.pyplot as mp

            mp.show()

        if clear:
            fig.clf()


def savefig(fig: Any, filename: PathLike, **kwargs: Any) -> None:
    logger.info("Saving '%s'", filename)
    fig.savefig(filename, **kwargs)


# }}}
