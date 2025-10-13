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

.. class:: Array
    :canonical: pyshocks.tools.Array

    A :class:`jax.Array` like object.

.. class:: Scalar
    :canonical: pyshocks.tools.Scalar

    A :class:`jax.Array` object with an empty shape `()`.

.. autoclass:: ScalarFunction
.. autoclass:: VectorFunction
.. autoclass:: SpatialFunction
.. autoclass:: TemporalFunction

Convergence
-----------

.. autoclass:: EOCRecorder
    :no-show-inheritance:
    :members:

.. autofunction:: estimate_order_of_convergence

Timing and Profiling
--------------------

.. autoclass:: TimeResult
    :no-show-inheritance:
    :members:
.. autoclass:: BlockTimer
    :no-show-inheritance:
    :members:
.. autoclass:: IterationTimer
    :no-show-inheritance:
    :members:

.. autofunction:: timeit
.. autofunction:: repeatit
.. autofunction:: profileit
"""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from pyshocks.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from types import TracebackType

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

PathLike = pathlib.Path | str

Array: TypeAlias = jax.Array
Scalar: TypeAlias = jax.Array
ScalarLike = float | np.floating[Any] | Scalar
ArrayOrNumpy = Array | np.ndarray[Any, Any]

# {{{ callable protocols


class ScalarFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: ScalarLike, x: Array) -> Scalar: ...


class VectorFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`(t, \mathbf{x})`.

    .. automethod:: __call__
    """

    def __call__(self, t: ScalarLike, x: Array) -> Array: ...


class SpatialFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`\mathbf{x}`.

    .. automethod:: __call__
    """

    def __call__(self, x: Array) -> Array: ...


class TemporalFunction(Protocol):
    r"""A generic callable that can be evaluated at :math:`t`.

    .. automethod:: __call__
    """

    def __call__(self, t: ScalarLike) -> Scalar: ...


# }}}


# {{{ eoc


def estimate_order_of_convergence(x: Array, y: Array) -> tuple[Scalar, Scalar]:
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    eps = jnp.finfo(x.dtype).eps  # type: ignore[no-untyped-call]
    c = jnp.polyfit(jnp.log10(x + eps), jnp.log10(y + eps), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(
    x: Array, y: Array, *, gliding_mean: int | None = None
) -> Array:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    if gliding_mean is None:
        gliding_mean = x.size

    npoints = x.size - gliding_mean + 1
    return jnp.array(
        [
            estimate_order_of_convergence(
                x[i : i + gliding_mean], y[i : i + gliding_mean] + 1.0e-16
            )
            for i in range(npoints)
        ],
        dtype=x.dtype,
    )


class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs."""

    name: str
    """An identifier used for the data being estimated."""
    dtype: jnp.dtype[Any]
    """:class:`numpy.dtype` of the error values."""
    history: list[tuple[Scalar, Scalar]]
    """History of ``(h, error)``."""

    def __init__(self, *, name: str = "Error", dtype: Any = None) -> None:
        if dtype is None:
            dtype = jnp.float64
        dtype = jnp.dtype(dtype)

        self.name = name
        self.dtype = dtype
        self.history = []

    @property
    def _history(self) -> Array:
        return jnp.array(self.history, dtype=self.dtype).T

    def add_data_point(self, h: ScalarLike, error: ScalarLike) -> None:
        """
        :arg h: abscissa, a value representative of the "grid size".
        :arg error: error at given *h*.
        """
        self.history.append((
            jnp.array(h, dtype=self.dtype),
            jnp.array(error, dtype=self.dtype),
        ))

    @property
    def estimated_order(self) -> Scalar:
        """Estimated order of convergence for currently available data. The
        order is estimated by least squares through the given data
        (see :func:`estimate_order_of_convergence`).
        """
        import numpy as np

        if not self.history:
            return jnp.array(np.nan, dtype=self.dtype)

        h, error = self._history
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> Scalar:
        """Largest error (in absolute value) in current data."""
        return jnp.amax(
            jnp.array([error for _, error in self.history], dtype=self.dtype),
            initial=jnp.array(0.0, dtype=self.dtype),
        )

    def satisfied(
        self, order: float, atol: float | None = None, *, slack: float = 0
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
            atol = 1.0e2 * float(jnp.finfo(error.dtype).eps)  # type: ignore[no-untyped-call]

        return bool(self.estimated_order >= (order - slack) or jnp.max(error) < atol)

    def __str__(self) -> str:
        return stringify_eoc(self)


def flatten(iterable: Iterable[Iterable[T]]) -> tuple[T, ...]:
    from itertools import chain

    return tuple(chain.from_iterable(iterable))


def stringify_eoc(*eocs: EOCRecorder) -> str:
    r"""
    :arg eocs: an iterable of :class:`EOCRecorder`\ s that are assumed to have
        the same number of entries in their histories.
    :returns: a string representing the results in *eocs* in the
        GitHub Markdown format.
    """
    histories = [eoc._history for eoc in eocs]
    orders = [
        estimate_gliding_order_of_convergence(h, error, gliding_mean=2)
        for h, error in histories
    ]

    h = histories[0][0]
    ncolumns = 1 + 2 * len(eocs)
    nrows = h.size

    lines = []
    lines.append(("h", *flatten([(eoc.name, "EOC") for eoc in eocs])))

    lines.append((":-:",) * ncolumns)

    for i in range(nrows):
        values = flatten([
            (f"{error[i]:.6e}", "---" if i == 0 else f"{order[i - 1, i]:.3f}")
            for (_, error), order in zip(histories, orders, strict=True)
        ])
        lines.append((f"{h[i]:.3e}", *values))

    lines.append((
        "Overall",
        *flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs]),
    ))

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]  # noqa: UP031

    return "\n".join([
        " | ".join(fmt.format(value) for fmt, value in zip(formats, line, strict=True))
        for line in lines
    ])


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

    h, error = eoc._history
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
        raise FileExistsError(f"Output file already exists: {filename!r}.")

    fig.savefig(filename)
    mp.close(fig)


# }}}


# {{{ timer


@dataclass
class BlockTimer:
    """A context manager for timing blocks of code.

    .. code:: python

        with BlockTimer("my-code-block") as bt:
            # do some code

        print(bt)
    """

    name: str = "block"
    """An identifier used to differentiate the timer."""

    t_wall_start: ScalarLike = field(init=False)
    t_wall: ScalarLike = field(init=False)
    """Total wall time, obtained from :func:`time.perf_counter`."""

    t_proc_start: ScalarLike = field(init=False)
    t_proc: ScalarLike = field(init=False)
    """Total process time, obtained from :func:`time.process_time`."""

    @property
    def t_cpu(self) -> Scalar:
        """Total CPU time, obtained from ``t_proc / t_wall``."""
        return cast("Scalar", self.t_proc / self.t_wall)

    def __enter__(self) -> BlockTimer:
        import time

        self.t_wall_start = jnp.array(time.perf_counter(), dtype=jnp.float64)
        self.t_proc_start = jnp.array(time.process_time(), dtype=jnp.float64)
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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.finalize()

    def __str__(self) -> str:
        return (
            f"{self.name}: completed ({self.t_wall:.3e}s wall, {self.t_cpu:.3f}x cpu)"
        )


@dataclass(frozen=True)
class TimeResult:
    __slots__ = ("mean", "std", "walltime")

    walltime: Scalar
    """Smallest value of the walltime for all the runs."""
    mean: Scalar
    """Mean value for the walltimes."""
    std: Scalar
    """Standard deviations for the walltimes."""

    def __str__(self) -> str:
        return f"wall {self.walltime:.3e}s mean {self.mean:.3e}s ± {self.std:.3e}"

    @classmethod
    def from_measurements(cls, deltas: Array, *, skip: int = 5) -> TimeResult:
        """
        :arg deltas: an array of run timings.
        :arg skip: number of initial entries to skip. The first few runs are
            likely to contain the JIT warming up, so they can be skipped to
            obtain more realistic estimates.
        """
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
    """

    name: str = "iteration"
    """An identifier used to differentiate the timer."""
    t_deltas: list[Scalar] = field(default_factory=list, init=False, repr=False)
    """A list of timings."""

    def tick(self) -> BlockTimer:
        """
        :returns: a :class:`BlockTimer` that can be used to time a single
            iteration.
        """

        @dataclass
        class _BlockTimer(BlockTimer):
            t_deltas: list[Scalar] | None = None

            def finalize(self) -> None:
                super().finalize()

                assert self.t_deltas is not None
                self.t_deltas.append(jnp.array(self.t_wall))

        return _BlockTimer(name="inner", t_deltas=self.t_deltas)

    @property
    def total(self) -> Scalar:
        """Total time taken by all the iterations."""
        return jnp.sum(jnp.array(self.t_deltas, dtype=jnp.float64))

    def stats(self) -> TimeResult:
        """Compute statistics across all the iterations."""
        return TimeResult.from_measurements(
            jnp.array(self.t_deltas, dtype=jnp.float64), skip=5
        )


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that applies :class:`BlockTimer`."""

    from functools import wraps

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
    stmt: str | Callable[[], Any],
    *,
    setup: str | Callable[[], Any] = "pass",
    repeat: int = 16,
    number: int = 1,
) -> TimeResult:
    """Run *stmt* using :func:`timeit.repeat`.

    :returns: a :class:`TimeResult` with statistics about the runs.
    """

    import timeit as _timeit

    r = _timeit.repeat(stmt=stmt, setup=setup, repeat=repeat + 1, number=number)
    return TimeResult.from_measurements(jnp.array(r, dtype=jnp.float64), skip=3)


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


def join_or(strings: Sequence[str], *, preposition: str = "or") -> str:
    if len(strings) == 1:
        return strings[0]

    *rest, last = strings

    return "'{}' {} '{}'".format("', '".join(rest), preposition.strip(), last)


# }}}


# {{{ single-valued


def is_single_valued(
    iterable: Iterable[Any], predicate: Callable[[T, T], bool] | None = None
) -> bool:
    it = iter(iterable)
    try:
        first_item = next(it)
    except StopIteration:
        return True

    if predicate is None:
        import operator

        predicate = operator.eq

    return all(predicate(other_item, first_item) for other_item in it)


# }}}


# {{{ environment


# fmt: off
BOOLEAN_STATES = {
    1: True, "1": True, "yes": True, "true": True, "on": True, "y": True,
    0: False, "0": False, "no": False, "false": False, "off": False, "n": False,
}
# fmt: on


def get_environ_bool(name: str) -> bool:
    value = os.environ.get(name)
    return BOOLEAN_STATES.get(value.lower(), False) if value else False


# }}}


# {{{ matplotlib


def check_usetex(*, s: bool) -> bool:
    # NOTE: this function was deprecated from v3.6.0 and removed in v3.10.0

    try:
        import matplotlib

        return bool(matplotlib.checkdep_usetex(s))  # type: ignore[attr-defined]
    except ImportError:
        # NOTE: no matplotlib, just return false
        return False
    except AttributeError:
        # NOTE: simplified version from matplotlib
        # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

        import shutil

        if not shutil.which("tex"):
            return False

        if not shutil.which("dvipng"):
            return False

        if not shutil.which("gs"):  # noqa: SIM103
            return False

        return True


def set_recommended_matplotlib(*, use_tex: bool | None = None) -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    defaults: dict[str, dict[str, Any]] = {
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

    from contextlib import suppress

    # NOTE: since v1.1.0 an import is required to import the styles
    with suppress(ImportError):
        import scienceplots  # noqa: F401

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
        with suppress(KeyError):
            mp.rc(group, **params)


@contextmanager
def figure(filename: PathLike | None = None, **kwargs: Any) -> Iterator[Any]:
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
    filename: PathLike | None = None,
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


# {{{ save animation


def save_animation(
    filename: str | pathlib.Path | None,
    x: ArrayOrNumpy,
    ys: ArrayOrNumpy | tuple[ArrayOrNumpy, ...],
    *,
    fps: int | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymin: ScalarLike | None = None,
    ymax: ScalarLike | None = None,
    legends: Sequence[str] | None = None,
    fig_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    anim_kwargs: dict[str, Any] | None = None,
) -> None:
    # {{{ handle inputs

    if not isinstance(ys, tuple):
        ys = (ys,)

    if not is_single_valued(y.shape for y in ys):
        raise ValueError("All 'ys' should be the same shape.")

    if (ys[0].shape[0],) != x.shape:
        raise ValueError("'x' and 'ys' should have the same shape.")

    if xlabel is None:
        xlabel = "$x$"

    if ylabel is None:
        ylabel = "$u(x, t)$"

    if legends is not None and len(legends) != len(ys):
        raise ValueError("Must provide 'legends' for all inputs.")

    if fig_kwargs is None:
        fig_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    if anim_kwargs is None:
        anim_kwargs = {}

    if fps is None:
        nframes = ys[0].shape[-1]
        fps = max(int(nframes / 10), 10)

    # }}}

    # {{{ plot

    def _anim_init() -> tuple[Any, ...]:
        for line in lines:
            line.set_ydata(x)

        return lines

    def _anim_func(n: int) -> tuple[Any, ...]:
        for line, y in zip(lines, ys, strict=True):
            line.set_ydata(y[:, n])

        return lines

    import matplotlib.pyplot as mp

    fig = mp.figure(**fig_kwargs)
    ax = fig.gca()

    if ymin is None:
        ymin = jnp.amin(jnp.array([y.min() for y in ys], dtype=x.dtype))

    if ymax is None:
        ymax = jnp.amax(jnp.array([y.max() for y in ys], dtype=x.dtype))

    ax.set_xlabel(xlabel)
    if legends is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim((float(x[0]), float(x[-1])))
    ax.set_ylim((float(ymin - 0.1 * jnp.abs(ymin)), float(ymax + 0.1 * jnp.abs(ymax))))
    ax.margins(0.05)

    if legends:
        lines = tuple(
            ax.plot(x, x, "o-", **plot_kwargs, label=label)[0] for label in legends
        )
        ax.legend(loc=1)
    else:
        lines = tuple(ax.plot(x, x, "o-", **plot_kwargs)[0] for _ in ys)

    # }}}

    # {{{ animation

    from matplotlib import animation

    anim_kwargs = {"interval": 25, "blit": True, **anim_kwargs}
    anim = animation.FuncAnimation(
        fig,
        _anim_func,
        jnp.arange(1, ys[0].shape[1]),
        init_func=_anim_init,
        **anim_kwargs,
    )

    if filename is None:
        mp.show()
    else:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)

    # }}}


# }}}
