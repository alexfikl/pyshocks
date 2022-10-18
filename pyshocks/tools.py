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

.. autoclass:: BlockTimer
    :no-show-inheritance:
.. autoclass:: IterationTimer
    :no-show-inheritance:

.. autofunction:: timeit
.. autofunction:: profileit
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Union,
    Tuple,
    Type,
    TypeVar,
)

try:
    from typing import ParamSpec  # noqa: F811
except ImportError:
    from typing_extensions import ParamSpec

from types import TracebackType
import pathlib
import logging

import jax.numpy as jnp

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


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

    c = jnp.polyfit(jnp.log10(x), jnp.log10(y), 1)
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
    return jnp.array(  # type: ignore[no-untyped-call]
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
    .. automethod:: as_table
    """

    def __init__(self, *, name: str = "Error") -> None:
        self.name = name
        self.history: List[Tuple[jnp.ndarray, jnp.ndarray]] = []

    def add_data_point(self, h: jnp.ndarray, error: jnp.ndarray) -> None:
        """
        :param h: abscissa, a value representative of the "grid size"
        :param error: error at given *h*.
        """
        self.history.append((h, error))

    @property
    def estimated_order(self) -> jnp.ndarray:
        import numpy as np

        if not self.history:
            return np.nan

        h, error = np.array(self.history).T
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> jnp.ndarray:
        return max(error for _, error in self.history)

    def satisfied(self, order: float, atol: float = 1.0e-12) -> bool:
        if not self.history:
            return True

        _, error = jnp.array(self.history).T  # type: ignore[no-untyped-call]
        return bool(self.estimated_order >= order or jnp.max(error) < atol)

    def as_table(self) -> str:
        """
        :return: a table representation of the errors and estimated order
            of convergence of the current data.
        """
        import numpy as np

        # header
        lines = []
        lines.append(("h", self.name, "EOC"))
        # NOTE: these make it into a centered markdown table
        lines.append((":-:", ":-:", ":-:"))

        if self.history:
            h, error = np.array(self.history).T
            orders = estimate_gliding_order_of_convergence(h, error, gliding_mean=2)

            # rows
            for i in range(h.size):
                lines.append(
                    (
                        f"{h[i]:.3e}",
                        f"{error[i]:.6e}",
                        "---" if i == 0 else f"{orders[i - 1, 1]:.3f}",
                    )
                )

        # footer
        lines.append(("Overall", "", f"{self.estimated_order:.3f}"))

        # figure out column width
        widths = [max(len(line[i]) for line in lines) for i in range(3)]
        formats = ["{:%s}" % w for w in widths]  # pylint: disable=C0209

        return "\n".join(
            [
                " | ".join(fmt.format(value) for fmt, value in zip(formats, line))
                for line in lines
            ]
        )

    def __str__(self) -> str:
        return self.as_table()


def visualize_eoc(
    filename: Union[str, pathlib.Path],
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

    h, error = jnp.array(eoc.history).T  # type: ignore[no-untyped-call]
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

    ax.grid(True, which="major", linestyle="-", alpha=0.75)
    ax.grid(True, which="minor", linestyle="--", alpha=0.5)

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
    .. attribute:: callback

        A callback taking the :class:`BlockTimer` itself that is called when
        the context manager is exited.

    .. attribute:: t_wall
    .. attribute:: t_proc
    """

    name: str = "block"
    callback: Optional[Callable[["BlockTimer"], None]] = None

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

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        import time

        self.t_wall = time.perf_counter() - self.t_wall_start
        self.t_proc = time.process_time() - self.t_proc_start

        if self.callback is not None:
            self.callback(self)

    def __str__(self) -> str:
        return (
            f"{self.name}: completed "
            f"({self.t_wall:.3}s wall, {self.t_cpu:.3f}x cpu)"
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
        return BlockTimer(
            name="inner",
            callback=lambda subtimer: self.t_deltas.append(subtimer.t_wall),
        )

    @property
    def total(self) -> jnp.ndarray:
        return jnp.sum(self.t_deltas)

    def stats(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute statistics across all the iterations.

        :returns: a :class:`tuple` of ``(total, mean, std)``.
        """

        t_deltas = jnp.array(self.t_deltas)  # type: ignore[no-untyped-call]

        # NOTE: skipping the first few iterations because they mostly measure
        # the jit warming up, so they'll skew the standard deviation
        return (jnp.sum(t_deltas), jnp.mean(t_deltas[5:-1]), jnp.std(t_deltas[5:-1]))


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
    def message(s: Any, success: bool = True) -> str:
        return Color.info(s) if success else Color.warn(s)

    @staticmethod
    def wrap(s: Any, color: str) -> str:
        return f"{color}{s}{Color.Normal}"


# }}}


# {{{ logging


def get_logger(
    module: str,
    level: Optional[Union[int, str]] = None,
) -> logging.Logger:
    if level is None:
        level = logging.INFO

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    logger = logging.getLogger(module)
    logger.propagate = False
    logger.setLevel(level)

    try:
        from rich.logging import RichHandler

        logger.addHandler(RichHandler())
    except ImportError:
        # NOTE: rich is vendored by pip since November 2021
        try:
            from pip._vendor.rich.logging import RichHandler  # type: ignore

            logger.addHandler(RichHandler())
        except ImportError:
            logger.addHandler(logging.StreamHandler())

    return logger


# }}}


# {{{ matplotlib


def set_recommended_matplotlib(use_tex: Optional[bool] = None) -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    if use_tex is None:
        import os
        import matplotlib

        use_tex = not os.environ.get(
            "GITHUB_REPOSITORY"
        ) and matplotlib.checkdep_usetex(True)

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

    if "science" in mp.style.available:
        mp.style.use(["science", "ieee"])
    else:
        mp.style.use("seaborn-white")

    for group, params in defaults.items():
        try:
            mp.rc(group, **params)
        except KeyError:
            pass


# }}}
