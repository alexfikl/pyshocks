# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: pyshocks

.. autoclass:: EOCRecorder
    :no-show-inheritance:
.. autofunction:: estimate_order_of_convergence

.. autoclass:: BlockTimer
    :no-show-inheritance:
.. autoclass:: IterationTimer
    :no-show-inheritance:
.. autofunction:: timeme
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Union, TypeVar
import logging

import jax.numpy as jnp

F = TypeVar("F", bound=Callable[..., Any])


# {{{ memoize method


def memoize_on_first_arg(function, cache_dict_name=None):
    if cache_dict_name is None:
        cache_dict_name = f"_memoize_dict_{function.__module__}{function.__name__}"

    def wrapper(obj, *args, **kwargs):
        if kwargs:
            raise RuntimeError("keyword arguments are not supported")

        key = args
        try:
            return getattr(obj, cache_dict_name)[key]
        except AttributeError:
            attribute_error = True
        except KeyError:
            attribute_error = False

        result = function(obj, *args, **kwargs)
        if attribute_error:  # pylint: disable=no-else-return
            object.__setattr__(obj, cache_dict_name, {key: result})
            return result
        else:
            getattr(obj, cache_dict_name)[key] = result
            return result

    def clear_cache(obj):
        object.__delattr__(obj, cache_dict_name)

    from functools import update_wrapper

    new_wrapper = update_wrapper(wrapper, function)
    new_wrapper.clear_cache = clear_cache

    return new_wrapper


def memoize_method(method: F) -> F:
    return memoize_on_first_arg(method, f"_memoize_dict_{method.__name__}")


# }}}


# {{{ eoc


def estimate_order_of_convergence(x, y):
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("need at least two values to estimate order")

    import numpy as np

    c = np.polyfit(np.log10(x), np.log10(y), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(x, y, *, gliding_mean=None):
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("need at least two values to estimate order")

    if gliding_mean is None:
        gliding_mean = x.size

    import numpy as np

    npoints = x.size - gliding_mean + 1
    eocs = np.zeros((npoints, 2), dtype=np.float64)

    for i in range(npoints):
        eocs[i] = estimate_order_of_convergence(
            x[i : i + gliding_mean], y[i : i + gliding_mean]
        )

    return eocs


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
    .. automethod:: as_table
    """

    def __init__(self, *, name="Error"):
        self.name = name
        self.history = []

    def add_data_point(self, h: float, error: float):
        """
        :param h: abscissa, a value representative of the "grid size"
        :param error: error at given *h*.
        """
        self.history.append((h, error))

    @property
    def estimated_order(self):
        import numpy as np

        h, error = np.array(self.history).T
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self):
        return max(error for _, error in self.history)

    def as_table(self) -> str:
        """
        :return: a table representation of the errors and estimated order
            of convergence of the current data.
        """
        import numpy as np

        h, error = np.array(self.history).T
        orders = estimate_gliding_order_of_convergence(h, error, gliding_mean=2)

        # header
        lines = []
        lines.append(("h", self.name, "EOC"))
        # NOTE: these make it into a centered markdown table
        lines.append((":-:", ":-:", ":-:"))

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

    def __str__(self):
        return self.as_table()


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
    callback: object = None

    t_wall: float = field(default=-1, init=False)
    t_wall_start: float = field(default=-1, init=False)

    t_proc: float = field(default=-1, init=False)
    t_proc_start: float = field(default=-1, init=False)

    @property
    def t_cpu(self):
        return self.t_proc / self.t_wall

    def __enter__(self):
        import time

        self.t_wall_start = time.perf_counter()
        self.t_proc_start = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.t_wall = time.perf_counter() - self.t_wall_start
        self.t_proc = time.process_time() - self.t_proc_start

        if self.callback is not None:
            self.callback(self)

    def __str__(self):
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
    t_deltas: list = field(default_factory=list, init=False, repr=False)

    def tick(self):
        """
        :returns: a :class:`BlockTimer` that can be used to time a single
            iteration.
        """
        return BlockTimer(
            name="inner",
            callback=lambda subtimer: self.t_deltas.append(subtimer.t_wall),
        )

    @property
    def total(self):
        return jnp.sum(self.t_deltas)

    def stats(self):
        """Compute statistics across all the iterations.

        :returns: a :class:`tuple` of ``(total, mean, std)``.
        """

        t_deltas = jnp.array(self.t_deltas)

        # NOTE: skipping the first few iterations because they mostly measure
        # the jit warming up, so they'll skew the standard deviation
        return (jnp.sum(t_deltas), jnp.mean(t_deltas[5:-1]), jnp.std(t_deltas[5:-1]))


def timeme(fun):
    """Decorator that applies :class:`BlockTimer`."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        with BlockTimer(fun.__qualname__) as t:
            ret = fun(*args, **kwargs)

        print(t)
        return ret

    return wrapper


# }}}


# {{{ logging

# https://misc.flogisoft.com/bash/tip_colors_and_formatting
LOG_BOLD = "\033[1m"
LOG_RESET = "\033[0m"
LOGLEVEL_TO_COLOR = {
    "WARNING": "\033[1;33m",  # yellow
    "INFO": "\033[1;36m",  # cyan
    "DEBUG": "\033[1;37m",  # light gray
    "CRITICAL": "\033[1;33m",  # yellow
    "ERROR": "\033[1;31m",  # red
}

PYSHOCKS_LOG_FORMAT = (
    f"[{LOG_BOLD}%(name)s{LOG_RESET}][%(levelname)s] "
    f"{LOG_BOLD}%(filename)s:%(lineno)-4d{LOG_RESET} :: %(message)s "
)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname

        if levelname in LOGLEVEL_TO_COLOR:
            record.levelname = f"{LOGLEVEL_TO_COLOR[levelname]}{levelname}{LOG_RESET}"

        return super().format(record)


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

    if not logger.hasHandlers():
        formatter = ColoredFormatter(PYSHOCKS_LOG_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

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
