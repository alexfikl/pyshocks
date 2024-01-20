# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os


def get_logger(
    module: str,
    level: int | str | None = None,
) -> logging.Logger:
    if level is None:
        level = os.environ.get("PYSHOCKS_LOGGING_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    root = logging.getLogger("pyshocks")
    if not root.handlers:
        from rich.logging import RichHandler

        root.setLevel(level)
        root.addHandler(RichHandler())

    return logging.getLogger(module)
