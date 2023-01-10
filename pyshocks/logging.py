# SPDX-FileCopyrightText: 2020 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

import logging
from typing import Optional, Union


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
