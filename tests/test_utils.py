# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from pyshocks import get_logger, set_recommended_matplotlib

import pytest

logger = get_logger("test_utils")
set_recommended_matplotlib()


# {{{ test_convolve_vs_scipy


@pytest.mark.parametrize(("n", "m"))
def test_convolve_vs_scipy(n: int, m: int) -> None:
    pass


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
