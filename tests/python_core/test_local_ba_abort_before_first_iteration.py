"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*
* Local BA whose abort flag is raised just before g2o's first iteration, so g2o returns
* without ever computing the edge errors (see tests/slam/local_ba_abort_support.py).
*
* Python core only: g2opy hands g2o a pointer to the flag, so the test can raise it exactly
* there. The C++ core checks the flag right before optimizing and receives it from Python
* through a thread that polls every 5 ms, so it cannot be reached deterministically; the
* timed aborts of tests/slam/test_local_ba_abort.py cover it on both cores.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slam"))

from local_ba_abort_support import LocalBaAbortCase  # noqa: E402


class TestLocalBaAbortBeforeFirstIteration(LocalBaAbortCase):
    def test_abort_before_first_iteration(self):
        err, interrupted = self._local_ba_aborted_after(0)
        self.assertTrue(interrupted)
        self._assert_error_of_accepted_state(err)


if __name__ == "__main__":
    unittest.main()
