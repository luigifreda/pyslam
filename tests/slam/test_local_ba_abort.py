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
* Local bundle adjustment aborted at timed delays, on the core PYSLAM_USE_CPP selects
* (see local_ba_abort_support.py). The Python core also has a deterministic check, in
* tests/python_core/test_local_ba_abort_before_first_iteration.py.
"""

import unittest

import numpy as np
from local_ba_abort_support import ABORTED_BEFORE_OPTIMIZING, LocalBaAbortCase
from pyslam.config_parameters import Parameters

# Abort delays, counted from the start of the call (C++ core) or of the optimization (Python
# core). Most are short, to stop g2o before its first iteration. On the C++ core the flag
# reaches g2o through a monitor thread that polls every 5 ms.
ABORT_DELAYS_S = [0.00025 * i for i in range(1, 9)] * 3 + [0.0025 * i for i in range(1, 9)]


class TestAbortedLocalBundleAdjustment(LocalBaAbortCase):
    def test_converged_map_has_small_error(self):
        self.assertGreater(len(self.keyframes), 5)
        self.assertGreater(len(self.points), 1000)
        self.assertTrue(np.isfinite(self.converged_error))
        self.assertLess(self.converged_error, 1.0)

    def test_aborted_run_reports_error_of_accepted_state(self):
        num_interrupted = 0
        for delay in ABORT_DELAYS_S:
            self._restore_map()
            err, interrupted = self._local_ba_aborted_after(delay)
            if err == ABORTED_BEFORE_OPTIMIZING:
                continue
            num_interrupted += interrupted
            with self.subTest(delay_ms=delay * 1e3):
                self._assert_error_of_accepted_state(err)
        self.assertGreater(num_interrupted, 0, "no abort landed inside the optimization")

    def test_aborted_mean_excludes_deleted_outlier(self):
        # A point just in front of a camera has an enormous reprojection chi2. Local BA
        # deletes that observation. An abort during the robust phase used to leave the
        # edge active, so the reported mean included it.
        candidates = [
            p for p in self.points if not p.is_bad() and p.num_observations() > 2
        ]
        candidates.sort(key=lambda p: p.num_observations(), reverse=True)
        delays = (0.0, 0.001, 0.002, 0.004)
        num_reported = 0
        for delay, point in zip(delays, candidates):
            self._restore_map()
            nobs = point.num_observations()
            kf = point.observations()[0][0]
            point.update_position(kf.Ow() + kf.Rwc() @ np.array([0.05, 0.05, 1e-4]))
            err, _interrupted = self._local_ba_aborted_after(delay)
            if err == ABORTED_BEFORE_OPTIMIZING:
                continue
            num_reported += 1
            with self.subTest(delay_ms=delay * 1e3):
                self.assertTrue(np.isfinite(err), err)
                self.assertLessEqual(err, Parameters.kChi2Stereo)
                self.assertLess(point.num_observations(), nobs)
        self.assertGreater(num_reported, 0, "no abort reached the chi2 report")


if __name__ == "__main__":
    unittest.main()
