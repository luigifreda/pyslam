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
* Local bundle adjustment interrupted through its abort flag, for the core PYSLAM_USE_CPP selects.
*
* Local mapping raises the flag whenever a new keyframe arrives. g2o stops at the next iteration
* boundary and leaves the edge errors as they were: never computed (uninitialized memory, often
* NaN, inf or ~1e37) if it stopped before its first iteration, or those of a rejected Levenberg
* step. The reported mean chi2 must describe the state written back to the map.
"""

import threading
import unittest

import numpy as np
from slam_test_support import make_slam, track_sequence

import g2o  # noqa: E402  (importable once pyslam.config has set the library paths)
from pyslam.slam import USE_CPP, optimizer_g2o  # noqa: E402

NUM_FRAMES = 60
ABORTED_BEFORE_OPTIMIZING = -1


class LocalBaAbortCase(unittest.TestCase):
    """A converged synthetic map, and local BA runs on it that are aborted after a delay."""

    @classmethod
    def setUpClass(cls):
        cls.slam = make_slam(cls)
        track_sequence(cls.slam, range(NUM_FRAMES))
        cls.keyframes = [kf for kf in cls.slam.map.get_keyframes() if not kf.is_bad()]
        cls.points = [p for p in cls.slam.map.get_points() if not p.is_bad()]
        cls.converged_error = cls._local_ba(g2o.Flag(False))
        cls.poses = [kf.Tcw().copy() for kf in cls.keyframes]
        cls.positions = [p.pt().copy() for p in cls.points]

    def setUp(self):
        self._restore_map()

    def _restore_map(self):
        """Every run starts from the same map, since an aborted run still moves it."""
        for kf, Tcw in zip(self.keyframes, self.poses):
            kf.update_pose(Tcw)
        for p, position in zip(self.points, self.positions):
            p.update_position(position)

    @classmethod
    def _local_ba(cls, abort_flag):
        err, _ = optimizer_g2o.local_bundle_adjustment(
            cls.keyframes, cls.points, [], False, False, 10, abort_flag=abort_flag
        )
        return err

    def _local_ba_aborted_after(self, delay):
        """Return the LBA error and whether the flag was raised while LBA was running."""
        abort_flag = g2o.Flag(False)
        timer = threading.Timer(delay, setattr, (abort_flag, "value", True))
        if USE_CPP:
            timer.start()
            err = self._local_ba(abort_flag)
        else:
            # The Python core spends most of the call building the graph, so start the timer
            # when g2o starts optimizing. A zero delay raises the flag before the first iteration.
            original_optimize = g2o.SparseOptimizer.optimize

            def optimize(optimizer, *args, **kwargs):
                if delay == 0:
                    abort_flag.value = True
                elif timer.ident is None:
                    timer.start()
                return original_optimize(optimizer, *args, **kwargs)

            g2o.SparseOptimizer.optimize = optimize
            try:
                err = self._local_ba(abort_flag)
            finally:
                g2o.SparseOptimizer.optimize = original_optimize
        interrupted = bool(abort_flag.value)
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        return err, interrupted

    def _assert_error_of_accepted_state(self, err):
        # The map is already converged, so every accepted state has about the same error.
        self.assertTrue(np.isfinite(err), err)
        self.assertGreater(err, 0.5 * self.converged_error)
        self.assertLess(err, 2.0 * self.converged_error)
