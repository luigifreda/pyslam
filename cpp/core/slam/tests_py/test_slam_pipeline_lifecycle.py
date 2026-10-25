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
* Native frontend lifecycle and lock-order regression tests.
*
* Run in a subprocess with a timeout, since a lock-order regression can deadlock:
*   timeout 120 python cpp/core/slam/tests_py/test_slam_pipeline_lifecycle.py
"""

import os
import gc
import sys
import threading
import tempfile
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
os.environ["PYSLAM_USE_CPP"] = "1"

import pyslam.config  # Library paths before importing the native core.
from pyslam.config_parameters import Parameters
from pyslam.io.dataset_types import SensorType
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.slam import Slam
from pyslam.slam.slam_commons import SlamState
from compare_cpp_python_trajectory import _make_camera, _texture, _frame


class PipelineLifecycleTests(unittest.TestCase):
    def make_slam(self, threaded):
        Parameters.kUseLoopClosing = False
        Parameters.kLocalMappingOnSeparateThread = threaded
        Parameters.kDoVolumetricIntegration = False
        Parameters.kDoSparseSemanticMappingAndSegmentation = False
        Parameters.kUseLargeWindowBA = False
        Parameters.kUseMotionBlurDection = False
        tracker = dict(FeatureTrackerConfigs.ORB2, deterministic=True, num_features=800)
        slam = Slam(_make_camera(True), tracker, sensor_type=SensorType.RGBD, headless=True)
        self.addCleanup(slam.quit)
        return slam

    def initialize(self, slam):
        texture = _texture()
        for i in range(2):
            image, depth = _frame(texture, i)
            slam.track(image, None, depth, i, float(i))
        self.assertEqual(slam.tracking.state, SlamState.OK)
        return texture

    def test_immediate_start_quit_and_reset_after_quit(self):
        slam = self.make_slam(True)
        for _ in range(20):
            slam.local_mapping.quit()
            slam.local_mapping.start()
        slam.local_mapping.quit()
        slam.local_mapping.request_reset()

    def test_reset_while_stopped(self):
        slam = self.make_slam(True)
        self.initialize(slam)
        slam.local_mapping.request_stop()
        deadline = time.monotonic() + 3
        while not slam.local_mapping.is_stopped() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(slam.local_mapping.is_stopped())
        slam.reset_session()
        self.assertEqual(slam.tracking.state, SlamState.NO_IMAGES_YET)
        slam.local_mapping.release()
        self.initialize(slam)

    def test_track_releases_gil_when_waiting_for_map(self):
        slam = self.make_slam(False)
        texture = self.initialize(slam)
        locked = threading.Event()

        def hold_map():
            with slam.map.update_lock:
                locked.set()
                time.sleep(0.2)

        worker = threading.Thread(target=hold_map)
        worker.start()
        self.assertTrue(locked.wait(3))
        image, depth = _frame(texture, 2)
        slam.track(image, None, depth, 2, 2.0)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())

    def test_relocalize_releases_map_lock(self):
        slam = self.make_slam(False)
        texture = self.initialize(slam)

        class LoopClosing:
            def is_closing(self):
                return False

            def relocalize(self, frame, image):
                def inspect_map():
                    with slam.map.update_lock:
                        return slam.map.num_keyframes()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    self.num_keyframes = pool.submit(inspect_map).result(timeout=3)
                return False

            def quit(self):
                pass

        collaborator = LoopClosing()
        slam.loop_closing = collaborator
        slam.set_tracking_state(SlamState.LOST)
        image, depth = _frame(texture, 2)
        slam.track(image, None, depth, 2, 2.0)
        self.assertEqual(collaborator.num_keyframes, 2)

    def test_missing_state_does_not_stop_mapper(self):
        slam = self.make_slam(True)
        with self.assertRaises(RuntimeError):
            slam.load_system_state("/tmp/pyslam-nonexistent-review-state")
        self.assertTrue(slam.local_mapping.is_running())

    def test_quit_releases_python_collaborator_cycle(self):
        slam = self.make_slam(False)

        class Collaborator:
            def __init__(self, owner):
                self.slam = owner

            def quit(self):
                pass

        collaborator = Collaborator(slam)
        reference = weakref.ref(collaborator)
        slam.loop_closing = collaborator
        del collaborator
        slam.quit()
        gc.collect()
        self.assertIsNone(reference())

    def test_stop_drains_queued_keyframes(self):
        slam = self.make_slam(False)
        self.initialize(slam)
        mapper = slam.local_mapping
        mapper.push_keyframe(slam.tracking.kf_ref)
        mapper.request_stop()
        mapper.step()
        self.assertEqual(mapper.queue_size(), 0)
        self.assertTrue(mapper.is_idle())
        mapper.step()
        self.assertTrue(mapper.is_stopped())
        # Single-thread callers do not have a worker that can acknowledge reset.
        mapper.request_reset()

    def test_threaded_tracking_snapshots_and_save_load(self):
        slam = self.make_slam(True)
        texture = self.initialize(slam)
        stop = threading.Event()
        failures = []

        def inspect_snapshots():
            try:
                while not stop.is_set():
                    tracking = slam.tracking
                    _ = tracking.f_cur, tracking.kf_ref, tracking.predicted_pose
                    _ = tracking.cur_R, tracking.cur_t, tracking.descriptor_distance_sigma
                    time.sleep(0.001)
            except Exception as error:
                failures.append(error)

        reader = threading.Thread(target=inspect_snapshots)
        reader.start()
        try:
            for i in range(2, 20):
                image, depth = _frame(texture, i)
                slam.track(image, None, depth, i, float(i))
        finally:
            stop.set()
            reader.join(timeout=3)
        self.assertFalse(reader.is_alive())
        self.assertFalse(failures)
        self.assertEqual(slam.tracking.state, SlamState.OK)
        deadline = time.monotonic() + 10
        while (slam.local_mapping.queue_size() or not slam.local_mapping.is_idle()) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(slam.local_mapping.queue_size(), 0)
        self.assertTrue(slam.local_mapping.is_idle())
        with tempfile.TemporaryDirectory(prefix="pyslam-review-state-") as path:
            slam.save_system_state(path)
            counts = slam.map.num_keyframes(), slam.map.num_points()
            slam.load_system_state(path)
            self.assertEqual(counts, (slam.map.num_keyframes(), slam.map.num_points()))
            self.assertTrue(slam.local_mapping.is_running())


if __name__ == "__main__":
    unittest.main()
