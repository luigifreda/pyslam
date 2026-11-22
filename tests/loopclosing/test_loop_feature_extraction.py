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
* Worker feature transport without requiring the native SLAM dependencies.
"""

import multiprocessing as mp
import os
import pickle
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pyslam.loop_closing.loop_feature_extraction import LoopKeyframeFeatureExtractor


def fake_dependencies():
    tracker = ModuleType("pyslam.local_features.feature_tracker")
    tracker.feature_tracker_factory = Mock(side_effect=lambda **config: config)
    shared = ModuleType("pyslam.slam.feature_tracker_shared")
    shared.FeatureTrackerShared = SimpleNamespace(
        set_feature_tracker=Mock(), set_feature_tracker_right=Mock())
    slam = ModuleType("pyslam.slam")

    def make_frame(camera, image, **kwargs):
        frame = SimpleNamespace(**{
            name: np.array([[1., 2.]]) if name.startswith("kps") else np.array([1])
            for name in LoopKeyframeFeatureExtractor.fields})
        frame.des = np.array([[7, 8]], dtype=np.uint8)
        frame.des_r = None if kwargs["img_right"] is None else frame.des.copy()
        return frame

    slam.Frame = Mock(side_effect=make_frame)
    return {tracker.__name__: tracker, shared.__name__: shared, slam.__name__: slam}


def task(stereo=False):
    return SimpleNamespace(
        feature_extraction=dict(config={"detector_type": "test"}, camera="calibration",
                                image_right=np.ones((4, 4)) if stereo else None,
                                depth=None if stereo else np.ones((4, 4))),
        keyframe_data=SimpleNamespace(id=42, img=np.zeros((4, 4)), g_des=None),
        covisible_keyframes_data=[])


def extract_in_child(output):
    with patch.dict(sys.modules, fake_dependencies()):
        output.put((os.getpid(), LoopKeyframeFeatureExtractor().extract(task(stereo=True))))


class LoopFeatureExtractionTests(unittest.TestCase):
    def test_standard_tasks_do_not_extract(self):
        extractor = LoopKeyframeFeatureExtractor()
        self.assertIsNone(extractor.extract(SimpleNamespace()))
        self.assertIsNone(extractor.extract(SimpleNamespace(feature_extraction=None)))

    def test_rgbd_features_are_owned_and_separate_from_retrieval_descriptors(self):
        modules = fake_dependencies()
        with patch.dict(sys.modules, modules):
            request = task()
            extractor = LoopKeyframeFeatureExtractor()
            features = extractor.extract(request)
            frame_factory = modules["pyslam.slam"].Frame
            self.assertEqual(frame_factory.call_args.args[0], "calibration")
            self.assertIs(frame_factory.call_args.kwargs["depth"], request.feature_extraction["depth"])
            self.assertEqual(frame_factory.call_args.kwargs["id"], 42)
            self.assertEqual(set(features), set(extractor.fields))
            # Retrieval can replace its descriptor field without altering geometry.
            request.keyframe_data.des = np.array([[99]], dtype=np.uint8)
            np.testing.assert_array_equal(features["des"], [[7, 8]])
            restored = pickle.loads(pickle.dumps(features))
            np.testing.assert_array_equal(restored["des"], features["des"])
            self.assertIsNone(restored["des_r"])

    def test_trackers_are_reused_and_stereo_tracker_is_initialized(self):
        modules = fake_dependencies()
        with patch.dict(sys.modules, modules):
            extractor = LoopKeyframeFeatureExtractor()
            extractor.extract(task(stereo=True))
            extractor.extract(task(stereo=True))
            shared = modules["pyslam.slam.feature_tracker_shared"].FeatureTrackerShared
            self.assertEqual(shared.set_feature_tracker.call_count, 1)
            self.assertEqual(shared.set_feature_tracker_right.call_count, 1)
            self.assertEqual(modules["pyslam.local_features.feature_tracker"].feature_tracker_factory.call_count, 2)

    def test_queued_neighbors_use_worker_global_descriptors_and_reset_clears_them(self):
        with patch.dict(sys.modules, fake_dependencies()):
            extractor = LoopKeyframeFeatureExtractor()
            first = task()
            first.keyframe_data.g_des = [0.2, 0.8]
            extractor.remember(first, SimpleNamespace(g_des_vec=None))
            second = task()
            neighbor = SimpleNamespace(id=42, g_des=None)
            second.covisible_keyframes_data = [neighbor]
            extractor.extract(second)
            self.assertEqual(neighbor.g_des, [0.2, 0.8])
            extractor.reset()
            self.assertEqual(extractor.global_descriptors, {})

    def test_global_descriptor_history_is_bounded_and_keeps_recent_keyframes(self):
        extractor = LoopKeyframeFeatureExtractor()
        limit = extractor.max_global_descriptors
        for kid in range(limit + 25):
            request = SimpleNamespace(feature_extraction={}, keyframe_data=SimpleNamespace(id=kid, g_des=[kid]))
            extractor.remember(request, SimpleNamespace(g_des_vec=None))
        self.assertEqual(len(extractor.global_descriptors), limit)
        self.assertNotIn(0, extractor.global_descriptors)
        self.assertEqual(extractor.global_descriptors[limit + 24], [limit + 24])
        self.assertEqual(next(iter(extractor.global_descriptors)), 25)

    def test_features_cross_spawn_process_boundary(self):
        context = mp.get_context("spawn")
        output = context.Queue()
        worker = context.Process(target=extract_in_child, args=(output,))
        worker.start()
        try:
            pid, features = output.get(timeout=15)
            self.assertNotEqual(pid, os.getpid())
            np.testing.assert_array_equal(features["des"], [[7, 8]])
            np.testing.assert_array_equal(features["des_r"], [[7, 8]])
            worker.join(timeout=5)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            output.close()
            output.join_thread()


if __name__ == "__main__":
    unittest.main()
