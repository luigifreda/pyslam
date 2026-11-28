"""Binding regression: run with python -m unittest discover -s this_directory."""
import unittest

import cv2
import numpy as np

from pyslam.config import Config  # Configure paths for the compiled dependencies.
from pyslam.slam.cpp import DirectSlam
import cpp_core


class DirectSlamBindingTest(unittest.TestCase):
    def test_snapshot_ownership_and_reset(self):
        camera = cpp_core.PinholeCamera()
        camera.width, camera.height = 320, 240
        camera.fx = camera.fy = 200.0
        camera.cx, camera.cy = 160.0, 120.0
        camera.u_min = camera.v_min = 0.0
        camera.u_max, camera.v_max = 320.0, 240.0
        camera.sensor_type = cpp_core.SensorType.RGBD
        camera.is_distorted = False
        camera.depth_factor = 1.0
        camera.set_intrinsic_matrices()
        slam = DirectSlam(camera, {"DIRECT_SLAM": {"num_features": 200, "max_keyframe_interval": 2}})
        rng = np.random.default_rng(42)
        image = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), dtype=np.uint8), (5, 5), 0.8)
        depth = np.full(image.shape, 4.0, dtype=np.float32)
        slam.track(image, None, depth, 7, 1.25)
        ids, pixels, points = slam.tracked_features()
        poses, timestamps = slam.keyframes()
        self.assertGreater(len(ids), 20)
        self.assertEqual(pixels.shape, (len(ids), 2))
        self.assertEqual(points.shape, (len(ids), 3))
        np.testing.assert_allclose(points[:, 2], 4.0)
        self.assertEqual(timestamps, [1.25])
        np.testing.assert_allclose(poses[0], np.eye(4))
        original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rendered = slam.draw_feature_trails(original)
        self.assertEqual(rendered.shape, original.shape)
        self.assertGreater(np.count_nonzero(rendered != original), 0)
        np.testing.assert_array_equal(original[:, :, 0], image)
        for i in range(1, 5):
            transform = np.float32([[1, 0, i * 2], [0, 1, 0]])
            current = cv2.warpAffine(image, transform, (320, 240))
            slam.track(current, None, depth, 7 + i, 1.25 + i * 0.1)
        result = slam.bundle_adjustment()
        self.assertTrue(result.applied)
        self.assertGreater(result.landmarks, 20)
        self.assertLessEqual(result.final_cost, result.initial_cost * 1.05 + 1e-6)
        np.testing.assert_allclose(slam.trajectory()[-1], slam.pose().Tcw)
        self.assertEqual(slam.map_points().shape[1], 3)
        # Viewer snapshots must remain valid after the core releases its map.
        saved = points.copy()
        slam.reset()
        np.testing.assert_equal(points, saved)
        self.assertEqual(slam.tracked_features()[1].shape, (0, 2))
        self.assertEqual(slam.tracked_features()[2].shape, (0, 3))
        self.assertEqual(slam.trajectory(), [])
        self.assertEqual(slam.map_points().shape, (0, 3))
        self.assertFalse(slam.last_ba().applied)
        self.assertEqual(slam.keyframes(), ([], []))
        with self.assertRaises(ValueError):
            slam.track(image, None, None)


if __name__ == "__main__":
    unittest.main()
