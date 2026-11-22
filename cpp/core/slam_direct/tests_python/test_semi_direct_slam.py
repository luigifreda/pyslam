"""semi_direct (SVO-style) binding regression: run with python -m unittest discover -s this_directory.

A camera translating parallel to a fronto-parallel textured plane sees an exact image
translation of fx * t / Z pixels, so cv2.warpAffine renders the ground truth.
"""
import unittest

import cv2
import numpy as np

from pyslam.config import Config  # Configure paths for the compiled dependencies.
from pyslam.slam.cpp import DirectSlam
import cpp_core

WIDTH, HEIGHT = 320, 240
FX = 200.0


def make_camera(sensor_type):
    camera = cpp_core.PinholeCamera()
    camera.width, camera.height = WIDTH, HEIGHT
    camera.fx = camera.fy = FX
    camera.cx, camera.cy = WIDTH / 2.0, HEIGHT / 2.0
    camera.u_min = camera.v_min = 0.0
    camera.u_max, camera.v_max = float(WIDTH), float(HEIGHT)
    camera.sensor_type = sensor_type
    camera.is_distorted = False
    camera.depth_factor = 1.0
    camera.set_intrinsic_matrices()
    return camera


def textured_plane(seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8), (5, 5), 1.2)


def shifted(image, dx):
    # Camera moves +x, so scene content moves -x in the image.
    transform = np.float32([[1, 0, -dx], [0, 1, 0]])
    return cv2.warpAffine(image, transform, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


class SemiDirectBindingTest(unittest.TestCase):
    def test_mode_aliases(self):
        camera = make_camera(cpp_core.SensorType.RGBD)
        for name in ("semi_direct", "semidirect"):
            slam = DirectSlam(camera, {"DIRECT_SLAM": {"mode": name}})
            self.assertEqual(slam.mode, cpp_core.DirectTrackingMode.SEMI_DIRECT)
        with self.assertRaises(Exception):
            DirectSlam(camera, {"DIRECT_SLAM": {"mode": "svo"}})

    def test_monocular_requires_semi_direct(self):
        camera = make_camera(cpp_core.SensorType.MONOCULAR)
        for mode in ("klt", "hybrid"):
            with self.assertRaises(ValueError):
                DirectSlam(camera, {"DIRECT_SLAM": {"mode": mode}})
        DirectSlam(camera, {"DIRECT_SLAM": {"mode": "semi_direct"}})

    def test_monocular_init_median_depth_follows_resolved_seed_range(self):
        camera = make_camera(cpp_core.SensorType.MONOCULAR)
        default = DirectSlam(camera, {"DIRECT_SLAM": {"mode": "semi_direct"}})
        self.assertAlmostEqual(default.monocular_init_median_depth, np.sqrt(0.5 * 80.0))
        # Flat YAML keys, with the monocular override taking precedence.
        slam = DirectSlam(camera, {"DIRECT_SLAM.mode": "semi_direct", "DIRECT_SLAM.seed_min_depth": 1.0,
                                   "DIRECT_SLAM.seed_max_depth": 50.0,
                                   "DIRECT_SLAM.monocular.seed_max_depth": 16.0})
        self.assertAlmostEqual(slam.monocular_init_median_depth, 4.0)

    def test_monocular_overrides(self):
        # A settings file shared by stereo and monocular runs: the monocular block applies
        # only to a monocular camera, in nested or flat (OpenCV YAML) form.
        nested = {"DIRECT_SLAM": {"mode": "klt", "monocular": {"mode": "semi_direct"}}}
        flat = {"DIRECT_SLAM.mode": "klt", "DIRECT_SLAM.monocular.mode": "semi_direct"}
        for settings in (nested, flat):
            mono = DirectSlam(make_camera(cpp_core.SensorType.MONOCULAR), settings)
            self.assertEqual(mono.mode, cpp_core.DirectTrackingMode.SEMI_DIRECT)
            rgbd = DirectSlam(make_camera(cpp_core.SensorType.RGBD), settings)
            self.assertEqual(rgbd.mode, cpp_core.DirectTrackingMode.KLT)

    def test_monocular_two_view_initialization(self):
        depth = 4.0
        settings = {
            "DIRECT_SLAM": {
                "mode": "semi_direct",
                "corner_detector": "gftt",
                "num_features": 120,
                "gftt_quality": 0.001,
                "gftt_min_distance": 10.0,
                "num_features_tracking": 15,
                "num_features_tracking_bad": 10,
                "num_features_needed_for_keyframe": 10000,
                "max_keyframe_interval": 1000,
                "seed_min_depth": 2.0,
                "seed_max_depth": 8.0,
                "seed_convergence_sigma2": 1e-3,
                "init_min_disparity": 15.0,
                "local_ba": False,
            }
        }
        slam = DirectSlam(make_camera(cpp_core.SensorType.MONOCULAR), settings)
        image = textured_plane(5)
        slam.track(image, None, None, 0, 0.0)
        self.assertEqual(slam.status_name(), "INIT")
        self.assertEqual(slam.num_keyframes(), 0)  # no keyframe before the two-view geometry
        self.assertEqual(len(slam.tracked_features()[0]), 0)
        step = 0.05
        for i in range(1, 12):
            slam.track(shifted(image, FX * step * i / depth), None, None, i, float(i))
            if slam.status_name() != "INIT":
                break
        self.assertIn(slam.status_name(), ("TRACKING_GOOD", "TRACKING_BAD"))
        ids, pixels, points = slam.tracked_features()
        self.assertGreater(len(ids), settings["DIRECT_SLAM"]["num_features_tracking_bad"])
        # Triangulated landmarks lie on the plane. The median depth is scaled to
        # sqrt(seed_min_depth * seed_max_depth) = 4, the true plane depth here.
        z = points[:, 2]
        self.assertLess(np.std(z) / np.mean(z), 0.05)
        self.assertAlmostEqual(float(np.median(z)), depth, delta=0.2)
        # Reference and current frame are both keyframes.
        self.assertGreaterEqual(slam.num_keyframes(), 2)

    def test_rgbd_with_missing_depth_spawns_and_promotes_seeds(self):
        # The right half of the depth image is invalid: corners there become depth seeds.
        depth_value = 3.0
        depth = np.full((HEIGHT, WIDTH), depth_value, dtype=np.float32)
        depth[:, WIDTH // 2:] = 0.0
        settings = {
            "DIRECT_SLAM": {
                "mode": "semi_direct",
                "num_features": 150,
                "num_features_tracking": 20,
                "num_features_tracking_bad": 10,
                "max_keyframe_interval": 3,
                "seed_convergence_sigma2": 1e-3,
                "local_ba": True,
            }
        }
        slam = DirectSlam(make_camera(cpp_core.SensorType.RGBD), settings)
        image = textured_plane(9)
        slam.track(image, None, depth, 0, 0.0)
        ids0 = set(slam.tracked_features()[0])
        self.assertGreater(len(ids0), 10)
        step = 0.03
        frames = 14
        for i in range(1, frames + 1):
            slam.track(shifted(image, FX * step * i / depth_value), None, depth, i, float(i))
            self.assertIn(slam.status_name(), ("TRACKING_GOOD", "TRACKING_BAD"), f"lost at frame {i}")
        Twc = np.linalg.inv(slam.pose().Tcw)
        self.assertAlmostEqual(Twc[0, 3], step * frames, delta=0.03)
        self.assertLess(np.linalg.norm(Twc[1:3, 3]), 0.03)
        ids, _, points = slam.tracked_features()
        promoted = [k for k, pid in enumerate(ids) if pid not in ids0]
        self.assertGreater(len(promoted), 0)
        # Landmarks from seeds or depth both lie on the plane z = depth_value (world frame).
        np.testing.assert_allclose(points[:, 2], depth_value, atol=0.25)


if __name__ == "__main__":
    unittest.main()
