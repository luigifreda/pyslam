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
* Pinhole compatibility and regression tests for the resolved camera defects.
"""

import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from camera_test_support import calibration_cases, model_config, sample_points
import cv2
import numpy as np
from pyslam.slam.camera import (
    CameraType, camera_from_config, camera_from_json, PinholeCamera,
)
from pyslam.slam.cpp import cpp_module
from pyslam.slam.frame import Frame
from pyslam.slam.map import Map
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared


def add_perfect_point(model, backward=False, turned=False, native=False):
    """Exercise the actual map geometry checks, isolating landmark bookkeeping."""
    with contextlib.redirect_stdout(io.StringIO()):
        config = model_config(model)
        if backward or turned:
            config.cam_settings["Camera.fx"] = config.cam_settings["Camera.fy"] = 100.0
        camera = camera_from_config(config, cpp_module.classes if native else None)
    point = np.ascontiguousarray([[1.2, 0.4, -0.2 if backward else (0.2 if turned else 2.0)]])
    frames = []
    for x in (0.0, -0.5):
        pose = np.eye(4)
        pose[0, 3] = x
        if turned and x != 0.0:
            theta = -np.pi / 3
            pose[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]]
        frame = (cpp_module.Frame if native else Frame)(camera=camera, img=None)
        frame.update_pose(pose)
        frame.is_keyframe = True
        frame.kps = camera.project(frame.transform_points(point))[0]
        frame.kpsu = camera.undistort_points(frame.kps)
        raw = frame.kps if getattr(camera, 'projects_raw_keypoints', lambda: False)() else frame.kpsu
        frame.kpsn = camera.unproject_points(raw)
        frame.octaves = np.array([0])
        if native:
            frame.des = np.zeros((1, 32), dtype=np.uint8)
            frame.points = np.array([None], dtype=object)
            frame.outliers = np.array([False])
            frames.append(cpp_module.KeyFrame(frame=frame))
        else:
            frames.append(frame)
    if native:
        resources = cpp_module.FeatureSharedResources
        fields = ('inv_level_sigmas2', 'scale_factors', 'scale_factor', 'num_levels', 'norm_type')
        original = {field: getattr(resources, field) for field in fields}
        try:
            resources.inv_level_sigmas2 = resources.scale_factors = [1.] * 8
            resources.scale_factor = 1.2
            resources.num_levels = 8
            resources.norm_type = cv2.NORM_HAMMING
            count, _mask, _points = cpp_module.Map().add_points(
                point, None, *frames, [0], [0], np.zeros((480, 640, 3), dtype=np.uint8))
            return count
        finally:
            for field, value in original.items():
                setattr(resources, field, value)
    manager = SimpleNamespace(inv_level_sigmas2=np.ones(8), scale_factors=np.ones(8), scale_factor=1.2)
    with patch.object(FeatureTrackerShared, "feature_manager", manager), \
            patch("pyslam.slam.map.MapPoint"):
        count, _mask, _points = Map.add_points(
            MagicMock(), point, None, *frames, [0], [0], np.zeros((480, 640, 3), dtype=np.uint8)
        )
    return count


class TestPinholeDatasetCompatibility(unittest.TestCase):
    def test_raw_camera_scale_check_uses_range_under_rotation(self):
        self.assertEqual(add_perfect_point("kb4", turned=True), 1)

    def test_native_raw_camera_scale_check_uses_range_under_rotation(self):
        self.assertEqual(add_perfect_point("kb4", turned=True, native=True), 1)

    def test_pinhole_jacobian_rejects_nonfinite_and_backward_points(self):
        config = model_config("pinhole")
        for constructors in (None, cpp_module.classes):
            camera = camera_from_config(config, constructors)
            for point in ([np.nan, 0., 1.], [0., np.inf, 1.], [0., 0., -1.], [0., 0., 0.]):
                with self.subTest(backend=constructors is not None, point=point):
                    uv, valid, jacobian = camera.project_with_jacobian(np.asarray(point))
                    self.assertFalse(valid)
                    self.assertTrue(np.isfinite(uv).all())
                    np.testing.assert_array_equal(jacobian, np.zeros((2, 3)))

    def test_map_accepts_oriented_backward_bearings(self):
        self.assertEqual(add_perfect_point("kb4", backward=True), 1)

    def test_map_accepts_perfect_pinhole_correspondence(self):
        self.assertEqual(add_perfect_point("pinhole"), 1)

    def test_all_calibrations_and_sensor_modes(self):
        points = sample_points()
        modes = set()
        calibrations = set()
        for name, sensor, config in calibration_cases():
            with self.subTest(calibration=name, sensor=sensor):
                modes.add(sensor)
                calibrations.add(name)
                with contextlib.redirect_stdout(io.StringIO()):
                    python = camera_from_config(config)
                    native = camera_from_config(config, cpp_module.classes)
                self.assertIsInstance(python, PinholeCamera)
                self.assertEqual(int(native.type), CameraType.PINHOLE.value)
                expected_uv = (python.K @ points.T).T
                expected_uv = expected_uv[:, :2] / points[:, 2, None]
                for camera in (python, native):
                    uv, z = camera.project(points)
                    np.testing.assert_allclose(uv, expected_uv, atol=1e-10)
                    np.testing.assert_allclose(z, points[:, 2], atol=1e-12)
                    xy = camera.unproject_points(np.ascontiguousarray(uv))
                    np.testing.assert_allclose(xy, points[:, :2] / points[:, 2, None], atol=1e-12)
                    if sensor != "mono":
                        stereo, z = camera.project_stereo(points)
                        np.testing.assert_allclose(stereo[:, :2], expected_uv, atol=1e-10)
                        np.testing.assert_allclose(
                            stereo[:, 2], expected_uv[:, 0] - camera.bf / z, atol=1e-10
                        )
                        reconstructed = camera.unproject_points_3d(
                            np.ascontiguousarray(uv), np.ascontiguousarray(z)
                        )
                        np.testing.assert_allclose(reconstructed, points, atol=1e-10)
                # Distortion must still be applied only during undistortion.
                pixels = np.ascontiguousarray(expected_uv)
                expected = cv2.undistortPoints(
                    pixels.reshape(-1, 1, 2), python.K, python.D, P=python.K
                ).reshape(-1, 2)
                for camera in (python, native):
                    np.testing.assert_allclose(camera.undistort_points(pixels), expected, atol=1e-8)
        self.assertEqual(modes, {"mono", "stereo", "rgbd"})
        self.assertGreaterEqual(len(calibrations), 18)

    def test_new_models_use_raw_pixels_for_kdtree(self):
        for model in ("kb4", "radtan8", "ds"):
            with self.subTest(model=model):
                camera = camera_from_config(model_config(model))
                frame = Frame(camera=camera, img=None)
                frame.kps = np.array([[100.0, 120.0], [500.0, 360.0]])
                frame.kpsu = camera.undistort_points(frame.kps)
                distances, indices = frame.kd.query(frame.kps)
                np.testing.assert_array_equal(indices, [0, 1])
                np.testing.assert_allclose(distances, 0.0)

    def test_python_camera_json_factory_preserves_model(self):
        for model in ("kb4", "radtan8", "ds"):
            with self.subTest(model=model):
                camera = camera_from_config(model_config(model))
                restored = camera_from_json(camera.to_json())
                self.assertIs(type(restored), type(camera))
                np.testing.assert_allclose(restored.project(sample_points())[0], camera.project(sample_points())[0])


class TestKnownCameraDefects(unittest.TestCase):
    def test_map_accepts_perfect_fisheye_correspondence(self):
        """R6: map admission still subtracts kpsu from raw-model projections."""
        self.assertEqual(add_perfect_point("kb4"), 1)

    def test_rgbd_projection_search_supports_selected_model(self):
        """R1: accepted RGB-D config must support the projection tracking requests."""
        camera = camera_from_config(model_config(sensor="rgbd"), cpp_module.classes)
        self.assertTrue(camera.is_stereo())
        uv, depth = camera.project_stereo(sample_points())
        self.assertEqual(uv.shape, (512, 3))
        self.assertTrue(np.isfinite(depth).all())

    def test_python_frame_json_preserves_camera_subclass(self):
        """R2: use the real frame loader, not just camera_from_json."""
        camera = camera_from_config(model_config())
        frame = Frame(camera=camera, img=None, id=1, timestamp=0.0, img_id=0)
        # The loader checks feature/point counts, including for an empty frame.
        frame.kps = np.empty((0, 2))
        frame.points = np.empty(0, dtype=object)
        restored = Frame.from_json(frame.to_json())
        self.assertIs(type(restored.camera), type(camera))
        np.testing.assert_allclose(restored.camera.project(sample_points())[0], camera.project(sample_points())[0])

    def test_python_model_accepts_serialized_config(self):
        """R3: Camera converts a config dict but the model reuses the original dict."""
        config = model_config()
        camera = camera_from_config(config.to_json())
        np.testing.assert_allclose(camera.project(sample_points())[0], camera_from_config(config).project(sample_points())[0])

    def test_kb4_inverse_rejects_pixels_outside_angular_domain(self):
        """R4: zero-distortion KB4 cannot represent a radius > pi radians."""
        config = model_config()
        for key in ("Camera.k1", "Camera.k2", "Camera.k3", "Camera.k4"):
            config.cam_settings[key] = 0.0
        camera = camera_from_config(config, cpp_module.classes)
        pixel = np.ascontiguousarray([camera.cx + 4.0 * camera.fx, camera.cy])
        _bearing, valid, _jacobian = camera.unproject_bearing(pixel)
        self.assertFalse(valid)

    def test_invalid_g2o_projection_does_not_poison_graph(self):
        """R5: graph error evaluation runs before outlier rejection."""
        import g2o
        import cpp_core

        camera = camera_from_config(model_config(), cpp_module.classes)
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        ))
        pose = g2o.VertexSE3Expmap()
        pose.set_id(0)
        pose.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros(3)))
        optimizer.add_vertex(pose)
        edge = cpp_core.EdgeCameraProjectXYZOnlyPose(camera)
        edge.Xw = np.zeros(3)
        edge.set_vertex(0, pose)
        edge.set_measurement(np.array([camera.cx, camera.cy]))
        edge.set_information(np.eye(2))
        optimizer.add_edge(edge)
        optimizer.initialize_optimization()
        optimizer.compute_active_errors()
        self.assertFalse(edge.is_depth_positive())
        self.assertTrue(np.isfinite(np.asarray(edge.error())).all())


if __name__ == "__main__":
    if "--strict" in sys.argv:
        sys.argv.remove("--strict")
        for test in vars(TestKnownCameraDefects).values():
            if getattr(test, "__unittest_expecting_failure__", False):
                test.__unittest_expecting_failure__ = False
    unittest.main()
