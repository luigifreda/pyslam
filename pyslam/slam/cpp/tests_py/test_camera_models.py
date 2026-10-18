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
"""

import os
import sys
import time
import unittest

kRootFolder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../.."))
if kRootFolder not in sys.path:
    sys.path.insert(0, kRootFolder)

import pyslam.config as config  # noqa: F401  — puts cpp/lib on sys.path before the SLAM imports

import numpy as np

from pyslam.slam.camera import CameraType
from pyslam.slam.camera import PinholeCamera as PythonPinholeCamera
from pyslam.slam.cpp import (
    CPP_AVAILABLE,
    DoubleSphereCamera,
    KannalaBrandt4Camera,
    PinholeCamera,
    PinholeRadtan8Camera,
)


def _parallel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@unittest.skipUnless(CPP_AVAILABLE, "C++ core module is not built")
class TestCameraModels(unittest.TestCase):
    def test_pinhole_project_into_matches_linear_projection(self):
        reference = PythonPinholeCamera(None)
        reference.width = 640
        reference.height = 480
        reference.fx = 525.0
        reference.fy = 520.0
        reference.cx = 320.0
        reference.cy = 240.0
        reference.set_intrinsic_matrices()
        camera = PinholeCamera()
        camera.width = 640
        camera.height = 480
        camera.fx = 525.0
        camera.fy = 520.0
        camera.cx = 320.0
        camera.cy = 240.0
        camera.set_intrinsic_matrices()
        points = np.ascontiguousarray(
            np.array([[0.2, -0.1, 2.0], [1.0, 0.5, 4.0]], dtype=np.float64)
        )
        uv_ref, depth_ref = reference.project(points)
        uv = np.empty_like(points[:, :2])
        depth = np.empty((points.shape[0],), dtype=np.float64)
        valid = np.empty((points.shape[0],), dtype=np.float64)
        camera.project_into(points, uv, depth, valid)
        np.testing.assert_allclose(uv, uv_ref, atol=1e-12)
        np.testing.assert_allclose(depth, depth_ref, atol=1e-12)
        self.assertTrue(np.all(valid > 0.5))

        _, _, jacobian = camera.project_with_jacobian(np.ascontiguousarray(points[0]))
        inv_z = 1.0 / points[0, 2]
        expected = np.array(
            [
                [camera.fx * inv_z, 0.0, -camera.fx * points[0, 0] * inv_z * inv_z],
                [0.0, camera.fy * inv_z, -camera.fy * points[0, 1] * inv_z * inv_z],
            ]
        )
        np.testing.assert_allclose(jacobian, expected, atol=1e-12)

    def _check_roundtrip(self, camera, points):
        points = np.ascontiguousarray(points)
        uv = np.empty((points.shape[0], 2), dtype=np.float64)
        depth = np.empty((points.shape[0],), dtype=np.float64)
        valid = np.empty((points.shape[0],), dtype=np.float64)
        camera.project_into(points, uv, depth, valid)
        self.assertTrue(np.all(valid > 0.5))
        # The buffers passed in are the ones that were written.
        self.assertTrue(uv.flags["OWNDATA"])
        bearings = np.empty((points.shape[0], 3), dtype=np.float64)
        bearing_valid = np.empty((points.shape[0],), dtype=np.float64)
        camera.unproject_bearings_into(np.ascontiguousarray(uv), bearings, bearing_valid)
        for source, restored in zip(points, bearings):
            self.assertGreater(_parallel(source, restored), 1.0 - 1e-8)
        point = np.ascontiguousarray(points[0])
        projected, is_valid, jacobian = camera.project_with_jacobian(point)
        self.assertTrue(is_valid)
        eps = 1e-6
        numeric = np.zeros((2, 3), dtype=np.float64)
        for axis in range(3):
            plus = point.copy()
            minus = point.copy()
            plus[axis] += eps
            minus[axis] -= eps
            uv_plus, _, _ = camera.project_with_jacobian(plus)
            uv_minus, _, _ = camera.project_with_jacobian(minus)
            numeric[:, axis] = (np.asarray(uv_plus) - np.asarray(uv_minus)) / (2.0 * eps)
        scale = max(1.0, np.linalg.norm(numeric))
        self.assertLess(np.linalg.norm(np.asarray(jacobian) - numeric) / scale, 1e-6)
        self.assertTrue(np.isfinite(projected).all())

    def test_kannala_brandt4(self):
        camera = KannalaBrandt4Camera(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002)
        self.assertEqual(int(camera.type), int(CameraType.KANNALA_BRANDT4.value))
        self._check_roundtrip(camera, np.array([[0.4, -0.2, 1.5], [0.0, 0.0, 2.0]]))

    def test_radtan8_matches_pinhole_when_undistorted(self):
        camera = PinholeRadtan8Camera(640, 480, 400.0, 410.0, 320.0, 240.0, 0, 0, 0, 0, 0, 0, 0, 0)
        point = np.ascontiguousarray(np.array([0.3, -0.2, 2.0], dtype=np.float64))
        uv, valid, jacobian = camera.project_with_jacobian(point)
        expected_uv = np.array(
            [400.0 * point[0] / point[2] + 320.0, 410.0 * point[1] / point[2] + 240.0]
        )
        np.testing.assert_allclose(uv, expected_uv, atol=1e-12)
        self.assertTrue(valid)
        inv_z = 1.0 / point[2]
        expected_j = np.array(
            [
                [400.0 * inv_z, 0.0, -400.0 * point[0] * inv_z * inv_z],
                [0.0, 410.0 * inv_z, -410.0 * point[1] * inv_z * inv_z],
            ]
        )
        np.testing.assert_allclose(jacobian, expected_j, atol=1e-9)
        distorted = PinholeRadtan8Camera(
            640, 480, 400.0, 410.0, 320.0, 240.0, -0.1, 0.02, 0.001, -0.0005, 0.003, 0.01, -0.004, 0.001
        )
        self._check_roundtrip(distorted, np.array([[0.3, -0.2, 2.0]]))
        self._check_undistort_is_pinhole_image(distorted, 400.0, 410.0, 320.0, 240.0)

    def test_double_sphere_accepts_negative_z(self):
        camera = DoubleSphereCamera(640, 480, 250.0, 260.0, 320.0, 240.0, -0.2, 0.6)
        self._check_roundtrip(camera, np.array([[0.25, -0.15, 1.2], [1.0, 0.0, -0.05]]))
        self._check_undistort_is_pinhole_image(camera, 250.0, 260.0, 320.0, 240.0)

    def _check_undistort_is_pinhole_image(self, camera, fx, fy, cx, cy):
        self.assertTrue(camera.projects_raw_keypoints())
        point = np.ascontiguousarray([[0.2, -0.1, 1.5]], dtype=np.float64)
        detected, _depth = camera.project(point)
        detected = np.ascontiguousarray(np.asarray(detected, dtype=np.float64).reshape(1, 2))
        undistorted = np.asarray(camera.undistort_points(detected), dtype=np.float64).reshape(2)
        expected = np.array([fx * 0.2 / 1.5 + cx, fy * -0.1 / 1.5 + cy])
        np.testing.assert_allclose(undistorted, expected, atol=1e-6)
        self.assertGreater(abs(undistorted[0] - detected[0, 0]), 1e-4)
        normalized = np.asarray(camera.unproject_points(detected), dtype=np.float64).reshape(2)
        np.testing.assert_allclose(
            normalized, np.array([(undistorted[0] - cx) / fx, (undistorted[1] - cy) / fy]), atol=1e-6
        )

    def test_tracking_keeps_raw_pixels_and_negative_depth(self):
        camera = KannalaBrandt4Camera(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002)
        self._check_undistort_is_pinhole_image(camera, 300.0, 310.0, 320.0, 240.0)
        self.assertFalse(camera.requires_positive_depth())
        self.assertTrue(camera.is_in_image(np.array([100.0, 120.0]), -0.05))

        pinhole = PinholeCamera()
        pinhole.u_min = 0.0
        pinhole.u_max = 640.0
        pinhole.v_min = 0.0
        pinhole.v_max = 480.0
        self.assertFalse(pinhole.is_in_image(np.array([100.0, 120.0]), -0.05))

        sphere = DoubleSphereCamera(640, 480, 250.0, 260.0, 320.0, 240.0, -0.2, 0.6)
        invalid = np.ascontiguousarray([[0.2, 0.0, -0.3]], dtype=np.float64)
        _uv, depth = sphere.project(invalid)
        self.assertTrue(np.isnan(np.asarray(depth)).all())
        valid_ray = np.ascontiguousarray([[1.0, 0.0, -0.05]], dtype=np.float64)
        _uv_ok, depth_ok = sphere.project(valid_ray)
        self.assertLess(float(np.asarray(depth_ok).reshape(-1)[0]), 0.0)

    def test_camera_type_selects_the_model(self):
        from pyslam.io.dataset_types import SensorType
        from pyslam.slam import camera_from_config as active_camera_from_config
        from pyslam.slam.camera import camera_from_config

        constructors = {
            "PinholeCamera": PinholeCamera,
            "KannalaBrandt4Camera": KannalaBrandt4Camera,
            "PinholeRadtan8Camera": PinholeRadtan8Camera,
            "DoubleSphereCamera": DoubleSphereCamera,
        }

        def make_config(camera_type=None, sensor=SensorType.MONOCULAR, **extra):
            settings = {
                "Camera.width": 640,
                "Camera.height": 480,
                "Camera.fx": 300.0,
                "Camera.fy": 310.0,
                "Camera.cx": 320.0,
                "Camera.cy": 240.0,
                "Camera.k1": -0.02,
                "Camera.k2": 0.005,
                "Camera.p1": 0.001,
                "Camera.p2": -0.0005,
                "Camera.k3": -0.001,
                "Camera.k4": 0.0002,
                "Camera.k5": 0.01,
                "Camera.k6": -0.004,
                "Camera.xi": -0.2,
                "Camera.alpha": 0.6,
                "Camera.fps": 30,
            }
            settings.update(extra)
            if camera_type is not None:
                settings["Camera.type"] = camera_type

            class _Config:
                cam_settings = settings
                dataset_settings = {
                    "sensor_type": "stereo" if sensor == SensorType.STEREO else "mono"
                }
                sensor_type = sensor

                @property
                def DistCoef(self):
                    return np.array(
                        [
                            settings["Camera.k1"],
                            settings["Camera.k2"],
                            settings["Camera.p1"],
                            settings["Camera.p2"],
                            settings["Camera.k3"],
                        ],
                        dtype=np.float64,
                    )

                def to_json(self):
                    return {
                        "cam_settings": self.cam_settings,
                        "dataset_settings": self.dataset_settings,
                    }

            return _Config()

        point = np.ascontiguousarray([[0.2, -0.1, 1.5]], dtype=np.float64)
        linear = np.array([300.0 * 0.2 / 1.5 + 320.0, 310.0 * -0.1 / 1.5 + 240.0])

        pinhole = camera_from_config(make_config(), constructors)
        self.assertEqual(int(pinhole.type), CameraType.PINHOLE.value)
        uv, _depth = pinhole.project(point)
        np.testing.assert_allclose(np.asarray(uv, dtype=np.float64).reshape(2), linear, atol=1e-9)

        active = active_camera_from_config(make_config())
        self.assertEqual(int(active.type), CameraType.PINHOLE.value)

        kannala = camera_from_config(make_config("kannala-brandt4"), constructors)
        self.assertEqual(int(kannala.type), CameraType.KANNALA_BRANDT4.value)
        self.assertTrue(kannala.projects_raw_keypoints())
        np.testing.assert_allclose(np.asarray(kannala.D, dtype=np.float64), [-0.02, 0.005, -0.001, 0.0002])
        kannala_uv, _depth = kannala.project(point)
        self.assertGreater(np.linalg.norm(np.asarray(kannala_uv, dtype=np.float64).reshape(2) - linear), 1e-3)

        rational = camera_from_config(make_config(3), constructors)
        self.assertEqual(int(rational.type), CameraType.PINHOLE_RADTAN8.value)
        self.assertEqual(len(np.asarray(rational.D).reshape(-1)), 8)

        sphere = camera_from_config(make_config("double_sphere"), constructors)
        self.assertEqual(int(sphere.type), CameraType.DOUBLE_SPHERE.value)
        np.testing.assert_allclose(np.asarray(sphere.D, dtype=np.float64), [-0.2, 0.6])

        with self.assertRaises(ValueError):
            camera_from_config(make_config("fisheye"), constructors)
        with self.assertRaises(ValueError):
            camera_from_config(make_config("kannala_brandt4", sensor=SensorType.STEREO), constructors)

    def test_output_buffer_is_written_in_place(self):
        camera = KannalaBrandt4Camera(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002)
        points = np.ascontiguousarray(np.array([[0.2, 0.1, 1.0]], dtype=np.float64))
        uv = np.zeros((1, 2), dtype=np.float64)
        depth = np.zeros((1,), dtype=np.float64)
        valid = np.zeros((1,), dtype=np.float64)
        uv_ptr = uv.__array_interface__["data"][0]
        camera.project_into(points, uv, depth, valid)
        self.assertEqual(uv.__array_interface__["data"][0], uv_ptr)
        self.assertGreater(abs(uv[0, 0]) + abs(uv[0, 1]), 0.0)


    def test_python_mono_factors_match_the_camera_model(self):
        import g2o
        import gtsam
        import cpp_core

        from pyslam.slam.camera_factors import (
            camera_projection_factor,
            camera_resectioning_factor,
            reprojection_measurement,
        )

        camera = PinholeCamera()
        camera.width = 640
        camera.height = 480
        camera.fx = 500.0
        camera.fy = 510.0
        camera.cx = 320.0
        camera.cy = 240.0
        camera.set_intrinsic_matrices()
        point = np.array([0.2, -0.1, 2.0], dtype=np.float64)
        uv, valid, _jacobian = camera.project_with_jacobian(np.ascontiguousarray(point))
        self.assertTrue(valid)
        measured = uv + np.array([1.0, -0.5])

        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(
            g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
        )
        pose = g2o.VertexSE3Expmap()
        pose.set_id(0)
        pose.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros(3)))
        optimizer.add_vertex(pose)
        landmark = g2o.VertexSBAPointXYZ()
        landmark.set_id(1)
        landmark.set_estimate(point.copy())
        landmark.set_marginalized(True)
        optimizer.add_vertex(landmark)
        edge = cpp_core.EdgeCameraProjectXYZ(camera)
        edge.set_vertex(0, landmark)
        edge.set_vertex(1, pose)
        edge.set_measurement(measured)
        edge.set_information(np.eye(2))
        optimizer.add_edge(edge)
        optimizer.initialize_optimization()
        optimizer.compute_active_errors()
        np.testing.assert_allclose(np.asarray(edge.error()).reshape(2), measured - uv, atol=1e-9)
        self.assertAlmostEqual(edge.chi2(), float(np.dot(measured - uv, measured - uv)), places=8)
        self.assertTrue(edge.is_depth_positive())

        pose_optimizer = g2o.SparseOptimizer()
        pose_optimizer.set_algorithm(
            g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
        )
        pose_only = g2o.VertexSE3Expmap()
        pose_only.set_id(0)
        pose_only.set_estimate(g2o.SE3Quat(np.eye(3), np.zeros(3)))
        pose_optimizer.add_vertex(pose_only)
        pose_edge = cpp_core.EdgeCameraProjectXYZOnlyPose(camera)
        pose_edge.Xw = point.copy()
        pose_edge.set_vertex(0, pose_only)
        pose_edge.set_measurement(uv)
        pose_edge.set_information(np.eye(2))
        pose_optimizer.add_edge(pose_edge)
        pose_optimizer.initialize_optimization()
        pose_optimizer.compute_active_errors()
        np.testing.assert_allclose(np.asarray(pose_edge.error()).reshape(2), np.zeros(2), atol=1e-9)

        values = gtsam.Values()
        pose_key = gtsam.symbol("x", 0)
        point_key = gtsam.symbol("l", 0)
        values.insert(pose_key, gtsam.Pose3())
        values.insert(point_key, gtsam.Point3(point[0], point[1], point[2]))
        noise = gtsam.noiseModel.Unit.Create(2)
        landmark = gtsam.Point3(point[0], point[1], point[2])
        kannala = KannalaBrandt4Camera(640, 480, 500.0, 510.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002)
        kannala_uv, kannala_valid, _ = kannala.project_with_jacobian(np.ascontiguousarray(point))
        self.assertTrue(kannala_valid)
        self.assertGreater(np.linalg.norm(kannala_uv - uv), 1e-3)
        offset = np.array([1.0, -0.5])
        factor = camera_projection_factor(kannala_uv + offset, noise, pose_key, point_key, kannala)
        self.assertEqual(type(factor).__module__, "cpp_core")
        self.assertAlmostEqual(factor.error(values), 0.5 * float(offset @ offset), places=8)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(factor)
        self.assertAlmostEqual(graph.error(values), factor.error(values), places=12)
        resection, _ = camera_resectioning_factor(noise, pose_key, kannala, kannala_uv, landmark)
        self.assertAlmostEqual(resection.error(values), 0.0, places=10)

        class _Frame:
            def __init__(self, model):
                self.camera = model
                self.kps = np.array([[10.0, 20.0]])
                self.kpsu = np.array([[11.0, 21.0]])

            def projection_keypoints(self):
                return self.kps

        raw = _Frame(kannala)
        np.testing.assert_allclose(reprojection_measurement(raw, 0, False), raw.kps[0])
        np.testing.assert_allclose(reprojection_measurement(raw, 0, True), raw.kps[0])
        linear_frame = _Frame(camera)
        np.testing.assert_allclose(reprojection_measurement(linear_frame, 0, False), linear_frame.kpsu[0])

    def test_models_match_opencv(self):
        import cv2

        kannala = KannalaBrandt4Camera(
            640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002
        )
        points = np.array(
            [[0.2, -0.1, 1.5], [0.4, 0.3, 2.0], [-0.5, 0.2, 0.8], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        kannala_uv, _depth = kannala.project(np.ascontiguousarray(points))
        opencv_kb, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            np.array([[300.0, 0.0, 320.0], [0.0, 310.0, 240.0], [0.0, 0.0, 1.0]]),
            np.array([-0.02, 0.005, -0.001, 0.0002]),
        )
        np.testing.assert_allclose(np.asarray(kannala_uv), opencv_kb.reshape(-1, 2), atol=1e-8)

        rational = PinholeRadtan8Camera(
            640, 480, 400.0, 410.0, 320.0, 240.0, -0.1, 0.02, 0.001, -0.0005, 0.003, 0.01, -0.004, 0.001
        )
        rational_uv, _depth = rational.project(np.ascontiguousarray(points))
        opencv_radtan, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            np.array([[400.0, 0.0, 320.0], [0.0, 410.0, 240.0], [0.0, 0.0, 1.0]]),
            np.array([-0.1, 0.02, 0.001, -0.0005, 0.003, 0.01, -0.004, 0.001]),
        )
        np.testing.assert_allclose(np.asarray(rational_uv), opencv_radtan.reshape(-1, 2), atol=1e-8)

    def test_singular_rational_jacobian_is_finite(self):
        camera = PinholeRadtan8Camera(640, 480, 500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
        _uv, valid, jacobian = camera.project_with_jacobian(np.ascontiguousarray([1.0, 0.0, 1.0]))
        self.assertFalse(valid)
        self.assertTrue(np.all(np.isfinite(jacobian)))
        np.testing.assert_allclose(jacobian, np.zeros((2, 3)), atol=0.0)

    def test_dataset_pinhole_matches_linear_projection_and_legacy_edge(self):
        import g2o
        import cpp_core

        from pyslam.slam import camera_from_config
        from pyslam.slam.camera import CameraUtils
        from pyslam.slam.camera import PinholeCamera as PythonPinholeCamera

        settings_dir = os.path.join(kRootFolder, "settings")
        checked = 0
        for name in sorted(os.listdir(settings_dir)):
            if not name.endswith(".yaml"):
                continue
            settings = _read_camera_settings(os.path.join(settings_dir, name))
            if "Camera.fx" not in settings:
                continue
            camera = camera_from_config(_SettingsConfig(settings))
            self.assertEqual(int(camera.type), CameraType.PINHOLE.value, name)
            point = np.ascontiguousarray([[0.3, -0.2, 4.0]], dtype=np.float64)
            uv, _depth = camera.project(point)
            expected = np.array(
                [
                    float(settings["Camera.fx"]) * 0.3 / 4.0 + float(settings["Camera.cx"]),
                    float(settings["Camera.fy"]) * -0.2 / 4.0 + float(settings["Camera.cy"]),
                ]
            )
            np.testing.assert_allclose(np.asarray(uv, dtype=np.float64).reshape(2), expected, atol=1e-8)
            checked += 1
        self.assertGreaterEqual(checked, 8)

        reference = PythonPinholeCamera(None)
        reference.width = 1241
        reference.height = 376
        reference.fx = 718.856
        reference.fy = 718.856
        reference.cx = 607.1928
        reference.cy = 185.2157
        reference.set_intrinsic_matrices()
        camera = PinholeCamera()
        camera.width = reference.width
        camera.height = reference.height
        camera.fx = reference.fx
        camera.fy = reference.fy
        camera.cx = reference.cx
        camera.cy = reference.cy
        camera.set_intrinsic_matrices()

        rng = np.random.default_rng(0)
        points = np.column_stack(
            (
                rng.uniform(-8.0, 8.0, 4096),
                rng.uniform(-2.0, 2.0, 4096),
                rng.uniform(2.0, 40.0, 4096),
            )
        ).astype(np.float64)
        points = np.ascontiguousarray(points)
        uv = np.empty((points.shape[0], 2), dtype=np.float64)
        depth = np.empty((points.shape[0],), dtype=np.float64)
        valid = np.empty((points.shape[0],), dtype=np.float64)
        K = np.array(reference.K, dtype=np.float64)

        def time_calls(function, repeats=12):
            for _ in range(2):
                function()
            started = time.perf_counter()
            for _ in range(repeats):
                function()
            return (time.perf_counter() - started) / repeats

        def project_cpp():
            camera.project_into(points, uv, depth, valid)

        def project_numba():
            return CameraUtils.project_numba(points, K)

        project_cpp()
        numba_uv, numba_depth = project_numba()
        np.testing.assert_allclose(uv, numba_uv, atol=1e-9)
        np.testing.assert_allclose(depth, numba_depth, atol=1e-9)
        self.assertTrue(np.all(valid > 0.5))
        cpp_seconds = time_calls(project_cpp)
        numba_seconds = time_calls(project_numba)
        self.assertLess(
            cpp_seconds,
            4.0 * numba_seconds,
            f"pinhole project_into took {cpp_seconds:.4f}s vs numba {numba_seconds:.4f}s",
        )

        true_points = np.column_stack(
            (
                rng.uniform(-3.0, 3.0, 80),
                rng.uniform(-1.0, 1.0, 80),
                rng.uniform(4.0, 20.0, 80),
            )
        )
        measurements = np.column_stack(
            (
                camera.fx * true_points[:, 0] / true_points[:, 2] + camera.cx,
                camera.fy * true_points[:, 1] / true_points[:, 2] + camera.cy,
            )
        )
        angle = 0.02
        rotation = np.array(
            [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        )
        translation = np.array([0.05, -0.02, 0.03])
        perturbed = (rotation @ true_points.T).T + translation

        def optimize(make_edge):
            optimizer = g2o.SparseOptimizer()
            optimizer.set_algorithm(
                g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
            )
            pose = g2o.VertexSE3Expmap()
            pose.set_id(0)
            pose.set_estimate(g2o.SE3Quat(rotation, translation))
            optimizer.add_vertex(pose)
            landmarks = []
            for index, (point, measured) in enumerate(zip(perturbed, measurements)):
                landmark = g2o.VertexSBAPointXYZ()
                landmark.set_id(index + 1)
                landmark.set_estimate(point.copy())
                landmark.set_marginalized(True)
                optimizer.add_vertex(landmark)
                landmarks.append(landmark)
                edge = make_edge()
                edge.set_vertex(0, landmark)
                edge.set_vertex(1, pose)
                edge.set_measurement(measured)
                edge.set_information(np.eye(2))
                optimizer.add_edge(edge)
            optimizer.initialize_optimization()
            optimizer.optimize(8)
            estimate = pose.estimate()
            solved = np.array([np.array(landmark.estimate()) for landmark in landmarks])
            return estimate.rotation().matrix(), np.array(estimate.translation()), solved

        def legacy_edge():
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.fx = camera.fx
            edge.fy = camera.fy
            edge.cx = camera.cx
            edge.cy = camera.cy
            return edge

        def model_edge():
            return cpp_core.EdgeCameraProjectXYZ(camera)

        started = time.perf_counter()
        legacy_R, legacy_t, legacy_points = optimize(legacy_edge)
        legacy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model_R, model_t, model_points = optimize(model_edge)
        model_seconds = time.perf_counter() - started
        np.testing.assert_allclose(model_R, legacy_R, atol=1e-6)
        np.testing.assert_allclose(model_t, legacy_t, atol=1e-6)
        np.testing.assert_allclose(model_points, legacy_points, atol=1e-5)

        def reproject(rotation_cw, translation_cw, landmarks):
            camera_points = (rotation_cw @ landmarks.T).T + translation_cw
            return np.column_stack(
                (
                    camera.fx * camera_points[:, 0] / camera_points[:, 2] + camera.cx,
                    camera.fy * camera_points[:, 1] / camera_points[:, 2] + camera.cy,
                )
            )

        np.testing.assert_allclose(reproject(model_R, model_t, model_points), measurements, atol=1e-4)
        np.testing.assert_allclose(reproject(legacy_R, legacy_t, legacy_points), measurements, atol=1e-4)
        self.assertLess(
            model_seconds,
            4.0 * max(legacy_seconds, 1e-4),
            f"pinhole camera edge took {model_seconds:.4f}s vs EdgeSE3ProjectXYZ {legacy_seconds:.4f}s",
        )


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _kb4_rig():
    """Wide-angle KB4 stereo rig (10 cm baseline, slightly toed-in right camera)."""
    left = KannalaBrandt4Camera(512, 512, 190.0, 190.0, 256.0, 255.0, 0.003, 0.003, -0.002, 0.0004)
    right = KannalaBrandt4Camera(512, 512, 191.0, 190.5, 255.0, 257.0, 0.004, 0.002, -0.001, 0.0003)
    T_rl = np.eye(4)
    T_rl[:3, :3] = _rotation_y(0.02)
    T_rl[:3, 3] = [-0.10, 0.002, 0.001]
    left.right_camera = right
    left.T_right_left = T_rl
    return left, right, T_rl


def _project_valid(camera, points):
    uv, depth = camera.project(np.ascontiguousarray(points, dtype=np.float64))
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 512) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
    return uv, inside & np.isfinite(np.asarray(depth, dtype=np.float64).reshape(-1))


@unittest.skipUnless(CPP_AVAILABLE, "C++ core module is not built")
class TestNativeStereoRigMatching(unittest.TestCase):
    """EpipolarMatcher.search_stereo_rig: epipolar-band stereo matching on raw fisheye pixels."""

    def setUp(self):
        import cpp_core

        self.match = cpp_core.EpipolarMatcher.search_stereo_rig
        self.left, self.right, self.T_rl = _kb4_rig()
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(400, 3))
        directions[:, 2] = np.abs(directions[:, 2]) + 0.3
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = directions * rng.uniform(0.8, 6.0, size=(400, 1))
        uv_l, ok_l = _project_valid(self.left, points)
        points_r = points @ self.T_rl[:3, :3].T + self.T_rl[:3, 3]
        uv_r, ok_r = _project_valid(self.right, points_r)
        keep = np.flatnonzero(ok_l & ok_r)[:80]
        self.assertGreater(len(keep), 60)
        self.points = points[keep]
        self.kps = uv_l[keep].astype(np.float32)
        self.kps_r = uv_r[keep].astype(np.float32)
        self.des = rng.integers(0, 256, size=(len(keep), 32), dtype=np.uint8)
        self.sigmas2 = [1.2 ** (2 * level) for level in range(8)]

    def _run(self, kps_r, des_r, octaves=None, octaves_r=None):
        n, m = len(self.kps), len(kps_r)
        return self.match(
            self.left, np.ascontiguousarray(self.kps), octaves or [0] * n, self.des,
            np.ascontiguousarray(kps_r, dtype=np.float32), octaves_r or [0] * m, des_r,
            100.0, 0.8, self.sigmas2,
        )

    def test_recovers_true_matches_and_depths(self):
        order = np.random.default_rng(4).permutation(len(self.kps_r))
        idxs, idxs_r, depths = self._run(self.kps_r[order], self.des[order])
        idxs, idxs_r = np.asarray(idxs), np.asarray(idxs_r)
        self.assertGreaterEqual(len(idxs), int(0.95 * len(self.kps)))
        np.testing.assert_array_equal(order[idxs_r], idxs)
        np.testing.assert_allclose(depths, self.points[idxs, 2], rtol=0.02)
        self.assertEqual(len(set(idxs_r.tolist())), len(idxs_r))

    def test_rejects_candidates_off_the_epipolar_curve(self):
        # Near the image centre the curves of a horizontal rig are near-horizontal: shifting the
        # true right keypoint 8 px vertically moves it out of the band.
        k = int(np.argmin(np.linalg.norm(self.kps - np.array([256.0, 255.0]), axis=1)))
        kps_r = self.kps_r.copy()
        kps_r[k] += np.array([0.0, 8.0], dtype=np.float32)
        idxs, idxs_r, _ = self._run(kps_r, self.des.copy())
        self.assertNotIn(k, list(idxs))
        self.assertGreaterEqual(len(idxs), int(0.95 * (len(self.kps) - 1)))

    def test_rejects_candidates_beyond_infinity(self):
        # On the epipolar great circle but past the point at infinity: no positive depth explains it.
        f = self.points[0]
        R, t = self.T_rl[:3, :3], self.T_rl[:3, 3]
        at_infinity = R @ (f / np.linalg.norm(f))
        normal = np.cross(t, at_infinity)
        normal /= np.linalg.norm(normal)
        away = np.cross(normal, at_infinity)  # in-plane, pointing away from t
        if np.dot(away, t) > 0:
            away = -away
        beyond = np.cos(0.05) * at_infinity + np.sin(0.05) * away
        uv_bad, ok = _project_valid(self.right, beyond[None, :])
        self.assertTrue(ok[0])
        kps_r = self.kps_r.copy()
        kps_r[0] = uv_bad[0]
        idxs, _, _ = self._run(kps_r, self.des.copy())
        self.assertNotIn(0, list(idxs))

    def test_octave_gap_and_one_to_one(self):
        octaves_r = [0] * len(self.kps_r)
        octaves_r[1] = 3
        kps_r = np.vstack([self.kps_r, self.kps_r[2:3]])
        des_r = np.vstack([self.des, self.des[2:3]])
        octaves_r.append(0)
        idxs, idxs_r, _ = self._run(kps_r, des_r, octaves_r=octaves_r)
        self.assertNotIn(1, list(idxs))  # octave difference 3 > 1
        # Two identical right candidates for left 2 fail the ratio test.
        self.assertNotIn(2, list(idxs))
        self.assertEqual(len(set(idxs_r)), len(idxs_r))

    def test_without_rig_returns_nothing(self):
        mono = KannalaBrandt4Camera(512, 512, 190.0, 190.0, 256.0, 255.0, 0.0, 0.0, 0.0, 0.0)
        result = self.match(mono, np.ascontiguousarray(self.kps), [0] * len(self.kps), self.des,
                            np.ascontiguousarray(self.kps_r), [0] * len(self.kps_r), self.des,
                            100.0, 0.8, self.sigmas2)
        self.assertEqual([len(part) for part in result], [0, 0, 0])


@unittest.skipUnless(CPP_AVAILABLE, "C++ core module is not built")
class TestBearingEpipolarMask(unittest.TestCase):
    """Triangulation-search epipolar test for cameras whose keypoints are raw model pixels."""

    def test_true_correspondences_pass_and_offsets_fail(self):
        from pyslam.slam.camera import KannalaBrandt4Camera as PyKannalaBrandt4Camera
        from pyslam.slam.geometry_matchers import _bearing_epipolar_mask

        camera = PyKannalaBrandt4Camera(fx=190.0, fy=190.0, cx=256.0, cy=255.0, width=512,
                                        height=512, k=(0.003, 0.003, -0.002, 0.0004))
        rng = np.random.default_rng(5)
        directions = rng.normal(size=(300, 3))
        directions[:, 2] = np.abs(directions[:, 2]) + 0.4
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points_w = directions * rng.uniform(1.0, 5.0, size=(300, 1))
        R2 = _rotation_y(0.15)
        t2 = np.array([-0.3, 0.05, 0.02])

        class _KF:
            def __init__(self, Rcw, tcw, kps):
                self.camera, self._R, self._t = camera, Rcw, tcw
                self.kps = kps
                self.octaves = np.zeros(len(kps), dtype=np.int64)

            def Rcw(self):
                return self._R

            def tcw(self):
                return self._t

        uv1, ok1 = _project_valid(camera, points_w)
        uv2, ok2 = _project_valid(camera, points_w @ R2.T + t2)
        keep = np.flatnonzero(ok1 & ok2)
        kf1 = _KF(np.eye(3), np.zeros(3), uv1[keep])
        idxs = np.arange(len(keep))
        scale_factors = np.ones(8)
        sigmas2 = np.ones(8)

        kf2 = _KF(R2, t2, uv2[keep])
        good = _bearing_epipolar_mask(kf1, kf2, idxs, idxs, scale_factors, sigmas2)
        self.assertGreater(good.mean(), 0.95)  # the few misses sit near the epipole

        # Offsetting every kf2 keypoint 6 px across its epipolar curve must fail (band ~2 px).
        bearings1 = points_w[keep] / np.linalg.norm(points_w[keep], axis=1)[:, None]
        normals = np.cross(t2, bearings1 @ R2.T)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        bearings2 = (points_w[keep] @ R2.T + t2)
        bearings2 /= np.linalg.norm(bearings2, axis=1)[:, None]
        uv_a, _ = camera.project(bearings2)
        uv_b, _ = camera.project(bearings2 + 1e-4 * normals)
        across = (uv_b - uv_a) / np.linalg.norm(uv_b - uv_a, axis=1)[:, None]
        kf2_off = _KF(R2, t2, uv2[keep] + 6.0 * across)
        off = _bearing_epipolar_mask(kf1, kf2_off, idxs, idxs, scale_factors, sigmas2)
        self.assertLess(off.mean(), 0.05)


def _read_camera_settings(path):
    settings = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.split("#", 1)[0].strip()
            if not text.startswith("Camera.") or ":" not in text:
                continue
            key, value = text.split(":", 1)
            value = value.strip()
            if not value:
                continue
            try:
                settings[key.strip()] = float(value)
            except ValueError:
                settings[key.strip()] = value
    return settings


class _SettingsConfig:
    def __init__(self, settings):
        self.cam_settings = settings
        self.dataset_settings = {"sensor_type": "mono"}

    def to_json(self):
        return {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}


if __name__ == "__main__":
    unittest.main()
