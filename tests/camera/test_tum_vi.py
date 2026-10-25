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
* Synthetic TUM-VI layout and independent OpenCV checks for both stereo paths.
"""

import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from camera_test_support import ROOT, SettingsConfig
import cv2
import numpy as np
import yaml

from pyslam.io.tum_vi_dataset import TumViDataset, prepare_calibration
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.slam.camera import camera_from_config, camera_from_json
from pyslam.slam.cpp import cpp_module, python_module
from pyslam.slam.frame import Frame
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.slam.camera_factors import (
    camera_projection_factor, camera_resectioning_factor,
    stereo_camera_projection_factor, stereo_camera_resectioning_factor,
    g2o_stereo_camera_edge,
)


def config_for(mode):
    settings = yaml.safe_load((ROOT / "settings/TUM_VI.yaml").read_text())
    config = SettingsConfig(settings, "stereo")
    config.dataset_settings.update(type="tum_vi", stereo_mode=mode)
    return config


def opencv_pixels(settings, index, points):
    prefix = f"Camera{index}."
    fx, fy, cx, cy = [settings[prefix + field] for field in ("fx", "fy", "cx", "cy")]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    D = np.array([settings[prefix + f"k{i}"] for i in range(1, 5)])
    return cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


class TestTumVi(unittest.TestCase):
    def test_frame_json_roundtrip_keeps_native_rig_in_both_backends(self):
        config = config_for("native")
        prepare_calibration(config)
        points = np.array([[0.3, -0.1, 2.]])
        for module in (python_module, cpp_module):
            with self.subTest(backend=module):
                camera = camera_from_config(config, module.classes)
                frame = module.Frame(camera=camera, img=None, timestamp=1., img_id=1)
                frame.kps = camera.project(points)[0].astype(np.float32)
                frame.kpsu = camera.undistort_points(frame.kps)
                frame.kpsn = camera.unproject_points(frame.kps)
                frame.octaves = np.zeros(1, dtype=np.int32)
                frame.des = np.zeros((1, 32), dtype=np.uint8)
                frame.points = np.array([None], dtype=object)
                frame.outliers = np.zeros(1, dtype=bool)
                restored = module.Frame.from_json(frame.to_json())
                self.assertTrue(restored.camera.is_native_stereo())
                np.testing.assert_allclose(restored.camera.project_stereo(points)[0],
                                           camera.project_stereo(points)[0], atol=1e-9)

    def test_calibration_rejects_nonrigid_transforms(self):
        for key in ("Stereo.T_c1_c2", "TUMVI.T_body_left"):
            for value in (np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]), np.eye(3)):
                with self.subTest(key=key, value=value):
                    config = config_for("rectified")
                    config.cam_settings[key] = value.tolist()
                    with self.assertRaisesRegex(ValueError, key.replace('.', r'\.')):
                        prepare_calibration(config)

    def test_mono_keeps_unpaired_left_exposures_and_groundtruth_indices(self):
        from pyslam.io.dataset_types import SensorType
        from pyslam.io.tum_vi_ground_truth import TumViGroundTruth

        with tempfile.TemporaryDirectory() as tmp:
            sequence = Path(tmp) / "sequence"
            left = sequence / "mav0/cam0"
            (left / "data").mkdir(parents=True)
            times = [1_000_000_000, 1_050_000_000, 1_100_000_000]
            (left / "data.csv").write_text("".join(f"{t},{i}.png\n" for i, t in enumerate(times)))
            for i in range(3):
                cv2.imwrite(str(left / f"data/{i}.png"), np.full((512, 512), 30 + i, np.uint8))
            mocap = sequence / "mav0/mocap0"
            mocap.mkdir()
            (mocap / "data.csv").write_text("".join(
                f"{t},{i},0,0,1,0,0,0\n" for i, t in enumerate(times)))
            for mode in ("native", "rectified"):
                with self.subTest(mode=mode):
                    config = config_for(mode)
                    config.sensor_type = SensorType.MONOCULAR
                    config.dataset_settings.update(sensor_type="mono", base_path=tmp, name="sequence",
                                                   start_frame_id=1, clahe=False)
                    config.cam_stereo_settings = None
                    dataset = dataset_factory(config)
                    self.assertEqual(dataset.scale_viewer_3d, 0.02)
                    self.assertEqual(dataset.timestamps_ns, times)
                    self.assertEqual(dataset.getImageColor(0).shape, (512, 512, 3))
                    self.assertAlmostEqual(dataset.getTimestamp(), 1.05)
                    self.assertIsNone(dataset.getImageRight(0))
                    for constructors in (None, cpp_module.classes):
                        camera = camera_from_config(config, constructors)
                        self.assertFalse(camera.is_native_stereo())
                        self.assertEqual(int(camera.type.value) if constructors is None else int(camera.type),
                                         2 if mode == "native" else 1)
                    gt = TumViGroundTruth(config.dataset_settings)
                    self.assertEqual(len(gt.image_timestamps), 3)
                    sample = gt.getTimestampPositionAndAbsoluteScale(0)
                    self.assertAlmostEqual(sample[0], 1.05)
                    self.assertAlmostEqual(sample[1], 1 + config.dataset_settings["T_body_camera"][0][3])

    def test_python_core_selection_in_fresh_process(self):
        script = """
import sys
sys.path.insert(0, 'tests/camera')
from camera_test_support import SettingsConfig, ROOT
from pyslam.config_parameters import Parameters
Parameters.USE_CPP_CORE = False
from pyslam.slam import camera_from_config
from pyslam.io.tum_vi_dataset import prepare_calibration
from pyslam.slam.camera_factors import camera_projection_factor
import yaml, numpy as np, gtsam
c = SettingsConfig(yaml.safe_load((ROOT/'settings/TUM_VI.yaml').read_text()), 'stereo')
c.dataset_settings.update(type='tum_vi', stereo_mode='native')
prepare_calibration(c)
camera = camera_from_config(c)
assert type(camera).__module__ == 'pyslam.slam.camera'
point = np.array([0.2, -0.1, 2.0])
uv = camera.project(point)[0]
factor = camera_projection_factor(uv, gtsam.noiseModel.Unit.Create(2), 0, 1, camera)
values = gtsam.Values()
values.insert(0, gtsam.Pose3())
values.insert(1, point)
assert factor.error(values) < 1e-12
assert camera.project_stereo(point[None])[0].shape == (1, 3)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_projection_triangulation_and_jacobians(self):
        config = config_for("native")
        prepare_calibration(config)
        points = np.array([[0.3, -0.1, 2.0], [-0.6, 0.4, 3.0], [0.0, 0.0, 1.5]])
        T = np.asarray(config.cam_settings["Stereo.T_right_left"])
        right_points = points @ T[:3, :3].T + T[:3, 3]
        left = np.ascontiguousarray(opencv_pixels(config.cam_settings, 1, points))
        right = np.ascontiguousarray(opencv_pixels(config.cam_settings, 2, right_points))
        for constructors in (None, cpp_module.classes):
            camera = camera_from_config(config, constructors)
            self.assertTrue(camera.is_native_stereo())
            uvur, z = camera.project_stereo(points)
            np.testing.assert_allclose(uvur[:, :2], left, atol=1e-9)
            np.testing.assert_allclose(uvur[:, 2], right[:, 0], atol=1e-9)
            recovered, valid = camera.triangulate_stereo(left, right)
            np.testing.assert_array_equal(valid, np.ones(3))
            np.testing.assert_allclose(recovered, points, atol=1e-9)
            bad_right = right.copy()
            bad_right[:, 1] += 30.0
            _bad, valid = camera.triangulate_stereo(left, bad_right)
            self.assertTrue(np.all(valid == 0.0))
            _, valid, J = camera.project_stereo_with_jacobian(points[0])
            self.assertTrue(valid)
            numeric = np.zeros((3, 3))
            for axis in range(3):
                step = np.zeros(3)
                step[axis] = 1e-6
                numeric[:, axis] = (camera.project_stereo((points[0] + step)[None])[0][0]
                                    - camera.project_stereo((points[0] - step)[None])[0][0]) / 2e-6
            np.testing.assert_allclose(J, numeric, atol=1e-6)
            if constructors is None:
                restored = camera_from_json(camera.to_json())
            else:
                import pickle
                restored = pickle.loads(pickle.dumps(camera))
                json_restored = type(camera).from_json(camera.to_json())
                self.assertTrue(json_restored.is_native_stereo())
                np.testing.assert_allclose(json_restored.project_stereo(points)[0], uvur, atol=1e-9)
            self.assertTrue(restored.is_native_stereo())
            np.testing.assert_allclose(restored.project_stereo(points)[0], uvur, atol=1e-9)

    def test_dataset_pairing_start_index_and_rectification(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = Path(tmp) / "sequence"
            times = [1_500_000_000_000_000_001, 1_500_000_000_050_000_001]
            # A missing right exposure must not shift later stereo associations.
            for side in (0, 1):
                folder = sequence / "mav0" / f"cam{side}"
                (folder / "data").mkdir(parents=True)
                records = [(times[0], "first.png"), (times[0] + 1, "unmatched.png"), (times[1], "last.png")]
                if side == 1:
                    records.pop(1)
                with (folder / "data.csv").open("w") as handle:
                    handle.write("#timestamp [ns],filename\n")
                    for timestamp, name in records:
                        handle.write(f"{timestamp},{name}\n")
                        cv2.imwrite(str(folder / "data" / name), np.full((512, 512), 30 + side, dtype=np.uint8))
            for mode in ("native", "rectified"):
                config = config_for(mode)
                config.dataset_settings.update(base_path=tmp, name="sequence", start_frame_id=1)
                config.cam_stereo_settings = None
                dataset = dataset_factory(config)
                self.addCleanup(dataset.close)
                self.assertIsInstance(dataset, TumViDataset)
                self.assertEqual(dataset.timestamps_ns, times)
                self.assertEqual(dataset.num_frames, 2)
                color = dataset.getImageColor(0)
                self.assertEqual(color.shape, (512, 512, 3))
                np.testing.assert_array_equal(color, cv2.cvtColor(dataset.getImage(1), cv2.COLOR_GRAY2RGB))
                timestamp = dataset.getTimestamp()
                self.assertAlmostEqual(timestamp, times[1] * 1e-9)
                right = dataset.getImageColorRight(0)
                self.assertEqual(right.shape, (512, 512, 3))
                np.testing.assert_array_equal(right, cv2.cvtColor(dataset.getImageRight(1), cv2.COLOR_GRAY2RGB))
                self.assertEqual(dataset.getTimestamp(), timestamp)
                if mode == "native":
                    self.assertGreater(float(dataset.getImage(0).mean()), 30.0)
                self.assertIsNone(dataset.getImageColor(1))
                camera = camera_from_config(config, cpp_module.classes)
                self.assertEqual(camera.is_native_stereo(), mode == "native")
                if mode == "rectified":
                    self.assertGreater(camera.bf, 0)
                    self.assertEqual(int(camera.type), 1)
                    self.assertTrue(np.allclose(camera.D, 0))
                    self.assertIs(prepare_calibration(config), dataset.maps)
            config = config_for("native")
            config.dataset_settings.update(base_path=tmp, name="sequence", clahe=False)
            config.cam_stereo_settings = None
            dataset = dataset_factory(config)
            np.testing.assert_array_equal(dataset.getImage(0), np.full((512, 512), 30, dtype=np.uint8))

    def test_config_roundtrip_recomputes_mode_and_caches(self):
        from pyslam.config import Config
        original = config_for("native")
        original.dataset_settings.update(base_path="/tmp/unused", name="sequence")
        prepare_calibration(original)
        restored = Config()
        restored.from_json(original.to_json())
        self.assertTrue(camera_from_config(restored).is_native_stereo())
        payload = restored.to_json()
        payload["dataset_settings"] = dict(payload["dataset_settings"], stereo_mode="rectified")
        restored.from_json(payload)
        self.assertFalse(camera_from_config(restored).is_native_stereo())
        self.assertIsNotNone(restored._tum_vi_maps)

    def test_native_frame_matching_uses_general_geometry(self):
        config = config_for("native")
        prepare_calibration(config)
        camera = camera_from_config(config)
        points = np.array([[0.2, 0.1, 2.0], [-0.3, -0.1, 3.0]])
        T = camera.T_right_left
        left = opencv_pixels(config.cam_settings, 1, points)
        right = opencv_pixels(config.cam_settings, 2, points @ T[:3, :3].T + T[:3, 3])
        self.assertGreater(np.max(np.abs(left[:, 1] - right[:, 1])), 1.1)
        # Native rigs match along the epipolar curves of the raw fisheye images (the rows differ).
        descriptors = np.array([[0] * 32, [255] * 32], dtype=np.uint8)
        frame = Frame(camera=camera, img=None)
        frame.kps, frame.kps_r = left, right[::-1].copy()
        frame.octaves = frame.octaves_r = np.zeros(2, dtype=np.int32)
        frame.des, frame.des_r = descriptors, descriptors[::-1].copy()
        frame.depths = np.full(2, -1.0)
        frame.kps_ur = np.full(2, -1.0)
        manager = SimpleNamespace(level_sigmas2=np.ones(8, dtype=np.float32))
        with patch.object(FeatureTrackerShared, "feature_manager", manager):
            frame.compute_stereo_matches(np.zeros((512, 512), dtype=np.uint8), np.zeros((512, 512), dtype=np.uint8))
        np.testing.assert_allclose(frame.depths, points[:, 2], atol=1e-5)
        np.testing.assert_allclose(frame.kps_ur, right[:, 0], atol=1e-4)

        native_camera = camera_from_config(config, cpp_module.classes)
        native_frame = cpp_module.Frame(camera=native_camera, img=None)
        native_frame.kps = left.astype(np.float32)
        native_frame.kps_r = right[::-1].astype(np.float32)
        native_frame.octaves = native_frame.octaves_r = np.zeros(2, dtype=np.int32)
        native_frame.des, native_frame.des_r = descriptors, descriptors[::-1].copy()
        native_frame.depths = np.full(2, -1.0, dtype=np.float32)
        native_frame.kps_ur = np.full(2, -1.0, dtype=np.float32)
        native_frame.compute_stereo_matches(np.zeros((512, 512), dtype=np.uint8), np.zeros((512, 512), dtype=np.uint8))
        np.testing.assert_allclose(native_frame.depths, points[:, 2], atol=1e-4)
        np.testing.assert_allclose(native_frame.kps_ur, right[:, 0], atol=1e-4)

    def test_native_g2o_stereo_pose_solve(self):
        import g2o
        config = config_for("native")
        prepare_calibration(config)
        camera = camera_from_config(config)
        rng = np.random.default_rng(7)
        points = rng.uniform([-0.8, -0.6, 2.0], [0.8, 0.6, 5.0], (64, 3))
        measurements = camera.project_stereo(points)[0]
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        ))
        pose = g2o.VertexSE3Expmap()
        pose.set_id(0)
        pose.set_estimate(g2o.SE3Quat(np.eye(3), np.array([0.04, -0.02, 0.03])))
        optimizer.add_vertex(pose)
        edges = []
        for point, measured in zip(points, measurements):
            edge = g2o_stereo_camera_edge(camera, pose_only=True)
            edge.Xw = point
            edge.set_vertex(0, pose)
            edge.set_measurement(measured)
            edge.set_information(np.eye(3))
            optimizer.add_edge(edge)
            edges.append(edge)
        optimizer.initialize_optimization()
        optimizer.optimize(10)
        np.testing.assert_allclose(pose.estimate().translation(), np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(pose.estimate().rotation().matrix(), np.eye(3), atol=1e-7)

    def test_native_optimizer_entry_points(self):
        config = config_for("native")
        prepare_calibration(config)
        rng = np.random.default_rng(9)
        points = rng.uniform([-0.8, -0.6, 2.0], [0.8, 0.6, 5.0], (64, 3))
        resources = cpp_module.FeatureSharedResources
        old_inverse, old_sigmas = resources.inv_level_sigmas2, resources.level_sigmas
        resources.inv_level_sigmas2 = [1.0] * 8
        resources.level_sigmas = [1.0] * 8
        try:
            for module in (cpp_module, python_module):
                camera = camera_from_config(config, module.classes)
                measurements = camera.project_stereo(points)[0]
                manager = SimpleNamespace(inv_level_sigmas2=np.ones(8), level_sigmas=np.ones(8))
                with patch.object(FeatureTrackerShared, "feature_manager", manager):
                    for optimizer in (module.optimizer_g2o, module.optimizer_gtsam):
                        with self.subTest(optimizer=optimizer):
                            frame = module.Frame(camera=camera, img=None)
                            frame.kps = measurements[:, :2].astype(np.float32)
                            frame.kpsu = camera.undistort_points(np.ascontiguousarray(measurements[:, :2])).astype(np.float32)
                            frame.kps_ur = measurements[:, 2].astype(np.float32)
                            frame.octaves = np.zeros(64, dtype=np.int32)
                            frame.points = np.array([
                                module.MapPoint(point, np.array([255, 0, 0], dtype=np.uint8)) for point in points
                            ], dtype=object)
                            frame.outliers = np.zeros(64, dtype=bool)
                            pose = np.eye(4)
                            pose[:3, 3] = [0.04, -0.02, 0.03]
                            frame.update_pose(pose)
                            _error, success, inliers = optimizer.pose_optimization(frame, rounds=10)
                            self.assertTrue(success)
                            self.assertEqual(inliers, 64)
                            np.testing.assert_allclose(frame.pose(), np.eye(4), atol=2e-5)
        finally:
            resources.inv_level_sigmas2, resources.level_sigmas = old_inverse, old_sigmas

    def test_sparse_mocap_quaternion_and_camera_extrinsic(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence = Path(tmp) / "sequence"
            for camera in ("cam0", "cam1"):
                folder = sequence / "mav0" / camera
                folder.mkdir(parents=True)
                (folder / "data.csv").write_text("#timestamp,filename\n1000000000,a.png\n2000000000,b.png\n")
            mocap = sequence / "mav0/mocap0"
            mocap.mkdir()
            (mocap / "data.csv").write_text("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n1000000000,1,2,3,1,0,0,0\n")
            T = np.eye(4)
            T[0, 3] = 0.1
            settings = dict(type="tum_vi", base_path=tmp, name="sequence", groundtruth_file="auto", T_body_camera=T.tolist())
            gt = groundtruth_factory(settings)
            sample = gt.getTimestampPoseAndAbsoluteScale(0)
            np.testing.assert_allclose(sample[1:4], [1.1, 2.0, 3.0])
            np.testing.assert_allclose(sample[4:8], [0, 0, 0, 1])
            self.assertEqual(sample[-1], 1.0)
            self.assertEqual(len(gt.getFull6dTrajectory()[0]), 1)
            from pyslam.io.ground_truth import GroundTruth
            gt.save(tmp)
            restored = GroundTruth.load(tmp)
            self.assertEqual(restored.getNumSamples(), 1)
            for actual, expected in zip(restored.getFull6dTrajectory(), gt.getFull6dTrajectory()):
                np.testing.assert_allclose(actual, expected)
            with self.assertRaises(ValueError):
                gt.getTimestampPoseAndAbsoluteScale(1)
            (mocap / "data.csv").write_text(
                "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
                "1000000000,1,2,3,1,0,0,0\n"
                "1900000000,2,2,3,1,0,0,0\n"
                "2000000000,4,2,3,1,0,0,0\n"
            )
            gt = groundtruth_factory(settings)
            self.assertAlmostEqual(gt.getTimestampPoseAndAbsoluteScale(1)[-1], 3.0)
            converted = Path(tmp) / 'simple.txt'
            gt.convertToSimpleDataset(str(converted))
            samples = np.loadtxt(converted, ndmin=2)
            np.testing.assert_allclose(samples[:, 0], [1., 1.9, 2.])
            np.testing.assert_allclose(samples[:, 1], [1.1, 2.1, 4.1])

    def test_native_gtsam_factors_and_lifetimes(self):
        import gtsam
        config = config_for("native")
        prepare_calibration(config)
        camera = camera_from_config(config)
        point = np.array([0.2, -0.1, 2.0])
        uv = camera.project(point)[0]
        uvur = camera.project_stereo(point[None])[0][0]
        values = gtsam.Values()
        x, l = gtsam.symbol("x", 0), gtsam.symbol("l", 0)
        values.insert(x, gtsam.Pose3())
        values.insert(l, point)
        noise2, noise3 = gtsam.noiseModel.Unit.Create(2), gtsam.noiseModel.Unit.Create(3)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(camera_projection_factor(uv, noise2, x, l, camera))
        graph.add(camera_resectioning_factor(noise2, x, camera, uv, point)[0])
        measured = gtsam.StereoPoint2(uvur[0], uvur[2], uvur[1])
        graph.add(stereo_camera_projection_factor(measured, noise3, x, l, camera))
        graph.add(stereo_camera_resectioning_factor(noise3, x, camera, measured, point)[0])
        del camera
        import gc
        gc.collect()
        self.assertLess(graph.error(values), 1e-12)
        H, b = graph.linearize(values).hessian()
        self.assertTrue(np.isfinite(H).all() and np.isfinite(b).all())

    def test_native_gtsam_stereo_pose_solve(self):
        import gtsam
        config = config_for("native")
        prepare_calibration(config)
        camera = camera_from_config(config)
        rng = np.random.default_rng(7)
        points = rng.uniform([-0.8, -0.6, 2.0], [0.8, 0.6, 5.0], (64, 3))
        measurements = camera.project_stereo(points)[0]
        key = gtsam.symbol("x", 0)
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        initial.insert(key, gtsam.Pose3(gtsam.Rot3(), np.array([0.04, -0.02, 0.03])))
        for point, measured in zip(points, measurements):
            observation = gtsam.StereoPoint2(measured[0], measured[2], measured[1])
            graph.add(stereo_camera_resectioning_factor(
                gtsam.noiseModel.Unit.Create(3), key, camera, observation, point
            )[0])
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        np.testing.assert_allclose(result.atPose3(key).matrix(), np.eye(4), atol=1e-7)

    def test_native_factor_shares_switchable_noise_state(self):
        import gtsam
        import gtsam_factors
        config = config_for("native")
        prepare_calibration(config)
        camera = camera_from_config(config)
        point = np.array([0.2, -0.1, 2.0])
        uv = camera.project(point)[0] + np.array([10.0, 0.0])
        x, l = gtsam.symbol("x", 0), gtsam.symbol("l", 0)
        values = gtsam.Values()
        values.insert(x, gtsam.Pose3())
        values.insert(l, point)
        noise = gtsam_factors.SwitchableRobustNoiseModel(2, 1.0, 1.0)
        factor = camera_projection_factor(uv, noise, x, l, camera)
        robust_error = factor.error(values)
        noise.set_robust_model_active(False)
        self.assertGreater(factor.error(values), robust_error)
        self.assertAlmostEqual(factor.error(values), 50.0, places=8)


if __name__ == "__main__":
    unittest.main()
