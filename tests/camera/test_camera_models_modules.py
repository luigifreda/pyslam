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
* Python-core path (USE_CPP_CORE=False): camera_models, camera_models_g2o, camera_models_gtsam.
*
* Run directly: python tests/camera/test_camera_models_modules.py
"""

import os
import subprocess
import sys
import unittest

from camera_test_support import ROOT  # noqa: E402  (imports pyslam.config)
import numpy as np  # noqa: E402

from pyslam.slam.camera import KannalaBrandt4Camera, import_camera_models  # noqa: E402
from pyslam.slam.camera_factors import (  # noqa: E402
    bound_camera,
    camera_projection_factor,
    camera_resectioning_factor,
    g2o_camera_project_pose_edge,
    g2o_stereo_camera_edge,
    projection_module,
    stereo_camera_resectioning_factor,
)


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _kb4(fx=190.0, cx=256.0):
    return KannalaBrandt4Camera(fx=fx, fy=fx + 0.5, cx=cx, cy=255.0, width=512, height=512,
                                k=(0.003, 0.003, -0.002, 0.0004))


def _kb4_rig():
    left, right = _kb4(), _kb4(fx=191.0, cx=254.0)
    T = np.eye(4)
    T[:3, :3] = _rotation_y(0.02)
    T[:3, 3] = [-0.101, 0.001, 0.002]
    left.right_camera = right
    left.T_right_left = T
    left.b = 0.101
    left.bf = left.b * left.fx
    return left


def _project(camera, points):
    uv = np.empty((len(points), 2))
    depth = np.empty(len(points))
    valid = np.empty(len(points))
    camera.project_into(np.ascontiguousarray(points, dtype=np.float64), uv, depth, valid)
    return uv, valid > 0


def _scene(n, seed, z_range=(1.0, 6.0)):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.5
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * rng.uniform(*z_range, size=(n, 1))


def _pose(angle, t):
    T = np.eye(4)
    T[:3, :3] = _rotation_y(angle)
    T[:3, 3] = t
    return T


class TestModuleSeparation(unittest.TestCase):
    def _loaded_after(self, module):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import pyslam.config; "
                f"sys.path.insert(0, {str(ROOT / 'cpp' / 'lib')!r}); before = set(sys.modules); "
                f"import {module}; "
                "print(','.join(m for m in ('gtsam', 'g2o', 'cpp_core') if m in set(sys.modules) - before))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env=dict(os.environ, PYSLAM_USE_CPP="false"))
        lines = out.stdout.splitlines()
        return set(filter(None, lines[-1].split(","))) if lines else set()

    def test_each_module_loads_only_its_optimizer(self):
        self.assertEqual(self._loaded_after("camera_models"), set())
        self.assertEqual(self._loaded_after("camera_models_g2o"), {"g2o"})
        self.assertEqual(self._loaded_after("camera_models_gtsam"), {"gtsam"})

    def test_python_camera_binds_to_camera_models(self):
        camera = _kb4_rig()
        native = bound_camera(camera)
        self.assertEqual(type(native).__module__, "camera_models")
        self.assertEqual(projection_module(camera), import_camera_models())
        self.assertEqual(type(native.right_camera).__module__, "camera_models")


class TestCameraModelsStereoRig(unittest.TestCase):
    def setUp(self):
        self.camera = _kb4_rig()
        self.native = bound_camera(self.camera)
        T = self.camera.T_right_left
        points = _scene(250, seed=3)
        uv_l, ok_l = _project(self.camera, points)
        uv_r, ok_r = _project(self.camera.right_camera, points @ T[:3, :3].T + T[:3, 3])
        keep = np.flatnonzero(ok_l & ok_r)
        self.points = points[keep]
        self.kps = np.ascontiguousarray(uv_l[keep], dtype=np.float32)
        self.kps_r = np.ascontiguousarray(uv_r[keep], dtype=np.float32)
        rng = np.random.default_rng(7)
        self.des = rng.integers(0, 256, size=(len(keep), 32), dtype=np.uint8)
        self.octaves = np.zeros(len(keep), dtype=np.int32)

    def _match(self, kps_r, des_r, octaves_r):
        return projection_module(self.camera).EpipolarMatcher.search_stereo_rig(
            self.native, self.kps, self.octaves, self.des, kps_r, octaves_r, des_r,
            100.0, 0.8, np.ones(8, dtype=np.float32))

    def test_recovers_correspondences_and_depths(self):
        order = np.arange(len(self.kps))[::-1]
        il, ir, depths = self._match(np.ascontiguousarray(self.kps_r[order]), self.des[order], self.octaves[order])
        self.assertGreater(len(il), 0.95 * len(self.kps))
        np.testing.assert_array_equal(order[ir], il)
        np.testing.assert_allclose(depths, self.points[il, 2], rtol=2e-3)

    def test_rejects_off_epipolar_duplicates(self):
        shifted = np.ascontiguousarray(self.kps_r + np.float32([0.0, 15.0]))
        il, _, _ = self._match(shifted, self.des, self.octaves)
        self.assertLess(len(il), 0.05 * len(self.kps))

    def test_no_rig_returns_nothing(self):
        mono = bound_camera(_kb4())
        il, ir, depths = projection_module(self.camera).EpipolarMatcher.search_stereo_rig(
            mono, self.kps, self.octaves, self.des, self.kps_r, self.octaves, self.des, 100.0, 0.8)
        self.assertEqual((len(il), len(ir), len(depths)), (0, 0, 0))


class TestCameraModelsOptimizerTerms(unittest.TestCase):
    """The shared edges/factors must pull a perturbed pose back to the true one."""

    def setUp(self):
        self.camera = _kb4_rig()
        self.Tcw = _pose(0.1, [0.05, -0.02, 0.1])
        points_c = _scene(60, seed=11)
        self.points_w = (points_c - self.Tcw[:3, 3]) @ self.Tcw[:3, :3]
        self.uv, ok = _project(self.camera, points_c)
        T = self.camera.T_right_left
        uv_r, ok_r = _project(self.camera.right_camera, points_c @ T[:3, :3].T + T[:3, 3])
        keep = ok & ok_r
        self.points_w, self.uv, self.ur = self.points_w[keep], self.uv[keep], uv_r[keep, 0]
        self.Tcw_init = _pose(0.13, [0.09, -0.05, 0.14])

    def _assert_pose(self, Tcw, tol=1e-6):
        np.testing.assert_allclose(Tcw[:3, :3], self.Tcw[:3, :3], atol=tol)
        np.testing.assert_allclose(Tcw[:3, 3], self.Tcw[:3, 3], atol=tol)

    def _g2o_pose(self, make_edge, measurement):
        import g2o
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())))
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(0)
        vertex.set_estimate(g2o.SE3Quat(self.Tcw_init[:3, :3], self.Tcw_init[:3, 3]))
        optimizer.add_vertex(vertex)
        edges = []
        for i, Xw in enumerate(self.points_w):
            edge = make_edge()
            edge.set_vertex(0, vertex)
            m = measurement(i)
            edge.set_measurement(m)
            edge.set_information(np.eye(len(m)))
            edge.Xw = Xw
            optimizer.add_edge(edge)
            edges.append(edge)
        optimizer.initialize_optimization()
        optimizer.optimize(20)
        return vertex.estimate().matrix(), edges

    def test_g2o_mono_pose_edge(self):
        Tcw, edges = self._g2o_pose(lambda: g2o_camera_project_pose_edge(self.camera), lambda i: self.uv[i])
        self._assert_pose(Tcw)
        self.assertTrue(all(edge.is_depth_positive() for edge in edges))

    def test_g2o_stereo_pose_edge(self):
        Tcw, _ = self._g2o_pose(lambda: g2o_stereo_camera_edge(self.camera, pose_only=True),
                                lambda i: np.array([self.uv[i, 0], self.uv[i, 1], self.ur[i]]))
        self._assert_pose(Tcw)

    def _gtsam_pose(self, make_factor):
        import gtsam
        graph = gtsam.NonlinearFactorGraph()
        for i, Xw in enumerate(self.points_w):
            graph.add(make_factor(i, Xw))
        Twc_init = np.linalg.inv(self.Tcw_init)
        values = gtsam.Values()
        values.insert(0, gtsam.Pose3(Twc_init))
        result = gtsam.LevenbergMarquardtOptimizer(graph, values, gtsam.LevenbergMarquardtParams()).optimize()
        return np.linalg.inv(result.atPose3(0).matrix()), graph

    def test_gtsam_mono_resectioning(self):
        import gtsam
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        Tcw, graph = self._gtsam_pose(lambda i, Xw: camera_resectioning_factor(
            noise, 0, self.camera, gtsam.Point2(*self.uv[i]), gtsam.Point3(*Xw))[0])
        self._assert_pose(Tcw)
        self.assertEqual(graph.clone().size(), graph.size())  # gtsam's default clone() throws

    def test_gtsam_stereo_resectioning(self):
        import gtsam
        noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        Tcw, _ = self._gtsam_pose(lambda i, Xw: stereo_camera_resectioning_factor(
            noise, 0, self.camera, gtsam.StereoPoint2(self.uv[i, 0], self.ur[i], self.uv[i, 1]),
            gtsam.Point3(*Xw))[0])
        self._assert_pose(Tcw)

    def test_gtsam_projection_factor_error(self):
        import gtsam
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        offset = np.array([1.5, -2.0])
        factor = camera_projection_factor(gtsam.Point2(*(self.uv[0] + offset)), noise, 0, 1, self.camera)
        values = gtsam.Values()
        values.insert(0, gtsam.Pose3(np.linalg.inv(self.Tcw)))
        values.insert(1, gtsam.Point3(*self.points_w[0]))
        self.assertAlmostEqual(factor.error(values), 0.5 * offset @ offset, places=6)


if __name__ == "__main__":
    unittest.main()
