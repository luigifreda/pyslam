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
* Pose prediction of MotionModel and MotionModelDamping, for the core PYSLAM_USE_CPP selects.
"""

import unittest

import numpy as np
import slam_test_support  # noqa: F401  (library paths)
from pyslam.slam import CameraPose, MotionModel, MotionModelDamping
from pyslam.utilities.geometry import poseRt, rotation_matrix_from_yaw_pitch_roll


def quaternion(yaw_deg):
    return CameraPose(poseRt(rotation_matrix_from_yaw_pitch_roll(yaw_deg, 0, 0), np.zeros(3))).quaternion


def predicted(model, timestamp):
    pose, _ = model.predict_pose(timestamp)
    return np.asarray(pose.matrix())


class TestMotionModel(unittest.TestCase):
    def test_uninitialized_model_predicts_last_pose(self):
        model = MotionModel()
        self.assertFalse(model.initialized)
        np.testing.assert_allclose(predicted(model, 0.0), np.eye(4), atol=1e-12)

    def test_constant_velocity_translation(self):
        model = MotionModel()
        model.update_pose(0.0, np.array([0.0, 0.0, 0.0]), quaternion(0))
        model.update_pose(1.0, np.array([1.0, -0.5, 0.2]), quaternion(0))
        self.assertTrue(model.initialized)
        np.testing.assert_allclose(predicted(model, 2.0)[:3, 3], [2.0, -1.0, 0.4], atol=1e-9)

    def test_constant_velocity_rotation(self):
        model = MotionModel()
        model.update_pose(0.0, np.zeros(3), quaternion(0))
        model.update_pose(1.0, np.zeros(3), quaternion(10))
        expected = rotation_matrix_from_yaw_pitch_roll(20, 0, 0)
        np.testing.assert_allclose(predicted(model, 2.0)[:3, :3], expected, atol=1e-9)

    def test_correction_moves_prediction_and_keeps_velocity(self):
        model = MotionModel()
        model.update_pose(0.0, np.zeros(3), quaternion(0))
        model.update_pose(1.0, np.array([1.0, 0.0, 0.0]), quaternion(0))
        model.apply_correction(poseRt(np.eye(3), np.array([0.0, 5.0, 0.0])))
        np.testing.assert_allclose(predicted(model, 2.0)[:3, 3], [2.0, 5.0, 0.0], atol=1e-9)

    def test_reset_forgets_motion(self):
        model = MotionModel()
        model.update_pose(0.0, np.zeros(3), quaternion(0))
        model.update_pose(1.0, np.array([1.0, 0.0, 0.0]), quaternion(0))
        model.reset()
        self.assertFalse(model.initialized)
        np.testing.assert_allclose(predicted(model, 2.0), np.eye(4), atol=1e-12)


class TestMotionModelDamping(unittest.TestCase):
    def test_damped_velocity_scales_with_elapsed_time(self):
        model = MotionModelDamping()
        model.update_pose(0.0, np.zeros(3), quaternion(0))
        model.update_pose(0.5, np.array([1.0, 0.0, 0.0]), quaternion(0))
        # v = 2 m/s, damping 0.95, dt = 1 s
        np.testing.assert_allclose(predicted(model, 1.5)[:3, 3], [1.0 + 2.0 * 0.95, 0.0, 0.0], atol=1e-9)

    def test_damped_rotation(self):
        model = MotionModelDamping()
        model.update_pose(0.0, np.zeros(3), quaternion(0))
        model.update_pose(1.0, np.zeros(3), quaternion(10))
        expected = rotation_matrix_from_yaw_pitch_roll(10 + 10 * 0.95, 0, 0)
        np.testing.assert_allclose(predicted(model, 2.0)[:3, :3], expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
