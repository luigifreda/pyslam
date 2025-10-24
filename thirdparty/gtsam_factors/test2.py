#!/usr/bin/env -S python3 -O
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
import sys

sys.path.append("./lib")

import unittest
import numpy as np
import gtsam
import gtsam_factors


class TestResectioningFactors(unittest.TestCase):
    def test_resectioning_factor(self):
        # Create a calibration object
        calib = gtsam.Cal3_S2(500.0, 500.0, 0.0, 320.0, 240.0)

        # Create a noise model
        noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        # Create a world point and a measured point
        world_point = gtsam.Point3(1.0, 2.0, 5.0)
        measured_point = gtsam.Point2(320.0, 240.0)

        # Create a key
        key = gtsam.symbol("x", 1)

        # Create a factor
        factor = gtsam_factors.ResectioningFactor(
            noise_model, key, calib, measured_point, world_point
        )

        # Create a graph
        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(factor)

        # Create initial estimates
        initial_estimates = gtsam.Values()
        initial_pose = gtsam.Pose3()  # create a Pose3 object.
        initial_estimates.insert(key, initial_pose)  # insert the pose3 object.

        # Debugging: Check the type of the stored value
        stored_value = initial_estimates.atPose3(key)
        print(f"Type of stored value: {type(stored_value)}")

        print(f"graph: {graph}")

        # Create optimizer
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)

        # Optimize
        result = optimizer.optimize()

        # Check result
        self.assertTrue(result.exists(key))

    def test_stereo_resectioning_factor(self):
        # Create a stereo calibration object
        calib = gtsam.Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 100.0)

        # Create a noise model
        noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # Create a world point and a measured stereo point
        world_point = gtsam.Point3(1.0, 2.0, 5.0)
        measured_stereo_point = gtsam.StereoPoint2(320.0, 310.0, 240.0)

        # Create a key
        key = gtsam.symbol("x", 1)

        # Create a factor
        factor = gtsam_factors.ResectioningFactorStereo(
            noise_model, key, calib, measured_stereo_point, world_point
        )

        # Create a graph
        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(factor)

        # Create initial estimates
        initial_estimates = gtsam.Values()
        initial_pose = gtsam.Pose3()  # create a Pose3 object.
        initial_estimates.insert(key, initial_pose)  # insert the pose3 object.

        # Debugging: Check the type of the stored value
        stored_value = initial_estimates.atPose3(key)
        print(f"Type of stored value: {type(stored_value)}")

        print(f"graph: {graph}")

        # Create optimizer
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)

        # Optimize
        result = optimizer.optimize()

        # Check result
        self.assertTrue(result.exists(key))


if __name__ == "__main__":
    unittest.main()
