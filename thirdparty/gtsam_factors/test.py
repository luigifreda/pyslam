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

import gtsam
from gtsam.symbol_shorthand import X, L
import numpy as np

import gtsam_factors

if __name__ == "__main__":
    # Print gtsam version and location of library
    print(f"gtsam location: {gtsam.__file__}")

    # Create a NonlinearFactorGraph to hold the factors
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    # Monocular Camera Calibration (Simple intrinsic parameters for demonstration)
    K_mono = gtsam.Cal3_S2(600, 600, 0, 320, 240)

    # Stereo Camera Calibration (Simple intrinsic parameters for demonstration)
    K_stereo = gtsam.Cal3_S2Stereo(600, 600, 0, 320, 240, 0.1)

    # Noise Model (Isotropic Gaussian)
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # Monocular (2D)
    stereo_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)  # Stereo (3D)

    # Define a known 3D point in front of the camera (ensure Z > 0)
    P = np.array([1, 2, 10], dtype=np.float64).reshape(3, 1)  # Point in front of the camera

    # Define 2D projection (Monocular), make sure this projection is valid for the given P
    p_2d = np.array([320, 240], dtype=np.float64).reshape(2, 1)

    # Define StereoPoint2 (Stereo camera model)
    p_stereo = gtsam.StereoPoint2(145, 140, 120)  # Right and Left projections

    # Add Monocular Resectioning Factor (ensure the 3D point is in front of the camera)
    factor_mono = gtsam_factors.ResectioningFactor(noise_model, X(1), K_mono, p_2d, P)
    print(f"factor_mono: {factor_mono}")
    graph.add(factor_mono)
    print(f"added factor_mono: {factor_mono}")

    # Add Stereo Resectioning Factor (ensure the 3D point is visible from both cameras)
    factor_stereo = gtsam_factors.ResectioningFactorStereo(
        stereo_noise_model, X(1), K_stereo, p_stereo, P
    )
    print(f"factor_stereo: {factor_stereo}")
    graph.add(factor_stereo)
    print(f"added factor_stereo: {factor_stereo}")

    # Initial Estimates (initial guess for pose)
    init_pose = gtsam.Pose3(
        np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    )  # Start with an identity pose as a guess
    initial_estimates.insert(X(1), init_pose)

    value = initial_estimates.atPose3(X(1))
    print(type(value))  # This should print something like <class 'gtsam.Pose3'>

    # Levenberg-Marquardt Optimizer Setup
    params = gtsam.LevenbergMarquardtParams()
    print(f"creating optimizer: {params}")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)

    # Perform Optimization
    print(f"optimizing...")
    result = optimizer.optimize()

    # Print the result (optimized pose)
    print("Optimized Pose:", result.atPose3(X(1)))
