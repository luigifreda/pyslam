/**
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
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace utils {

class Sim3PointRegistrationSolverInput {
  public:
    // matches data
    std::vector<Eigen::Vector3f> mvX3Dw1; // matched 3D points of frame 1
    std::vector<Eigen::Vector3f> mvX3Dw2; // matched 3D points of frame 2
    float mSigma2 = -1;                   // squared sigma on a the 3D point error
    bool bFixScale = false;
};

// Sim3 Point Registration Solver: Estimate the Sim3 transformation between two sets of 3D points.
//                                 Inliers are evaluated using the reprojection error (3D-3D
//                                 associations).
class Sim3PointRegistrationSolver {
    static constexpr float kSigma = 0.03;             // [m]
    static constexpr float kSigma2 = kSigma * kSigma; // [m^2]

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Sim3PointRegistrationSolver(const Sim3PointRegistrationSolverInput &input);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6,
                             int maxIterations = 300);
    void SetSigma(float sigma) { mSigma2 = sigma * sigma; }

    Eigen::Matrix4f find(std::vector<uint8_t> &vbInliers12, int &nInliers, bool &bConverged);

    Eigen::Matrix4f iterate(const int nIterations, bool &bNoMore, std::vector<uint8_t> &vbInliers,
                            int &nInliers, bool &bConverged);

    Eigen::Matrix4f GetEstimatedTransformation();
    Eigen::Matrix3f GetEstimatedRotation();
    Eigen::Vector3f GetEstimatedTranslation();
    float GetEstimatedScale();

    float Compute3dRegistrationError();

  protected:
    void ComputeCentroid(Eigen::Matrix3f &P, Eigen::Matrix3f &Pr, Eigen::Vector3f &C);

    bool ComputeSim3(Eigen::Matrix3f &P1, Eigen::Matrix3f &P2);

    void CheckInliers();

  protected:
    std::vector<Eigen::Vector3f> mvX3Dw1;
    std::vector<Eigen::Vector3f> mvX3Dw2;
    std::vector<size_t> mvnIndices1;

    int mN;

    // Current Estimation
    Eigen::Matrix3f mR12i;
    Eigen::Vector3f mt12i;
    float ms12i;
    Eigen::Matrix4f mT12i;
    Eigen::Matrix4f mT21i;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Matrix4f mBestT12;
    Eigen::Matrix3f mBestRotation;
    Eigen::Vector3f mBestTranslation;
    float mBestScale;

    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;

    // Indices for random selection
    std::vector<size_t> mvAllIndices;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 7.81*mSigma2 (considering 95% of the 3D
    // points are inliers)
    float mThChi2;
    float mSigma2 = kSigma2; // [m^2] single-component squared sigma on a the 3D point error
};

} // namespace utils
