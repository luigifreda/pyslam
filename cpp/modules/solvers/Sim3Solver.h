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

/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel
 * and Juan D. Tardós, University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M.
 * Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace utils {

// Input for the Sim3Solver
class Sim3SolverInput {
  public:
    // matches data
    std::vector<Eigen::Vector3f> mvX3Dw1; // matched 3D points of KF1
    std::vector<Eigen::Vector3f> mvX3Dw2; // matched 3D points of KF2

    std::vector<float> mvSigmaSquare1; // mvLevelSigma2[kp.octave] corresponding to mvX3Dw1
    std::vector<float> mvSigmaSquare2; // mvLevelSigma2[kp.octave] corresponding to mvX3Dw2

  public:
    // keyframes data
    Eigen::Matrix3f K1;
    Eigen::Matrix3f Rcw1;
    Eigen::Vector3f tcw1;

    Eigen::Matrix3f K2;
    Eigen::Matrix3f Rcw2;
    Eigen::Vector3f tcw2;

    bool bFixScale = false;
};

// Input for the Sim3Solver
class Sim3SolverInput2 {
  public:
    // matches data
    std::vector<Eigen::Vector3f> mvX3Dc1; // matched 3D points of KF1 in KF1 frame
    std::vector<Eigen::Vector3f> mvX3Dc2; // matched 3D points of KF2 in KF2 frame

    std::vector<float> mvSigmaSquare1; // mvLevelSigma2[kp.octave] corresponding to mvX3Dw1
    std::vector<float> mvSigmaSquare2; // mvLevelSigma2[kp.octave] corresponding to mvX3Dw2

  public:
    // keyframes data
    Eigen::Matrix3f K1;
    Eigen::Matrix3f K2;

    bool bFixScale = false;
};

// Sim3 Solver: Estimate the Sim3 transformation between two sets of 3D points back-projected from
// two camera frames.
//              Inliers are evaluated using the reprojection error (2D-3D associations).
class Sim3Solver {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit Sim3Solver(const Sim3SolverInput &input);
    explicit Sim3Solver(const Sim3SolverInput2 &input);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6,
                             int maxIterations = 300);

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

    void Project(const std::vector<Eigen::Vector3f> &vP3Dw, const Eigen::Matrix4f &Tcw,
                 const Eigen::Matrix3f &K, std::vector<Eigen::Vector2f> &vP2D,
                 std::vector<float> &depths);
    void FromCameraToImage(const std::vector<Eigen::Vector3f> &vP3Dc, const Eigen::Matrix3f &K,
                           std::vector<Eigen::Vector2f> &vP2D);

  protected:
    std::vector<Eigen::Vector3f> mvX3Dc1;
    std::vector<Eigen::Vector3f> mvX3Dc2;
    std::vector<size_t> mvnIndices1;
    std::vector<float> mvnMaxError1;
    std::vector<float> mvnMaxError2;

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

    // Projections
    std::vector<Eigen::Vector2f> mvP1im1;
    std::vector<Eigen::Vector2f> mvP2im2;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
    float mTh;
    float mSigma2;

    // Calibration
    Eigen::Matrix3f mK1;
    Eigen::Matrix3f mK2;
};

} // namespace utils
