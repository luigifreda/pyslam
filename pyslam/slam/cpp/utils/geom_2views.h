/*
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

#include "eigen_aliases.h"
#include "utils/eigen_helpers.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include <cmath>

// Forward declarations
class FramePtr;

namespace pyslam {

namespace geom_2views {

/**
 * Compute fundamental matrix F12 and infinite homography H21 between two frames
 * Optimized version using pre-computed matrices
 */
inline std::pair<Mat3d, Mat3d> computeF12(const Mat3d &R1w, const Vec3d &t1w, const Mat3d &R2w,
                                          const Vec3d &t2w, const Mat3d &K1inv, const Mat3d &K2,
                                          const Mat3d &K2inv) {

    // Relative rotation and translation
    Mat3d R12 = R1w * R2w.transpose();
    Vec3d t12 = -R1w * (R2w.transpose() * t2w) + t1w;

    // Skew-symmetric matrix of translation
    Mat3d t12x = skew(t12);

    // Infinite homography from frame 1 to frame 2
    Mat3d R21 = R12.transpose();
    Mat3d H21 = K2 * R21 * K1inv;

    // Fundamental matrix
    Mat3d F12 = K1inv.transpose() * t12x * R12 * K2inv;

    return std::make_pair(F12, H21);
}

/**
 * Compute fundamental matrix F12 and infinite homography H21 from two frames
 * This is the main interface function
 */
std::pair<Mat3d, Mat3d> computeF12(const FramePtr &f1, const FramePtr &f2);

/**
 * Check if a keypoint satisfies the epipolar constraint
 * @param kp1 Keypoint in first image [x, y]
 * @param kp2 Keypoint in second image [x, y]
 * @param F12 Fundamental matrix from frame 1 to frame 2
 * @param sigma2_kp2 Variance of keypoint 2 (squared)
 * @return true if keypoint satisfies epipolar constraint
 */
inline bool check_dist_epipolar_line(const Vec2d &kp1, const Vec2d &kp2, const Mat3d &F12,
                                     double sigma2_kp2) {

    // Epipolar line in second image l = kp1' * F12 = [a b c]
    const Vec3d kp1_hom(kp1(0), kp1(1), 1.0);
    const Vec3d l = F12.transpose() * kp1_hom;

    // Distance from kp2 to epipolar line
    const double num = l(0) * kp2(0) + l(1) * kp2(1) + l(2); // kp1' * F12 * kp2
    const double den = l(0) * l(0) + l(1) * l(1);            // a*a + b*b

    if (den == 0.0) {
        return false;
    }

    const double dist_sqr = num * num / den; // squared distance from epipolar line

    // Chi-square test for 1 DOF (Hartley Zisserman pag 567)
    return dist_sqr < 3.84 * sigma2_kp2;
}

/**
 * Estimate pose from essential matrix using RANSAC
 *
 * @param kpn_ref Normalized keypoints in reference frame [Nx2]
 * @param kpn_cur Normalized keypoints in current frame [Nx2]
 * @param method RANSAC method (default: cv::RANSAC)
 * @param prob Probability (default: 0.999)
 * @param threshold Threshold for RANSAC (default: 0.0003)
 * @return Pair of (pose matrix Trc, inlier mask)
 *
 * Notes:
 * - Uses five-point algorithm solver by D. Nister
 * - Translation is estimated up to scale (||trc||=1)
 * - Works well in degenerate planar cases
 * - Requires sufficient parallax for pure rotation cases
 */
inline std::pair<Mat4d, std::vector<uchar>>
estimate_pose_ess_mat(const MatNx2d &kpn_ref, const MatNx2d &kpn_cur, int method = cv::RANSAC,
                      double prob = 0.999, double threshold = 0.0003) {

    // Convert Eigen matrices to OpenCV format
    std::vector<cv::Point2f> points_ref, points_cur;
    points_ref.reserve(kpn_ref.rows());
    points_cur.reserve(kpn_cur.rows());

    for (int i = 0; i < kpn_ref.rows(); ++i) {
        points_ref.emplace_back(kpn_ref(i, 0), kpn_ref(i, 1));
        points_cur.emplace_back(kpn_cur(i, 0), kpn_cur(i, 1));
    }

    // Find essential matrix using five-point algorithm
    cv::Mat E, mask_match;
    E = cv::findEssentialMat(points_cur, points_ref, 1.0, cv::Point2d(0.0, 0.0), method, prob,
                             threshold, mask_match);

    // Recover pose from essential matrix
    cv::Mat R, t, mask;
    cv::recoverPose(E, points_cur, points_ref, R, t, 1.0, cv::Point2d(0.0, 0.0), mask);

    // Convert OpenCV matrices to Eigen
    Mat3d R_eigen;
    Vec3d t_eigen;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_eigen(i, j) = R.at<double>(i, j);
        }
        t_eigen(i) = t.at<double>(i);
    }

    // Create pose matrix
    const Mat4d Trc = poseRt(R_eigen, t_eigen);

    // Convert mask to vector
    std::vector<uchar> mask_vec;
    mask_vec.reserve(mask_match.rows);
    for (int i = 0; i < mask_match.rows; ++i) {
        mask_vec.push_back(mask_match.at<uchar>(i));
    }

    return std::make_pair(Trc, mask_vec);
}

} // namespace geom_2views

} // namespace pyslam