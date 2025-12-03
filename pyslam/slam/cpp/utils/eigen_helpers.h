#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pyslam {

// Helper functions

// Create a 4x4 matrix from a 3x3 rotation matrix and a 3x1 translation vector
Eigen::Matrix4d poseRt(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

// Extract a 3x3 rotation matrix and a 3x1 translation vector from a 4x4 matrix
std::pair<Eigen::Matrix3d, Eigen::Vector3d> extractRt(const Eigen::Matrix4d &T) {
    return {T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)};
}

} // namespace pyslam