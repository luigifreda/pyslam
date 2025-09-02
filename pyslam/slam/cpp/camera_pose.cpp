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

#include "camera_pose.h"
#include <sstream>

namespace pyslam {

CameraPose::CameraPose()
    : pose_(std::make_unique<Eigen::Isometry3d>()), Tcw(Eigen::Matrix4d::Identity()),
      Rcw(Eigen::Matrix3d::Identity()), tcw(Eigen::Vector3d::Zero()),
      Rwc(Eigen::Matrix3d::Identity()), Ow(Eigen::Vector3d::Zero()) {
    // Ensure identity initialization for default constructor
    *pose_ = Eigen::Isometry3d::Identity();
    initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const Eigen::Isometry3d &pose)
    : pose_(std::make_unique<Eigen::Isometry3d>(pose)), Tcw(Eigen::Matrix4d::Identity()),
      Rcw(Eigen::Matrix3d::Identity()), tcw(Eigen::Vector3d::Zero()),
      Rwc(Eigen::Matrix3d::Identity()), Ow(Eigen::Vector3d::Zero()) {
    initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const CameraPose &other)
    : pose_(std::make_unique<Eigen::Isometry3d>(*other.pose_)), Tcw(other.Tcw), Rcw(other.Rcw),
      tcw(other.tcw), Rwc(other.Rwc), Ow(other.Ow), covariance(other.covariance) {}

CameraPose &CameraPose::operator=(const CameraPose &other) {
    if (this != &other) {
        *pose_ = *other.pose_;
        Tcw = other.Tcw;
        Rcw = other.Rcw;
        tcw = other.tcw;
        Rwc = other.Rwc;
        Ow = other.Ow;
        covariance = other.covariance;
    }
    return *this;
}

CameraPose CameraPose::copy() const { return CameraPose(*this); }

void CameraPose::set(const Eigen::Isometry3d &pose) {
    *pose_ = pose;
    update_cached_matrices();
}

void CameraPose::update(const Eigen::Isometry3d &pose) { set(pose); }

void CameraPose::update(const Eigen::Matrix4d &Tcw) {
    const auto pose = Eigen::Isometry3d(Tcw);
    set(pose);
}

void CameraPose::set_mat(const Eigen::Matrix4d &Tcw) {
    this->Tcw = Tcw;
    Rcw = Tcw.block<3, 3>(0, 0);
    tcw = Tcw.block<3, 1>(0, 3);
    Rwc = Rcw.transpose();
    Ow = -(Rwc * tcw);

    // Update the pose
    *pose_ = Eigen::Isometry3d(this->Tcw);
}

void CameraPose::update_mat(const Eigen::Matrix4d &Tcw) { set_mat(Tcw); }

Eigen::AngleAxisd CameraPose::get_rotation_angle_axis() const {
    return Eigen::AngleAxisd(pose_->rotation());
}

Eigen::Matrix3d CameraPose::get_rotation_matrix() const { return pose_->rotation(); }

Eigen::Matrix3d CameraPose::get_inverse_rotation_matrix() const {
    return pose_->inverse().rotation();
}

Eigen::Matrix4d CameraPose::get_matrix() const { return pose_->matrix(); }

Eigen::Matrix4d CameraPose::get_inverse_matrix() const { return pose_->inverse().matrix(); }

void CameraPose::set_from_quaternion_and_position(const Eigen::Quaterniond &quaternion,
                                                  const Eigen::Vector3d &position) {
    *pose_ = Eigen::Isometry3d::Identity();
    pose_->linear() = quaternion.normalized().toRotationMatrix(); // set rotation
    pose_->translation() = position;                              // set translation
    update_cached_matrices();
}

void CameraPose::set_from_matrix(const Eigen::Matrix4d &Tcw) {
    *pose_ = Eigen::Isometry3d(Tcw);
    update_cached_matrices();
}

void CameraPose::set_from_rotation_and_translation(const Eigen::Matrix3d &Rcw,
                                                   const Eigen::Vector3d &tcw) {
    *pose_ = Eigen::Isometry3d::Identity();
    pose_->linear() = Rcw;      // set rotation
    pose_->translation() = tcw; // set translation
    update_cached_matrices();
}

void CameraPose::set_quaternion(const Eigen::Quaterniond &quaternion) {
    *pose_ = Eigen::Isometry3d::Identity();
    pose_->linear() = quaternion.normalized().toRotationMatrix(); // set rotation
    pose_->translation() = pose_->translation();                  // set translation
    update_cached_matrices();
}

void CameraPose::set_rotation_matrix(const Eigen::Matrix3d &Rcw) {
    *pose_ = Eigen::Isometry3d::Identity();
    pose_->linear() = Rcw;                       // set rotation
    pose_->translation() = pose_->translation(); // set translation
    update_cached_matrices();
}

void CameraPose::set_translation(const Eigen::Vector3d &tcw) {
    *pose_ = Eigen::Isometry3d::Identity();
    pose_->linear() = Eigen::Quaterniond(Rcw).normalized().toRotationMatrix(); // set rotation
    pose_->translation() = tcw;                                                // set translation
    update_cached_matrices();
}

void CameraPose::update_cached_matrices() {
    Tcw = pose_->matrix();
    Rcw = Tcw.block<3, 3>(0, 0);
    tcw = Tcw.block<3, 1>(0, 3);
    Rwc = Rcw.transpose();
    Ow = -(Rwc * tcw);
}

void CameraPose::initialize_covariance() { covariance = Eigen::Matrix<double, 6, 6>::Identity(); }

// Comparison operators - matches Python interface
bool CameraPose::operator==(const CameraPose &other) const { return pose_->isApprox(*other.pose_); }

bool CameraPose::operator!=(const CameraPose &other) const { return !(*this == other); }

// String representation - matches Python interface
std::string CameraPose::to_string() const {
    std::ostringstream oss;
    oss << "CameraPose(";
    oss << "Tcw=[" << Tcw(0, 0) << "," << Tcw(0, 1) << "," << Tcw(0, 2) << "," << Tcw(0, 3) << ";";
    oss << Tcw(1, 0) << "," << Tcw(1, 1) << "," << Tcw(1, 2) << "," << Tcw(1, 3) << ";";
    oss << Tcw(2, 0) << "," << Tcw(2, 1) << "," << Tcw(2, 2) << "," << Tcw(2, 3) << ";";
    oss << Tcw(3, 0) << "," << Tcw(3, 1) << "," << Tcw(3, 2) << "," << Tcw(3, 3) << "])";
    return oss.str();
}

std::string CameraPose::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"Tcw\":[";
    for (int i = 0; i < 4; ++i) {
        if (i > 0)
            oss << ",";
        oss << "[";
        for (int j = 0; j < 4; ++j) {
            if (j > 0)
                oss << ",";
            oss << Tcw(i, j);
        }
        oss << "]";
    }
    oss << "],";
    oss << "\"covariance\":[";
    for (int i = 0; i < 6; ++i) {
        if (i > 0)
            oss << ",";
        oss << "[";
        for (int j = 0; j < 6; ++j) {
            if (j > 0)
                oss << ",";
            oss << covariance(i, j);
        }
        oss << "]";
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

CameraPose CameraPose::from_json(const std::string &json_str) {
    // This is a simplified implementation
    // The actual implementation would parse JSON properly
    return CameraPose();
}

} // namespace pyslam
