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
#include <nlohmann/json.hpp>
#include <sstream>

namespace pyslam {

CameraPose::CameraPose()
    : pose_(Eigen::Isometry3d::Identity()), Tcw_(Eigen::Matrix4d::Identity()),
      Rcw_(Eigen::Matrix3d::Identity()), tcw_(Eigen::Vector3d::Zero()),
      Rwc_(Eigen::Matrix3d::Identity()), Ow_(Eigen::Vector3d::Zero()) {
    // Ensure identity initialization for default constructor
    // pose_ = Eigen::Isometry3d::Identity();
    // initialize_covariance();
    // update_cached_matrices();
}

CameraPose::CameraPose(const Eigen::Isometry3d &pose) : pose_(pose) {
    // initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const Eigen::Matrix4d &Tcw) : pose_(Eigen::Isometry3d(Tcw)) {
    // initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw) {
    pose_ = Eigen::Isometry3d::Identity();
    pose_.linear() = Rcw;
    pose_.translation() = tcw;
    // initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const Eigen::Quaterniond &quaternion, const Eigen::Vector3d &position) {
    pose_ = Eigen::Isometry3d::Identity();
    pose_.linear() = quaternion.normalized().toRotationMatrix();
    pose_.translation() = position;
    // initialize_covariance();
    update_cached_matrices();
}

CameraPose::CameraPose(const CameraPose &other)
    : pose_(other.pose_), Tcw_(other.Tcw_), Rcw_(other.Rcw_), tcw_(other.tcw_), Rwc_(other.Rwc_),
      Ow_(other.Ow_), covariance_(other.covariance_) {}

CameraPose &CameraPose::operator=(const CameraPose &other) {
    if (this != &other) {
        pose_ = other.pose_;
        Tcw_ = other.Tcw_;
        Rcw_ = other.Rcw_;
        tcw_ = other.tcw_;
        Rwc_ = other.Rwc_;
        Ow_ = other.Ow_;
        covariance_ = other.covariance_;
    }
    return *this;
}

CameraPose CameraPose::copy() const { return CameraPose(*this); }

void CameraPose::set(const Eigen::Isometry3d &pose) {
    pose_ = pose;
    update_cached_matrices();
}

void CameraPose::update(const Eigen::Isometry3d &pose) { set(pose); }

void CameraPose::update(const Eigen::Matrix4d &Tcw) {
    const auto pose = Eigen::Isometry3d(Tcw);
    set(pose);
}

void CameraPose::set_mat(const Eigen::Matrix4d &Tcw) {
    Tcw_ = Tcw;
    Rcw_ = Tcw.block<3, 3>(0, 0);
    tcw_ = Tcw.block<3, 1>(0, 3);
    Rwc_ = Rcw_.transpose();
    Ow_ = -(Rwc_ * tcw_);

    // Update the pose
    pose_ = Eigen::Isometry3d(this->Tcw_);
}

void CameraPose::update_mat(const Eigen::Matrix4d &Tcw) { set_mat(Tcw); }

Eigen::AngleAxisd CameraPose::get_rotation_angle_axis() const {
    return Eigen::AngleAxisd(pose_.rotation());
}

Eigen::Matrix3d CameraPose::get_rotation_matrix() const { return pose_.rotation(); }

Eigen::Matrix3d CameraPose::get_inverse_rotation_matrix() const {
    return pose_.inverse().rotation();
}

Eigen::Matrix4d CameraPose::get_matrix() const { return pose_.matrix(); }

Eigen::Matrix4d CameraPose::get_inverse_matrix() const { return pose_.inverse().matrix(); }

void CameraPose::set_from_quaternion_and_position(const Eigen::Quaterniond &quaternion,
                                                  const Eigen::Vector3d &position) {
    pose_ = Eigen::Isometry3d::Identity();
    pose_.linear() = quaternion.normalized().toRotationMatrix(); // set rotation
    pose_.translation() = position;                              // set translation
    update_cached_matrices();
}

void CameraPose::set_from_matrix(const Eigen::Matrix4d &Tcw) {
    pose_ = Eigen::Isometry3d(Tcw);
    update_cached_matrices();
}

void CameraPose::set_from_rotation_and_translation(const Eigen::Matrix3d &Rcw,
                                                   const Eigen::Vector3d &tcw) {
    pose_ = Eigen::Isometry3d::Identity();
    pose_.linear() = Rcw;      // set rotation
    pose_.translation() = tcw; // set translation
    update_cached_matrices();
}

void CameraPose::set_quaternion(const Eigen::Quaterniond &quaternion) {
    pose_.linear() = quaternion.normalized().toRotationMatrix(); // set rotation
    update_cached_matrices();
}

void CameraPose::set_rotation_matrix(const Eigen::Matrix3d &Rcw) {
    pose_.linear() = Rcw; // set rotation
    update_cached_matrices();
}

void CameraPose::set_translation(const Eigen::Vector3d &tcw) {
    pose_.translation() = tcw; // set translation
    update_cached_matrices();
}

void CameraPose::update_cached_matrices(bool initialize) {
    if (initialize) {
        Tcw_ = Eigen::Matrix4d::Identity();
        Rcw_ = Eigen::Matrix3d::Identity();
        tcw_ = Eigen::Vector3d::Zero();
        Rwc_ = Eigen::Matrix3d::Identity();
        Ow_ = Eigen::Vector3d::Zero();
    } else {
        Tcw_ = pose_.matrix();
        Rcw_ = Tcw_.block<3, 3>(0, 0);
        tcw_ = Tcw_.block<3, 1>(0, 3);
        Rwc_ = Rcw_.transpose();
        Ow_ = -(Rwc_ * tcw_);
    }
}

void CameraPose::initialize_covariance() { covariance_ = Eigen::Matrix<double, 6, 6>::Identity(); }

// Comparison operators - matches Python interface
bool CameraPose::operator==(const CameraPose &other) const { return pose_.isApprox(other.pose_); }

bool CameraPose::operator!=(const CameraPose &other) const { return !(*this == other); }

// String representation - matches Python interface
std::string CameraPose::to_string() const {
    std::ostringstream oss;
    oss << "CameraPose(";
    oss << "Tcw=[" << Tcw_(0, 0) << "," << Tcw_(0, 1) << "," << Tcw_(0, 2) << "," << Tcw_(0, 3)
        << ";";
    oss << Tcw_(1, 0) << "," << Tcw_(1, 1) << "," << Tcw_(1, 2) << "," << Tcw_(1, 3) << ";";
    oss << Tcw_(2, 0) << "," << Tcw_(2, 1) << "," << Tcw_(2, 2) << "," << Tcw_(2, 3) << ";";
    oss << Tcw_(3, 0) << "," << Tcw_(3, 1) << "," << Tcw_(3, 2) << "," << Tcw_(3, 3) << "])";
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
            oss << Tcw_(i, j);
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
            oss << covariance_(i, j);
        }
        oss << "]";
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

CameraPose CameraPose::from_json(const std::string &json_str) {
    try {
        nlohmann::json json_obj = nlohmann::json::parse(json_str);

        // Parse Tcw matrix
        if (json_obj.contains("Tcw") && json_obj["Tcw"].is_array()) {
            Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
            auto tcw_array = json_obj["Tcw"];

            if (tcw_array.size() == 4) {
                for (size_t i = 0; i < 4; ++i) {
                    if (tcw_array[i].is_array() && tcw_array[i].size() == 4) {
                        for (size_t j = 0; j < 4; ++j) {
                            Tcw(i, j) = tcw_array[i][j].get<double>();
                        }
                    }
                }
            }

            CameraPose pose;
            pose.set_mat(Tcw);

            // Parse covariance matrix if present
            if (json_obj.contains("covariance") && json_obj["covariance"].is_array()) {
                auto cov_array = json_obj["covariance"];
                if (cov_array.size() == 6) {
                    for (size_t i = 0; i < 6; ++i) {
                        if (cov_array[i].is_array() && cov_array[i].size() == 6) {
                            for (size_t j = 0; j < 6; ++j) {
                                pose.covariance_(i, j) = cov_array[i][j].get<double>();
                            }
                        }
                    }
                }
            }

            return pose;
        }
    } catch (const std::exception &e) {
        // If JSON parsing fails, return default pose
        // In a production environment, you might want to log the error
    }

    return CameraPose();
}

} // namespace pyslam
