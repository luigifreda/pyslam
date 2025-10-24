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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>

namespace pyslam {

class CameraPose {
  private:
    // Core pose data - matches Python's _pose
    Eigen::Isometry3d pose_ = Eigen::Isometry3d::Identity();

    // Cached matrices for efficiency - matches Python's public attributes
    Eigen::Matrix4d Tcw_ = Eigen::Matrix4d::Identity(); // homogeneous transformation matrix
    Eigen::Matrix3d Rcw_ = Eigen::Matrix3d::Identity(); // rotation matrix
    Eigen::Vector3d tcw_ = Eigen::Vector3d::Zero();     // translation vector
    Eigen::Matrix3d Rwc_ = Eigen::Matrix3d::Identity(); // inverse rotation matrix
    Eigen::Vector3d Ow_ = Eigen::Vector3d::Zero();      // camera origin in world coordinates

    // Pose covariance - matches Python's covariance
    Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();

  public:
    // Constructor - matches Python's __init__(self, pose=None)
    CameraPose();
    explicit CameraPose(const Eigen::Isometry3d &pose);
    explicit CameraPose(const Eigen::Matrix4d &Tcw);
    explicit CameraPose(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);
    explicit CameraPose(const Eigen::Quaterniond &quaternion, const Eigen::Vector3d &position);

    // Destructor
    ~CameraPose() = default;

    // Copy constructor and assignment
    CameraPose(const CameraPose &other);
    CameraPose &operator=(const CameraPose &other);
    CameraPose(CameraPose &&other) noexcept = default;
    CameraPose &operator=(CameraPose &&other) noexcept = default;

    // copy() method - matches Python's copy()
    CameraPose copy() const;

    const Eigen::Matrix4d &Tcw() const { return Tcw_; }
    const Eigen::Matrix3d &Rcw() const { return Rcw_; }
    const Eigen::Vector3d &tcw() const { return tcw_; }
    const Eigen::Matrix3d &Rwc() const { return Rwc_; }
    const Eigen::Vector3d &Ow() const { return Ow_; }

    Eigen::Matrix<double, 6, 6> &covariance() { return covariance_; }
    const Eigen::Matrix<double, 6, 6> &covariance() const { return covariance_; }

    // Core operations - matches Python interface exactly
    void set(const Eigen::Isometry3d &pose);
    void update(const Eigen::Isometry3d &pose);
    void update(const Eigen::Matrix4d &Tcw);
    void set_mat(const Eigen::Matrix4d &Tcw);
    void update_mat(const Eigen::Matrix4d &Tcw);

    // Properties - matches Python's @property methods
    const Eigen::Isometry3d &isometry3d() const { return pose_; }
    Eigen::Quaterniond quaternion() const { return Eigen::Quaterniond(pose_.rotation()); }
    Eigen::Quaterniond orientation() const { return Eigen::Quaterniond(pose_.rotation()); }
    Eigen::Vector3d position() const { return pose_.translation(); }

    // Utility methods - matches Python interface exactly
    Eigen::AngleAxisd get_rotation_angle_axis() const;
    Eigen::Matrix4d get_matrix() const;
    Eigen::Matrix4d get_inverse_matrix() const;
    Eigen::Matrix3d get_rotation_matrix() const;
    Eigen::Matrix3d get_inverse_rotation_matrix() const;

    // Setter methods - matches Python interface exactly
    void set_from_quaternion_and_position(const Eigen::Quaterniond &quaternion,
                                          const Eigen::Vector3d &position);
    void set_from_matrix(const Eigen::Matrix4d &Tcw);
    void set_from_rotation_and_translation(const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw);
    void set_quaternion(const Eigen::Quaterniond &quaternion);
    void set_rotation_matrix(const Eigen::Matrix3d &Rcw);
    void set_translation(const Eigen::Vector3d &tcw);

    // Comparison operators - matches Python interface
    bool operator==(const CameraPose &other) const;
    bool operator!=(const CameraPose &other) const;

    // String representation - matches Python interface
    std::string to_string() const;
    std::string to_json() const;
    static CameraPose from_json(const std::string &json_str);

  private:
    // Helper methods
    void update_cached_matrices(bool initialize = false);
    void initialize_covariance();
};

} // namespace pyslam
