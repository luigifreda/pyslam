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

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>

#include <sstream>

namespace pyslam {

class Sim3Pose {
  public:
    // Default constructor
    Sim3Pose() : R_(Eigen::Matrix3d::Identity()), t_(Eigen::Vector3d::Zero()), s_(1.0) {}

    // Constructor with rotation, translation, and scale
    Sim3Pose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double s = 1.0)
        : R_(R), t_(t), s_(s) {
        assert(s > 0 && "Scale must be positive");
    }

    // Constructor from 4x4 transformation matrix
    Sim3Pose(const Eigen::Matrix4d &T) { from_matrix(T); }

    // Copy constructor
    Sim3Pose(const Sim3Pose &other) : R_(other.R_), t_(other.t_), s_(other.s_) {}

    // Assignment operator
    Sim3Pose &operator=(const Sim3Pose &other) {
        if (this != &other) {
            R_ = other.R_;
            t_ = other.t_;
            s_ = other.s_;
        }
        return *this;
    }

    // Destructor
    ~Sim3Pose() = default;

    // Getters
    Eigen::Matrix3d &R() { return R_; }
    Eigen::Vector3d &t() { return t_; }
    double &s() { return s_; }
    const Eigen::Matrix3d &R() const { return R_; }
    const Eigen::Vector3d &t() const { return t_; }
    const double &s() const { return s_; }

    // Initialize from 4x4 transformation matrix
    Sim3Pose &from_matrix(const Eigen::Matrix4d &T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        // Compute scale as the average norm of the rows of the rotation matrix
        Eigen::Vector3d row_norms = R.rowwise().norm();
        s_ = row_norms.mean();
        R_ = R / s_;
        t_ = T.block<3, 1>(0, 3);
        return *this;
    }

    // Initialize from SE(3) matrix (scale = 1.0)
    Sim3Pose &from_se3_matrix(const Eigen::Matrix4d &T) {
        s_ = 1.0;
        R_ = T.block<3, 3>(0, 0);
        t_ = T.block<3, 1>(0, 3);
        return *this;
    }

    // Get 4x4 transformation matrix
    Eigen::Matrix4d matrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = s_ * R_;
        T.block<3, 1>(0, 3) = t_;
        return T;
    }

    // Get inverse transformation
    Sim3Pose inverse() const {
        Eigen::Matrix3d R_inv = R_.transpose();
        double s_inv = 1.0 / s_;
        Eigen::Vector3d t_inv = -s_inv * R_inv * t_;
        return Sim3Pose(R_inv, t_inv, s_inv);
    }

    // Get 4x4 inverse transformation matrix
    Eigen::Matrix4d inverse_matrix() const {
        Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d sR_inv = (1.0 / s_) * R_.transpose();
        T_inv.block<3, 3>(0, 0) = sR_inv;
        T_inv.block<3, 1>(0, 3) = -sR_inv * t_;
        return T_inv;
    }

    // Convert to SE(3) matrix
    Eigen::Matrix4d to_se3_matrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R_;
        T.block<3, 1>(0, 3) = t_ / s_;
        return T;
    }

    // Copy this transformation
    Sim3Pose copy() const { return Sim3Pose(R_, t_, s_); }

    // Transform a 3D point
    Eigen::Vector3d map(const Eigen::Vector3d &p3d) const { return s_ * R_ * p3d + t_; }

    // Transform a set of 3D points [Nx3]
    Eigen::MatrixXd map_points(const Eigen::MatrixXd &points) const {
        assert(points.cols() == 3 && "Points must be Nx3 matrix");
        return ((s_ * R_ * points.transpose()).colwise() + t_).transpose();
    }

    // Matrix multiplication operator
    Sim3Pose operator*(const Sim3Pose &other) const {
        double s_res = s_ * other.s_;
        Eigen::Matrix3d R_res = R_ * other.R_;
        Eigen::Vector3d t_res = s_ * R_ * other.t_ + t_;
        return Sim3Pose(R_res, t_res, s_res);
    }

    // Matrix multiplication with 4x4 matrix
    Sim3Pose operator*(const Eigen::Matrix4d &other) const {
        Eigen::Matrix3d R_other = other.block<3, 3>(0, 0);
        Eigen::Vector3d row_norms = R_other.rowwise().norm();
        double s_other = row_norms.mean();
        R_other = R_other / s_other;
        Eigen::Vector3d t_other = other.block<3, 1>(0, 3);

        double s_res = s_ * s_other;
        Eigen::Matrix3d R_res = R_ * R_other;
        Eigen::Vector3d t_res = s_ * R_ * t_other + t_;
        return Sim3Pose(R_res, t_res, s_res);
    }

    // Equality operator
    bool operator==(const Sim3Pose &other) const {
        const double epsilon = 1e-10;
        return R_.isApprox(other.R_, epsilon) && t_.isApprox(other.t_, epsilon) &&
               std::abs(s_ - other.s_) < epsilon;
    }

    // Inequality operator
    bool operator!=(const Sim3Pose &other) const { return !(*this == other); }

    // String output operator
    std::string to_string() const {
        std::stringstream ss;
        ss << "Sim3Pose(R=" << R_.format(Eigen::IOFormat(Eigen::FullPrecision))
           << ", t=" << t_.transpose().format(Eigen::IOFormat(Eigen::FullPrecision)) << ", s=" << s_
           << ")";
        return ss.str();
    }

    // Stream output operator
    friend std::ostream &operator<<(std::ostream &os, const Sim3Pose &pose) {
        os << "Sim3Pose(R=" << pose.R_ << ", t=" << pose.t_.transpose() << ", s=" << pose.s_ << ")";
        return os;
    }

  private:
    Eigen::Matrix3d R_; // Rotation matrix (normalized)
    Eigen::Vector3d t_; // Translation vector
    double s_;          // Scale factor
};

} // namespace pyslam