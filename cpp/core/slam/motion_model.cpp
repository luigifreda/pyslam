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
#include "motion_model.h"

#include <cmath>

namespace pyslam {

Eigen::Isometry3d isometry_from_quaternion_position(const Eigen::Quaterniond &orientation,
                                                    const Eigen::Vector3d &position) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = orientation.normalized().toRotationMatrix();
    pose.translation() = position;
    return pose;
}

MotionModelBase::MotionModelBase() = default;

MotionModelBase::MotionModelBase(std::optional<double> timestamp_,
                                 const Eigen::Vector3d &initial_position,
                                 const Eigen::Quaterniond &initial_orientation)
    : timestamp(timestamp_), position(initial_position), orientation(initial_orientation) {}

Eigen::Isometry3d MotionModelBase::current_pose() const {
    return isometry_from_quaternion_position(orientation, position);
}

std::pair<Eigen::Isometry3d, bool>
MotionModelBase::predict_pose(std::optional<double>, const Eigen::Vector3d *,
                              const Eigen::Quaterniond *) {
    return {current_pose(), false};
}

void MotionModelBase::update_pose(std::optional<double>, const Eigen::Vector3d &,
                                  const Eigen::Quaterniond &) {}

void MotionModelBase::apply_correction(const Eigen::Isometry3d &) {}

void MotionModelBase::reset() {
    timestamp.reset();
    position.setZero();
    orientation = Eigen::Quaterniond::Identity();
    is_ok = false;
    initialized = false;
}

MotionModel::MotionModel() = default;

std::pair<Eigen::Isometry3d, bool>
MotionModel::predict_pose(std::optional<double> timestamp_in, const Eigen::Vector3d *prev_position,
                          const Eigen::Quaterniond *prev_orientation) {
    if (prev_position != nullptr) {
        position = *prev_position;
    }
    if (prev_orientation != nullptr) {
        orientation = *prev_orientation;
    }
    if (!initialized) {
        return {current_pose(), false};
    }
    Eigen::Quaterniond predicted_orientation = delta_orientation * orientation;
    const Eigen::Vector3d predicted_position =
        position + delta_orientation._transformVector(delta_position);
    predicted_orientation.normalize();
    (void)timestamp_in;
    return {isometry_from_quaternion_position(predicted_orientation, predicted_position), false};
}

void MotionModel::update_pose(std::optional<double> timestamp_in, const Eigen::Vector3d &new_position,
                              const Eigen::Quaterniond &new_orientation) {
    if (initialized) {
        delta_position = new_position - position;
        delta_orientation = new_orientation * orientation.inverse();
        delta_orientation.normalize();
    }
    timestamp = timestamp_in;
    position = new_position;
    orientation = new_orientation;
    initialized = true;
}

void MotionModel::apply_correction(const Eigen::Isometry3d &correction) {
    Eigen::Isometry3d current = correction * current_pose();
    position = current.translation();
    orientation = Eigen::Quaterniond(current.linear());
    orientation.normalize();
    const Eigen::Quaterniond q_corr(correction.linear());
    delta_orientation = q_corr * delta_orientation;
    delta_position = q_corr._transformVector(delta_position);
}

void MotionModel::reset() {
    MotionModelBase::reset();
    delta_position.setZero();
    delta_orientation = Eigen::Quaterniond::Identity();
}

MotionModelDamping::MotionModelDamping(double damping) : damp(damping) {}

std::pair<Eigen::Isometry3d, bool> MotionModelDamping::predict_pose(
    std::optional<double> timestamp_in, const Eigen::Vector3d *prev_position,
    const Eigen::Quaterniond *prev_orientation) {
    if (prev_position != nullptr) {
        position = *prev_position;
    }
    if (prev_orientation != nullptr) {
        orientation = *prev_orientation;
    }
    if (!initialized) {
        return {current_pose(), false};
    }
    const double dt = (timestamp && timestamp_in) ? (*timestamp_in - *timestamp) : 0.0;
    Eigen::AngleAxisd delta_angle(v_angular_angle * dt * damp, v_angular_axis.normalized());
    Eigen::Quaterniond delta_orientation(delta_angle);
    Eigen::Quaterniond predicted_orientation = delta_orientation * orientation;
    const Eigen::Vector3d predicted_position =
        position + delta_orientation._transformVector(v_linear * dt * damp);
    predicted_orientation.normalize();
    return {isometry_from_quaternion_position(predicted_orientation, predicted_position), false};
}

void MotionModelDamping::update_pose(std::optional<double> timestamp_in,
                                     const Eigen::Vector3d &new_position,
                                     const Eigen::Quaterniond &new_orientation) {
    if (initialized && timestamp && timestamp_in) {
        const double dt = *timestamp_in - *timestamp;
        if (dt != 0.0) {
            v_linear = (new_position - position) / dt;
            Eigen::Quaterniond delta_q = new_orientation * orientation.inverse();
            delta_q.normalize();
            Eigen::AngleAxisd delta_angle(delta_q);
            double angle = delta_angle.angle();
            Eigen::Vector3d axis = delta_angle.axis();
            constexpr double kPi = 3.14159265358979323846;
            if (angle > kPi) {
                axis *= -1.0;
                angle = 2.0 * kPi - angle;
            }
            v_angular_axis = axis;
            v_angular_angle = angle / dt;
        }
    }
    timestamp = timestamp_in;
    position = new_position;
    orientation = new_orientation;
    initialized = true;
}

void MotionModelDamping::apply_correction(const Eigen::Isometry3d &correction) {
    Eigen::Isometry3d current = correction * current_pose();
    position = current.translation();
    orientation = Eigen::Quaterniond(current.linear());
    orientation.normalize();
}

void MotionModelDamping::reset() {
    MotionModelBase::reset();
    v_linear.setZero();
    v_angular_angle = 0.0;
    v_angular_axis = Eigen::Vector3d::UnitX();
}

} // namespace pyslam
