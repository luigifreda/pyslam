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

#include <Eigen/Geometry>
#include <optional>

namespace pyslam {

// Same names and fields as pyslam/slam/motion_model.py.
class MotionModelBase {
  public:
    MotionModelBase();
    MotionModelBase(std::optional<double> timestamp, const Eigen::Vector3d &initial_position,
                    const Eigen::Quaterniond &initial_orientation);
    virtual ~MotionModelBase() = default;

    std::optional<double> timestamp;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    bool is_ok = false;
    bool initialized = false;

    Eigen::Isometry3d current_pose() const;
    virtual std::pair<Eigen::Isometry3d, bool> predict_pose(std::optional<double> timestamp,
                                                            const Eigen::Vector3d *prev_position,
                                                            const Eigen::Quaterniond *prev_orientation);
    virtual void update_pose(std::optional<double> timestamp, const Eigen::Vector3d &new_position,
                             const Eigen::Quaterniond &new_orientation);
    virtual void apply_correction(const Eigen::Isometry3d &correction);
    virtual void reset();
};

class MotionModel : public MotionModelBase {
  public:
    MotionModel();
    std::pair<Eigen::Isometry3d, bool> predict_pose(std::optional<double> timestamp,
                                                    const Eigen::Vector3d *prev_position,
                                                    const Eigen::Quaterniond *prev_orientation) override;
    void update_pose(std::optional<double> timestamp, const Eigen::Vector3d &new_position,
                     const Eigen::Quaterniond &new_orientation) override;
    void apply_correction(const Eigen::Isometry3d &correction) override;
    void reset() override;

    Eigen::Vector3d delta_position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond delta_orientation = Eigen::Quaterniond::Identity();
};

class MotionModelDamping : public MotionModelBase {
  public:
    explicit MotionModelDamping(double damping = 0.95);
    std::pair<Eigen::Isometry3d, bool> predict_pose(std::optional<double> timestamp,
                                                    const Eigen::Vector3d *prev_position,
                                                    const Eigen::Quaterniond *prev_orientation) override;
    void update_pose(std::optional<double> timestamp, const Eigen::Vector3d &new_position,
                     const Eigen::Quaterniond &new_orientation) override;
    void apply_correction(const Eigen::Isometry3d &correction) override;
    void reset() override;

    Eigen::Vector3d v_linear = Eigen::Vector3d::Zero();
    double v_angular_angle = 0.0;
    Eigen::Vector3d v_angular_axis = Eigen::Vector3d::UnitX();
    double damp = 0.95;
};

Eigen::Isometry3d isometry_from_quaternion_position(const Eigen::Quaterniond &orientation,
                                                    const Eigen::Vector3d &position);

} // namespace pyslam
