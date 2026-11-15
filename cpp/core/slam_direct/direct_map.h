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

#include "camera/camera_pose.h"
#include "common/eigen_aliases.h"

#include <opencv2/core.hpp>

#include <utility>
#include <map>
#include <cstdint>
#include <vector>

namespace pyslam {

// A keyframe observation references one persistent landmark. Sensor depth/right
// pixels remain immutable measurements; position_c is the initial geometry.
struct DirectPoint {
    int64_t id = -1;
    Vec2d uv_right = Vec2d::Zero();
    bool has_right = false;
    bool outlier = false;
    bool right_outlier = false;
    bool depth_outlier = false;
    Vec2d uv = Vec2d::Zero();
    double depth = 0.0;
    Vec3d position_c = Vec3d::Zero();
    bool has_depth = false;
};

// Keyframe IDs are insertion indices; observations refer to persistent landmark IDs.
struct DirectKeyframe {
    int id = -1;
    int img_id = -1;
    double timestamp = 0.0;
    CameraPose pose;
    cv::Mat gray; // Kept only by semi_direct, while this keyframe is the tracking reference or owns seeds.
    std::vector<DirectPoint> points;
};

// Optimized world position, shared by all keyframe observations of this ID.
struct DirectLandmark {
    Vec3d position_w = Vec3d::Zero();
    bool bad = false;
    // Monocular scale anchor: inverse depth in prior_keyframe's camera.
    bool has_inverse_prior = false;
    double inverse_depth_mu = 0.0;
    double inverse_depth_sigma2 = 0.0;
    int prior_keyframe = -1;
};

// An accepted frame pose with a reference-keyframe transform for later BA corrections.
struct DirectTrajectoryPose {
    int reference_keyframe = -1; // Index into keyframes_; -1 means no reference was available.
    // T_frame_world = T_frame_reference * T_reference_world.
    Eigen::Matrix4d T_frame_reference = Eigen::Matrix4d::Identity();
    int img_id = -1;
    double timestamp = 0.0;
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
};

// The archive is not window-limited; only the backend graph uses the latest keyframes.
// Tracking, optimization and snapshots run sequentially; this container has no worker locks.
class DirectMap {
  public:
    const std::vector<DirectKeyframe> &keyframes() const { return keyframes_; }
    const std::vector<DirectTrajectoryPose> &trajectory() const { return trajectory_; }
    const std::map<int64_t, DirectLandmark> &landmarks() const { return landmarks_; }
    std::vector<DirectKeyframe> &keyframes_mutable() { return keyframes_; }
    std::map<int64_t, DirectLandmark> &landmarks_mutable() { return landmarks_; }

    // Non-keyframe poses keep their transform relative to their reference keyframe.
    // Reconstruct them after BA so exported trajectories use the corrected map.
    void update_trajectory() {
        for (auto &pose : trajectory_) {
            if (pose.reference_keyframe >= 0)
                pose.Tcw = pose.T_frame_reference * keyframes_.at(pose.reference_keyframe).pose.Tcw();
        }
    }

    // A seed promoted after the keyframe was stored still has to join the observer index.
    void note_observation(int keyframe_id, int64_t landmark_id) {
        auto &list = observers_[landmark_id];
        if (list.empty() || list.back() != keyframe_id) {
            list.push_back(keyframe_id);
        }
    }

    void add_keyframe(DirectKeyframe keyframe) {
        for (const DirectPoint &obs : keyframe.points) {
            auto &list = observers_[obs.id];
            if (list.empty() || list.back() != keyframe.id) list.push_back(keyframe.id);
        }
        keyframes_.push_back(std::move(keyframe));
    }

    // Keyframe IDs that observed a landmark when they were added, in insertion
    // order, or nullptr. Outlier flags are not tracked here; callers recheck them.
    const std::vector<int> *observers(int64_t landmark_id) const {
        const auto it = observers_.find(landmark_id);
        return it == observers_.end() ? nullptr : &it->second;
    }

    void add_pose(int img_id, double timestamp, const Eigen::Isometry3d &Tcw) {
        DirectTrajectoryPose pose;
        pose.img_id = img_id;
        pose.timestamp = timestamp;
        pose.Tcw = Tcw.matrix();
        if (!keyframes_.empty()) {
            pose.reference_keyframe = keyframes_.back().id;
            pose.T_frame_reference = Tcw.matrix() * keyframes_.back().pose.get_inverse_matrix();
        }
        trajectory_.push_back(std::move(pose));
    }

    const DirectKeyframe *latest_keyframe() const {
        return keyframes_.empty() ? nullptr : &keyframes_.back();
    }

    void clear() {
        keyframes_.clear();
        trajectory_.clear();
        landmarks_.clear();
        observers_.clear();
    }

  private:
    std::map<int64_t, DirectLandmark> landmarks_;
    std::vector<DirectKeyframe> keyframes_;
    std::vector<DirectTrajectoryPose> trajectory_;
    std::map<int64_t, std::vector<int>> observers_; // landmark ID -> observing keyframe IDs
};

} // namespace pyslam
