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

#include "slam_direct/direct_frontend.h"
#include "common/feature_trails.h"

#include "slam_direct/depth_measurement.h"
#include "slam_direct/features/corner_detection.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace pyslam {
namespace {

// Keep measured pixels as camera-frame points for points_from_camera.
std::vector<DirectPoint> measured_points(const std::vector<Vec2d> &pixels,
                                         const std::vector<PixelMeasurement> &measured) {
    std::vector<DirectPoint> points;
    points.reserve(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (!measured[i].ok) {
            continue;
        }
        DirectPoint point;
        point.uv = pixels[i];
        point.depth = measured[i].X.z();
        point.position_c = measured[i].X;
        point.has_depth = true;
        points.push_back(point);
    }
    return points;
}

} // namespace

DirectSlam::DirectSlam(CameraPtr camera, const ConfigDict &config)
    : camera_(std::move(camera)),
      params_(DirectSlamParams::from_config(config, camera_ && camera_->sensor_type == SensorType::MONOCULAR)) {
    if (!camera_) {
        throw std::invalid_argument("DirectSlam: camera is null");
    }
    tracker_ = make_pose_tracker(params_.mode, camera_, params_);
    // Raw-pixel models project with their distortion Jacobian. A distorted linear
    // pinhole does not, so it is still rejected.
    if (!camera_->projects_raw_keypoints() &&
        (camera_->type != CameraType::PINHOLE || camera_->is_distorted)) {
        throw std::invalid_argument(
            "DirectSlam: distorted images require a raw-pixel camera model");
    }
    if (camera_->sensor_type == SensorType::MONOCULAR && !tracker_->supports_monocular()) {
        throw std::invalid_argument("DirectSlam: monocular input requires semi_direct mode");
    }
    if (camera_->sensor_type == SensorType::MONOCULAR) {
        mono_init_ = std::make_unique<MonocularInitializer>(camera_, params_);
    }
    if (!(camera_->fx > 0.0) || !(camera_->fy > 0.0)) {
        throw std::invalid_argument("DirectSlam: camera intrinsics are not set");
    }
}

std::string DirectSlam::status_name() const {
    switch (status_) {
    case DirectTrackingStatus::Init:
        return "INIT";
    case DirectTrackingStatus::Good:
        return "TRACKING_GOOD";
    case DirectTrackingStatus::Bad:
        return "TRACKING_BAD";
    case DirectTrackingStatus::Lost:
        return "LOST";
    }
    return "LOST";
}

// The shared Frame renderer keeps direct-mode overlays identical to main_slam.py.
cv::Mat DirectSlam::draw_feature_trails(const cv::Mat &image, bool with_level_radius,
                                       int trail_max_length) const {
    cv::Mat shown = image.clone();
    if (status_ == DirectTrackingStatus::Lost) return shown;
    // Before the monocular map exists, show the initializer's reference tracks.
    const bool initializing = mono_init_ && map_.keyframes().empty();
    for (const auto &point : initializing ? mono_init_->tracks() : live_points_) {
        const size_t length = std::min(point.trail.size(), size_t(std::max(0, trail_max_length)));
        std::vector<cv::Point> trail(point.trail.begin(), point.trail.begin() + length);
        const cv::Point uv(std::floor(point.uv.x()), std::floor(point.uv.y()));
        if (with_level_radius)
            draw_feature_trail<true>(shown, uv, point.id, trail, point.observations);
        else
            draw_feature_trail<false>(shown, uv, point.id, trail, point.observations);
    }
    return shown;
}

void DirectSlam::reset() {
    map_.clear();
    status_ = DirectTrackingStatus::Init;
    current_pose_ = CameraPose();
    velocity_ = Eigen::Isometry3d::Identity();
    has_velocity_ = false;
    previous_pose_ = Eigen::Isometry3d::Identity();
    last_gray_.release();
    live_points_.clear();
    tracker_->reset();
    if (mono_init_) mono_init_->reset();
    consecutive_failures_ = 0;
    last_inliers_ = 0;
    next_point_id_ = 0;
    last_img_id_ = -1;
    last_ba_ = DirectBAResult();
}

std::vector<DirectPoint> DirectSlam::make_points(const cv::Mat &gray, const cv::Mat &gray_right,
                                                 const cv::Mat &depth) const {
    const std::vector<Vec2d> pixels = detect_corners(gray, params_);
    if (camera_->sensor_type != SensorType::RGBD && gray_right.empty()) {
        throw std::invalid_argument("DirectSlam: stereo tracking requires the right image");
    }
    return measured_points(pixels, measure_pixels(*camera_, params_, gray, gray_right, depth, pixels));
}

void DirectSlam::record_keyframe(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                                 int img_id, double timestamp, const Eigen::Isometry3d &Tcw) {
    DirectKeyframe keyframe;
    keyframe.id = static_cast<int>(map_.keyframes().size());
    keyframe.img_id = img_id;
    keyframe.timestamp = timestamp;
    keyframe.pose = CameraPose(Tcw);
    const std::vector<PixelMeasurement> measured =
        measure_live_points(*camera_, params_, gray, gray_right, depth, Tcw, live_points_);
    for (size_t i = 0; i < live_points_.size(); ++i) {
        const auto &point = live_points_[i];
        DirectPoint observation;
        observation.id = point.id;
        observation.uv = point.uv;
        observation.position_c = Tcw * point.position_w;
        if (camera_->sensor_type == SensorType::RGBD) {
            observation.depth = measured[i].sensor_depth;
            observation.has_depth = measured[i].ok;
        } else if (measured[i].has_right) {
            // Store the sensor measurement, never a depth derived from the pose
            // being optimized. Otherwise BA would constrain the estimate to itself.
            observation.uv_right = measured[i].uv_right;
            observation.has_right = true;
            observation.depth = measured[i].X.z();
            observation.has_depth = true;
        }
        keyframe.points.push_back(observation);
    }
    if (tracker_->uses_keyframe_images()) {
        // The reference image is needed while this keyframe owns the tracking patches or immature seeds.
        keyframe.gray = gray.clone();
    }
    const int keyframe_id = keyframe.id;
    map_.add_keyframe(std::move(keyframe));
    if (tracker_->uses_keyframe_images()) {
        for (LivePoint &point : live_points_) {
            point.uv_ref = point.uv;
            point.reference_keyframe = keyframe_id;
        }
    }
}

// Apply one coherent map correction to trajectory, motion prediction and live landmarks.
DirectBAResult DirectSlam::bundle_adjustment() {
    last_ba_ = optimize_direct_map(map_, *camera_, params_);
    if (last_ba_.applied) refresh_tracking_state_from_map();
    return last_ba_;
}

void DirectSlam::refresh_tracking_state_from_map() {
    const auto &trajectory = map_.trajectory();
    if (!trajectory.empty()) {
        previous_pose_.matrix() = trajectory.back().Tcw;
        current_pose_ = CameraPose(previous_pose_);
        if (trajectory.size() >= 2) {
            velocity_.matrix() = trajectory.back().Tcw * trajectory[trajectory.size() - 2].Tcw.inverse();
        } else has_velocity_ = false;
    }
    std::set<int64_t> rejected_current;
    const auto *latest = map_.latest_keyframe();
    if (latest && latest->img_id == last_img_id_)
        for (const auto &obs : latest->points)
            if (obs.outlier) rejected_current.insert(obs.id);
    std::vector<LivePoint> corrected;
    for (auto point : live_points_) {
        const auto it = map_.landmarks().find(point.id);
        if (it == map_.landmarks().end() || it->second.bad || rejected_current.count(point.id)) continue;
        point.position_w = it->second.position_w;
        corrected.push_back(std::move(point));
    }
    live_points_ = std::move(corrected);
}

void DirectSlam::apply_keyframe_poses(const std::vector<Eigen::Matrix4d> &poses, const std::vector<double> &scales) {
    auto &keyframes = map_.keyframes_mutable();
    if (poses.size() != keyframes.size()) {
        throw std::invalid_argument("DirectSlam::apply_keyframe_poses: one pose is required per keyframe");
    }
    for (const Eigen::Matrix4d &pose : poses) {
        if (!pose.allFinite() || !pose.inverse().allFinite()) {
            throw std::invalid_argument("DirectSlam::apply_keyframe_poses: pose is not finite");
        }
    }
    if (!scales.empty() && scales.size() != keyframes.size()) {
        throw std::invalid_argument("DirectSlam::apply_keyframe_poses: one scale is required per keyframe");
    }
    for (const double scale : scales) {
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::invalid_argument("DirectSlam::apply_keyframe_poses: scale must be positive and finite");
        }
    }
    const auto scale_of = [&](size_t i) { return scales.empty() ? 1.0 : scales[i]; };

    std::vector<Eigen::Matrix4d> previous;
    previous.reserve(keyframes.size());
    for (const DirectKeyframe &keyframe : keyframes) previous.push_back(keyframe.pose.Tcw());

    // The earliest inlier observation anchors the landmark. A later keyframe pose
    // change must not drag a point that was created on an earlier, fixed keyframe.
    std::map<int64_t, size_t> reference;
    for (size_t i = 0; i < keyframes.size(); ++i) {
        for (const DirectPoint &observation : keyframes[i].points) {
            if (!observation.outlier) reference.emplace(observation.id, i);
        }
    }
    for (size_t i = 0; i < keyframes.size(); ++i) {
        for (const DirectPoint &observation : keyframes[i].points) {
            reference.emplace(observation.id, i);
        }
    }

    for (size_t i = 0; i < keyframes.size(); ++i) {
        keyframes[i].pose = CameraPose(poses[i]);
        const double scale = scale_of(i);
        if (scale != 1.0) {
            // Initial camera-frame geometry of the keyframe's observations follows its scale.
            for (DirectPoint &observation : keyframes[i].points) {
                observation.position_c /= scale;
                if (camera_->sensor_type == SensorType::MONOCULAR) observation.depth /= scale;
            }
        }
    }
    for (auto &[id, landmark] : map_.landmarks_mutable()) {
        if (landmark.has_inverse_prior && landmark.prior_keyframe >= 0 &&
            static_cast<size_t>(landmark.prior_keyframe) < keyframes.size()) {
            // Depth in the prior keyframe shrinks by 1/s, so inverse depth grows by s.
            const double scale = scale_of(static_cast<size_t>(landmark.prior_keyframe));
            landmark.inverse_depth_mu *= scale;
            landmark.inverse_depth_sigma2 *= scale * scale;
        }
        const auto found = reference.find(id);
        if (found == reference.end()) continue;
        const size_t anchor = found->second;
        const double scale = scale_of(anchor);
        const Eigen::Matrix4d delta = poses[anchor].inverse() * previous[anchor];
        if (scale == 1.0 && (delta - Eigen::Matrix4d::Identity()).norm() < 1e-9) continue;
        // Into the anchor camera with the old pose, rescale, back out with the new pose.
        const Eigen::Vector3d X_anchor =
            (previous[anchor].topLeftCorner<3, 3>() * landmark.position_w + previous[anchor].topRightCorner<3, 1>()) /
            scale;
        const Eigen::Matrix4d Twc = poses[anchor].inverse();
        const Eigen::Vector3d moved = Twc.topLeftCorner<3, 3>() * X_anchor + Twc.topRightCorner<3, 1>();
        if (!moved.allFinite()) {
            throw std::invalid_argument("DirectSlam::apply_keyframe_poses: landmark correction is not finite");
        }
        landmark.position_w = moved;
    }
    if (!scales.empty()) {
        // A frame's offset from its reference keyframe is camera-frame geometry too.
        for (DirectTrajectoryPose &pose : map_.trajectory_mutable()) {
            if (pose.reference_keyframe >= 0 && static_cast<size_t>(pose.reference_keyframe) < scales.size())
                pose.T_frame_reference.topRightCorner<3, 1>() /= scales[static_cast<size_t>(pose.reference_keyframe)];
        }
        tracker_->apply_keyframe_scales(scales);
    }
    map_.update_trajectory();
    refresh_tracking_state_from_map();
}

void DirectSlam::set_status_from_inliers(int inliers) {
    last_inliers_ = inliers;
    if (inliers > params_.num_features_tracking) {
        status_ = DirectTrackingStatus::Good;
    } else if (inliers > params_.num_features_tracking_bad) {
        status_ = DirectTrackingStatus::Bad;
    } else {
        status_ = DirectTrackingStatus::Lost;
    }
}

std::vector<LivePoint> DirectSlam::points_from_camera(const std::vector<DirectPoint> &points,
                                                           const Eigen::Isometry3d &Tcw) {
    std::vector<LivePoint> live;
    live.reserve(points.size());
    const Eigen::Isometry3d Twc = Tcw.inverse();
    for (const DirectPoint &point : points) {
        if (!point.has_depth || !metric_point_ok(*camera_, point.position_c)) {
            continue;
        }
        // IDs survive temporal tracking and connect later keyframe observations in BA.
        LivePoint live_point = make_live_point(next_point_id_++, point.uv, Twc * point.position_c);
        map_.landmarks_mutable()[live_point.id] = {live_point.position_w, false};
        live.push_back(live_point);
    }
    return live;
}

void DirectSlam::supplement_live_points(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                                        const Eigen::Isometry3d &Tcw, int img_id, double timestamp) {
    const int room = params_.num_features - static_cast<int>(live_points_.size());
    if (room < 10) {
        return;
    }
    OccupancyMask occupied(gray.size(), params_);
    for (const LivePoint &point : live_points_) {
        occupied.block(point.uv);
    }
    const std::vector<Vec2d> pixels = detect_corners(gray, params_, occupied.mask(), room);
    if (pixels.empty()) {
        return;
    }
    const std::vector<DirectPoint> fresh =
        measured_points(pixels, measure_pixels(*camera_, params_, gray, gray_right, depth, pixels));
    if (fresh.empty()) {
        return;
    }
    const std::vector<LivePoint> extra = points_from_camera(fresh, Tcw);
    live_points_.insert(live_points_.end(), extra.begin(), extra.end());
}

void DirectSlam::initialize(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                            int img_id, double timestamp) {
    if (camera_->sensor_type == SensorType::STEREO && gray_right.empty()) {
        throw std::invalid_argument("DirectSlam: stereo tracking requires the right image");
    }
    if (camera_->sensor_type == SensorType::RGBD && depth.empty()) {
        throw std::invalid_argument("DirectSlam: RGBD tracking requires a depth image");
    }
    if (camera_->sensor_type == SensorType::MONOCULAR) {
        // No sensor depth: wait for a two-view geometry with enough parallax.
        status_ = DirectTrackingStatus::Init;
        if (mono_init_->add_frame(gray, img_id, timestamp)) {
            initialize_monocular_map(gray, img_id, timestamp, mono_init_->result());
            mono_init_->reset();
        } else {
            last_inliers_ = mono_init_->num_tracks();
        }
        return;
    }
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const TrackingFrame frame{gray, gray_right, depth, img_id};
    TrackerMapState state = map_state();
    // The first depth-backed frame defines the world origin; recovery never re-enters here.
    const auto points = make_points(gray, gray_right, depth);
    last_inliers_ = static_cast<int>(points.size());
    if (last_inliers_ <= params_.num_features_tracking_bad) {
        status_ = DirectTrackingStatus::Lost;
        return;
    }
    live_points_ = points_from_camera(points, identity);
    record_keyframe(gray, gray_right, depth, img_id, timestamp, identity);
    tracker_->on_keyframe(frame, map_.latest_keyframe()->id, state);
    set_status_from_inliers(last_inliers_);
    last_img_id_ = img_id;
    last_gray_ = gray.clone();
    current_pose_ = CameraPose(identity);
    previous_pose_ = identity;
    has_velocity_ = false;
    consecutive_failures_ = 0;
    map_.add_pose(img_id, timestamp, identity);
}

void DirectSlam::initialize_monocular_map(const cv::Mat &gray, int img_id, double timestamp,
                                          const MonocularInitResult &init) {
    const cv::Mat empty;
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const Eigen::Isometry3d &T_cur_w = init.T_cur_ref; // The reference camera is the world frame.
    live_points_.clear();
    live_points_.reserve(init.X_ref.size());
    for (size_t i = 0; i < init.X_ref.size(); ++i) {
        // uv_ref is reset by record_keyframe below (semi_direct keeps keyframe images).
        LivePoint point = make_live_point(next_point_id_++, init.uv_ref[i], init.X_ref[i]);
        point.trail = init.trails[i];
        point.observations = static_cast<int>(point.trail.size());
        DirectLandmark landmark;
        landmark.position_w = point.position_w;
        // Same monocular scale anchor as a promoted seed, on the reference keyframe.
        if (init.inverse_depth_sigma2[i] > 0.0) {
            landmark.set_inverse_prior(1.0 / init.X_ref[i].z(), init.inverse_depth_sigma2[i],
                                       static_cast<int>(map_.keyframes().size()));
        }
        map_.landmarks_mutable()[point.id] = landmark;
        live_points_.push_back(std::move(point));
    }
    // Reference keyframe at the identity, then the current frame.
    record_keyframe(init.ref_gray, empty, empty, init.ref_img_id, init.ref_timestamp, identity);
    map_.add_pose(init.ref_img_id, init.ref_timestamp, identity);
    for (size_t i = 0; i < live_points_.size(); ++i) live_points_[i].uv = init.uv_cur[i];
    record_keyframe(gray, empty, empty, img_id, timestamp, T_cur_w);
    map_.add_pose(img_id, timestamp, T_cur_w);
    TrackerMapState state = map_state();
    const TrackingFrame frame{gray, empty, empty, img_id};
    tracker_->on_keyframe(frame, map_.latest_keyframe()->id, state);

    // Constant-velocity seed: the average per-frame motion since the reference.
    const int frames = std::max(1, img_id - init.ref_img_id);
    const Eigen::AngleAxisd rotation(T_cur_w.linear());
    velocity_ = Eigen::Isometry3d::Identity();
    velocity_.linear() = Eigen::AngleAxisd(rotation.angle() / frames, rotation.axis()).toRotationMatrix();
    velocity_.translation() = T_cur_w.translation() / static_cast<double>(frames);
    has_velocity_ = true;
    previous_pose_ = T_cur_w;
    current_pose_ = CameraPose(T_cur_w);
    last_gray_ = gray.clone();
    last_img_id_ = img_id;
    consecutive_failures_ = 0;
    set_status_from_inliers(static_cast<int>(live_points_.size()));
    if (params_.local_ba) bundle_adjustment();
}

void DirectSlam::track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
                       std::optional<double> timestamp) {
    const cv::Mat gray = as_gray8(img);
    if (gray.empty()) {
        throw std::invalid_argument("DirectSlam::track: empty image");
    }
    const cv::Mat gray_right = as_gray8(img_right);
    if (gray.cols != camera_->width || gray.rows != camera_->height)
        throw std::invalid_argument("DirectSlam: image size does not match camera calibration");
    if (camera_->sensor_type == SensorType::STEREO &&
        (gray_right.empty() || gray_right.size() != gray.size()))
        throw std::invalid_argument("DirectSlam: stereo requires matching left/right image sizes");
    if (camera_->sensor_type == SensorType::RGBD &&
        (depth.empty() || depth.size() != gray.size() ||
         (depth.type() != CV_16UC1 && depth.type() != CV_32FC1 && depth.type() != CV_64FC1)))
        throw std::invalid_argument("DirectSlam: RGBD requires a matching uint16/float depth image");
    const double stamp = timestamp.value_or(static_cast<double>(img_id));
    if (map_.keyframes().empty()) {
        initialize(gray, gray_right, depth, img_id, stamp);
        return;
    }

    const TrackingFrame frame{gray, gray_right, depth, img_id};
    const Eigen::Isometry3d T_pred_w = has_velocity_ ? velocity_ * previous_pose_ : previous_pose_;
    const TrackingResult result =
        tracker_->track(TrackingInput{frame, last_gray_, live_points_, map_, previous_pose_, T_pred_w});
    if (!result.accepted) {
        note_tracking_failure(result.inliers);
        return; // Keep the last accepted image and landmarks for a possible recovery.
    }
    const Eigen::Isometry3d &T_cur_w = result.T_cur_w;
    consecutive_failures_ = 0;
    set_status_from_inliers(result.inliers);
    velocity_ = T_cur_w * previous_pose_.inverse();
    has_velocity_ = true;

    std::vector<LivePoint> kept;
    kept.reserve(live_points_.size());
    for (size_t i = 0; i < live_points_.size(); ++i) {
        if (!result.tracked_ok[i]) continue;
        LivePoint point = live_points_[i];
        point.uv = result.tracked_uv[i];
        // Do not draw a continuous trail across an input-frame gap or a tracking failure.
        if (img_id != last_img_id_ + 1) point.trail.clear();
        point.trail.insert(point.trail.begin(), cv::Point(std::floor(point.uv.x()), std::floor(point.uv.y())));
        if (point.trail.size() > 64) point.trail.resize(64);
        ++point.observations;
        kept.push_back(std::move(point));
    }
    live_points_ = std::move(kept);

    TrackerMapState state = map_state();
    const AcceptOutcome outcome = tracker_->after_accept(frame, T_cur_w, state);

    // Low track support triggers replenishment; automatic BA also forces periodic keyframes.
    const bool needs_features = static_cast<int>(live_points_.size()) < params_.num_features_needed_for_keyframe;
    if (needs_features)
        supplement_live_points(gray, gray_right, depth, T_cur_w, img_id, stamp);
    const bool new_keyframe =
        outcome.force_keyframe || needs_features ||
        (params_.local_ba && img_id - map_.latest_keyframe()->img_id >= params_.max_keyframe_interval);
    if (new_keyframe) {
        record_keyframe(gray, gray_right, depth, img_id, stamp, T_cur_w);
        tracker_->on_keyframe(frame, map_.latest_keyframe()->id, state);
    }

    previous_pose_ = T_cur_w;
    current_pose_ = CameraPose(T_cur_w);
    last_gray_ = gray.clone();
    last_img_id_ = img_id;
    // Register the accepted frame before BA so its reference-relative pose is corrected too.
    map_.add_pose(img_id, stamp, T_cur_w);
    if (new_keyframe && params_.local_ba) bundle_adjustment();
}

void DirectSlam::note_tracking_failure(int inliers) {
    ++consecutive_failures_;
    has_velocity_ = false;
    last_inliers_ = inliers;
    status_ = DirectTrackingStatus::Lost;
}

} // namespace pyslam
