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
#include "utils/feature_trails.h"

#include "slam_direct/direct_aligner.h"
#include "slam_direct/semi_direct_tracker.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace pyslam {
namespace {

constexpr double kMinDepth = 1e-2;

cv::Mat as_gray8(const cv::Mat &img) {
    if (img.empty()) {
        return cv::Mat();
    }
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img;
    } else if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("DirectSlam: unsupported image channel count");
    }
    if (gray.type() != CV_8UC1) {
        cv::Mat eight;
        gray.convertTo(eight, CV_8U);
        return eight;
    }
    return gray;
}

// Convert the configured sensor units to meters; zero denotes an unusable depth sample.
double sample_depth(const cv::Mat &depth, const Vec2d &uv, double depth_factor) {
    if (depth.empty()) {
        return 0.0;
    }
    const int u = static_cast<int>(std::lround(uv.x()));
    const int v = static_cast<int>(std::lround(uv.y()));
    if (u < 0 || v < 0 || u >= depth.cols || v >= depth.rows) {
        return 0.0;
    }
    double raw = 0.0;
    switch (depth.type()) {
    case CV_16UC1:
        raw = static_cast<double>(depth.at<uint16_t>(v, u));
        break;
    case CV_32FC1:
        raw = static_cast<double>(depth.at<float>(v, u));
        break;
    case CV_64FC1:
        raw = static_cast<double>(depth.at<double>(v, u));
        break;
    default:
        return 0.0;
    }
    if (!(raw > 0.0) || !std::isfinite(raw)) {
        return 0.0;
    }
    const double factor = depth_factor > 0.0 ? depth_factor : 1.0;
    return raw * factor;
}

bool depth_is_usable(const Camera &camera, double depth) {
    (void)camera;
    // depth_threshold marks "close" points in the feature SLAM. Far points are still valid
    // here as long as the stereo/RGBD depth is finite and inside a metric range.
    return depth > 0.5 && depth < 80.0 && std::isfinite(depth);
}

// Rectified KITTI-style cameras store the baseline in Camera::bf and have no
// right camera. A calibrated rig uses Camera::triangulate_stereo_point.
bool triangulate_match(const Camera &camera, const Vec2d &left, const Vec2d &right, double max_error,
                       Vec3d &X) {
    if (camera.right_camera) {
        return camera.triangulate_stereo_point(left, right, max_error, X) && depth_is_usable(camera, X.z());
    }
    if (!(camera.bf > 0.0) || std::abs(left.y() - right.y()) > 2.0) {
        return false;
    }
    const double disparity = left.x() - right.x();
    if (!(disparity > 0.1)) {
        return false;
    }
    const double depth = camera.bf / disparity;
    if (!(depth > kMinDepth)) {
        return false;
    }
    X = camera.unproject_point_3d(left.x(), left.y(), depth);
    return depth_is_usable(camera, X.z());
}

} // namespace

DirectSlam::DirectSlam(CameraPtr camera, const ConfigDict &config)
    : camera_(std::move(camera)), params_(DirectSlamParams::from_config(config)) {
    if (!camera_) {
        throw std::invalid_argument("DirectSlam: camera is null");
    }
    if (camera_->type != CameraType::PINHOLE || camera_->is_distorted) {
        throw std::invalid_argument(
            "DirectSlam: this slice requires an undistorted pinhole camera");
    }
    if (camera_->sensor_type == SensorType::MONOCULAR) {
        throw std::invalid_argument("DirectSlam: monocular is not supported; use stereo or RGBD");
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
    for (const auto &point : live_points_) {
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
    consecutive_failures_ = 0;
    last_inliers_ = 0;
    next_point_id_ = 0;
    last_img_id_ = -1;
    last_ba_ = DirectBAResult();
}

std::vector<DirectPoint> DirectSlam::make_points(const cv::Mat &gray, const cv::Mat &gray_right,
                                                 const cv::Mat &depth) const {
    const std::vector<Vec2d> pixels = detect_gftt(gray, params_);
    std::vector<DirectPoint> points;
    points.reserve(pixels.size());
    if (camera_->sensor_type == SensorType::RGBD) {
        for (const Vec2d &uv : pixels) {
            const double z = sample_depth(depth, uv, camera_->depth_factor);
            if (!depth_is_usable(*camera_, z)) {
                continue;
            }
            DirectPoint point;
            point.uv = uv;
            point.depth = z;
            point.position_c = camera_->unproject_point_3d(uv.x(), uv.y(), z);
            point.has_depth = true;
            points.push_back(point);
        }
        return points;
    }

    if (gray_right.empty()) {
        throw std::invalid_argument("DirectSlam: stereo tracking requires the right image");
    }
    const LkTrack tracked = track_pyramidal_lk(gray, gray_right, pixels, pixels, params_);
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (!tracked.status[i]) {
            continue;
        }
        Vec3d X = Vec3d::Zero();
        if (!triangulate_match(*camera_, pixels[i], tracked.points_cur[i], params_.stereo_max_error, X)) {
            continue;
        }
        DirectPoint point;
        point.uv = pixels[i];
        point.depth = X.z();
        point.position_c = X;
        point.has_depth = true;
        points.push_back(point);
    }
    return points;
}

void DirectSlam::record_keyframe(const cv::Mat &gray, const cv::Mat &gray_right, const cv::Mat &depth,
                                 int img_id, double timestamp, const Eigen::Isometry3d &Tcw) {
    DirectKeyframe keyframe;
    keyframe.id = static_cast<int>(map_.keyframes().size());
    keyframe.img_id = img_id;
    keyframe.timestamp = timestamp;
    keyframe.pose = CameraPose(Tcw);
    std::vector<Vec2d> left, right_init;
    for (const auto &point : live_points_) {
        left.push_back(point.uv);
        const Vec3d Xc = Tcw * point.position_w;
        if (camera_->right_camera) {
            const Vec3d Xr = camera_->T_right_left.block<3, 3>(0, 0) * Xc +
                             camera_->T_right_left.block<3, 1>(0, 3);
            right_init.push_back(camera_->right_camera->project_point(Xr).first);
        } else {
            right_init.emplace_back(point.uv.x() - camera_->bf / Xc.z(), point.uv.y());
        }
    }
    LkTrack stereo;
    if (camera_->sensor_type == SensorType::STEREO)
        stereo = track_pyramidal_lk(gray, gray_right, left, right_init, params_);
    for (size_t i = 0; i < live_points_.size(); ++i) {
        const auto &point = live_points_[i];
        DirectPoint observation;
        observation.id = point.id;
        observation.uv = point.uv;
        observation.position_c = Tcw * point.position_w;
        if (camera_->sensor_type == SensorType::RGBD) {
            observation.depth = sample_depth(depth, point.uv, camera_->depth_factor);
            observation.has_depth = depth_is_usable(*camera_, observation.depth);
        } else if (stereo.status[i]) {
            Vec3d measured;
            if (triangulate_match(*camera_, point.uv, stereo.points_cur[i], params_.stereo_max_error, measured)) {
                // Store the sensor measurement, never a depth derived from the pose
                // being optimized. Otherwise BA would constrain the estimate to itself.
                observation.uv_right = stereo.points_cur[i];
                observation.has_right = true;
                observation.depth = measured.z();
                observation.has_depth = true;
            }
        }
        keyframe.points.push_back(observation);
    }
    // Only the last accepted image is needed by the photometric frontend.
    // Keyframes retain observations, not full-resolution image copies.
    map_.add_keyframe(std::move(keyframe));
}

// Apply one coherent map correction to trajectory, motion prediction and live landmarks.
DirectBAResult DirectSlam::bundle_adjustment() {
    last_ba_ = optimize_direct_map(map_, *camera_, params_);
    if (!last_ba_.applied) return last_ba_;
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
    return last_ba_;
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
        if (!point.has_depth || !depth_is_usable(*camera_, point.depth)) {
            continue;
        }
        LivePoint live_point;
        // IDs survive temporal tracking and connect later keyframe observations in BA.
        live_point.id = next_point_id_++;
        live_point.uv = point.uv;
        live_point.position_w = Twc * point.position_c;
        live_point.trail.emplace_back(std::floor(point.uv.x()), std::floor(point.uv.y()));
        live_point.observations = 1;
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
    cv::Mat occupied(gray.size(), CV_8U, cv::Scalar(255));
    const int radius = std::max(1, static_cast<int>(std::lround(params_.gftt_min_distance)));
    for (const LivePoint &point : live_points_) {
        cv::circle(occupied, cv::Point(static_cast<int>(std::lround(point.uv.x())),
                                       static_cast<int>(std::lround(point.uv.y()))),
                   radius, cv::Scalar(0), cv::FILLED);
    }
    const std::vector<Vec2d> pixels = detect_gftt(gray, params_, occupied, room);
    if (pixels.empty()) {
        return;
    }

    std::vector<DirectPoint> fresh;
    if (camera_->sensor_type == SensorType::RGBD) {
        for (const Vec2d &uv : pixels) {
            const double z = sample_depth(depth, uv, camera_->depth_factor);
            if (!depth_is_usable(*camera_, z)) {
                continue;
            }
            DirectPoint point;
            point.uv = uv;
            point.depth = z;
            point.position_c = camera_->unproject_point_3d(uv.x(), uv.y(), z);
            point.has_depth = true;
            fresh.push_back(point);
        }
    } else if (!gray_right.empty()) {
        const LkTrack tracked = track_pyramidal_lk(gray, gray_right, pixels, pixels, params_);
        for (size_t i = 0; i < pixels.size(); ++i) {
            if (!tracked.status[i]) {
                continue;
            }
            Vec3d X = Vec3d::Zero();
            if (!triangulate_match(*camera_, pixels[i], tracked.points_cur[i], params_.stereo_max_error, X)) {
                continue;
            }
            DirectPoint point;
            point.uv = pixels[i];
            point.depth = X.z();
            point.position_c = X;
            point.has_depth = true;
            fresh.push_back(point);
        }
    }
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
    // The first depth-backed frame defines the world origin; recovery never re-enters here.
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    const auto points = make_points(gray, gray_right, depth);
    last_inliers_ = static_cast<int>(points.size());
    if (last_inliers_ <= params_.num_features_tracking_bad) {
        status_ = DirectTrackingStatus::Lost;
        return;
    }
    live_points_ = points_from_camera(points, identity);
    record_keyframe(gray, gray_right, depth, img_id, timestamp, identity);
    set_status_from_inliers(last_inliers_);
    last_img_id_ = img_id;
    last_gray_ = gray.clone();
    current_pose_ = CameraPose(identity);
    previous_pose_ = identity;
    has_velocity_ = false;
    consecutive_failures_ = 0;
    map_.add_pose(img_id, timestamp, identity);
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

    const Eigen::Isometry3d T_pred_w = has_velocity_ ? velocity_ * previous_pose_ : previous_pose_;
    std::vector<char> tracked_ok(live_points_.size(), 0);
    std::vector<Vec2d> tracked_uv(live_points_.size(), Vec2d::Zero());
    int inliers = 0;
    Eigen::Isometry3d T_cur_w = T_pred_w;

    std::vector<Vec2d> pixels_last;
    std::vector<Vec2d> pixels_init;
    pixels_last.reserve(live_points_.size());
    pixels_init.reserve(live_points_.size());
    for (const LivePoint &point : live_points_) {
        const Vec3d Xc = T_pred_w * point.position_w;
        Vec2d predicted = point.uv;
        if (Xc.z() > kMinDepth) {
            const auto projection = camera_->project_point(Xc);
            if (std::isfinite(projection.first.x()) && std::isfinite(projection.first.y())) {
                predicted = projection.first;
            }
        }
        pixels_last.push_back(point.uv);
        pixels_init.push_back(predicted);
    }
    const LkTrack tracked = track_pyramidal_lk(last_gray_, gray, pixels_last, pixels_init, params_);
    std::vector<PoseObservation> observations;
    observations.reserve(live_points_.size());
    // Geometric observations omit failed LK tracks; retain a map back to the live-point indices.
    std::vector<size_t> observation_indices;
    for (size_t i = 0; i < live_points_.size(); ++i) {
        tracked_ok[i] = tracked.status[i];
        tracked_uv[i] = tracked.points_cur[i];
        if (!tracked.status[i]) {
            continue;
        }
        PoseObservation observation;
        observation.point_ref = live_points_[i].position_w;
        observation.pixel_cur = tracked.points_cur[i];
        observations.push_back(observation);
        observation_indices.push_back(i);
    }
    const SemiDirectPoseResult pose =
        estimate_pose_reprojection(*camera_, observations, T_pred_w, params_);
    T_cur_w = pose.T_cur_ref;
    inliers = pose.inliers;
    // Only geometric inliers may carry a landmark into the next image.
    for (size_t j = 0; j < observation_indices.size(); ++j) {
        tracked_ok[observation_indices[j]] = pose.inlier_mask[j];
    }
    if (params_.mode == DirectTrackingMode::Direct && inliers <= params_.num_features_tracking_bad) {
        // Photometric tracking can still resolve motion when corner flow fails
        // (for example at a repetitive pattern's half-period displacement).
        std::vector<Vec2d> pixels;
        std::vector<double> depths;
        for (const auto &point : live_points_) {
            pixels.push_back(point.uv);
            depths.push_back((previous_pose_ * point.position_w).z());
        }
        const auto aligned = align_direct_multilayer(*camera_, params_, last_gray_, gray, pixels,
                                                     depths, T_pred_w * previous_pose_.inverse());
        if (aligned.inliers > params_.num_features_tracking_bad &&
            aligned.inliers >= 0.6 * live_points_.size()) {
            T_cur_w = aligned.T_cur_ref * previous_pose_;
            inliers = 0;
            for (size_t i = 0; i < live_points_.size(); ++i) {
                const Vec3d X = T_cur_w * live_points_[i].position_w;
                tracked_uv[i] = camera_->project_point(X).first;
                tracked_ok[i] = aligned.inlier_mask[i] && X.z() > kMinDepth &&
                                tracked_uv[i].x() >= params_.border && tracked_uv[i].y() >= params_.border &&
                                tracked_uv[i].x() < gray.cols - params_.border &&
                                tracked_uv[i].y() < gray.rows - params_.border;
                inliers += tracked_ok[i];
            }
        }
    } else if (params_.mode == DirectTrackingMode::Direct && inliers > params_.num_features_tracking_bad) {
        // Use geometric tracks to seed and validate photometric refinement. Outdoor
        // brightness changes and moving objects can produce a low photometric cost
        // at the wrong pose, so retain the geometric solution if refinement disagrees.
        std::vector<Vec2d> pixels;
        std::vector<double> depths;
        for (size_t i = 0; i < live_points_.size(); ++i) {
            if (!tracked_ok[i]) continue;
            pixels.push_back(live_points_[i].uv);
            depths.push_back((previous_pose_ * live_points_[i].position_w).z());
        }
        const auto aligned = align_direct_multilayer(*camera_, params_, last_gray_, gray, pixels,
                                                     depths, T_cur_w * previous_pose_.inverse());
        const Eigen::Isometry3d candidate = aligned.T_cur_ref * previous_pose_;
        double geometric_cost = 0.0, candidate_cost = 0.0;
        int candidate_inliers = 0;
        for (size_t i = 0; i < live_points_.size(); ++i) {
            if (!tracked_ok[i]) continue;
            const Vec3d X = candidate * live_points_[i].position_w;
            const Vec2d uv = camera_->project_point(X).first;
            const double error = (uv - tracked_uv[i]).squaredNorm();
            if (X.z() > 0.0 && std::isfinite(error)) {
                candidate_cost += error;
                if (error <= params_.chi2_threshold) ++candidate_inliers;
            } else {
                candidate_cost = std::numeric_limits<double>::infinity();
            }
            geometric_cost += (camera_->project_point(T_cur_w * live_points_[i].position_w).first -
                               tracked_uv[i]).squaredNorm();
        }
        if (aligned.inliers > params_.num_features_tracking_bad &&
            candidate_inliers >= 0.9 * inliers && candidate_cost <= geometric_cost + 0.25 * inliers) {
            T_cur_w = candidate;
            inliers = 0;
            for (size_t i = 0; i < live_points_.size(); ++i) {
                if (!tracked_ok[i]) continue;
                const Vec3d X = T_cur_w * live_points_[i].position_w;
                tracked_ok[i] = (camera_->project_point(X).first - tracked_uv[i]).squaredNorm() <=
                                params_.chi2_threshold;
                inliers += tracked_ok[i];
            }
        }
    }

    // A prediction is only an optimizer seed. Never integrate unmeasured motion
    // or triangulate landmarks at a predicted pose after tracking has failed.
    const bool accepted = inliers > params_.num_features_tracking_bad && T_cur_w.matrix().allFinite();
    if (!accepted) {
        ++consecutive_failures_;
        has_velocity_ = false;
        last_inliers_ = inliers;
        status_ = DirectTrackingStatus::Lost;
        return; // Keep the last accepted image and landmarks for a possible recovery.
    }
    consecutive_failures_ = 0;
    set_status_from_inliers(inliers);
    velocity_ = T_cur_w * previous_pose_.inverse();
    has_velocity_ = true;

    std::vector<LivePoint> kept;
    kept.reserve(live_points_.size());
    for (size_t i = 0; i < live_points_.size(); ++i) {
        if (!tracked_ok[i]) continue;
        LivePoint point = live_points_[i];
        point.uv = tracked_uv[i];
        // Do not draw a continuous trail across an input-frame gap or a tracking failure.
        if (img_id != last_img_id_ + 1) point.trail.clear();
        point.trail.insert(point.trail.begin(), cv::Point(std::floor(point.uv.x()), std::floor(point.uv.y())));
        if (point.trail.size() > 64) point.trail.resize(64);
        ++point.observations;
        kept.push_back(std::move(point));
    }
    live_points_ = std::move(kept);
    if (params_.mode == DirectTrackingMode::Direct && !params_.local_ba) {
        // Photometric warps need depth at the current reference pixel. Refresh it
        // from the current sensor measurement instead of carrying stereo errors forever.
        const Eigen::Isometry3d Twc = T_cur_w.inverse();
        std::vector<Vec2d> left, right_init;
        for (const auto &point : live_points_) {
            left.push_back(point.uv);
            const Vec3d Xc = T_cur_w * point.position_w;
            if (camera_->right_camera) {
                right_init.push_back(camera_->right_camera->project_point((camera_->T_right_left.block<3, 3>(0, 0) * Xc + camera_->T_right_left.block<3, 1>(0, 3)).eval()).first);
            } else {
                right_init.emplace_back(point.uv.x() - camera_->bf / Xc.z(), point.uv.y());
            }
        }
        LkTrack stereo;
        if (camera_->sensor_type == SensorType::STEREO)
            stereo = track_pyramidal_lk(gray, gray_right, left, right_init, params_);
        std::vector<LivePoint> measured;
        for (size_t i = 0; i < live_points_.size(); ++i) {
            Vec3d X;
            if (camera_->sensor_type == SensorType::RGBD) {
                const double z = sample_depth(depth, left[i], camera_->depth_factor);
                if (!depth_is_usable(*camera_, z)) continue;
                X = camera_->unproject_point_3d(left[i].x(), left[i].y(), z);
            } else if (!stereo.status[i] || !triangulate_match(*camera_, left[i], stereo.points_cur[i],
                                                             params_.stereo_max_error, X)) {
                continue;
            }
            LivePoint updated = live_points_[i];
            updated.position_w = Twc * X;
            measured.push_back(updated);
        }
        live_points_ = std::move(measured);
    }
    // Low track support triggers replenishment; automatic BA also forces periodic keyframes.
    const bool needs_features = static_cast<int>(live_points_.size()) < params_.num_features_needed_for_keyframe;
    if (needs_features)
        supplement_live_points(gray, gray_right, depth, T_cur_w, img_id, stamp);
    const bool new_keyframe = needs_features ||
        (params_.local_ba && img_id - map_.latest_keyframe()->img_id >= params_.max_keyframe_interval);
    if (new_keyframe)
        record_keyframe(gray, gray_right, depth, img_id, stamp, T_cur_w);

    previous_pose_ = T_cur_w;
    current_pose_ = CameraPose(T_cur_w);
    last_gray_ = gray.clone();
    last_img_id_ = img_id;
    // Register the accepted frame before BA so its reference-relative pose is corrected too.
    map_.add_pose(img_id, stamp, T_cur_w);
    if (new_keyframe && params_.local_ba) bundle_adjustment();
}

} // namespace pyslam
