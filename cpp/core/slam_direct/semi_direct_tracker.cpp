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

#include "slam_direct/semi_direct_tracker.h"

#include "camera_model_math.h"
#include "slam_direct/se3.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

using Mat66d = Eigen::Matrix<double, 6, 6>;
using Mat26d = Eigen::Matrix<double, 2, 6>;

std::vector<cv::Point2f> to_cv(const std::vector<Vec2d> &points) {
    std::vector<cv::Point2f> out;
    out.reserve(points.size());
    for (const Vec2d &point : points) {
        out.emplace_back(static_cast<float>(point.x()), static_cast<float>(point.y()));
    }
    return out;
}

std::vector<Vec2d> from_cv(const std::vector<cv::Point2f> &points) {
    std::vector<Vec2d> out;
    out.reserve(points.size());
    for (const cv::Point2f &point : points) {
        out.emplace_back(point.x, point.y);
    }
    return out;
}

void require_gray(const cv::Mat &img, const char *name) {
    if (img.empty() || img.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty CV_8UC1 image");
    }
}

bool inside_image(const Camera &camera, const Vec2d &uv, double z) {
    const double u_min = camera.u_max > camera.u_min ? camera.u_min : 0.0;
    const double v_min = camera.v_max > camera.v_min ? camera.v_min : 0.0;
    const double u_max = camera.u_max > camera.u_min ? camera.u_max : static_cast<double>(camera.width);
    const double v_max = camera.v_max > camera.v_min ? camera.v_max : static_cast<double>(camera.height);
    return z > 0.0 && uv.x() >= u_min && uv.x() < u_max && uv.y() >= v_min && uv.y() < v_max;
}

} // namespace

std::vector<Vec2d> detect_gftt(const cv::Mat &gray, const DirectSlamParams &params,
                               const cv::Mat &occupied, int max_corners) {
    require_gray(gray, "detect_gftt");
    const int corners_to_find = max_corners > 0 ? max_corners : params.num_features;
    if (corners_to_find < 1) {
        return {};
    }
    // Preserve existing tracks by combining their occupancy mask with border exclusion.
    cv::Mat mask(gray.size(), CV_8U, cv::Scalar(255));
    const int border = std::min(params.border, std::min(gray.cols, gray.rows) / 4);
    if (border > 0) {
        mask.rowRange(0, border).setTo(0);
        mask.rowRange(gray.rows - border, gray.rows).setTo(0);
        mask.colRange(0, border).setTo(0);
        mask.colRange(gray.cols - border, gray.cols).setTo(0);
    }
    if (!occupied.empty() && occupied.size() == gray.size() && occupied.type() == CV_8UC1) {
        cv::bitwise_and(mask, occupied, mask);
    }
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, corners_to_find, params.gftt_quality, params.gftt_min_distance,
                            mask, params.gftt_block_size, false, 0.04);
    return from_cv(corners);
}

LkTrack track_pyramidal_lk(const cv::Mat &img_ref, const cv::Mat &img_cur,
                           const std::vector<Vec2d> &points_ref,
                           const std::vector<Vec2d> &points_cur_init, const DirectSlamParams &params) {
    require_gray(img_ref, "track_pyramidal_lk img_ref");
    require_gray(img_cur, "track_pyramidal_lk img_cur");
    LkTrack tracked;
    tracked.points_cur.resize(points_ref.size(), Vec2d::Zero());
    tracked.status.assign(points_ref.size(), 0);
    if (points_ref.empty()) {
        return tracked;
    }
    std::vector<Vec2d> initial = points_cur_init.size() == points_ref.size() ? points_cur_init : points_ref;
    std::vector<cv::Point2f> prev = to_cv(points_ref);
    std::vector<cv::Point2f> curr = to_cv(initial);
    std::vector<uchar> status;
    std::vector<float> error;
    const cv::Size window(params.lk_patch_size, params.lk_patch_size);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img_ref, img_cur, prev, curr, status, error, window, params.lk_max_level, criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    // A forward-backward displacement check rejects plausible-looking but unstable matches.
    std::vector<cv::Point2f> backward = prev;
    std::vector<uchar> backward_status;
    cv::calcOpticalFlowPyrLK(img_cur, img_ref, curr, backward, backward_status, error, window,
                             params.lk_max_level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    const int border = params.border;
    constexpr float kForwardBackwardPx = 1.5f;
    for (size_t i = 0; i < status.size(); ++i) {
        const float dx = backward[i].x - prev[i].x;
        const float dy = backward[i].y - prev[i].y;
        const bool consistent = backward_status[i] && dx * dx + dy * dy <= kForwardBackwardPx * kForwardBackwardPx;
        const bool in_image = curr[i].x >= border && curr[i].y >= border &&
                              curr[i].x < img_cur.cols - border && curr[i].y < img_cur.rows - border;
        tracked.status[i] = status[i] && consistent && in_image ? 1 : 0;
        tracked.points_cur[i] = Vec2d(curr[i].x, curr[i].y);
    }
    return tracked;
}

SemiDirectPoseResult estimate_pose_reprojection(const Camera &camera,
                                                const std::vector<PoseObservation> &observations,
                                                const Eigen::Isometry3d &T_cur_ref_init,
                                                const DirectSlamParams &params) {
    SemiDirectPoseResult result;
    result.T_cur_ref = T_cur_ref_init;
    result.inlier_mask.assign(observations.size(), 0);
    if (observations.size() < 6) {
        return result;
    }
    // RANSAC provides a geometric seed independent of an inaccurate motion prediction.
    // LK consistency alone cannot reject matches on moving objects or repetitive texture.
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    for (const auto &obs : observations) {
        object_points.emplace_back(obs.point_ref.x(), obs.point_ref.y(), obs.point_ref.z());
        image_points.emplace_back(obs.pixel_cur.x(), obs.pixel_cur.y());
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3) << camera.fx, 0, camera.cx,
                       0, camera.fy, camera.cy, 0, 0, 1);
    cv::Mat rvec, tvec, indices;
    if (!cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec,
                            false, 100, std::sqrt(params.chi2_threshold), 0.99, indices,
                            cv::SOLVEPNP_EPNP) || indices.total() < 6) {
        return result;
    }
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    for (int r = 0; r < 3; ++r) {
        result.T_cur_ref.translation()[r] = tvec.at<double>(r);
        for (int c = 0; c < 3; ++c) result.T_cur_ref.linear()(r, c) = R.at<double>(r, c);
    }
    std::vector<char> outlier(observations.size(), 1);
    for (int i = 0; i < indices.rows; ++i) outlier[indices.at<int>(i)] = 0;
    const int outer_iterations = std::max(1, params.pose_iterations);
    const int inner_iterations = std::max(1, params.pose_inner_iterations);
    for (int outer = 0; outer < outer_iterations; ++outer) {
        for (int inner = 0; inner < inner_iterations; ++inner) {
            Mat66d H = Mat66d::Zero();
            Vec6d gradient = Vec6d::Zero();
            double cost = 0.0;
            int used = 0;
            const double huber = std::sqrt(std::max(params.chi2_threshold, 1e-6));
            for (size_t i = 0; i < observations.size(); ++i) {
                if (outlier[i]) {
                    continue;
                }
                const Vec3d Xc = result.T_cur_ref * observations[i].point_ref;
                Vec2d uv = Vec2d::Zero();
                Eigen::Matrix<double, 2, 3> J_uv_Xc;
                if (!camera.project_with_jacobian(Xc, uv, &J_uv_Xc) || !inside_image(camera, uv, Xc.z())) {
                    continue;
                }
                Eigen::Matrix<double, 3, 6> dXc_dxi;
                camera_model::g2o_left_se3_jacobian(Xc, dXc_dxi);
                // Residual is observed - projected, so the projection derivative changes sign.
                const Mat26d J = -J_uv_Xc * dXc_dxi;
                const Vec2d residual = observations[i].pixel_cur - uv;
                const double residual_norm = residual.norm();
                const double weight = residual_norm > huber ? huber / residual_norm : 1.0;
                H += weight * J.transpose() * J;
                gradient += weight * J.transpose() * residual;
                cost += residual.squaredNorm();
                ++used;
            }
            result.cost = cost;
            result.inliers = used;
            if (used < 6) {
                break;
            }
            const Eigen::LDLT<Mat66d> solver(H);
            if (solver.info() != Eigen::Success) {
                break;
            }
            Vec6d update = solver.solve(-gradient);
            if (!update.allFinite()) {
                break;
            }
            // Limit each optimizer increment, not the final inter-frame motion estimate.
            const double rotation_limit = 0.01;
            const double translation_limit = 0.2;
            const double rotation_norm = update.head<3>().norm();
            const double translation_norm = update.tail<3>().norm();
            double scale = 1.0;
            if (rotation_norm > rotation_limit) {
                scale = std::min(scale, rotation_limit / rotation_norm);
            }
            if (translation_norm > translation_limit) {
                scale = std::min(scale, translation_limit / translation_norm);
            }
            update *= scale;
            const Eigen::Isometry3d previous = result.T_cur_ref;
            left_multiply_rotation_translation(result.T_cur_ref, update);
            if (update.norm() < params.eps_convergence) {
                break;
            }
            if (cost > 0.0) {
                double new_cost = 0.0;
                int projected = 0;
                for (size_t i = 0; i < observations.size(); ++i) {
                    if (outlier[i]) {
                        continue;
                    }
                    const Vec3d Xc = result.T_cur_ref * observations[i].point_ref;
                    Vec2d uv = Vec2d::Zero();
                    if (!camera.project_with_jacobian(Xc, uv, nullptr) || Xc.z() <= 0.0) {
                        new_cost = cost + 1.0;
                        break;
                    }
                    new_cost += (observations[i].pixel_cur - uv).squaredNorm();
                    ++projected;
                }
                if (projected < used || new_cost > cost) {
                    result.T_cur_ref = previous;
                    break;
                }
            }
        }

        // Re-evaluate all observations so a previously excluded match can become an inlier.
        result.inliers = 0;
        result.cost = 0.0;
        for (size_t i = 0; i < observations.size(); ++i) {
            const Vec3d Xc = result.T_cur_ref * observations[i].point_ref;
            Vec2d uv = Vec2d::Zero();
            if (!camera.project_with_jacobian(Xc, uv, nullptr) || !inside_image(camera, uv, Xc.z())) {
                outlier[i] = 1;
                continue;
            }
            const double chi2 = (observations[i].pixel_cur - uv).squaredNorm();
            outlier[i] = chi2 > params.chi2_threshold ? 1 : 0;
            if (!outlier[i]) {
                result.cost += chi2;
                ++result.inliers;
            }
        }
    }
    for (size_t i = 0; i < outlier.size(); ++i) result.inlier_mask[i] = !outlier[i];
    return result;
}

} // namespace pyslam
