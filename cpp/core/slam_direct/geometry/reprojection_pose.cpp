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

#include "slam_direct/geometry/reprojection_pose.h"

#include "camera/camera_model_math.h"
#include "slam_direct/se3.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

namespace pyslam {
namespace {

using Mat66d = Eigen::Matrix<double, 6, 6>;
using Mat26d = Eigen::Matrix<double, 2, 6>;

bool inside_image(const Camera &camera, const Vec2d &uv, double z) {
    const double u_min = camera.u_max > camera.u_min ? camera.u_min : 0.0;
    const double v_min = camera.v_max > camera.v_min ? camera.v_min : 0.0;
    const double u_max = camera.u_max > camera.u_min ? camera.u_max : static_cast<double>(camera.width);
    const double v_max = camera.v_max > camera.v_min ? camera.v_max : static_cast<double>(camera.height);
    return z > 0.0 && uv.x() >= u_min && uv.x() < u_max && uv.y() >= v_min && uv.y() < v_max;
}

void pose_from_rvec_tvec(const cv::Mat &rvec, const cv::Mat &tvec, Eigen::Isometry3d &pose) {
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    pose = Eigen::Isometry3d::Identity();
    for (int row = 0; row < 3; ++row) {
        pose.translation()[row] = tvec.at<double>(row);
        for (int col = 0; col < 3; ++col) {
            pose.linear()(row, col) = rotation.at<double>(row, col);
        }
    }
}

// Score the single accepted seed in raw pixels. The mask is what Gauss-Newton consumes.
int raw_pixel_inliers(const Camera &camera, const std::vector<PoseObservation> &observations,
                      const Eigen::Isometry3d &pose, double chi2, std::vector<char> &outlier) {
    int inliers = 0;
    outlier.assign(observations.size(), 1);
    for (size_t i = 0; i < observations.size(); ++i) {
        const Vec3d Xc = pose * observations[i].point_ref;
        Vec2d uv = Vec2d::Zero();
        if (!camera.project_with_jacobian(Xc, uv, nullptr) || !inside_image(camera, uv, Xc.z())) {
            continue;
        }
        if ((observations[i].pixel_cur - uv).squaredNorm() <= chi2) {
            outlier[i] = 0;
            ++inliers;
        }
    }
    return inliers;
}

// Undistorted pinhole: one OpenCV RANSAC on the detected pixels.
bool seed_pinhole_pose(const Camera &camera, const std::vector<PoseObservation> &observations,
                       const DirectSlamParams &params, Eigen::Isometry3d &pose,
                       std::vector<char> &outlier) {
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(observations.size());
    image_points.reserve(observations.size());
    for (const auto &obs : observations) {
        object_points.emplace_back(obs.point_ref.x(), obs.point_ref.y(), obs.point_ref.z());
        image_points.emplace_back(obs.pixel_cur.x(), obs.pixel_cur.y());
    }
    const cv::Mat K = (cv::Mat_<double>(3, 3) << camera.fx, 0, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1);
    cv::Mat rvec, tvec, indices;
    if (!cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec, false, 100,
                            std::sqrt(params.chi2_threshold), 0.99, indices, cv::SOLVEPNP_EPNP) ||
        indices.total() < 6) {
        return false;
    }
    pose_from_rvec_tvec(rvec, tvec, pose);
    outlier.assign(observations.size(), 1);
    for (int i = 0; i < indices.rows; ++i) {
        outlier[indices.at<int>(i)] = 0;
    }
    return true;
}

// Distorted models: unproject each pixel once, then one OpenCV RANSAC on the normalized
// bearings. Distortion is not evaluated inside the sampling loop.
bool seed_bearing_pose(const Camera &camera, const std::vector<PoseObservation> &observations,
                       const DirectSlamParams &params, Eigen::Isometry3d &pose,
                       std::vector<char> &outlier) {
    if (!(camera.fx > 0.0)) {
        return false;
    }
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(observations.size());
    image_points.reserve(observations.size());
    constexpr double kMinBearingZ = 0.05;
    for (const auto &obs : observations) {
        Vec3d bearing;
        if (!camera.unproject_bearing(obs.pixel_cur, bearing, nullptr) || !bearing.allFinite() ||
            !(bearing.z() > kMinBearingZ)) {
            continue;
        }
        object_points.emplace_back(obs.point_ref.x(), obs.point_ref.y(), obs.point_ref.z());
        image_points.emplace_back(bearing.x() / bearing.z(), bearing.y() / bearing.z());
    }
    if (object_points.size() < 6) {
        return false;
    }
    const cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rvec, tvec, indices;
    const double normalized_threshold = std::sqrt(params.chi2_threshold) / camera.fx;
    if (!cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec, false, 100,
                            normalized_threshold, 0.99, indices, cv::SOLVEPNP_EPNP) ||
        indices.total() < 6) {
        return false;
    }
    Eigen::Isometry3d seeded = Eigen::Isometry3d::Identity();
    pose_from_rvec_tvec(rvec, tvec, seeded);
    if (raw_pixel_inliers(camera, observations, seeded, params.chi2_threshold, outlier) < 6) {
        return false;
    }
    pose = seeded;
    return true;
}

} // namespace

ReprojectionPoseResult estimate_pose_reprojection(const Camera &camera,
                                                const std::vector<PoseObservation> &observations,
                                                const Eigen::Isometry3d &T_cur_ref_init,
                                                const DirectSlamParams &params) {
    ReprojectionPoseResult result;
    result.T_cur_ref = T_cur_ref_init;
    result.inlier_mask.assign(observations.size(), 0);
    if (observations.size() < 6) {
        return result;
    }
    // RANSAC provides a geometric seed independent of an inaccurate motion prediction.
    // LK consistency alone cannot reject matches on moving objects or repetitive texture.
    // An undistorted pinhole keeps one solvePnPRansac on raw pixels. A raw-pixel camera
    // model lifts each observation once and runs that same call on normalized bearings.
    std::vector<char> outlier(observations.size(), 1);
    const bool seeded = camera.projects_raw_keypoints()
                            ? seed_bearing_pose(camera, observations, params, result.T_cur_ref, outlier)
                            : seed_pinhole_pose(camera, observations, params, result.T_cur_ref, outlier);
    if (!seeded) {
        result.T_cur_ref = T_cur_ref_init;
        return result;
    }
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
