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

#include "slam_direct/direct/direct_aligner.h"

#include "camera/camera_model_math.h"
#include "slam_direct/se3.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

using Mat66d = Eigen::Matrix<double, 6, 6>;

struct LiftedPoint {
    Vec3d X = Vec3d::Zero();
    bool valid = false;
};

// Bilinear sampling supports subpixel warps; clamp edge samples used by image gradients.
float pixel_value(const cv::Mat &img, float x, float y) {
    if (img.cols < 2 || img.rows < 2) return img.at<uchar>(0, 0);
    if (x < 0.0f) {
        x = 0.0f;
    }
    if (y < 0.0f) {
        y = 0.0f;
    }
    if (x >= static_cast<float>(img.cols - 1)) {
        x = static_cast<float>(img.cols - 2);
    }
    if (y >= static_cast<float>(img.rows - 1)) {
        y = static_cast<float>(img.rows - 2);
    }
    const float xx = x - std::floor(x);
    const float yy = y - std::floor(y);
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(img.cols - 1, x0 + 1);
    const int y1 = std::min(img.rows - 1, y0 + 1);
    const float i00 = static_cast<float>(img.at<uchar>(y0, x0));
    const float i10 = static_cast<float>(img.at<uchar>(y0, x1));
    const float i01 = static_cast<float>(img.at<uchar>(y1, x0));
    const float i11 = static_cast<float>(img.at<uchar>(y1, x1));
    return (1.0f - xx) * (1.0f - yy) * i00 + xx * (1.0f - yy) * i10 + (1.0f - xx) * yy * i01 +
           xx * yy * i11;
}

void require_gray(const cv::Mat &img, const char *name) {
    if (img.empty() || img.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty CV_8UC1 image");
    }
}

struct Accumulated {
    Mat66d H = Mat66d::Zero();
    Vec6d b = Vec6d::Zero();
    double cost = 0.0;
    int visible = 0;
    int inliers = 0;
    std::vector<unsigned char> inlier_mask;
};

Accumulated accumulate(const Camera &camera, double scale, const DirectSlamParams &params,
                       const cv::Mat &img_ref, const cv::Mat &img_cur, const std::vector<Vec2d> &pixels_ref,
                       const std::vector<LiftedPoint> &points_ref, const Eigen::Isometry3d &T_cur_ref) {
    Accumulated acc;
    acc.inlier_mask.assign(pixels_ref.size(), 0);
    const int half = params.half_patch_size;
    const double inlier_threshold = params.intensity_inlier_threshold;
    for (size_t i = 0; i < pixels_ref.size(); ++i) {
        if (!points_ref[i].valid) {
            continue;
        }
        const Vec3d point_cur = T_cur_ref * points_ref[i].X;
        Vec2d uv = Vec2d::Zero();
        Eigen::Matrix<double, 2, 6> J_pixel_xi;
        // One projection per point. The patch reuses this Jacobian.
        if (!photometric_warp_jacobian(camera, point_cur, scale, uv, J_pixel_xi)) {
            continue;
        }
        const double u = uv.x();
        const double v = uv.y();
        if (u < half || u > img_cur.cols - half - 1 || v < half || v > img_cur.rows - half - 1) {
            continue;
        }

        double abs_error = 0.0;
        int samples = 0;
        Mat66d H = Mat66d::Zero();
        Vec6d bias = Vec6d::Zero();
        double cost = 0.0;
        for (int py = -half; py <= half; ++py) {
            for (int px = -half; px <= half; ++px) {
                const double error = pixel_value(img_ref, static_cast<float>(pixels_ref[i].x() + px),
                                                 static_cast<float>(pixels_ref[i].y() + py)) -
                                     pixel_value(img_cur, static_cast<float>(u + px),
                                                 static_cast<float>(v + py));
                const Vec2d image_gradient(
                    0.5 * (pixel_value(img_cur, static_cast<float>(u + px + 1),
                                       static_cast<float>(v + py)) -
                           pixel_value(img_cur, static_cast<float>(u + px - 1),
                                       static_cast<float>(v + py))),
                    0.5 * (pixel_value(img_cur, static_cast<float>(u + px),
                                       static_cast<float>(v + py + 1)) -
                           pixel_value(img_cur, static_cast<float>(u + px),
                                       static_cast<float>(v + py - 1))));
                // Chain image gradient and projection Jacobian for I_ref - I_cur.
                // Huber weights limit individual brightness outliers in the normal equations.
                const Vec6d J = -J_pixel_xi.transpose() * image_gradient;
                const double weight = std::abs(error) > inlier_threshold ? inlier_threshold / std::abs(error) : 1.0;
                H += weight * J * J.transpose();
                bias += -weight * error * J;
                cost += error * error;
                abs_error += std::abs(error);
                ++samples;
            }
        }
        if (samples == 0) {
            continue;
        }
        acc.H += H;
        acc.b += bias;
        // Count patch inliers by mean absolute error; report unweighted squared error.
        acc.cost += cost / samples;
        ++acc.visible;
        if (abs_error / static_cast<double>(samples) <= inlier_threshold) {
            ++acc.inliers;
            acc.inlier_mask[i] = 1;
        }
    }
    if (acc.visible > 0) {
        acc.cost /= static_cast<double>(acc.visible);
    }
    return acc;
}

DirectAlignmentResult align_layer(const Camera &camera, double scale, const DirectSlamParams &params,
                                  const cv::Mat &img_ref, const cv::Mat &img_cur,
                                  const std::vector<Vec2d> &pixels_ref,
                                  const std::vector<LiftedPoint> &points_ref,
                                  const Eigen::Isometry3d &T_init) {
    DirectAlignmentResult result;
    result.T_cur_ref = T_init;
    for (int iter = 0; iter < params.num_iterations; ++iter) {
        const Accumulated acc =
            accumulate(camera, scale, params, img_ref, img_cur, pixels_ref, points_ref, result.T_cur_ref);
        result.iterations = iter + 1;
        result.visible = acc.visible;
        result.inliers = acc.inliers;
        result.cost = acc.cost;
        if (acc.visible < 6) {
            break;
        }
        const Eigen::LDLT<Mat66d> solver(acc.H);
        if (solver.info() != Eigen::Success) {
            break;
        }
        const Vec6d update = solver.solve(acc.b);
        if (!update.allFinite()) {
            break;
        }
        Eigen::Isometry3d candidate = result.T_cur_ref;
        left_multiply_translation_rotation(candidate, update);
        // Evaluate the proposed pose before committing it. Comparing the old cost
        // after applying an update used to retain the step that increased error.
        const Accumulated next =
            accumulate(camera, scale, params, img_ref, img_cur, pixels_ref, points_ref, candidate);
        if (!candidate.matrix().allFinite() || next.visible < 6 || next.cost > acc.cost) {
            break;
        }
        result.T_cur_ref = candidate;
        if (update.norm() < params.eps_convergence) {
            result.converged = true;
            break;
        }
    }
    const Accumulated final =
        accumulate(camera, scale, params, img_ref, img_cur, pixels_ref, points_ref, result.T_cur_ref);
    result.visible = final.visible;
    result.inliers = final.inliers;
    result.cost = final.cost;
    result.inlier_mask = final.inlier_mask;
    return result;
}

std::vector<LiftedPoint> lift_points(const Camera &camera, const std::vector<Vec2d> &pixels,
                                    const std::vector<double> &depths) {
    std::vector<LiftedPoint> lifted(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        lifted[i].valid = lift_reference_point(camera, pixels[i], depths[i], lifted[i].X);
    }
    return lifted;
}

} // namespace

bool lift_reference_point(const Camera &camera, const Vec2d &pixel, double depth, Vec3d &X_ref) {
    if (!(depth > 0.0) || !std::isfinite(depth)) {
        return false;
    }
    Vec3d bearing;
    constexpr double kMinBearingZ = 1e-4;
    if (!camera.unproject_bearing(pixel, bearing, nullptr) || !bearing.allFinite() ||
        !(bearing.z() > kMinBearingZ)) {
        return false;
    }
    X_ref = bearing * (depth / bearing.z());
    return X_ref.allFinite();
}

bool photometric_warp_jacobian(const Camera &camera, const Vec3d &X_cur, double scale, Vec2d &uv,
                               Eigen::Matrix<double, 2, 6> &J_pixel_xi) {
    if (!(scale > 0.0) || !X_cur.allFinite() || !(X_cur.z() > 1e-8)) {
        return false;
    }
    Vec2d uv_full = Vec2d::Zero();
    Eigen::Matrix<double, 2, 3> J_uv_X;
    if (!camera.project_with_jacobian(X_cur, uv_full, &J_uv_X) || !uv_full.allFinite() ||
        !J_uv_X.allFinite()) {
        return false;
    }
    // g2o stores [rotation, translation]. The photometric increment is [translation, rotation].
    Eigen::Matrix<double, 3, 6> dX_dxi_rotation_translation;
    camera_model::g2o_left_se3_jacobian(X_cur, dX_dxi_rotation_translation);
    Eigen::Matrix<double, 3, 6> dX_dxi;
    dX_dxi.leftCols<3>() = dX_dxi_rotation_translation.rightCols<3>();
    dX_dxi.rightCols<3>() = dX_dxi_rotation_translation.leftCols<3>();
    J_pixel_xi = scale * (J_uv_X * dX_dxi);
    uv = scale * uv_full;
    return uv.allFinite() && J_pixel_xi.allFinite();
}

DirectAlignmentResult align_direct_single_layer(const Camera &camera, const DirectSlamParams &params,
                                                const cv::Mat &img_ref, const cv::Mat &img_cur,
                                                const std::vector<Vec2d> &pixels_ref,
                                                const std::vector<double> &depth_ref,
                                                const Eigen::Isometry3d &T_cur_ref_init) {
    require_gray(img_ref, "align_direct img_ref");
    require_gray(img_cur, "align_direct img_cur");
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("align_direct: pixels and depths have different sizes");
    }
    const std::vector<LiftedPoint> points = lift_points(camera, pixels_ref, depth_ref);
    return align_layer(camera, 1.0, params, img_ref, img_cur, pixels_ref, points, T_cur_ref_init);
}

DirectAlignmentResult align_direct_multilayer(const Camera &camera, const DirectSlamParams &params,
                                              const cv::Mat &img_ref, const cv::Mat &img_cur,
                                              const std::vector<Vec2d> &pixels_ref,
                                              const std::vector<double> &depth_ref,
                                              const Eigen::Isometry3d &T_cur_ref_init) {
    require_gray(img_ref, "align_direct img_ref");
    require_gray(img_cur, "align_direct img_cur");
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("align_direct: pixels and depths have different sizes");
    }

    std::vector<cv::Mat> pyramid_ref;
    std::vector<cv::Mat> pyramid_cur;
    std::vector<double> scales;
    pyramid_ref.reserve(static_cast<size_t>(params.num_pyramid_levels));
    pyramid_cur.reserve(static_cast<size_t>(params.num_pyramid_levels));
    scales.reserve(static_cast<size_t>(params.num_pyramid_levels));
    for (int level = 0; level < params.num_pyramid_levels; ++level) {
        if (level == 0) {
            pyramid_ref.push_back(img_ref);
            pyramid_cur.push_back(img_cur);
            scales.push_back(1.0);
            continue;
        }
        cv::Mat ref_level;
        cv::Mat cur_level;
        const cv::Mat &prev_ref = pyramid_ref.back();
        const cv::Mat &prev_cur = pyramid_cur.back();
        const cv::Size size(std::max(1, static_cast<int>(std::lround(prev_ref.cols * params.pyramid_scale))),
                            std::max(1, static_cast<int>(std::lround(prev_ref.rows * params.pyramid_scale))));
        cv::resize(prev_ref, ref_level, size, 0.0, 0.0, cv::INTER_LINEAR);
        cv::resize(prev_cur, cur_level, size, 0.0, 0.0, cv::INTER_LINEAR);
        pyramid_ref.push_back(ref_level);
        pyramid_cur.push_back(cur_level);
        scales.push_back(scales.back() * params.pyramid_scale);
    }

    // Lift once at full resolution. Coarse levels only scale the projected pixel and Jacobian.
    const std::vector<LiftedPoint> points = lift_points(camera, pixels_ref, depth_ref);
    DirectAlignmentResult result;
    result.T_cur_ref = T_cur_ref_init;
    // Large pixel motion is easier to resolve at the coarsest level; depth stays in meters.
    for (int level = params.num_pyramid_levels - 1; level >= 0; --level) {
        const double scale = scales[static_cast<size_t>(level)];
        std::vector<Vec2d> pixels_level;
        pixels_level.reserve(pixels_ref.size());
        for (const Vec2d &pixel : pixels_ref) {
            pixels_level.push_back(scale * pixel);
        }
        result = align_layer(camera, scale, params, pyramid_ref[static_cast<size_t>(level)],
                             pyramid_cur[static_cast<size_t>(level)], pixels_level, points,
                             result.T_cur_ref);
    }
    return result;
}

} // namespace pyslam
