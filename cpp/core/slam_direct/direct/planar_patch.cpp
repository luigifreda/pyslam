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

#include "slam_direct/direct/planar_patch.h"

#include "slam_direct/direct/direct_aligner.h"
#include "slam_direct/image_utils.h"
#include "slam_direct/se3.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

using Mat66d = Eigen::Matrix<double, 6, 6>;

// One pixel of a fronto-parallel patch: unproject at the feature depth, then project.
bool warp_patch_pixel(const Camera &camera, const Eigen::Isometry3d &T_cur_ref, const Vec2d &pixel_ref,
                      double depth, double scale, Vec2d &pixel_cur, Eigen::Matrix<double, 2, 6> *jacobian) {
    Vec3d point_ref = Vec3d::Zero();
    if (!lift_reference_point(camera, pixel_ref, depth, point_ref)) {
        return false;
    }
    const Vec3d point_cur = T_cur_ref * point_ref;
    Vec2d unused = Vec2d::Zero();
    Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();
    if (!photometric_warp_jacobian(camera, point_cur, scale, jacobian ? pixel_cur : unused, J)) {
        return false;
    }
    if (!jacobian) {
        pixel_cur = unused;
    } else {
        *jacobian = J;
    }
    return pixel_cur.allFinite();
}

struct PoseAccumulation {
    Mat66d H = Mat66d::Zero();
    Vec6d b = Vec6d::Zero();
    double cost = 0.0;
    int visible = 0;
    int inliers = 0;
    std::vector<unsigned char> inlier_mask;
};

// The pose-independent part of one pyramid level: each patch pixel lifted to the reference
// camera at its feature depth, and its reference intensity. Built once per level instead of
// in every accumulate_pose() call. A point whose patch leaves the reference image or cannot
// be lifted is invisible at every pose, exactly as when it was checked per call.
struct ReferenceLevel {
    std::vector<unsigned char> usable;  // Per point.
    std::vector<Vec3d> points_ref;      // Per point and patch pixel (row-major patch order).
    std::vector<float> intensities_ref; // Same layout.
};

ReferenceLevel build_reference_level(const Camera &camera, double scale, const DirectSlamParams &params,
                                     const cv::Mat &img_ref, const std::vector<Vec2d> &pixels_ref,
                                     const std::vector<double> &depths) {
    const int half = params.feature_half_patch;
    const size_t patch = static_cast<size_t>((2 * half + 1) * (2 * half + 1));
    ReferenceLevel level;
    level.usable.assign(pixels_ref.size(), 0);
    level.points_ref.assign(pixels_ref.size() * patch, Vec3d::Zero());
    level.intensities_ref.assign(pixels_ref.size() * patch, 0.0f);
    for (size_t i = 0; i < pixels_ref.size(); ++i) {
        if (!(depths[i] > 0.0) || !std::isfinite(depths[i])) {
            continue;
        }
        bool usable = true;
        size_t k = i * patch;
        for (int py = -half; py <= half && usable; ++py) {
            for (int px = -half; px <= half; ++px, ++k) {
                const Vec2d ref_level = scale * pixels_ref[i] + Vec2d(px, py);
                const Vec2d ref_full = ref_level / scale;
                if (!lift_reference_point(camera, ref_full, depths[i], level.points_ref[k]) ||
                    ref_level.x() < 1.0 || ref_level.y() < 1.0 || ref_level.x() >= img_ref.cols - 2 ||
                    ref_level.y() >= img_ref.rows - 2) {
                    usable = false;
                    break;
                }
                level.intensities_ref[k] =
                    sample_gray(img_ref, static_cast<float>(ref_level.x()), static_cast<float>(ref_level.y()));
            }
        }
        level.usable[i] = usable;
    }
    return level;
}

PoseAccumulation accumulate_pose(const Camera &camera, double scale, const DirectSlamParams &params,
                                 const ReferenceLevel &reference, const cv::Mat &img_cur,
                                 const Eigen::Isometry3d &T_cur_ref) {
    PoseAccumulation acc;
    acc.inlier_mask.assign(reference.usable.size(), 0);
    const int half = params.feature_half_patch;
    const size_t patch = static_cast<size_t>((2 * half + 1) * (2 * half + 1));
    const double gate = params.intensity_inlier_threshold;
    for (size_t i = 0; i < reference.usable.size(); ++i) {
        if (!reference.usable[i]) {
            continue;
        }
        double abs_error = 0.0;
        int samples = 0;
        Mat66d H = Mat66d::Zero();
        Vec6d bias = Vec6d::Zero();
        double cost = 0.0;
        bool visible = true;
        for (size_t k = i * patch; k < (i + 1) * patch; ++k) {
            {
                // Same steps as warp_patch_pixel() after the lift, which is precomputed.
                const Vec3d point_cur = T_cur_ref * reference.points_ref[k];
                Vec2d cur_level = Vec2d::Zero();
                Eigen::Matrix<double, 2, 6> J_pixel = Eigen::Matrix<double, 2, 6>::Zero();
                if (!photometric_warp_jacobian(camera, point_cur, scale, cur_level, J_pixel) ||
                    !cur_level.allFinite()) {
                    visible = false;
                    break;
                }
                if (cur_level.x() < 1.0 || cur_level.y() < 1.0 || cur_level.x() >= img_cur.cols - 2 ||
                    cur_level.y() >= img_cur.rows - 2) {
                    visible = false;
                    break;
                }
                const double error = reference.intensities_ref[k] -
                                     sample_gray(img_cur, static_cast<float>(cur_level.x()),
                                                 static_cast<float>(cur_level.y()));
                const Vec2d gradient(
                    0.5 * (sample_gray(img_cur, static_cast<float>(cur_level.x() + 1.0),
                                       static_cast<float>(cur_level.y())) -
                           sample_gray(img_cur, static_cast<float>(cur_level.x() - 1.0),
                                       static_cast<float>(cur_level.y()))),
                    0.5 * (sample_gray(img_cur, static_cast<float>(cur_level.x()),
                                       static_cast<float>(cur_level.y() + 1.0)) -
                           sample_gray(img_cur, static_cast<float>(cur_level.x()),
                                       static_cast<float>(cur_level.y() - 1.0))));
                const Vec6d J = -J_pixel.transpose() * gradient;
                const double weight = std::abs(error) > gate ? gate / std::abs(error) : 1.0;
                H += weight * J * J.transpose();
                bias += -weight * error * J;
                cost += error * error;
                abs_error += std::abs(error);
                ++samples;
            }
        }
        if (!visible || samples == 0) {
            continue;
        }
        acc.H += H;
        acc.b += bias;
        acc.cost += cost / static_cast<double>(samples);
        ++acc.visible;
        if (abs_error / static_cast<double>(samples) <= gate) {
            ++acc.inliers;
            acc.inlier_mask[i] = 1;
        }
    }
    if (acc.visible > 0) {
        acc.cost /= static_cast<double>(acc.visible);
    }
    return acc;
}

PlanarPoseResult align_one_level(const Camera &camera, double scale, const DirectSlamParams &params,
                                 const cv::Mat &img_ref, const cv::Mat &img_cur,
                                 const std::vector<Vec2d> &pixels_ref, const std::vector<double> &depths,
                                 const Eigen::Isometry3d &T_init) {
    PlanarPoseResult result;
    result.T_cur_ref = T_init;
    result.inlier_mask.assign(pixels_ref.size(), 0);
    if (params.num_iterations <= 0) {
        return result;
    }
    const ReferenceLevel reference = build_reference_level(camera, scale, params, img_ref, pixels_ref, depths);
    // An accepted candidate becomes the next linearization point, so its accumulation is reused.
    PoseAccumulation acc = accumulate_pose(camera, scale, params, reference, img_cur, result.T_cur_ref);
    for (int iter = 0; iter < params.num_iterations; ++iter) {
        result.iterations = iter + 1;
        result.visible = acc.visible;
        result.inliers = acc.inliers;
        result.cost = acc.cost;
        result.inlier_mask = acc.inlier_mask;
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
        PoseAccumulation next = accumulate_pose(camera, scale, params, reference, img_cur, candidate);
        if (!candidate.matrix().allFinite() || next.visible < 6 || next.cost > acc.cost) {
            break;
        }
        result.T_cur_ref = candidate;
        result.visible = next.visible;
        result.inliers = next.inliers;
        result.cost = next.cost;
        result.inlier_mask = next.inlier_mask;
        if (update.norm() < params.eps_convergence) {
            result.converged = true;
            break;
        }
        acc = std::move(next);
    }
    return result;
}

} // namespace

PlanarPoseResult align_planar_patch_pose(const Camera &camera, const DirectSlamParams &params,
                                         const cv::Mat &img_ref, const cv::Mat &img_cur,
                                         const std::vector<Vec2d> &pixels_ref,
                                         const std::vector<double> &depth_ref,
                                         const Eigen::Isometry3d &T_cur_ref_init) {
    require_gray(img_ref, "planar patch reference");
    require_gray(img_cur, "planar patch current");
    return align_planar_patch_pose(camera, params, build_pyramid(img_ref, params), build_pyramid(img_cur, params),
                                   pixels_ref, depth_ref, T_cur_ref_init);
}

PlanarPoseResult align_planar_patch_pose(const Camera &camera, const DirectSlamParams &params,
                                         const ImagePyramid &pyramid_ref, const ImagePyramid &pyramid_cur,
                                         const std::vector<Vec2d> &pixels_ref,
                                         const std::vector<double> &depth_ref,
                                         const Eigen::Isometry3d &T_cur_ref_init) {
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("planar patch: pixels and depths have different sizes");
    }
    PlanarPoseResult result;
    result.T_cur_ref = T_cur_ref_init;
    result.inlier_mask.assign(pixels_ref.size(), 0);
    for (int level = params.num_pyramid_levels - 1; level >= 0; --level) {
        const size_t index = static_cast<size_t>(level);
        if (pyramid_ref.levels[index].cols <= 2 * params.feature_half_patch + 2 ||
            pyramid_ref.levels[index].rows <= 2 * params.feature_half_patch + 2) {
            continue;
        }
        result = align_one_level(camera, pyramid_ref.scales[index], params, pyramid_ref.levels[index],
                                 pyramid_cur.levels[index], pixels_ref, depth_ref, result.T_cur_ref);
    }
    return result;
}

PatchOffsetResult align_planar_patch_offsets(const Camera &camera, const DirectSlamParams &params,
                                             const cv::Mat &img_ref, const cv::Mat &img_cur,
                                             const std::vector<Vec2d> &pixels_ref,
                                             const std::vector<double> &depth_ref,
                                             const Eigen::Isometry3d &T_cur_ref) {
    require_gray(img_ref, "patch offset reference");
    require_gray(img_cur, "patch offset current");
    return align_planar_patch_offsets(camera, params, build_pyramid(img_ref, params),
                                      build_pyramid(img_cur, params), pixels_ref, depth_ref, T_cur_ref);
}

PatchOffsetResult align_planar_patch_offsets(const Camera &camera, const DirectSlamParams &params,
                                             const ImagePyramid &pyramid_ref, const ImagePyramid &pyramid_cur,
                                             const std::vector<Vec2d> &pixels_ref,
                                             const std::vector<double> &depth_ref,
                                             const Eigen::Isometry3d &T_cur_ref) {
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("patch offset: pixels and depths have different sizes");
    }
    const int half = params.feature_half_patch;
    const double gate = params.intensity_inlier_threshold;
    const double max_shift = std::max(2.0, 2.0 * static_cast<double>(half));
    PatchOffsetResult result;
    result.pixels_cur.assign(pixels_ref.size(), Vec2d::Zero());
    result.status.assign(pixels_ref.size(), 0);

    for (size_t i = 0; i < pixels_ref.size(); ++i) {
        if (!(depth_ref[i] > 0.0) || !std::isfinite(depth_ref[i])) {
            continue;
        }
        Vec2d offset = Vec2d::Zero();
        bool tracked = true;
        double mean_abs = 0.0;
        for (int level = params.num_pyramid_levels - 1; level >= 0 && tracked; --level) {
            const size_t index = static_cast<size_t>(level);
            const double scale = pyramid_ref.scales[index];
            const cv::Mat &ref_level = pyramid_ref.levels[index];
            const cv::Mat &cur_level = pyramid_cur.levels[index];
            if (level < params.num_pyramid_levels - 1) {
                offset *= scale / pyramid_ref.scales[index + 1];
            }
            // Reference gradients stay fixed: the incremental unknown is a translation
            // of the already projected patch in the current image.
            Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
            std::vector<Vec2d> gradients;
            std::vector<Vec2d> projected;
            std::vector<float> intensities_ref; // Fixed per level; sampled once, not per iteration.
            gradients.reserve(static_cast<size_t>((2 * half + 1) * (2 * half + 1)));
            projected.reserve(gradients.capacity());
            intensities_ref.reserve(gradients.capacity());
            for (int py = -half; py <= half && tracked; ++py) {
                for (int px = -half; px <= half; ++px) {
                    const Vec2d ref_level_px = scale * pixels_ref[i] + Vec2d(px, py);
                    const Vec2d ref_full = ref_level_px / scale;
                    Vec2d cur_level_px = Vec2d::Zero();
                    if (ref_level_px.x() < 1.0 || ref_level_px.y() < 1.0 ||
                        ref_level_px.x() >= ref_level.cols - 2 || ref_level_px.y() >= ref_level.rows - 2 ||
                        !warp_patch_pixel(camera, T_cur_ref, ref_full, depth_ref[i], scale, cur_level_px,
                                          nullptr)) {
                        tracked = false;
                        break;
                    }
                    const Vec2d gradient(
                        0.5 * (sample_gray(ref_level, static_cast<float>(ref_level_px.x() + 1.0),
                                           static_cast<float>(ref_level_px.y())) -
                               sample_gray(ref_level, static_cast<float>(ref_level_px.x() - 1.0),
                                           static_cast<float>(ref_level_px.y()))),
                        0.5 * (sample_gray(ref_level, static_cast<float>(ref_level_px.x()),
                                           static_cast<float>(ref_level_px.y() + 1.0)) -
                               sample_gray(ref_level, static_cast<float>(ref_level_px.x()),
                                           static_cast<float>(ref_level_px.y() - 1.0))));
                    const Vec2d J = -gradient;
                    normal += J * J.transpose();
                    gradients.push_back(J);
                    projected.push_back(cur_level_px);
                    intensities_ref.push_back(sample_gray(ref_level, static_cast<float>(ref_level_px.x()),
                                                          static_cast<float>(ref_level_px.y())));
                }
            }
            if (!tracked || gradients.empty()) {
                tracked = false;
                break;
            }
            const Eigen::LDLT<Eigen::Matrix2d> solver(normal);
            if (solver.info() != Eigen::Success) {
                tracked = false;
                break;
            }
            for (int iter = 0; iter < params.num_iterations; ++iter) {
                Vec2d rhs = Vec2d::Zero();
                double abs_error = 0.0;
                int samples = 0;
                bool inside = true;
                for (size_t k = 0; k < gradients.size(); ++k) {
                    const Vec2d cur = projected[k] + offset;
                    if (cur.x() < 1.0 || cur.y() < 1.0 || cur.x() >= cur_level.cols - 2 ||
                        cur.y() >= cur_level.rows - 2) {
                        inside = false;
                        break;
                    }
                    const double error = intensities_ref[k] -
                                         sample_gray(cur_level, static_cast<float>(cur.x()),
                                                     static_cast<float>(cur.y()));
                    rhs += -error * gradients[k];
                    abs_error += std::abs(error);
                    ++samples;
                }
                if (!inside || samples == 0) {
                    tracked = false;
                    break;
                }
                const Vec2d step = solver.solve(rhs);
                if (!step.allFinite()) {
                    tracked = false;
                    break;
                }
                offset += step;
                mean_abs = abs_error / static_cast<double>(samples);
                if (step.norm() < 1e-3) {
                    break;
                }
            }
        }
        if (!tracked) {
            continue;
        }
        const Vec2d full_offset = offset; // finest level is full resolution, scale 1
        Vec2d center = Vec2d::Zero();
        if (!warp_patch_pixel(camera, T_cur_ref, pixels_ref[i], depth_ref[i], 1.0, center, nullptr)) {
            continue;
        }
        const Vec2d aligned = center + full_offset;
        if (mean_abs > gate || full_offset.norm() > max_shift || !aligned.allFinite()) {
            continue;
        }
        result.pixels_cur[i] = aligned;
        result.status[i] = 1;
    }
    return result;
}

} // namespace pyslam
