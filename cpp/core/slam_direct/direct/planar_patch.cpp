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
#include "slam_direct/se3.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pyslam {
namespace {

using Mat66d = Eigen::Matrix<double, 6, 6>;

float sample_gray(const cv::Mat &img, float x, float y) {
    if (img.cols < 2 || img.rows < 2) {
        return img.empty() ? 0.0f : static_cast<float>(img.at<uchar>(0, 0));
    }
    x = std::clamp(x, 0.0f, static_cast<float>(img.cols - 2));
    y = std::clamp(y, 0.0f, static_cast<float>(img.rows - 2));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float i00 = static_cast<float>(img.at<uchar>(y0, x0));
    const float i10 = static_cast<float>(img.at<uchar>(y0, x0 + 1));
    const float i01 = static_cast<float>(img.at<uchar>(y0 + 1, x0));
    const float i11 = static_cast<float>(img.at<uchar>(y0 + 1, x0 + 1));
    return (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 +
           fx * fy * i11;
}

void require_gray(const cv::Mat &img, const char *name) {
    if (img.empty() || img.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty CV_8UC1 image");
    }
}

struct ImagePyramid {
    std::vector<cv::Mat> levels;
    std::vector<double> scales;
};

ImagePyramid build_pyramid(const cv::Mat &image, const DirectSlamParams &params) {
    ImagePyramid pyramid;
    pyramid.levels.reserve(static_cast<size_t>(params.num_pyramid_levels));
    pyramid.scales.reserve(static_cast<size_t>(params.num_pyramid_levels));
    for (int level = 0; level < params.num_pyramid_levels; ++level) {
        if (level == 0) {
            pyramid.levels.push_back(image);
            pyramid.scales.push_back(1.0);
            continue;
        }
        const cv::Mat &previous = pyramid.levels.back();
        const cv::Size size(std::max(1, static_cast<int>(std::lround(previous.cols * params.pyramid_scale))),
                            std::max(1, static_cast<int>(std::lround(previous.rows * params.pyramid_scale))));
        cv::Mat resized;
        cv::resize(previous, resized, size, 0.0, 0.0, cv::INTER_LINEAR);
        pyramid.levels.push_back(resized);
        pyramid.scales.push_back(pyramid.scales.back() * params.pyramid_scale);
    }
    return pyramid;
}

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

PoseAccumulation accumulate_pose(const Camera &camera, double scale, const DirectSlamParams &params,
                                 const cv::Mat &img_ref, const cv::Mat &img_cur,
                                 const std::vector<Vec2d> &pixels_ref, const std::vector<double> &depths,
                                 const Eigen::Isometry3d &T_cur_ref) {
    PoseAccumulation acc;
    acc.inlier_mask.assign(pixels_ref.size(), 0);
    const int half = params.feature_half_patch;
    const double gate = params.intensity_inlier_threshold;
    for (size_t i = 0; i < pixels_ref.size(); ++i) {
        if (!(depths[i] > 0.0) || !std::isfinite(depths[i])) {
            continue;
        }
        double abs_error = 0.0;
        int samples = 0;
        Mat66d H = Mat66d::Zero();
        Vec6d bias = Vec6d::Zero();
        double cost = 0.0;
        bool visible = true;
        for (int py = -half; py <= half && visible; ++py) {
            for (int px = -half; px <= half; ++px) {
                const Vec2d ref_level = scale * pixels_ref[i] + Vec2d(px, py);
                const Vec2d ref_full = ref_level / scale;
                Vec2d cur_level = Vec2d::Zero();
                Eigen::Matrix<double, 2, 6> J_pixel = Eigen::Matrix<double, 2, 6>::Zero();
                if (!warp_patch_pixel(camera, T_cur_ref, ref_full, depths[i], scale, cur_level, &J_pixel)) {
                    visible = false;
                    break;
                }
                if (cur_level.x() < 1.0 || cur_level.y() < 1.0 || cur_level.x() >= img_cur.cols - 2 ||
                    cur_level.y() >= img_cur.rows - 2 || ref_level.x() < 1.0 || ref_level.y() < 1.0 ||
                    ref_level.x() >= img_ref.cols - 2 || ref_level.y() >= img_ref.rows - 2) {
                    visible = false;
                    break;
                }
                const double error = sample_gray(img_ref, static_cast<float>(ref_level.x()),
                                                 static_cast<float>(ref_level.y())) -
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
    for (int iter = 0; iter < params.num_iterations; ++iter) {
        const PoseAccumulation acc =
            accumulate_pose(camera, scale, params, img_ref, img_cur, pixels_ref, depths, result.T_cur_ref);
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
        const PoseAccumulation next =
            accumulate_pose(camera, scale, params, img_ref, img_cur, pixels_ref, depths, candidate);
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
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("planar patch: pixels and depths have different sizes");
    }
    const ImagePyramid pyramid_ref = build_pyramid(img_ref, params);
    const ImagePyramid pyramid_cur = build_pyramid(img_cur, params);
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
    if (pixels_ref.size() != depth_ref.size()) {
        throw std::invalid_argument("patch offset: pixels and depths have different sizes");
    }
    const ImagePyramid pyramid_ref = build_pyramid(img_ref, params);
    const ImagePyramid pyramid_cur = build_pyramid(img_cur, params);
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
            gradients.reserve(static_cast<size_t>((2 * half + 1) * (2 * half + 1)));
            projected.reserve(gradients.capacity());
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
                    const int py = static_cast<int>(k) / (2 * half + 1) - half;
                    const int px = static_cast<int>(k) % (2 * half + 1) - half;
                    const Vec2d ref_level_px = scale * pixels_ref[i] + Vec2d(px, py);
                    const Vec2d cur = projected[k] + offset;
                    if (cur.x() < 1.0 || cur.y() < 1.0 || cur.x() >= cur_level.cols - 2 ||
                        cur.y() >= cur_level.rows - 2) {
                        inside = false;
                        break;
                    }
                    const double error = sample_gray(ref_level, static_cast<float>(ref_level_px.x()),
                                                     static_cast<float>(ref_level_px.y())) -
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
