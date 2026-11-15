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

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "camera/camera.h"
#include "camera/camera_models.h"
#include "slam_direct/direct/direct_aligner.h"
#include "slam_direct/direct/depth_filter.h"
#include "slam_direct/direct/planar_patch.h"
#include "slam_direct/direct/semi_direct_pose_tracker.h"
#include "slam_direct/se3.h"
#include "slam_direct/direct_frontend.h"
#include "slam_direct/direct_backend.h"
#include "common/feature_trails.h"
#include "slam_direct/features/corner_grid.h"
#include "slam_direct/features/corner_detection.h"
#include "slam_direct/geometry/reprojection_pose.h"
#include "slam_direct/geometry/monocular_initializer.h"
#include "slam_direct/klt/klt_tracker.h"

using namespace pyslam;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

PinholeCamera make_camera(int width, int height, double fx, double fy, double cx, double cy) {
    PinholeCamera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;
    camera.u_min = 0.0;
    camera.u_max = width;
    camera.v_min = 0.0;
    camera.v_max = height;
    camera.type = CameraType::PINHOLE;
    camera.is_distorted = false;
    camera.depth_factor = 1.0;
    camera.set_intrinsic_matrices();
    return camera;
}

KannalaBrandt4Camera make_fisheye(int width, int height, double fx, double fy, double cx, double cy);

cv::Mat textured_image(int width, int height, double shift_x) {
    cv::Mat image(height, width, CV_8U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double sx = static_cast<double>(x) - shift_x;
            const double sy = static_cast<double>(y);
            const double value = 128.0 + 50.0 * std::sin(sx * 0.21) + 35.0 * std::cos(sy * 0.17) +
                                 25.0 * std::sin((sx + sy) * 0.11);
            image.at<uchar>(y, x) =
                static_cast<uchar>(std::max(0.0, std::min(255.0, std::round(value))));
        }
    }
    return image;
}

cv::Mat checkerboard(int width, int height, int cell, int shift_x) {
    cv::Mat image(height, width, CV_8U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int sx = x - shift_x;
            const int cell_x = sx >= 0 ? sx / cell : (sx - cell + 1) / cell;
            const int cell_y = y / cell;
            image.at<uchar>(y, x) = ((cell_x + cell_y) & 1) ? 220 : 30;
        }
    }
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);
    return image;
}

// A known image shift and metric plane depth determine the expected SE3 translation.
void test_direct_alignment_recovers_translation() {
    const int width = 160;
    const int height = 120;
    const double fx = 120.0;
    const double fy = 120.0;
    const double depth = 2.0;
    const double shift = 8.0;
    PinholeCamera camera = make_camera(width, height, fx, fy, 80.0, 60.0);
    const cv::Mat reference = textured_image(width, height, 0.0);
    const cv::Mat current = textured_image(width, height, shift);

    std::vector<Vec2d> pixels;
    std::vector<double> depths;
    for (int y = 24; y < height - 24; y += 4) {
        for (int x = 24; x < width - 24; x += 4) {
            pixels.emplace_back(static_cast<double>(x), static_cast<double>(y));
            depths.push_back(depth);
        }
    }
    DirectSlamParams params;
    params.num_iterations = 15;
    params.eps_convergence = 1e-4;
    params.num_pyramid_levels = 4;
    params.half_patch_size = 1;

    const DirectAlignmentResult result = align_direct_multilayer(
        camera, params, reference, current, pixels, depths, Eigen::Isometry3d::Identity());
    const double expected_tx = shift * depth / fx;
    const Vec3d translation = result.T_cur_ref.translation();
    const double rotation = Eigen::AngleAxisd(result.T_cur_ref.linear()).angle();
    std::cout << "direct translation " << translation.transpose() << " expected " << expected_tx
              << " rotation " << rotation << " inliers " << result.inliers << std::endl;
    require(result.inliers > 50, "direct alignment produced too few inliers");
    require(std::abs(translation.x() - expected_tx) < 0.03, "direct translation x is wrong");
    require(std::abs(translation.y()) < 0.02 && std::abs(translation.z()) < 0.02,
            "direct translation leaked into y or z");
    require(rotation < 0.02, "direct alignment invented rotation");
}

Eigen::Matrix<double, 2, 6> pinhole_photometric_jacobian(const Camera &camera, const Vec3d &X) {
    const double z_inv = 1.0 / X.z();
    const double z2_inv = z_inv * z_inv;
    const double x = X.x();
    const double y = X.y();
    Eigen::Matrix<double, 2, 6> J;
    J << camera.fx * z_inv, 0.0, -camera.fx * x * z2_inv, -camera.fx * x * y * z2_inv,
        camera.fx + camera.fx * x * x * z2_inv, -camera.fx * y * z_inv, 0.0, camera.fy * z_inv,
        -camera.fy * y * z2_inv, -camera.fy - camera.fy * y * y * z2_inv, camera.fy * x * y * z2_inv,
        camera.fy * x * z_inv;
    return J;
}

void test_photometric_jacobian_matches_pinhole_and_finite_difference() {
    PinholeCamera pinhole = make_camera(320, 240, 200.0, 210.0, 150.0, 120.0);
    const Vec3d points[] = {Vec3d(0.2, -0.4, 3.0), Vec3d(-1.1, 0.7, 5.5), Vec3d(0.0, 0.0, 1.2)};
    for (const Vec3d &X : points) {
        Vec2d uv;
        Eigen::Matrix<double, 2, 6> J;
        require(photometric_warp_jacobian(pinhole, X, 1.0, uv, J), "pinhole warp rejected a front point");
        require((J - pinhole_photometric_jacobian(pinhole, X)).norm() < 1e-8,
                "photometric Jacobian does not match the pinhole closed form");
        const auto projected = pinhole.project_point(X);
        require((uv - projected.first).norm() < 1e-8, "pinhole warp pixel does not match project_point");
        Vec2d uv_level;
        Eigen::Matrix<double, 2, 6> J_level;
        require(photometric_warp_jacobian(pinhole, X, 0.5, uv_level, J_level), "scaled warp failed");
        require((uv_level - 0.5 * uv).norm() < 1e-8 && (J_level - 0.5 * J).norm() < 1e-8,
                "pyramid scale did not apply to the pixel and the Jacobian");
    }

    KannalaBrandt4Camera fisheye = make_fisheye(320, 240, 180.0, 180.0, 160.0, 120.0);
    Vec3d bearing;
    const Vec2d pixel(220.0, 70.0);
    require(fisheye.unproject_bearing(pixel, bearing, nullptr), "fisheye unprojection failed");
    const Vec3d X = bearing * (2.5 / bearing.z());
    Vec2d uv;
    Eigen::Matrix<double, 2, 6> J;
    require(photometric_warp_jacobian(fisheye, X, 1.0, uv, J), "fisheye warp rejected the lifted point");
    const double eps = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec6d xi = Vec6d::Zero();
        xi[k] = eps;
        Eigen::Isometry3d perturb = Eigen::Isometry3d::Identity();
        left_multiply_translation_rotation(perturb, xi);
        Vec2d uv_step;
        Eigen::Matrix<double, 2, 6> unused;
        require(photometric_warp_jacobian(fisheye, perturb * X, 1.0, uv_step, unused),
                "perturbed fisheye point did not project");
        const Vec2d numerical = (uv_step - uv) / eps;
        require((numerical - J.col(k)).norm() < 1e-4, "fisheye photometric Jacobian failed the finite difference");
    }
}

// Lifting a raw pixel at sensor z must reproject to that pixel, and invalid inputs are rejected.
void test_lift_reference_point_and_warp_rejections() {
    KannalaBrandt4Camera fisheye = make_fisheye(320, 240, 180.0, 180.0, 160.0, 120.0);
    PinholeCamera pinhole = make_camera(320, 240, 200.0, 210.0, 150.0, 120.0);
    const Vec2d pixels[] = {Vec2d(160.0, 120.0), Vec2d(15.0, 20.0), Vec2d(300.0, 225.0), Vec2d(220.0, 70.0)};
    for (const Camera *camera : {static_cast<const Camera *>(&fisheye), static_cast<const Camera *>(&pinhole)}) {
        for (const Vec2d &pixel : pixels) {
            for (const double depth : {0.6, 2.5, 40.0}) {
                Vec3d X;
                require(lift_reference_point(*camera, pixel, depth, X), "lift rejected a valid pixel");
                require(std::abs(X.z() - depth) < 1e-9, "lift did not keep camera-frame z");
                Vec2d uv;
                Eigen::Matrix<double, 2, 6> J;
                require(photometric_warp_jacobian(*camera, X, 1.0, uv, J), "lifted point did not warp");
                require((uv - pixel).norm() < 1e-6, "lifted point does not reproject to its raw pixel");
            }
        }
        Vec3d X;
        for (const double depth : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()}) {
            require(!lift_reference_point(*camera, pixels[0], depth, X), "lift accepted an invalid depth");
        }
        Vec2d uv;
        Eigen::Matrix<double, 2, 6> J;
        const Vec3d front(0.1, -0.2, 2.0);
        require(!photometric_warp_jacobian(*camera, front, 0.0, uv, J), "warp accepted a zero scale");
        require(!photometric_warp_jacobian(*camera, front, -0.5, uv, J), "warp accepted a negative scale");
        require(!photometric_warp_jacobian(*camera, Vec3d(0.1, 0.2, -2.0), 1.0, uv, J),
                "warp accepted a point behind the camera");
        require(!photometric_warp_jacobian(*camera, Vec3d(0.1, 0.2, 0.0), 1.0, uv, J),
                "warp accepted a point on the camera plane");
        require(!photometric_warp_jacobian(*camera, Vec3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 2.0),
                                           1.0, uv, J),
                "warp accepted a non-finite point");
    }

    // Raw-pixel models are accepted; a distorted linear pinhole has no distortion Jacobian.
    auto distorted = std::make_shared<PinholeCamera>(make_camera(160, 120, 100, 100, 80, 60));
    distorted->is_distorted = true;
    distorted->sensor_type = SensorType::RGBD;
    bool rejected = false;
    try { DirectSlam slam(distorted, ConfigDict()); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "DirectSlam accepted a distorted linear pinhole");
}

uchar sample_gray(const cv::Mat &image, double x, double y) {
    if (x < 0.0 || y < 0.0 || x >= image.cols - 1 || y >= image.rows - 1) {
        return 0;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double xx = x - x0;
    const double yy = y - y0;
    const double value = (1.0 - xx) * (1.0 - yy) * image.at<uchar>(y0, x0) +
                         xx * (1.0 - yy) * image.at<uchar>(y0, x0 + 1) +
                         (1.0 - xx) * yy * image.at<uchar>(y0 + 1, x0) +
                         xx * yy * image.at<uchar>(y0 + 1, x0 + 1);
    return static_cast<uchar>(std::lround(std::max(0.0, std::min(255.0, value))));
}

// A fronto-parallel plane, inverse-warped through the fisheye model, must be recovered
// from the raw pixels. One coarse pyramid level is included.
void check_kannala_brandt_direct_alignment(const Eigen::Isometry3d &expected, const char *label) {
    KannalaBrandt4Camera camera = make_fisheye(160, 120, 140.0, 140.0, 80.0, 60.0);
    const double depth = 2.0;
    const Eigen::Isometry3d T_ref_cur = expected.inverse();
    const cv::Mat reference = textured_image(160, 120, 0.0);
    cv::Mat current(120, 160, CV_8U, cv::Scalar(0));
    for (int y = 0; y < current.rows; ++y) {
        for (int x = 0; x < current.cols; ++x) {
            Vec3d bearing;
            if (!camera.unproject_bearing(Vec2d(x, y), bearing, nullptr) || !(bearing.z() > 1e-3)) {
                continue;
            }
            const double denom = (T_ref_cur.linear() * bearing).z();
            if (!(std::abs(denom) > 1e-8)) {
                continue;
            }
            const double ray = (depth - T_ref_cur.translation().z()) / denom;
            if (!(ray > 0.0)) {
                continue;
            }
            const Vec3d X_ref = T_ref_cur * (ray * bearing);
            const auto uv = camera.project_point(X_ref);
            if (!std::isfinite(uv.first.x()) || !std::isfinite(uv.first.y())) {
                continue;
            }
            current.at<uchar>(y, x) = sample_gray(reference, uv.first.x(), uv.first.y());
        }
    }

    std::vector<Vec2d> pixels;
    std::vector<double> depths;
    for (int y = 28; y < 92; y += 6) {
        for (int x = 28; x < 132; x += 6) {
            pixels.emplace_back(x, y);
            depths.push_back(depth);
        }
    }
    DirectSlamParams params;
    params.num_iterations = 15;
    params.eps_convergence = 1e-4;
    params.num_pyramid_levels = 2;
    params.pyramid_scale = 0.5;
    params.half_patch_size = 1;
    const DirectAlignmentResult result = align_direct_multilayer(
        camera, params, reference, current, pixels, depths, Eigen::Isometry3d::Identity());
    const Eigen::Isometry3d error = result.T_cur_ref * expected.inverse();
    if (result.inliers <= 40 || error.translation().norm() >= 0.01 ||
        Eigen::AngleAxisd(error.linear()).angle() >= 0.004) {
        throw std::runtime_error(std::string("fisheye direct alignment failed: ") + label);
    }
}

void test_kannala_brandt_direct_alignment() {
    Eigen::Isometry3d translation = Eigen::Isometry3d::Identity();
    translation.translation() = Vec3d(0.04, -0.01, 0.0);
    check_kannala_brandt_direct_alignment(translation, "translation");
    // Rotation exercises the rotational Jacobian columns through the distortion model.
    Eigen::Isometry3d rotation = translation;
    rotation.linear() = Eigen::AngleAxisd(0.015, Vec3d(0.2, 1.0, 0.1).normalized()).toRotationMatrix();
    check_kannala_brandt_direct_alignment(rotation, "rotation");
}

void test_grid_fast_spreads_corners() {
    const int width = 80;
    const int height = 60;
    cv::Mat image(height, width, CV_8U, cv::Scalar(0));
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            const int x = col * 40 + 16;
            const int y = row * 30 + 11;
            image(cv::Rect(x, y, 8, 8)).setTo(255);
        }
    }
    CornerGridRequest request;
    request.num_features = 4;
    request.grid_cols = 2;
    request.grid_rows = 2;
    request.fast_threshold = 10;
    // Axis-aligned square corners have tied FAST scores that suppress each other.
    request.nonmax_suppression = false;
    request.border = 0;
    request.refine_subpixel = false;
    const std::vector<Vec2d> corners = detect_grid_fast(image, request);
    std::array<int, 4> counts = {0, 0, 0, 0};
    for (const Vec2d &corner : corners) {
        const int bin = (corner.x() >= 40 ? 1 : 0) + (corner.y() >= 30 ? 2 : 0);
        ++counts[bin];
    }
    for (int count : counts) {
        require(count >= 1, "grid FAST left a cell without a corner");
    }

    cv::Mat allowed(image.size(), CV_8U, cv::Scalar(255));
    allowed(cv::Rect(0, 0, 40, 30)).setTo(0);
    request.cells = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    const std::vector<Vec2d> masked = detect_grid_fast(image, request, allowed);
    for (const Vec2d &corner : masked) {
        require(!(corner.x() < 40.0 && corner.y() < 30.0), "grid FAST kept a corner in a masked cell");
    }
    require(!masked.empty(), "grid FAST removed every unmasked cell");

    // num_features is a hard cap, for the full grid and for an explicit cell list,
    // and sub-pixel refinement must not move a corner into the masked region.
    // Smoothed noise has many isolated FAST corners in every cell.
    cv::Mat busy(120, 160, CV_8U);
    cv::RNG rng(7);
    rng.fill(busy, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(busy, busy, cv::Size(5, 5), 1.0);
    CornerGridRequest capped;
    capped.grid_cols = 5;
    capped.grid_rows = 5;
    capped.refine_subpixel = true;
    capped.border = 6;
    for (int budget : {1, 3, 7, 10, 24, 40}) {
        capped.num_features = budget;
        capped.cells.clear();
        // Every cell has corners, so the budget is filled exactly, not exceeded.
        require(int(detect_grid_fast(busy, capped).size()) == budget, "grid FAST did not return num_features");
        capped.cells = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
        require(int(detect_grid_fast(busy, capped).size()) <= budget,
                "grid FAST exceeded num_features for explicit cells");
    }
    capped.cells.clear();
    capped.num_features = 200;
    cv::Mat half(busy.size(), CV_8U, cv::Scalar(255));
    half.colRange(0, 80).setTo(0);
    const std::vector<Vec2d> refined = detect_grid_fast(busy, capped, half);
    require(refined.size() > 20, "grid FAST found too few corners on textured input");
    for (const Vec2d &corner : refined) {
        const int x = int(std::lround(corner.x()));
        const int y = int(std::lround(corner.y()));
        require(x >= 80 && x >= capped.border && y >= capped.border && x < busy.cols - capped.border &&
                    y < busy.rows - capped.border,
                "refined grid FAST corner left the allowed region");
    }

    ConfigDict config;
    config["corner_detector"] = "grid_fast";
    config["grid_cols"] = 6;
    config["grid_rows"] = 3;
    config["fast_threshold"] = 12;
    const DirectSlamParams params = DirectSlamParams::from_config(config);
    require(params.corner_detector == CornerDetector::GridFast && params.grid_cols == 6 &&
                params.grid_rows == 3 && params.fast_threshold == 12,
            "grid FAST settings were not parsed");
}

// Both detectors must treat a malformed occupancy mask the same way regardless of the
// border, and a border too large for the image is clamped for mask and corners alike.
void test_detect_corners_mask_consistency() {
    cv::Mat gray(120, 160, CV_8U);
    cv::RNG rng(11);
    rng.fill(gray, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
    const cv::Mat wrong_size(10, 10, CV_8U, cv::Scalar(0));
    for (const CornerDetector detector : {CornerDetector::GridFast, CornerDetector::Gftt}) {
        for (const int border : {0, 8}) {
            DirectSlamParams params;
            params.corner_detector = detector;
            params.num_features = 60;
            params.border = border;
            const auto plain = detect_corners(gray, params, cv::Mat(), 0);
            const auto ignored = detect_corners(gray, params, wrong_size, 0);
            require(!plain.empty(), "detect_corners found no corners");
            require(plain.size() == ignored.size(), "a malformed occupancy mask changed detection");
        }
        DirectSlamParams params;
        params.corner_detector = detector;
        params.num_features = 60;
        params.border = 1000; // Clamped to min(rows, cols) / 4 = 30.
        const auto clamped = detect_corners(gray, params, cv::Mat(), 0);
        require(!clamped.empty(), "an oversized border removed every corner");
        for (const Vec2d &corner : clamped) {
            require(corner.x() >= 29.5 && corner.y() >= 29.5 && corner.x() < 130.5 && corner.y() < 90.5,
                    "corner inside the clamped border");
        }
    }
}

// The landmark -> observer index used by local BA follows insertion and reset.
void test_direct_map_observer_index() {
    DirectMap map;
    for (int id = 0; id < 3; ++id) {
        DirectKeyframe keyframe;
        keyframe.id = id;
        for (int64_t landmark : {int64_t(5), int64_t(10 + id)}) {
            DirectPoint obs;
            obs.id = landmark;
            keyframe.points.push_back(obs);
        }
        keyframe.points.push_back(keyframe.points.front()); // Duplicate observation.
        map.add_keyframe(std::move(keyframe));
    }
    const std::vector<int> *shared = map.observers(5);
    require(shared && *shared == std::vector<int>({0, 1, 2}), "shared landmark observers are wrong");
    const std::vector<int> *single = map.observers(11);
    require(single && *single == std::vector<int>({1}), "single landmark observers are wrong");
    require(map.observers(99) == nullptr, "unknown landmark has observers");
    map.clear();
    require(map.observers(5) == nullptr, "clear() kept the observer index");
}

// Exercise the full corner -> LK -> reprojection path with a known metric displacement.
void test_lk_and_geometric_pose() {
    const int width = 320;
    const int height = 240;
    const int shift = 8;
    const double fx = 200.0;
    const double depth = 3.0;
    PinholeCamera camera = make_camera(width, height, fx, fx, 160.0, 120.0);
    const cv::Mat reference = checkerboard(width, height, 16, 0);
    const cv::Mat current = checkerboard(width, height, 16, shift);

    DirectSlamParams params;
    params.num_features = 250;
    params.gftt_min_distance = 10.0;
    params.border = 20;
    params.lk_patch_size = 15;
    params.lk_max_level = 3;
    const std::vector<Vec2d> detected = detect_gftt(reference, params);
    require(detected.size() >= 20, "GFTT found too few corners");
    const LkTrack tracked = track_pyramidal_lk(reference, current, detected, detected, params);

    std::vector<double> dx;
    std::vector<PoseObservation> observations;
    for (size_t i = 0; i < detected.size(); ++i) {
        if (!tracked.status[i]) {
            continue;
        }
        dx.push_back(tracked.points_cur[i].x() - detected[i].x());
        PoseObservation observation;
        observation.pixel_cur = tracked.points_cur[i];
        observation.point_ref = camera.unproject_point_3d(detected[i].x(), detected[i].y(), depth);
        observations.push_back(observation);
    }
    require(dx.size() >= 15, "pyramidal LK tracked too few corners");
    std::sort(dx.begin(), dx.end());
    const double median_dx = dx[dx.size() / 2];
    std::cout << "LK tracks " << dx.size() << " median dx " << median_dx << std::endl;
    require(std::abs(median_dx - shift) < 1.5, "LK displacement is not the image shift");

    const ReprojectionPoseResult pose =
        estimate_pose_reprojection(camera, observations, Eigen::Isometry3d::Identity(), params);
    const double expected_tx = static_cast<double>(shift) * depth / fx;
    const Vec3d translation = pose.T_cur_ref.translation();
    std::cout << "geometric translation " << translation.transpose() << " expected " << expected_tx
              << " inliers " << pose.inliers << std::endl;
    require(pose.inliers >= 10, "geometric pose kept too few inliers");
    require(std::abs(translation.x() - expected_tx) < 0.04, "geometric translation x is wrong");
    require(std::abs(translation.y()) < 0.03 && std::abs(translation.z()) < 0.05,
            "geometric translation leaked into y or z");
}

// Repetitive texture also exercises photometric recovery when geometric seeding is weak.
void test_frontend_direct_on_rgbd() {
    const int width = 160;
    const int height = 120;
    const double fx = 100.0;
    const double depth = 2.5;
    const double shift = 6.0;
    auto camera = std::make_shared<PinholeCamera>(make_camera(width, height, fx, fx, 80.0, 60.0));
    camera->sensor_type = SensorType::RGBD;
    ConfigDict config;
    config["mode"] = "hybrid";
    config["num_features"] = 150;
    config["gftt_quality"] = 0.001;
    config["gftt_min_distance"] = 6.0;
    config["num_features_tracking"] = 8;
    config["num_features_tracking_bad"] = 3;
    config["num_features_needed_for_keyframe"] = 10000;
    config["num_iterations"] = 12;

    DirectSlam slam(camera, config);
    const cv::Mat reference = checkerboard(width, height, 12, 0);
    const cv::Mat current = checkerboard(width, height, 12, static_cast<int>(shift));
    cv::Mat depth_ref(height, width, CV_32F, cv::Scalar(static_cast<float>(depth)));
    cv::Mat depth_cur = depth_ref.clone();
    slam.track(reference, cv::Mat(), depth_ref, 0, 0.0);
    require(slam.map().keyframes().size() == 1, "frontend did not create the first keyframe");
    slam.track(current, cv::Mat(), depth_cur, 1, 1.0);
    const double expected_tx = shift * depth / fx;
    const double tx = slam.pose().tcw().x();
    std::cout << "frontend tx " << tx << " expected " << expected_tx << " status " << slam.status_name()
              << " inliers " << slam.last_inliers() << std::endl;
    require(slam.status() == DirectTrackingStatus::Good || slam.status() == DirectTrackingStatus::Bad,
            "frontend lost the second frame");
    require(std::abs(tx - expected_tx) < 0.05, "frontend direct pose is wrong");
}

// Pose recovery must follow measured correspondences instead of trusting the motion seed.
void test_pose_with_outliers_and_bad_prediction() {
    auto camera = make_camera(640, 480, 400, 400, 320, 240);
    Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
    expected.linear() = Eigen::AngleAxisd(0.04, Vec3d::UnitY()).toRotationMatrix();
    expected.translation() = Vec3d(0.15, -0.02, -0.7);
    std::vector<PoseObservation> observations;
    int count = 0;
    for (int y = 60; y < 420; y += 30) {
        for (int x = 80; x < 560; x += 30) {
            const double depth = 4.0 + (count % 7);
            const Vec3d X = camera.unproject_point_3d(double(x), double(y), depth);
            Vec2d uv = camera.project_point(expected * X).first;
            if (count % 4 == 0) uv += Vec2d(35, -25);
            observations.push_back({X, uv});
            ++count;
        }
    }
    Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
    prediction.translation() = Vec3d(-0.4, 0.2, 0.4);
    const auto result = estimate_pose_reprojection(camera, observations, prediction, DirectSlamParams());
    require((result.T_cur_ref.matrix() - expected.matrix()).norm() < 1e-3,
            "pose did not recover from outliers and a bad prediction");
    require(result.inliers == count - (count + 3) / 4, "incorrect geometric inlier count");
    for (int i = 0; i < count; ++i)
        require(bool(result.inlier_mask[i]) == (i % 4 != 0), "outlier mask is not observation aligned");
}

// Blank frames must not integrate velocity; later recovery must retain the same world origin.
void test_stereo_frontend_loss_and_reset() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(320, 240, 200, 200, 160, 120));
    camera->sensor_type = SensorType::STEREO;
    camera->bf = 40.0;
    ConfigDict config;
    config["num_features"] = 250;
    config["num_features_tracking"] = 30;
    config["num_features_tracking_bad"] = 10;
    config["num_features_needed_for_keyframe"] = 240;
    config["lk_patch_size"] = 21;
    DirectSlam slam(camera, config);
    cv::Mat reference(240, 320, CV_8U);
    cv::RNG rng(42);
    rng.fill(reference, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(reference, reference, cv::Size(5, 5), 0.8);
    auto shifted = [&](int dx) {
        cv::Mat image;
        cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, 0);
        cv::warpAffine(reference, image, affine, reference.size());
        return image;
    };
    slam.track(reference, shifted(-10), cv::Mat(), 0, 0.0);
    require(slam.status() == DirectTrackingStatus::Good, "stereo initialization failed");
    for (const auto &point : slam.live_points())
        require(std::abs(point.position_w.z() - 4.0) < 0.1, "rectified stereo depth is wrong");
    slam.track(shifted(3), shifted(-7), cv::Mat(), 1, 0.1);
    require(slam.status() != DirectTrackingStatus::Lost, "stereo temporal tracking failed");
    require(std::abs(slam.pose().tcw().x() - 0.06) < 0.02, "stereo temporal metric scale is wrong");
    for (const auto &kf : slam.map().keyframes())
        require(kf.img_id >= 0 && kf.timestamp >= 0, "keyframe metadata was lost");
    const auto pose = slam.pose().Tcw();
    const auto size = slam.map().trajectory().size();
    cv::Mat blank(reference.size(), CV_8U, cv::Scalar(0));
    for (int i = 2; i < 7; ++i) {
        slam.track(blank, blank, cv::Mat(), i, i * 0.1);
        require(slam.status() == DirectTrackingStatus::Lost, "blank image did not report loss");
        require((slam.pose().Tcw() - pose).norm() < 1e-12, "tracking loss integrated phantom motion");
        require(slam.map().trajectory().size() == size, "tracking loss added an unmeasured pose");
    }
    slam.track(shifted(6), shifted(-4), cv::Mat(), 7, 0.7);
    require(slam.status() != DirectTrackingStatus::Lost, "did not recover against last accepted image");
    require(std::abs(slam.pose().tcw().x() - 0.12) < 0.03, "recovery changed the world origin");
    slam.reset();
    require(slam.map().keyframes().empty() && slam.map().trajectory().empty() && slam.live_points().empty(),
            "reset left stale map data");
    require(slam.status() == DirectTrackingStatus::Init, "reset did not restore INIT status");
}

// Verify sensor-unit conversion and rejection of zero/NaN depth before landmark creation.
void test_rgbd_depth_units_and_invalid_samples() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(160, 120, 100, 100, 80, 60));
    camera->sensor_type = SensorType::RGBD;
    camera->depth_factor = 0.001;
    ConfigDict config;
    config["num_features_tracking_bad"] = 3;
    DirectSlam slam(camera, config);
    cv::Mat image = checkerboard(160, 120, 12, 0);
    cv::Mat depth(120, 160, CV_16U, cv::Scalar(2500));
    slam.track(image, cv::Mat(), depth, 0, 0.0);
    require(!slam.live_points().empty(), "16-bit RGBD did not initialize");
    for (const auto &point : slam.live_points())
        require(std::abs(point.position_w.z() - 2.5) < 1e-9, "RGBD depth factor was not applied");
    slam.reset();
    depth.setTo(0);
    slam.track(image, cv::Mat(), depth, 0, 0.0);
    require(slam.map().keyframes().empty(), "zero depth seeded landmarks");
    cv::Mat invalid_depth(120, 160, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    slam.track(image, cv::Mat(), invalid_depth, 1, 0.1);
    require(slam.map().keyframes().empty(), "NaN depth seeded landmarks");
}

// Cover both settings formats and fail early on unsupported sensor input.
void test_configuration_and_missing_depth() {
    ConfigDict flat;
    flat["DIRECT_SLAM.mode"] = "hybrid";
    flat["DIRECT_SLAM.num_features"] = 123;
    const auto params = DirectSlamParams::from_config(flat);
    require(params.mode == DirectTrackingMode::Hybrid && params.num_features == 123 &&
                params.mode_name() == "hybrid",
            "flattened camera settings were not parsed");
    // Deprecated mode names still load older settings files.
    ConfigDict legacy;
    legacy["DIRECT_SLAM.mode"] = std::string("semi_direct");
    require(DirectSlamParams::from_config(legacy).mode == DirectTrackingMode::SemiDirect &&
                DirectSlamParams::from_config(legacy).mode_name() == "semi_direct",
            "semi_direct mode was not parsed");
    legacy["DIRECT_SLAM.mode"] = std::string("semidirect");
    require(DirectSlamParams::from_config(legacy).mode == DirectTrackingMode::SemiDirect,
            "semidirect alias was not parsed");
    legacy["DIRECT_SLAM.mode"] = std::string("direct");
    require(DirectSlamParams::from_config(legacy).mode == DirectTrackingMode::Hybrid,
            "legacy direct mode alias was rejected");
    require(DirectSlamParams::from_config(ConfigDict()).mode_name() == "klt", "default tracking mode is not klt");
    ConfigDict nested;
    ConfigDict block;
    block["mode"] = "hybrid";
    block["num_features"] = 123;
    nested["DIRECT_SLAM"] = Value(static_cast<Dict>(block));
    require(DirectSlamParams::from_config(nested).num_features == 123, "nested settings were not parsed");
    bool invalid = false;
    block["gftt_quality"] = 0.0;
    try { DirectSlamParams::from_config(block); }
    catch (const std::invalid_argument &) { invalid = true; }
    require(invalid, "invalid GFTT settings were accepted");
    auto camera = std::make_shared<PinholeCamera>(make_camera(160, 120, 100, 100, 80, 60));
    camera->sensor_type = SensorType::RGBD;
    DirectSlam slam(camera, ConfigDict());
    bool raised = false;
    try { slam.track(textured_image(160, 120, 0), cv::Mat(), cv::Mat(), 0, 0.0); }
    catch (const std::invalid_argument &) { raised = true; }
    require(raised, "RGBD accepted a missing depth image");
}

// Keep observations tied to ground truth while perturbing pose/landmark initial estimates.
DirectMap synthetic_ba_map(PinholeCamera &camera, bool rgbd, bool native) {
    camera.sensor_type = rgbd ? SensorType::RGBD : SensorType::STEREO;
    camera.bf = 40;
    if (native) {
        camera.right_camera = std::make_shared<PinholeCamera>(camera);
        camera.T_right_left.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(0.02, Vec3d::UnitZ()).toRotationMatrix();
        camera.T_right_left(0, 3) = -0.2;
    }
    DirectMap map;
    for (int j = 0; j < 60; ++j) {
        const Vec3d truth((j % 10 - 5) * 0.16, (j / 10 - 3) * 0.14, 3.0 + 0.07 * (j % 7));
        map.landmarks_mutable()[j] = {truth + Vec3d(0.025, -0.018, 0.08), false};
    }
    for (int i = 0; i < 4; ++i) {
        Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
        truth.translation().x() = -0.12 * i;
        Eigen::Isometry3d initial = truth;
        if (i > 0) {
            initial.translation() += Vec3d(0.025 * i, -0.01, 0.02);
            initial.linear() = Eigen::AngleAxisd(0.006, Vec3d::UnitY()).toRotationMatrix();
        }
        DirectKeyframe kf;
        kf.id = i;
        kf.img_id = i * 10;
        kf.pose = CameraPose(initial);
        for (int j = 0; j < 60; ++j) {
            const Vec3d Xw((j % 10 - 5) * 0.16, (j / 10 - 3) * 0.14, 3.0 + 0.07 * (j % 7));
            const Vec3d Xc = truth * Xw;
            DirectPoint obs;
            obs.id = j;
            obs.uv = camera.project_point(Xc).first;
            obs.position_c = initial * map.landmarks().at(j).position_w;
            obs.has_depth = true;
            obs.depth = Xc.z();
            if (!rgbd) {
                obs.has_right = true;
                if (native) {
                    const Vec3d Xr = camera.T_right_left.block<3, 3>(0, 0) * Xc +
                                     camera.T_right_left.block<3, 1>(0, 3);
                    obs.uv_right = camera.right_camera->project_point(Xr).first;
                } else obs.uv_right = obs.uv - Vec2d(camera.bf / Xc.z(), 0);
            }
            if (i == 2 && j == 0) obs.uv += Vec2d(60, -45);
            kf.points.push_back(obs);
        }
        map.add_keyframe(std::move(kf));
        map.add_pose(i * 10, i * 0.1, initial);
        Eigen::Isometry3d interframe = initial;
        interframe.translation().x() -= 0.03;
        map.add_pose(i * 10 + 1, i * 0.1 + 0.01, interframe);
    }
    return map;
}

// Check joint recovery, metric scale, channel outliers and corrected trajectory references.
void test_local_bundle_adjustment() {
    for (int mode = 0; mode < 3; ++mode) {
        auto camera = make_camera(320, 240, 200, 200, 160, 120);
        auto map = synthetic_ba_map(camera, mode == 1, mode == 2);
        const auto fixed = map.keyframes().front().pose.Tcw();
        const auto relative = map.trajectory().back().T_frame_reference;
        DirectSlamParams params;
        params.ba_iterations = 20;
        const auto result = optimize_direct_map(map, camera, params);
        require(result.applied, "local BA did not apply");
        require(result.final_cost < 0.25 * result.initial_cost, "local BA did not reduce error");
        require(result.keyframes == 4 && result.landmarks == 60, "local BA graph has wrong size");
        require((map.keyframes().front().pose.Tcw() - fixed).norm() < 1e-12, "BA changed gauge anchor");
        require(map.keyframes()[2].points[0].outlier, "BA did not reject the injected left outlier");
        require(result.outliers >= 1, "BA did not report rejected observations");
        require(std::abs(map.keyframes().back().pose.tcw().x() + 0.36) < 0.01,
                "BA did not recover the metric camera motion");
        const Vec3d true_point(-0.8, -0.42, 3.0);
        require((map.landmarks().at(0).position_w - true_point).norm() < 0.03,
                "BA did not refine the shared landmark");
        require((map.trajectory().back().Tcw - relative * map.keyframes().back().pose.Tcw()).norm() < 1e-12,
                "BA did not correct a non-keyframe trajectory pose");
        require((map.trajectory()[6].Tcw - map.keyframes().back().pose.Tcw()).norm() < 1e-12,
                "BA did not correct a keyframe trajectory pose");
    }
}

// Only free poses inside the bounded window may move; the oldest active pose is the anchor.
void test_ba_window_and_empty_graph() {
    auto camera = make_camera(320, 240, 200, 200, 160, 120);
    DirectMap empty;
    DirectSlamParams params;
    require(!optimize_direct_map(empty, camera, params).applied, "empty BA graph was accepted");
    auto map = synthetic_ba_map(camera, false, false);
    params.num_active_keyframes = 2;
    std::vector<Mat4d> before;
    for (const auto &kf : map.keyframes()) before.push_back(kf.pose.Tcw());
    const auto result = optimize_direct_map(map, camera, params);
    require(result.applied && result.keyframes == 2, "BA did not bound its active window");
    for (int i = 0; i < 3; ++i)
        require((map.keyframes()[i].pose.Tcw() - before[i]).norm() < 1e-12,
                "BA modified an inactive keyframe or the oldest active anchor");
}

// Landmarks remain shared with archived cameras after the active window slides.
// Recent biased measurements must not move them without those fixed constraints.
void test_ba_preserves_archived_observations() {
    auto camera = make_camera(320, 240, 200, 200, 160, 120);
    auto map = synthetic_ba_map(camera, true, false);
    for (int j = 0; j < 60; ++j)
        map.landmarks_mutable().at(j).position_w =
            Vec3d((j % 10 - 5) * 0.16, (j / 10 - 3) * 0.14, 3.0 + 0.07 * (j % 7));
    for (auto &kf : map.keyframes_mutable()) {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = -0.12 * kf.id;
        kf.pose = CameraPose(pose);
        for (auto &obs : kf.points) {
            const Vec3d Xc = pose * map.landmarks().at(obs.id).position_w;
            obs.uv = camera.project_point(Xc).first;
            if (kf.id >= 2) obs.uv.x() += 1.0;
            obs.depth = Xc.z();
        }
    }
    DirectSlamParams params;
    params.num_active_keyframes = 2;
    params.ba_iterations = 20;
    const auto result = optimize_direct_map(map, camera, params);
    require(result.applied, "boundary-observation BA did not apply");
    double archived_squared_error = 0.0;
    int count = 0;
    for (int i = 0; i < 2; ++i) {
        const auto &kf = map.keyframes()[i];
        for (const auto &obs : kf.points) {
            const Vec3d Xc = kf.pose.Tcw().block<3, 3>(0, 0) *
                                map.landmarks().at(obs.id).position_w + kf.pose.tcw();
            archived_squared_error += (camera.project_point(Xc).first - obs.uv).squaredNorm();
            ++count;
        }
    }
    require(std::sqrt(archived_squared_error / count) < 0.75,
            "sliding-window BA ignored archived observations of shared landmarks");
}

// Reject gauges that remain free despite one fixed pose; failed solves must be transactional.
void test_ba_rejects_unobservable_graphs() {
    auto camera = make_camera(320, 240, 200, 200, 160, 120);
    auto map = synthetic_ba_map(camera, false, false);
    for (auto &kf : map.keyframes_mutable())
        for (auto &obs : kf.points) obs.has_right = false;
    const auto before = map.keyframes().back().pose.Tcw();
    const auto point_before = map.landmarks().at(0).position_w;
    require(!optimize_direct_map(map, camera, DirectSlamParams()).applied,
            "BA accepted a graph with unconstrained scale");
    require((map.keyframes().back().pose.Tcw() - before).norm() == 0 &&
            (map.landmarks().at(0).position_w - point_before).norm() == 0,
            "failed BA changed the map");
    // A stereo point seen only once does not fix the monocular trajectory scale.
    auto single_metric = map.keyframes().back().points.back();
    single_metric.id = 500;
    single_metric.has_right = true;
    map.landmarks_mutable()[500] = map.landmarks().at(59);
    map.keyframes_mutable().back().points.push_back(single_metric);
    require(!optimize_direct_map(map, camera, DirectSlamParams()).applied,
            "a single-view metric landmark incorrectly fixed trajectory scale");
    map = synthetic_ba_map(camera, false, false);
    auto &last = map.keyframes_mutable().back();
    for (auto &obs : last.points) {
        const auto landmark = map.landmarks().at(obs.id);
        obs.id += 100;
        map.landmarks_mutable()[obs.id] = landmark;
    }
    require(!optimize_direct_map(map, camera, DirectSlamParams()).applied,
            "BA accepted a disconnected pose component");
}

// Detect stale frontend copies of positions or poses after automatic keyframe BA.
void test_frontend_applies_backend_updates() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(320, 240, 200, 200, 160, 120));
    camera->sensor_type = SensorType::STEREO;
    camera->bf = 40;
    ConfigDict config;
    config["max_keyframe_interval"] = 2;
    config["num_features_tracking_bad"] = 10;
    config["num_features_needed_for_keyframe"] = 180;
    config["num_active_keyframes"] = 3;
    config["lk_patch_size"] = 21;
    DirectSlam slam(camera, config);
    cv::Mat image(240, 320, CV_8U);
    cv::RNG rng(15);
    rng.fill(image, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.8);
    auto shifted = [&](int shift) {
        cv::Mat out;
        cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, shift, 0, 1, 0);
        cv::warpAffine(image, out, transform, image.size());
        return out;
    };
    bool applied = false;
    for (int i = 0; i < 9; ++i) {
        slam.track(shifted(i * 2), shifted(i * 2 - 10), cv::Mat(), i, 0.1 * i);
        require(slam.status() != DirectTrackingStatus::Lost, "frontend lost synthetic BA sequence");
        if (slam.last_ba().applied) {
            applied = true;
            require(slam.last_ba().keyframes <= 3, "automatic BA ignored its window bound");
        }
        require((slam.map().trajectory().back().Tcw - slam.pose().Tcw()).norm() < 1e-12,
                "frontend pose differs from corrected trajectory");
        for (const auto &point : slam.live_points())
            require((point.position_w - slam.map().landmarks().at(point.id).position_w).norm() < 1e-12,
                    "live tracking retained stale landmark coordinates after BA");
    }
    require(applied, "frontend never ran local BA at keyframe insertion");
    require(std::abs(slam.pose().tcw().x() - 0.32) < 0.025, "frontend BA broke metric scale");
    slam.reset();
    require(slam.map().landmarks().empty() && !slam.last_ba().applied, "reset retained backend state");
}

// A loop-closure pose update moves the referenced landmark and trajectory, and leaves other keyframes fixed.
// Monocular Sim(3) correction: with scale s on a keyframe, a landmark anchored there keeps its
// bearing and its depth shrinks by 1/s; frame offsets and inverse-depth priors follow.
void test_apply_keyframe_poses_with_scale() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(320, 240, 200, 200, 160, 120));
    camera->sensor_type = SensorType::RGBD;
    ConfigDict config;
    config["local_ba"] = false;
    config["max_keyframe_interval"] = 1;
    config["num_features_needed_for_keyframe"] = 100000;
    DirectSlam slam(camera, config);
    cv::Mat image = checkerboard(320, 240, 16, 0);
    cv::Mat depth(240, 320, CV_32F, cv::Scalar(4));
    slam.track(image, cv::Mat(), depth, 0, 0.0);
    cv::Mat shifted;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 6, 0, 1, 0);
    cv::warpAffine(image, shifted, transform, image.size());
    slam.track(shifted, cv::Mat(), depth, 1, 0.1);
    require(slam.map().keyframes().size() == 2, "scaled correction test did not create two keyframes");
    const Eigen::Matrix4d T0 = slam.map().keyframes()[0].pose.Tcw();
    const Eigen::Matrix4d T1 = slam.map().keyframes()[1].pose.Tcw();
    int64_t id = -1;
    for (const auto &observation : slam.map().keyframes()[0].points)
        if (!observation.outlier) { id = observation.id; break; }
    require(id >= 0, "the first keyframe has no inlier landmark");
    const Vec3d X_before = slam.map().landmarks().at(id).position_w;
    const Vec3d X_cam = T0.topLeftCorner<3, 3>() * X_before + T0.topRightCorner<3, 1>();

    bool rejected = false;
    try {
        slam.apply_keyframe_poses({T0, T1}, {1.0, -2.0});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "a negative keyframe scale was accepted");

    // Keyframe 1 moves to half its distance from keyframe 0, as a monocular correction with s = 2 would.
    Eigen::Matrix4d T1_new = T1;
    T1_new.topRightCorner<3, 1>() /= 2.0;
    slam.apply_keyframe_poses({T0, T1_new}, {2.0, 2.0});
    const Vec3d X_after = slam.map().landmarks().at(id).position_w;
    const Vec3d X_cam_after = T0.topLeftCorner<3, 3>() * X_after + T0.topRightCorner<3, 1>();
    require((X_cam_after - X_cam / 2.0).norm() < 1e-8, "a scaled correction did not shrink the anchored landmark");
    require((slam.map().trajectory()[1].Tcw - T1_new).norm() < 1e-8,
            "the keyframe trajectory entry did not follow the scaled correction");
}

void test_apply_keyframe_poses() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(320, 240, 200, 200, 160, 120));
    camera->sensor_type = SensorType::RGBD;
    ConfigDict config;
    config["local_ba"] = false;
    config["max_keyframe_interval"] = 1;
    config["num_features_needed_for_keyframe"] = 100000;
    DirectSlam slam(camera, config);
    cv::Mat image = checkerboard(320, 240, 16, 0);
    cv::Mat depth(240, 320, CV_32F, cv::Scalar(4));
    slam.track(image, cv::Mat(), depth, 0, 0.0);
    cv::Mat shifted;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, 6, 0, 1, 0);
    cv::warpAffine(image, shifted, transform, image.size());
    slam.track(shifted, cv::Mat(), depth, 1, 0.1);
    require(slam.map().keyframes().size() == 2, "pose correction test did not create two keyframes");
    require(!slam.map().landmarks().empty(), "pose correction test created no landmarks");

    const auto fixed_pose = slam.map().keyframes()[1].pose.Tcw();
    const auto fixed_trajectory = slam.map().trajectory()[1].Tcw;
    std::vector<Eigen::Matrix4d> invalid = {Eigen::Matrix4d::Identity(), Eigen::Matrix4d::Identity()};
    invalid[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try {
        slam.apply_keyframe_poses(invalid);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "a non-finite keyframe pose was accepted");
    rejected = false;
    try {
        slam.apply_keyframe_poses({Eigen::Matrix4d::Identity()});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "a short pose list was accepted");
    require((slam.map().keyframes()[1].pose.Tcw() - fixed_pose).norm() < 1e-12,
            "a rejected pose update changed a keyframe");

    int64_t moved_id = -1;
    Vec3d moved_before = Vec3d::Zero();
    for (const auto &observation : slam.map().keyframes()[0].points) {
        if (observation.outlier) continue;
        moved_id = observation.id;
        moved_before = slam.map().landmarks().at(observation.id).position_w;
        break;
    }
    require(moved_id >= 0, "the first keyframe has no inlier landmark to correct");
    int64_t fixed_id = -1;
    Vec3d fixed_before = Vec3d::Zero();
    for (const auto &observation : slam.map().keyframes()[1].points) {
        bool seen_earlier = false;
        for (const auto &earlier : slam.map().keyframes()[0].points)
            seen_earlier = seen_earlier || earlier.id == observation.id;
        if (seen_earlier || observation.outlier) continue;
        fixed_id = observation.id;
        fixed_before = slam.map().landmarks().at(observation.id).position_w;
        break;
    }

    const Eigen::Matrix4d old_pose = slam.map().keyframes()[0].pose.Tcw();
    Eigen::Matrix4d moved_pose = old_pose;
    moved_pose(0, 3) += 1.0;
    slam.apply_keyframe_poses({moved_pose, fixed_pose});
    Eigen::Vector4d homogeneous;
    homogeneous << moved_before, 1.0;
    const Vec3d expected = (moved_pose.inverse() * old_pose * homogeneous).head<3>();
    require((slam.map().landmarks().at(moved_id).position_w - expected).norm() < 1e-8,
            "the reference keyframe pose change did not move its landmark");
    if (fixed_id >= 0) {
        require((slam.map().landmarks().at(fixed_id).position_w - fixed_before).norm() < 1e-8,
                "a landmark anchored to the fixed keyframe moved");
    }
    require((slam.map().keyframes()[1].pose.Tcw() - fixed_pose).norm() < 1e-12,
            "pose correction changed the unrelated keyframe");
    require((slam.map().keyframes()[0].pose.Tcw() - moved_pose).norm() < 1e-12,
            "pose correction did not store the updated keyframe pose");
    require((slam.map().trajectory()[0].Tcw - moved_pose).norm() < 1e-8,
            "the trajectory entry referencing the moved keyframe was not corrected");
    require((slam.map().trajectory()[1].Tcw - fixed_trajectory).norm() < 1e-8,
            "the trajectory entry referencing the fixed keyframe moved");
    require((slam.pose().Tcw() - slam.map().trajectory().back().Tcw).norm() < 1e-12,
            "tracking pose was not refreshed from the corrected trajectory");
    for (const auto &point : slam.live_points()) {
        require((point.position_w - slam.map().landmarks().at(point.id).position_w).norm() < 1e-12,
                "live tracking retained a stale landmark after pose correction");
    }
}

KannalaBrandt4Camera make_fisheye(int width, int height, double fx, double fy, double cx, double cy) {
    return KannalaBrandt4Camera(width, height, fx, fy, cx, cy, 0.15, 0.02, -0.01, 0.005);
}

// Raw Kannala-Brandt images are valid in both modes. Depth lifting must reproject to the
// detected pixel, and off-axis points use ray length.
void test_kannala_brandt_klt() {
    auto camera = std::make_shared<KannalaBrandt4Camera>(make_fisheye(320, 240, 120, 120, 160, 120));
    camera->sensor_type = SensorType::RGBD;
    ConfigDict direct_config;
    direct_config["mode"] = "hybrid";
    DirectSlam direct_slam(camera, direct_config);
    require(direct_slam.params().mode == DirectTrackingMode::Hybrid,
            "hybrid mode rejected a Kannala-Brandt camera");

    const double z = 0.45;
    ConfigDict config;
    config["mode"] = "klt";
    config["local_ba"] = false;
    config["num_features"] = 200;
    config["num_features_tracking"] = 8;
    config["num_features_tracking_bad"] = 3;
    config["num_features_needed_for_keyframe"] = 10000;
    config["corner_detector"] = "gftt";
    config["gftt_quality"] = 0.001;
    config["gftt_min_distance"] = 10.0;
    config["border"] = 12;
    DirectSlam slam(camera, config);
    slam.track(checkerboard(320, 240, 16, 0), cv::Mat(), cv::Mat(240, 320, CV_32F, cv::Scalar(static_cast<float>(z))),
               0, 0.0);
    require(!slam.live_points().empty(), "Kannala-Brandt RGBD initialization found no points");
    int wider_than_pinhole = 0;
    for (const auto &point : slam.live_points()) {
        const auto projected = camera->project_point(point.position_w);
        require(std::abs(projected.second - z) < 1e-6, "lifted optical-axis depth is not the sensor z");
        require((projected.first - point.uv).norm() < 1e-4, "lifted point does not reproject to its raw pixel");
        require(point.position_w.norm() > 0.5, "ray-length gate kept a point closer than 0.5 m");
        const Vec3d pinhole((point.uv.x() - camera->cx) / camera->fx * z,
                            (point.uv.y() - camera->cy) / camera->fy * z, z);
        if ((point.position_w - pinhole).norm() > 1e-2) {
            ++wider_than_pinhole;
        }
    }
    require(wider_than_pinhole > 0, "fisheye lift matched the pinhole back-projection");

    KannalaBrandt4Camera pose_camera = make_fisheye(640, 480, 250, 250, 320, 240);
    Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
    expected.linear() = Eigen::AngleAxisd(0.04, Vec3d::UnitY()).toRotationMatrix();
    expected.translation() = Vec3d(0.12, -0.02, 0.08);
    std::vector<PoseObservation> observations;
    int count = 0;
    int outliers = 0;
    for (int y = 80; y < 400; y += 28) {
        for (int x = 80; x < 560; x += 28) {
            const double depth = 3.0 + (count % 5);
            Vec3d bearing;
            const Vec2d pixel(x, y);
            if (!pose_camera.unproject_bearing(pixel, bearing, nullptr) || !(bearing.z() > 0.2)) {
                continue;
            }
            const Vec3d X = bearing * (depth / bearing.z());
            const auto projected = pose_camera.project_point(expected * X);
            if (!std::isfinite(projected.first.x()) || projected.first.x() < 20 || projected.first.y() < 20 ||
                projected.first.x() >= 620 || projected.first.y() >= 460) {
                continue;
            }
            Vec2d uv = projected.first;
            if (count % 4 == 0) {
                uv += Vec2d(40, -30);
                ++outliers;
            }
            observations.push_back({X, uv});
            ++count;
        }
    }
    require(observations.size() >= 20, "fisheye pose test did not produce enough observations");
    Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
    prediction.translation() = Vec3d(-0.4, 0.2, 0.3);
    const auto result = estimate_pose_reprojection(pose_camera, observations, prediction, DirectSlamParams());
    require((result.T_cur_ref.matrix() - expected.matrix()).norm() < 1e-2,
            "fisheye pose did not recover from outliers and a bad prediction");
    require(result.inliers == static_cast<int>(observations.size()) - outliers,
            "fisheye inlier mask dropped a valid raw-pixel observation");
}

// Compare the direct overlay with the same drawing primitive used by sparse-SLAM Frame.
void test_shared_feature_trail_style() {
    const auto camera = std::make_shared<PinholeCamera>(make_camera(320, 240, 200, 200, 160, 120));
    camera->sensor_type = SensorType::RGBD;
    DirectSlam slam(camera, ConfigDict());
    cv::Mat image = checkerboard(320, 240, 16, 0);
    cv::Mat depth(240, 320, CV_32F, cv::Scalar(4));
    slam.track(image, cv::Mat(), depth, 0, 0.0);
    slam.track(image, cv::Mat(), depth, 1, 0.1);
    cv::Mat color;
    cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
    for (bool radius : {false, true}) {
        cv::Mat expected = color.clone();
        for (const auto &point : slam.live_points()) {
            const cv::Point uv(std::floor(point.uv.x()), std::floor(point.uv.y()));
            if (radius) draw_feature_trail<true>(expected, uv, point.id, point.trail, point.observations);
            else draw_feature_trail<false>(expected, uv, point.id, point.trail, point.observations);
        }
        const auto rendered = slam.draw_feature_trails(color, radius);
        require(cv::norm(rendered, expected, cv::NORM_INF) == 0,
                "direct features differ from the shared Frame drawing style");
        require(cv::norm(color, rendered, cv::NORM_INF) > 0, "feature drawing produced no overlay");
    }
}

float sample_test_image(const cv::Mat &img, float x, float y) {
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
    return (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 + fx * fy * i11;
}

// Render the current view of a fronto-parallel plane at `depth` in the reference camera.
cv::Mat render_plane(const Camera &camera, const cv::Mat &reference, double depth,
                     const Eigen::Isometry3d &T_cur_ref) {
    cv::Mat image(reference.rows, reference.cols, CV_8U, cv::Scalar(0));
    const Eigen::Isometry3d T_ref_cur = T_cur_ref.inverse();
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const Vec2d pixel(static_cast<double>(x), static_cast<double>(y));
            Vec3d bearing = Vec3d::Zero();
            if (!camera.unproject_bearing(pixel, bearing, nullptr) || !(bearing.z() > 1e-8)) {
                continue;
            }
            const Vec3d ray = bearing / bearing.z();
            const double denominator = (T_ref_cur.linear().row(2) * ray)(0);
            if (std::abs(denominator) < 1e-8) {
                continue;
            }
            const double z_cur = (depth - T_ref_cur.translation().z()) / denominator;
            if (!(z_cur > 0.1)) {
                continue;
            }
            const Vec2d uv = camera.project_point(T_ref_cur * (ray * z_cur)).first;
            if (!uv.allFinite() || uv.x() < 1.0 || uv.y() < 1.0 || uv.x() >= reference.cols - 2 ||
                uv.y() >= reference.rows - 2) {
                continue;
            }
            image.at<uchar>(y, x) = static_cast<uchar>(std::lround(sample_test_image(reference, static_cast<float>(uv.x()),
                                                                                     static_cast<float>(uv.y()))));
        }
    }
    return image;
}

void test_planar_patch_pose_and_feature_shift() {
    const int width = 160;
    const int height = 120;
    const double depth = 4.0;
    PinholeCamera camera = make_camera(width, height, 120.0, 120.0, 80.0, 60.0);
    cv::Mat reference(height, width, CV_8U);
    cv::RNG rng(7);
    rng.fill(reference, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(reference, reference, cv::Size(5, 5), 1.2);
    Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
    expected.linear() = Eigen::AngleAxisd(0.08, Vec3d::UnitY()).toRotationMatrix();
    expected.translation() = Vec3d(0.05, -0.02, 0.0);
    const cv::Mat current = render_plane(camera, reference, depth, expected);
    std::vector<Vec2d> pixels;
    std::vector<double> depths;
    for (int y = 24; y < height - 24; y += 10) {
        for (int x = 24; x < width - 24; x += 10) {
            pixels.emplace_back(x, y);
            depths.push_back(depth);
        }
    }
    DirectSlamParams params;
    params.feature_half_patch = 2;
    params.num_iterations = 15;
    params.num_pyramid_levels = 4;
    params.half_patch_size = 2;
    const PlanarPoseResult planar =
        align_planar_patch_pose(camera, params, reference, current, pixels, depths, Eigen::Isometry3d::Identity());
    const DirectAlignmentResult windowed =
        align_direct_multilayer(camera, params, reference, current, pixels, depths, Eigen::Isometry3d::Identity());
    const double planar_error = (planar.T_cur_ref.matrix() - expected.matrix()).norm();
    const double window_error = (windowed.T_cur_ref.matrix() - expected.matrix()).norm();
    std::cout << "planar pose error " << planar_error << " window error " << window_error << " inliers "
              << planar.inliers << std::endl;
    require(planar_error < 0.05, "planar patch did not recover the rotated pose");
    require(window_error > planar_error * 2.0, "flat 2D window matched the planar patch on a rotation");

    cv::Mat shifted;
    cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, 1.5, 0, 1, -0.75);
    cv::warpAffine(reference, shifted, affine, reference.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    const PatchOffsetResult offsets = align_planar_patch_offsets(
        camera, params, reference, shifted, pixels, depths, Eigen::Isometry3d::Identity());
    std::vector<double> du;
    std::vector<double> dv;
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (!offsets.status[i]) {
            continue;
        }
        du.push_back(offsets.pixels_cur[i].x() - pixels[i].x());
        dv.push_back(offsets.pixels_cur[i].y() - pixels[i].y());
    }
    require(du.size() > pixels.size() / 2, "feature alignment kept too few patches");
    std::sort(du.begin(), du.end());
    std::sort(dv.begin(), dv.end());
    require(std::abs(du[du.size() / 2] - 1.5) < 0.35, "feature alignment missed the horizontal shift");
    require(std::abs(dv[dv.size() / 2] + 0.75) < 0.35, "feature alignment missed the vertical shift");
}

void test_inverse_depth_filter() {
    InverseDepthBelief near_measurement;
    near_measurement.mu = 0.25;
    near_measurement.sigma2 = 0.01;
    near_measurement.a = 10.0;
    near_measurement.b = 10.0;
    require(fuse_inverse_depth(near_measurement, 0.26, 1e-4, 0.5, 80.0), "near inverse-depth update failed");
    require(near_measurement.sigma2 < 0.003, "a nearby inverse-depth measurement did not shrink the variance");

    InverseDepthBelief far_measurement;
    far_measurement.mu = 0.25;
    far_measurement.sigma2 = 0.01;
    far_measurement.a = 10.0;
    far_measurement.b = 10.0;
    require(fuse_inverse_depth(far_measurement, 1.8, 0.01, 0.5, 80.0), "far inverse-depth update failed");
    require(far_measurement.sigma2 > 0.008, "an outlier inverse-depth measurement collapsed the variance");

    const int width = 160;
    const int height = 120;
    const double truth = 4.0;
    PinholeCamera camera = make_camera(width, height, 140.0, 140.0, 80.0, 60.0);
    cv::Mat reference(height, width, CV_8U);
    cv::RNG rng(11);
    rng.fill(reference, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(reference, reference, cv::Size(5, 5), 1.0);
    Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
    motion.translation() = Vec3d(0.25, 0.0, 0.0);
    const cv::Mat current = render_plane(camera, reference, truth, motion);
    DirectSlamParams params;
    params.feature_half_patch = 2;
    params.seed_min_depth = 1.0;
    params.seed_max_depth = 20.0;
    params.intensity_inlier_threshold = 40.0;
    DepthSeed seed;
    seed.pixel = Vec2d(80.0, 60.0);
    seed.belief = make_inverse_depth_prior(params.seed_min_depth, params.seed_max_depth);
    const InverseDepthBelief before = seed.belief;
    EpipolarMatch missed = match_seed_on_epipolar_line(camera, params, reference, cv::Mat(height, width, CV_8U, cv::Scalar(0)),
                                                       seed.pixel, seed.belief.mu, seed.belief.sigma2, motion);
    missed.searched = true;
    missed.matched = false;
    require(observe_seed(seed, missed, params.seed_min_depth, params.seed_max_depth) == SeedObservation::Failed,
            "a blank frame was treated as a depth match");
    // SVO: a searched segment without a match is one outlier vote (b += 1). The depth
    // estimate itself is untouched.
    require(seed.failures == 1 && seed.belief.mu == before.mu && seed.belief.sigma2 == before.sigma2 &&
                seed.belief.a == before.a && seed.belief.b == before.b + 1.0,
            "a failed epipolar match must only add one outlier vote to the belief");
    EpipolarMatch skipped;
    require(observe_seed(seed, skipped, params.seed_min_depth, params.seed_max_depth) == SeedObservation::Skipped &&
                seed.belief.b == before.b + 1.0 && seed.failures == 1,
            "an unsearched seed was modified");

    seed.failures = 0;
    seed.belief = before;
    const EpipolarMatch found = match_seed_on_epipolar_line(camera, params, reference, current, seed.pixel,
                                                            seed.belief.mu, seed.belief.sigma2, motion);
    require(found.matched, "epipolar search missed the translated plane");
    std::cout << "epipolar depth " << found.depth << " truth " << truth << " sigma_inv "
              << std::sqrt(found.inverse_variance) << std::endl;
    // Subpixel refinement along the segment: well under the one-sample spacing.
    require(std::abs(found.depth - truth) < 0.15, "epipolar search depth is far from the plane");
    // The measurement variance is SVO's one-pixel tau, not the sample spacing.
    const double expected_tau = depth_uncertainty_one_pixel(camera, motion, seed.pixel, found.depth,
                                                            2.0 * std::atan(1.0 / (2.0 * camera.fx)));
    const double expected_sigma =
        0.5 * (1.0 / (found.depth - expected_tau) - 1.0 / (found.depth + expected_tau));
    require(std::abs(std::sqrt(found.inverse_variance) - expected_sigma) < 1e-9 * (1.0 + expected_sigma),
            "epipolar measurement variance is not the SVO one-pixel tau");
    require(observe_seed(seed, found, params.seed_min_depth, params.seed_max_depth) == SeedObservation::Updated,
            "a matched seed was not updated");
    require(std::abs(seed.belief.mu - 1.0 / truth) < std::abs(before.mu - 1.0 / truth),
            "the seed did not move toward the true inverse depth");
}

void test_semi_direct_stereo_and_monocular() {
    const int width = 160;
    const int height = 120;
    const double fx = 100.0;
    const double depth = 2.5;
    const double shift = 6.0;
    auto camera = std::make_shared<PinholeCamera>(make_camera(width, height, fx, fx, 80.0, 60.0));
    camera->sensor_type = SensorType::RGBD;
    ConfigDict config;
    config["mode"] = "semi_direct";
    config["num_features"] = 80;
    config["num_features_tracking"] = 8;
    config["num_features_tracking_bad"] = 3;
    config["num_features_needed_for_keyframe"] = 10000;
    config["local_ba"] = false;
    config["feature_half_patch"] = 2;
    config["num_iterations"] = 12;
    DirectSlam slam(camera, config);
    const cv::Mat reference = checkerboard(width, height, 12, 0);
    const cv::Mat current = checkerboard(width, height, 12, static_cast<int>(shift));
    cv::Mat depth_image(height, width, CV_32F, cv::Scalar(static_cast<float>(depth)));
    slam.track(reference, cv::Mat(), depth_image, 0, 0.0);
    slam.track(current, cv::Mat(), depth_image, 1, 1.0);
    const double expected_tx = shift * depth / fx;
    std::cout << "semi-direct tx " << slam.pose().tcw().x() << " expected " << expected_tx << " status "
              << slam.status_name() << " inliers " << slam.last_inliers() << std::endl;
    require(slam.status() == DirectTrackingStatus::Good || slam.status() == DirectTrackingStatus::Bad,
            "semi-direct stereo tracking lost the second frame");
    require(std::abs(slam.pose().tcw().x() - expected_tx) < 0.08, "semi-direct stereo translation is wrong");

    auto mono = std::make_shared<PinholeCamera>(make_camera(width, height, 140.0, 140.0, 80.0, 60.0));
    mono->sensor_type = SensorType::MONOCULAR;
    bool rejected = false;
    try {
        ConfigDict klt;
        klt["mode"] = "klt";
        DirectSlam rejected_slam(mono, klt);
        (void)rejected_slam;
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "klt accepted a monocular camera");
    rejected = false;
    try {
        ConfigDict hybrid;
        hybrid["mode"] = "hybrid";
        DirectSlam rejected_slam(mono, hybrid);
        (void)rejected_slam;
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "hybrid accepted a monocular camera");

    ConfigDict mono_config;
    mono_config["mode"] = "semi_direct";
    mono_config["corner_detector"] = "gftt";
    mono_config["num_features"] = 60;
    mono_config["gftt_quality"] = 0.001;
    mono_config["gftt_min_distance"] = 8.0;
    mono_config["num_features_tracking"] = 6;
    mono_config["num_features_tracking_bad"] = 5;
    mono_config["num_features_needed_for_keyframe"] = 10000;
    mono_config["max_keyframe_interval"] = 1000;
    mono_config["seed_min_depth"] = 2.0;
    mono_config["seed_max_depth"] = 8.0;
    mono_config["seed_convergence_sigma2"] = 1e-3;
    mono_config["feature_half_patch"] = 2;
    mono_config["num_iterations"] = 12;
    mono_config["local_ba"] = true;
    DirectSlam mono_slam(mono, mono_config);
    cv::Mat mono_ref(height, width, CV_8U);
    cv::RNG mono_rng(19);
    mono_rng.fill(mono_ref, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(mono_ref, mono_ref, cv::Size(5, 5), 1.2);
    mono_slam.track(mono_ref, cv::Mat(), cv::Mat(), 0, 0.0);
    require(mono_slam.status() == DirectTrackingStatus::Init, "monocular semi-direct left INIT on the first frame");
    // Keyframes are created only once the two-view initializer accepts a geometry.
    require(mono_slam.map().keyframes().empty(), "monocular initialization stored a keyframe without parallax");
    for (int step = 1; step <= 6 && mono_slam.status() == DirectTrackingStatus::Init; ++step) {
        Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
        motion.translation() = Vec3d(0.12 * step, 0.0, 0.0);
        const cv::Mat mono_cur = render_plane(*mono, mono_ref, 4.0, motion);
        mono_slam.track(mono_cur, cv::Mat(), cv::Mat(), step, static_cast<double>(step));
    }
    std::cout << "mono status " << mono_slam.status_name() << " inliers " << mono_slam.last_inliers()
              << " ba " << mono_slam.last_ba().applied << std::endl;
    require(mono_slam.status() == DirectTrackingStatus::Good || mono_slam.status() == DirectTrackingStatus::Bad,
            "monocular semi-direct did not leave INIT after two-view initialization");
    require(mono_slam.map().keyframes().size() == 2, "monocular initialization did not store two keyframes");
    require(mono_slam.last_ba().applied, "monocular local BA did not apply the inverse-depth prior");
}

// Verbatim transcription of rpg_svo DepthFilter::updateSeed (Forster et al. 2014),
// with z_range the width of the uniform outlier density.
void svo_reference_update(double x, double tau2, double z_range, InverseDepthBelief &seed) {
    const double norm_scale = std::sqrt(seed.sigma2 + tau2);
    const double pdf = std::exp(-0.5 * (x - seed.mu) * (x - seed.mu) / (norm_scale * norm_scale)) /
                       (norm_scale * std::sqrt(2.0 * M_PI));
    const double s2 = 1.0 / (1.0 / seed.sigma2 + 1.0 / tau2);
    const double m = s2 * (seed.mu / seed.sigma2 + x / tau2);
    double C1 = seed.a / (seed.a + seed.b) * pdf;
    double C2 = seed.b / (seed.a + seed.b) * 1.0 / z_range;
    const double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    const double f = C1 * (seed.a + 1.0) / (seed.a + seed.b + 1.0) + C2 * seed.a / (seed.a + seed.b + 1.0);
    const double e = C1 * (seed.a + 1.0) * (seed.a + 2.0) / ((seed.a + seed.b + 1.0) * (seed.a + seed.b + 2.0)) +
                     C2 * seed.a * (seed.a + 1.0) / ((seed.a + seed.b + 1.0) * (seed.a + seed.b + 2.0));
    const double mu_new = C1 * m + C2 * seed.mu;
    seed.sigma2 = C1 * (s2 + m * m) + C2 * (seed.sigma2 + seed.mu * seed.mu) - mu_new * mu_new;
    seed.mu = mu_new;
    seed.a = (e - f) / (f - e / f);
    seed.b = seed.a * (1.0 - f) / f;
}

// The fused Gaussian x Beta posterior must equal SVO's update, measurement by measurement.
void test_depth_filter_matches_svo_update() {
    const double near_m = 0.5;
    const double far_m = 80.0;
    const double z_range = 1.0 / near_m - 1.0 / far_m;
    InverseDepthBelief ours = make_inverse_depth_prior(near_m, far_m);
    InverseDepthBelief svo = ours;
    require(std::abs(ours.sigma2 - z_range * z_range / 36.0) < 1e-12, "prior variance is not z_range^2 / 36");
    // Inliers around 1/3 m^-1 with one gross outlier in the middle.
    const std::vector<std::pair<double, double>> measurements = {
        {0.34, 1e-3}, {0.33, 1e-3}, {1.70, 1e-3}, {0.335, 5e-4}, {0.332, 2e-4},
        {0.334, 2e-4}, {0.333, 2e-4}, {0.331, 2e-4}, {0.335, 2e-4}, {0.333, 2e-4}};
    for (const auto &[x, tau2] : measurements) {
        require(fuse_inverse_depth(ours, x, tau2, near_m, far_m), "inverse-depth fusion failed");
        svo_reference_update(x, tau2, z_range, svo);
        require(std::abs(ours.mu - svo.mu) < 1e-9 && std::abs(ours.sigma2 - svo.sigma2) < 1e-12 &&
                    std::abs(ours.a - svo.a) < 1e-7 && std::abs(ours.b - svo.b) < 1e-7,
                "inverse-depth fusion diverged from the SVO update");
    }
    require(std::abs(ours.mu - 1.0 / 3.0) < 0.01, "fused seed did not converge to the inlier depth");
    require(ours.sigma2 < 1e-4, "fused seed did not converge");

    // SVO starts a seed at the scene depth; out-of-range scene depths are clamped.
    require(std::abs(make_inverse_depth_prior(near_m, far_m, 4.0).mu - 0.25) < 1e-12,
            "prior mean ignores the scene depth");
    require(std::abs(make_inverse_depth_prior(near_m, far_m, 0.1).mu - 1.0 / near_m) < 1e-12,
            "prior mean was not clamped to the near limit");
    require(make_inverse_depth_prior(near_m, far_m, 4.0).sigma2 == make_inverse_depth_prior(near_m, far_m).sigma2,
            "scene depth changed the prior variance");

    // Repeated misses drive the outlier ratio up until the seed expires.
    DepthSeed seed;
    seed.belief = make_inverse_depth_prior(near_m, far_m);
    EpipolarMatch miss;
    miss.searched = true;
    int votes = 0;
    while (!seed_expired(seed, make_inverse_depth_prior(near_m, far_m).sigma2) && votes < 100) {
        observe_seed(seed, miss, near_m, far_m);
        ++votes;
    }
    require(votes > 0 && votes <= 20, "a seed without matches did not expire");
}

// computeTau: shifting the depth by tau moves the current projection by about one pixel.
void test_svo_one_pixel_tau() {
    PinholeCamera camera = make_camera(640, 480, 500.0, 500.0, 320.0, 240.0);
    const double angle = 2.0 * std::atan(1.0 / (2.0 * camera.fx));
    for (const Vec3d &baseline : {Vec3d(0.2, 0.0, 0.0), Vec3d(0.1, 0.05, 0.1)}) {
        Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
        T_cur_ref.translation() = -baseline; // current camera center at +baseline in the reference frame
        for (const Vec2d &pixel : {Vec2d(320.0, 240.0), Vec2d(150.0, 380.0)}) {
            for (double depth : {1.0, 4.0, 12.0}) {
                const double tau = depth_uncertainty_one_pixel(camera, T_cur_ref, pixel, depth, angle);
                require(tau > 0.0, "one-pixel tau was not positive");
                Vec3d X = Vec3d::Zero();
                Vec3d X_plus = Vec3d::Zero();
                require(lift_reference_point(camera, pixel, depth, X) &&
                            lift_reference_point(camera, pixel, depth + tau, X_plus),
                        "tau test lift failed");
                // SVO's tau is exact in angle: the current-frame rays to z and z + tau differ
                // by px_error_angle. In pixels this is one pixel at the principal point.
                const Vec3d ray = (T_cur_ref * X).normalized();
                const Vec3d ray_plus = (T_cur_ref * X_plus).normalized();
                const double seen = std::acos(std::clamp(ray.dot(ray_plus), -1.0, 1.0));
                require(std::abs(seen - angle) < 1e-3 * angle, "depth + tau does not subtend the pixel angle");
                if (pixel == Vec2d(320.0, 240.0)) {
                    const double shift = (camera.project_point(T_cur_ref * X_plus).first -
                                          camera.project_point(T_cur_ref * X).first).norm();
                    require(shift > 0.9 && shift < 1.1, "depth + tau did not move a central projection by one pixel");
                }
            }
        }
    }
    Eigen::Isometry3d no_baseline = Eigen::Isometry3d::Identity();
    require(depth_uncertainty_one_pixel(camera, no_baseline, Vec2d(320.0, 240.0), 3.0, angle) < 0.0,
            "tau without baseline must be reported as degenerate");
}

// A seed promoted after a newer keyframe keeps uv_ref on its own (older) keyframe.
// Tracking must sample the latest keyframe image at the landmark's projection, not at uv_ref.
void test_semi_direct_tracks_points_from_older_keyframes() {
    const int width = 160;
    const int height = 120;
    const double depth = 3.0;
    auto camera = std::make_shared<PinholeCamera>(make_camera(width, height, 120.0, 120.0, 80.0, 60.0));
    camera->sensor_type = SensorType::RGBD;
    DirectSlamParams params;
    params.mode = DirectTrackingMode::SemiDirect;
    params.feature_half_patch = 2;
    params.num_iterations = 12;
    params.num_features_tracking_bad = 5;
    cv::Mat kf0_image(height, width, CV_8U);
    cv::RNG rng(23);
    rng.fill(kf0_image, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(kf0_image, kf0_image, cv::Size(5, 5), 1.2);
    // Keyframe 1 sits 0.15 m to the right (about 6 px of image motion), the current frame
    // another 0.05 m.
    Eigen::Isometry3d T_kf1_w = Eigen::Isometry3d::Identity();
    T_kf1_w.translation() = Vec3d(-0.15, 0.0, 0.0);
    Eigen::Isometry3d T_cur_w = Eigen::Isometry3d::Identity();
    T_cur_w.translation() = Vec3d(-0.20, 0.0, 0.0);
    const cv::Mat kf1_image = render_plane(*camera, kf0_image, depth, T_kf1_w);
    const cv::Mat cur_image = render_plane(*camera, kf0_image, depth, T_cur_w);

    DirectMap map;
    DirectKeyframe kf0;
    kf0.id = 0;
    kf0.img_id = 0;
    kf0.pose = CameraPose(Eigen::Isometry3d::Identity());
    kf0.gray = kf0_image.clone();
    map.add_keyframe(kf0);
    DirectKeyframe kf1;
    kf1.id = 1;
    kf1.img_id = 5;
    kf1.pose = CameraPose(T_kf1_w);
    kf1.gray = kf1_image.clone();
    map.add_keyframe(kf1);

    std::vector<LivePoint> points;
    for (int y = 24; y < height - 24; y += 12) {
        for (int x = 30; x < width - 30; x += 12) {
            LivePoint point;
            point.id = static_cast<int64_t>(points.size());
            point.uv_ref = Vec2d(x, y); // measured on keyframe 0
            point.reference_keyframe = 0;
            Vec3d X = Vec3d::Zero();
            require(lift_reference_point(*camera, point.uv_ref, depth, X), "lift failed");
            point.position_w = X;
            point.uv = camera->project_point(T_kf1_w * X).first;
            points.push_back(point);
        }
    }
    SemiDirectPoseTracker tracker(camera, params);
    const cv::Mat empty;
    const TrackingFrame frame{cur_image, empty, empty, 6};
    const TrackingInput input{frame, kf1_image, points, map, T_kf1_w, T_kf1_w};
    const TrackingResult result = tracker.track(input);
    require(result.accepted, "semi-direct rejected landmarks anchored on an older keyframe");
    int accurate = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (!result.tracked_ok[i]) continue;
        const Vec2d truth = camera->project_point(T_cur_w * points[i].position_w).first;
        if ((result.tracked_uv[i] - truth).norm() < 0.5) ++accurate;
    }
    std::cout << "older-keyframe points tracked accurately " << accurate << "/" << points.size()
              << " tx " << result.T_cur_w.translation().x() << std::endl;
    require(accurate > static_cast<int>(points.size()) * 3 / 5, "older-keyframe points were tracked off target");
    require((result.T_cur_w.translation() - T_cur_w.translation()).norm() < 0.01,
            "pose from older-keyframe points is wrong");
}

// Monocular two-view initialization: LK tracks from a reference frame, essential matrix,
// triangulation. Scale puts the median depth at sqrt(seed_min_depth * seed_max_depth).
// A tracked correspondence is triangulated against the seed's keyframe: correct depth on the
// epipolar line, a failed match off it, and no measurement without parallax.
// DIRECT_SLAM.monocular.<key> overrides apply only when from_config is told the input is monocular.
void test_monocular_param_overrides() {
    ConfigDict flat;
    flat["DIRECT_SLAM.num_features"] = 200;
    flat["DIRECT_SLAM.max_seeds"] = 400;
    flat["DIRECT_SLAM.monocular.num_features"] = 400;
    flat["DIRECT_SLAM.monocular.max_seeds"] = 800;
    const DirectSlamParams stereo = DirectSlamParams::from_config(flat);
    const DirectSlamParams mono = DirectSlamParams::from_config(flat, true);
    require(stereo.num_features == 200 && stereo.max_seeds == 400, "monocular overrides leaked into stereo");
    require(mono.num_features == 400 && mono.max_seeds == 800, "flat monocular overrides were ignored");
    ConfigDict block;
    block["num_features"] = 200;
    ConfigDict overrides;
    overrides["num_features"] = 300;
    block["monocular"] = static_cast<Dict>(overrides);
    ConfigDict nested;
    nested["DIRECT_SLAM"] = static_cast<Dict>(block);
    require(DirectSlamParams::from_config(nested, true).num_features == 300, "nested monocular overrides were ignored");
    require(DirectSlamParams::from_config(nested).num_features == 200, "nested monocular overrides leaked");
}

void test_seed_measurement_from_track() {
    auto camera = std::make_shared<PinholeCamera>(make_camera(160, 120, 140.0, 140.0, 80.0, 60.0));
    DirectSlamParams params;
    params.seed_min_depth = 1.0;
    params.seed_max_depth = 20.0;
    const Vec2d pixel_ref(50.0, 70.0);
    const double depth = 4.0;
    Vec3d X_ref = Vec3d::Zero();
    require(lift_reference_point(*camera, pixel_ref, depth, X_ref), "lift failed");
    Eigen::Isometry3d T_cur_ref = Eigen::Isometry3d::Identity();
    T_cur_ref.translation() = Vec3d(-0.3, 0.05, -0.1);
    T_cur_ref.linear() = Eigen::AngleAxisd(0.02, Vec3d::UnitY()).toRotationMatrix();
    const Vec2d pixel_cur = camera->project_point(T_cur_ref * X_ref).first;

    const EpipolarMatch match = measure_seed_from_track(*camera, params, pixel_ref, pixel_cur, T_cur_ref);
    require(match.searched && match.matched, "tracked seed was not measured");
    require(std::abs(match.depth - depth) < 1e-6, "tracked seed depth is wrong");
    require(match.inverse_variance > 0.0 && match.parallax_px > 2.0, "tracked seed has no variance or parallax");

    // Five pixels off the epipolar line: inconsistent with the pose, counted as a failed match.
    const Vec2d off = pixel_cur + Vec2d(0.0, 5.0);
    const EpipolarMatch wrong = measure_seed_from_track(*camera, params, pixel_ref, off, T_cur_ref);
    require(wrong.searched && !wrong.matched, "off-epipolar track was accepted");

    // A tiny baseline carries no depth information: skipped, not an outlier.
    Eigen::Isometry3d tiny = Eigen::Isometry3d::Identity();
    tiny.translation() = Vec3d(-1e-4, 0.0, 0.0);
    const Vec2d still = camera->project_point(tiny * X_ref).first;
    require(!measure_seed_from_track(*camera, params, pixel_ref, still, tiny).searched,
            "a track without parallax was searched");
}

void test_monocular_two_view_initializer() {
    const int width = 160;
    const int height = 120;
    auto camera = std::make_shared<PinholeCamera>(make_camera(width, height, 140.0, 140.0, 80.0, 60.0));
    camera->sensor_type = SensorType::MONOCULAR;
    DirectSlamParams params;
    params.mode = DirectTrackingMode::SemiDirect;
    params.corner_detector = CornerDetector::Gftt;
    params.num_features = 60;
    params.gftt_quality = 0.001;
    params.gftt_min_distance = 8.0;
    params.border = 8;
    params.num_features_tracking = 10;
    params.num_features_tracking_bad = 5;
    params.seed_min_depth = 2.0;
    params.seed_max_depth = 8.0; // Median depth target 4 m, the true plane depth.
    params.init_min_disparity = 15.0;
    cv::Mat reference(height, width, CV_8U);
    cv::RNG rng(19);
    rng.fill(reference, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(reference, reference, cv::Size(5, 5), 1.2);
    const double plane_depth = 4.0;

    MonocularInitializer init(camera, params);
    require(!init.add_frame(reference, 0, 0.0), "initializer accepted a single frame");
    require(init.num_tracks() > params.num_features_tracking, "initializer found too few corners");
    // Small motion: below init_min_disparity, so no attempt yet.
    Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
    motion.translation() = Vec3d(-0.1, 0.0, 0.0); // Tcw: the camera moves +x.
    require(!init.add_frame(render_plane(*camera, reference, plane_depth, motion), 1, 1.0),
            "initializer accepted a geometry below the disparity threshold");
    bool accepted = false;
    Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
    for (int step = 2; step <= 8 && !accepted; ++step) {
        truth.translation() = Vec3d(-0.1 * step, 0.02 * step, 0.0);
        accepted = init.add_frame(render_plane(*camera, reference, plane_depth, truth), step, step);
    }
    require(accepted, "two-view initialization did not succeed");
    const MonocularInitResult &result = init.result();
    const Vec3d t = result.T_cur_ref.translation();
    std::cout << "mono init points " << result.X_ref.size() << " t " << t.transpose() << " truth "
              << truth.translation().transpose() << std::endl;
    require(static_cast<int>(result.X_ref.size()) > params.num_features_tracking, "too few initial points");
    require(Eigen::AngleAxisd(result.T_cur_ref.linear()).angle() < 0.01, "initial rotation is wrong");
    require((t - truth.translation()).norm() < 0.1 * truth.translation().norm(), "initial translation is wrong");
    for (size_t i = 0; i < result.X_ref.size(); ++i) {
        require(std::abs(result.X_ref[i].z() - plane_depth) < 0.3, "initial point is off the plane");
        require(result.inverse_depth_sigma2[i] > 0.0, "initial point has no inverse-depth prior");
    }

    // Unrelated content drops every track: the reference restarts on the new frame.
    MonocularInitializer restart(camera, params);
    restart.add_frame(reference, 0, 0.0);
    cv::Mat other(height, width, CV_8U);
    cv::RNG other_rng(77);
    other_rng.fill(other, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(other, other, cv::Size(5, 5), 1.2);
    require(!restart.add_frame(other, 1, 1.0), "initializer accepted unrelated images");
    require(restart.num_tracks() > params.num_features_tracking, "initializer did not restart its reference");
    for (const LivePoint &track : restart.tracks()) {
        require(track.uv == track.uv_ref, "restarted tracks are not on the new reference");
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--grid-only") {
            test_grid_fast_spreads_corners();
            std::cout << "grid FAST tests passed" << std::endl;
            return 0;
        }
        test_local_bundle_adjustment();
        test_ba_window_and_empty_graph();
        test_ba_preserves_archived_observations();
        test_ba_rejects_unobservable_graphs();
        test_frontend_applies_backend_updates();
        test_apply_keyframe_poses();
        test_apply_keyframe_poses_with_scale();
        test_direct_map_observer_index();
        if (argc > 1 && std::string(argv[1]) == "--backend-only") {
            std::cout << "direct backend tests passed" << std::endl;
            return 0;
        }
        test_shared_feature_trail_style();
        test_rgbd_depth_units_and_invalid_samples();
        test_configuration_and_missing_depth();
        test_pose_with_outliers_and_bad_prediction();
        test_kannala_brandt_klt();
        test_stereo_frontend_loss_and_reset();
        test_direct_alignment_recovers_translation();
        test_photometric_jacobian_matches_pinhole_and_finite_difference();
        test_lift_reference_point_and_warp_rejections();
        test_kannala_brandt_direct_alignment();
        test_grid_fast_spreads_corners();
        test_detect_corners_mask_consistency();
        test_lk_and_geometric_pose();
        test_frontend_direct_on_rgbd();
        test_planar_patch_pose_and_feature_shift();
        test_inverse_depth_filter();
        test_depth_filter_matches_svo_update();
        test_svo_one_pixel_tau();
        test_semi_direct_tracks_points_from_older_keyframes();
        test_monocular_param_overrides();
        test_seed_measurement_from_track();
        test_monocular_two_view_initializer();
        test_semi_direct_stereo_and_monocular();
    } catch (const std::exception &error) {
        std::cerr << "test_direct_slam failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "test_direct_slam passed" << std::endl;
    return 0;
}
