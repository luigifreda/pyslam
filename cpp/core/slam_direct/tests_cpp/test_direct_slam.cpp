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
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "camera.h"
#include "slam_direct/direct_aligner.h"
#include "slam_direct/direct_frontend.h"
#include "slam_direct/direct_backend.h"
#include "utils/feature_trails.h"
#include "slam_direct/semi_direct_tracker.h"

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

    const SemiDirectPoseResult pose =
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
    config["mode"] = "direct";
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
    flat["DIRECT_SLAM.mode"] = "direct";
    flat["DIRECT_SLAM.num_features"] = 123;
    const auto params = DirectSlamParams::from_config(flat);
    require(params.mode == DirectTrackingMode::Direct && params.num_features == 123,
            "flattened camera settings were not parsed");
    ConfigDict nested;
    ConfigDict block;
    block["mode"] = "direct";
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

} // namespace

int main() {
    try {
        test_local_bundle_adjustment();
        test_ba_window_and_empty_graph();
        test_ba_rejects_unobservable_graphs();
        test_frontend_applies_backend_updates();
        test_shared_feature_trail_style();
        test_rgbd_depth_units_and_invalid_samples();
        test_configuration_and_missing_depth();
        test_pose_with_outliers_and_bad_prediction();
        test_stereo_frontend_loss_and_reset();
        test_direct_alignment_recovers_translation();
        test_lk_and_geometric_pose();
        test_frontend_direct_on_rgbd();
    } catch (const std::exception &error) {
        std::cerr << "test_direct_slam failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "test_direct_slam passed" << std::endl;
    return 0;
}
