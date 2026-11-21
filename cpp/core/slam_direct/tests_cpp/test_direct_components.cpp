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

// Unit tests for the building blocks shared by the direct SLAM trackers: image helpers,
// corner occupancy, sensor measurement of live points, map/landmark bookkeeping, and the
// tracker hooks DirectSlam calls. test_direct_slam.cpp covers end-to-end scenarios.

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "camera/camera.h"
#include "camera/camera_models.h"
#include "slam_direct/depth_measurement.h"
#include "slam_direct/direct_frontend.h"
#include "slam_direct/direct_map.h"
#include "slam_direct/features/corner_detection.h"
#include "slam_direct/image_utils.h"
#include "slam_direct/live_point.h"
#include "slam_direct/pose_tracker.h"

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

cv::Mat random_texture(int width, int height, int seed) {
    cv::Mat image(height, width, CV_8U);
    cv::RNG rng(seed);
    rng.fill(image, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.8);
    return image;
}

template <typename Fn> bool throws_invalid_argument(Fn &&fn, const std::string &expected_text = "") {
    try {
        fn();
    } catch (const std::invalid_argument &error) {
        return expected_text.empty() || std::string(error.what()).find(expected_text) != std::string::npos;
    }
    return false;
}

// ---------------------------------------------------------------------------------------
// image_utils

void test_require_gray() {
    require_gray(cv::Mat(4, 4, CV_8UC1, cv::Scalar(0)), "ok");
    require(throws_invalid_argument([] { require_gray(cv::Mat(), "empty image"); }, "empty image"),
            "require_gray accepted an empty image or lost its name");
    require(throws_invalid_argument([] { require_gray(cv::Mat(4, 4, CV_8UC3), "color"); }),
            "require_gray accepted a 3-channel image");
    require(throws_invalid_argument([] { require_gray(cv::Mat(4, 4, CV_16UC1), "16 bit"); }),
            "require_gray accepted a 16-bit image");
}

// v(x, y) = 10 x + 100 y is linear, so every bilinear sample is exact in float.
cv::Mat linear_ramp() {
    cv::Mat image(3, 4, CV_8U);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x) image.at<uchar>(y, x) = static_cast<uchar>(10 * x + 100 * y);
    return image;
}

void test_bilinear_samplers() {
    const cv::Mat img = linear_ramp(); // 4 columns, 3 rows.
    // Interior: both samplers interpolate identically.
    for (float y : {0.0f, 0.25f, 0.5f, 1.0f})
        for (float x : {0.0f, 0.5f, 1.25f, 1.75f, 2.0f}) {
            const float expected = 10.0f * x + 100.0f * y;
            require(sample_gray(img, x, y) == expected, "sample_gray interior value is wrong");
            require(sample_gray_aligner(img, x, y) == expected, "sample_gray_aligner interior value is wrong");
        }
    // Left/top of the image clamps to the first pixel.
    require(sample_gray(img, -3.0f, -1.0f) == 0.0f, "sample_gray does not clamp negative coordinates");
    require(sample_gray_aligner(img, -3.0f, -1.0f) == 0.0f,
            "sample_gray_aligner does not clamp negative coordinates");
    // Last strip (cols - 2, cols - 1) = (2, 3): the two samplers deliberately differ.
    require(sample_gray(img, 2.5f, 0.0f) == 20.0f, "sample_gray must clamp the last column strip to cols - 2");
    require(sample_gray_aligner(img, 2.5f, 0.0f) == 25.0f,
            "sample_gray_aligner must interpolate inside the last column strip");
    require(sample_gray(img, 0.0f, 1.5f) == 100.0f, "sample_gray must clamp the last row strip to rows - 2");
    require(sample_gray_aligner(img, 0.0f, 1.5f) == 150.0f,
            "sample_gray_aligner must interpolate inside the last row strip");
    // At and beyond the last pixel both read column/row size - 2 (the aligner snaps).
    for (float x : {3.0f, 7.0f}) {
        require(sample_gray(img, x, 0.0f) == 20.0f, "sample_gray past the last column is wrong");
        require(sample_gray_aligner(img, x, 0.0f) == 20.0f, "sample_gray_aligner past the last column is wrong");
    }
    // Degenerate images.
    const cv::Mat column(5, 1, CV_8U, cv::Scalar(42));
    require(sample_gray(column, 0.7f, 2.2f) == 42.0f && sample_gray_aligner(column, 0.7f, 2.2f) == 42.0f,
            "a one-pixel-wide image must return its first pixel");
    require(sample_gray(cv::Mat(), 1.0f, 1.0f) == 0.0f && sample_gray_aligner(cv::Mat(), 1.0f, 1.0f) == 0.0f,
            "an empty image must sample to zero");
}

void test_build_pyramid() {
    DirectSlamParams params;
    params.num_pyramid_levels = 4;
    params.pyramid_scale = 0.5;
    const cv::Mat image = random_texture(101, 75, 3);
    const ImagePyramid pyramid = build_pyramid(image, params);
    require(pyramid.levels.size() == 4 && pyramid.scales.size() == 4, "pyramid level count is wrong");
    require(pyramid.levels[0].data == image.data, "pyramid level 0 must share the input image");
    // lround rounds halves away from zero: 101 -> 51 -> 26 -> 13, 75 -> 38 -> 19 -> 10.
    const int widths[] = {101, 51, 26, 13};
    const int heights[] = {75, 38, 19, 10};
    for (size_t level = 0; level < 4; ++level) {
        require(pyramid.levels[level].cols == widths[level] && pyramid.levels[level].rows == heights[level],
                "pyramid level size is wrong");
        require(pyramid.levels[level].type() == CV_8UC1, "pyramid level type changed");
        require(pyramid.scales[level] == std::pow(0.5, static_cast<double>(level)), "pyramid scale is wrong");
    }
    // Repeated downscaling never produces an empty level.
    params.num_pyramid_levels = 12;
    const ImagePyramid deep = build_pyramid(image, params);
    require(deep.levels.back().cols >= 1 && deep.levels.back().rows >= 1, "pyramid produced an empty level");
}

void test_inside_border() {
    const cv::Size size(100, 80);
    require(inside_border(Vec2d(10.0, 10.0), size, 10), "border start must be inside");
    require(inside_border(Vec2d(89.999, 69.999), size, 10), "pixel before the far border must be inside");
    require(!inside_border(Vec2d(90.0, 50.0), size, 10), "far column border must be outside");
    require(!inside_border(Vec2d(50.0, 70.0), size, 10), "far row border must be outside");
    require(!inside_border(Vec2d(9.999, 50.0), size, 10), "near border must be outside");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    require(!inside_border(Vec2d(nan, 50.0), size, 10) && !inside_border(Vec2d(50.0, nan), size, 10),
            "NaN pixels must be outside");
}

// ---------------------------------------------------------------------------------------
// features: OccupancyMask

void test_occupancy_mask() {
    DirectSlamParams params;
    params.gftt_min_distance = 6.4; // radius lround(6.4) = 6
    OccupancyMask occupied(cv::Size(120, 90), params);
    require(occupied.radius() == 6, "occupancy radius must round gftt_min_distance");
    require(cv::countNonZero(occupied.mask()) == 120 * 90, "a new occupancy mask must be free");
    occupied.block(Vec2d(40.4, 30.6)); // Center rounds to (40, 31).
    require(occupied.mask().at<uchar>(31, 40) == 0 && occupied.mask().at<uchar>(31, 46) == 0 &&
                occupied.mask().at<uchar>(25, 40) == 0,
            "occupancy disk does not cover its radius");
    require(occupied.mask().at<uchar>(31, 47) == 255 && occupied.mask().at<uchar>(31, 33) == 255,
            "occupancy disk is larger than its radius");

    params.gftt_min_distance = 0.2; // Never below one pixel.
    require(OccupancyMask(cv::Size(10, 10), params).radius() == 1, "occupancy radius must be at least 1");

    // detect_corners never returns a corner at a blocked pixel.
    params = DirectSlamParams();
    params.corner_detector = CornerDetector::Gftt;
    params.num_features = 300;
    params.gftt_quality = 0.001;
    const cv::Mat image = random_texture(160, 120, 11);
    const std::vector<Vec2d> all = detect_corners(image, params);
    require(all.size() > 20, "test image has too few corners");
    OccupancyMask mask(image.size(), params);
    for (size_t i = 0; i < all.size(); i += 2) mask.block(all[i]);
    const std::vector<Vec2d> fresh = detect_corners(image, params, mask.mask());
    for (const Vec2d &corner : fresh) {
        const int u = static_cast<int>(std::lround(corner.x()));
        const int v = static_cast<int>(std::lround(corner.y()));
        require(mask.mask().at<uchar>(v, u) != 0, "detect_corners returned a corner inside a blocked disk");
    }
}

// ---------------------------------------------------------------------------------------
// depth_measurement: measure_live_points

std::vector<LivePoint> live_points_on_plane(const Camera &camera, const std::vector<Vec2d> &pixels, double z,
                                            const Eigen::Isometry3d &Tcw) {
    std::vector<LivePoint> points;
    const Eigen::Isometry3d Twc = Tcw.inverse();
    for (size_t i = 0; i < pixels.size(); ++i) {
        Vec3d Xc;
        require(lift_pixel_depth(camera, pixels[i], z, Xc), "could not lift a test pixel");
        points.push_back(make_live_point(static_cast<int64_t>(i), pixels[i], Twc * Xc));
    }
    return points;
}

void test_measure_live_points_rgbd() {
    auto camera = make_camera(160, 120, 100.0, 100.0, 80.0, 60.0);
    camera.sensor_type = SensorType::RGBD;
    DirectSlamParams params;
    cv::Mat depth(120, 160, CV_32F, cv::Scalar(2.0f));
    depth.at<float>(30, 40) = 0.0f; // A missing depth sample.
    const std::vector<Vec2d> pixels = {{40.0, 30.0}, {100.2, 70.4}, {20.0, 90.0}};
    const auto points = live_points_on_plane(camera, pixels, 2.0, Eigen::Isometry3d::Identity());
    const cv::Mat gray = random_texture(160, 120, 1);
    const auto measured =
        measure_live_points(camera, params, gray, cv::Mat(), depth, Eigen::Isometry3d::Identity(), points);
    const auto reference = measure_pixels(camera, params, gray, cv::Mat(), depth, pixels);
    require(measured.size() == 3, "one measurement per live point is required");
    require(!measured[0].ok && measured[0].sensor_depth == 0.0, "a missing depth sample was measured");
    for (size_t i = 1; i < 3; ++i) {
        require(measured[i].ok && measured[i].sensor_depth == 2.0 && std::abs(measured[i].X.z() - 2.0) < 1e-12,
                "RGBD live point depth is wrong");
        require(measured[i].X == reference[i].X, "measure_live_points differs from measure_pixels at uv");
    }
}

void test_measure_live_points_stereo() {
    // Rectified rig: a fronto-parallel plane at z has constant disparity bf / z.
    const double bf = 40.0, z = 2.0, disparity = bf / z;
    auto camera = make_camera(320, 240, 200.0, 200.0, 160.0, 120.0);
    camera.sensor_type = SensorType::STEREO;
    camera.bf = bf;
    DirectSlamParams params;
    const cv::Mat left = random_texture(320, 240, 7);
    cv::Mat right;
    // right(x) = left(x + disparity)
    const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, -disparity, 0, 1, 0);
    cv::warpAffine(left, right, shift, left.size());
    const std::vector<Vec2d> pixels = {{120.0, 80.0}, {200.5, 150.25}, {260.0, 60.0}};
    // A non-identity pose: the prediction must use the landmark seen from Tcw.
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Tcw.translation() = Vec3d(0.3, -0.1, 0.5);
    const auto points = live_points_on_plane(camera, pixels, z, Tcw);
    for (const LivePoint &point : points) {
        const Vec2d predicted = predict_right_pixel(camera, Tcw * point.position_w, point.uv);
        require(std::abs(predicted.x() - (point.uv.x() - disparity)) < 1e-9 &&
                    std::abs(predicted.y() - point.uv.y()) < 1e-12,
                "rectified right-pixel prediction is wrong");
    }
    const auto measured = measure_live_points(camera, params, left, right, cv::Mat(), Tcw, points);
    for (size_t i = 0; i < points.size(); ++i) {
        require(measured[i].ok && measured[i].has_right, "stereo live point was not matched");
        require(std::abs(measured[i].uv_right.x() - (pixels[i].x() - disparity)) < 0.1,
                "stereo right pixel is wrong");
        require(std::abs(measured[i].X.z() - z) < 0.02, "stereo depth is wrong");
    }
}

void test_measure_live_points_ignores_right_image_without_stereo() {
    // A right image given to a monocular camera must not be used for stereo matching.
    auto camera = make_camera(320, 240, 200.0, 200.0, 160.0, 120.0);
    camera.sensor_type = SensorType::MONOCULAR;
    camera.bf = 40.0;
    DirectSlamParams params;
    const cv::Mat left = random_texture(320, 240, 7);
    const auto points = live_points_on_plane(camera, {{120.0, 80.0}, {200.0, 150.0}}, 2.0,
                                             Eigen::Isometry3d::Identity());
    const auto measured =
        measure_live_points(camera, params, left, left.clone(), cv::Mat(), Eigen::Isometry3d::Identity(), points);
    require(measured.size() == 2 && !measured[0].ok && !measured[1].ok,
            "monocular measurement used the right image");
}

// ---------------------------------------------------------------------------------------
// live_point / direct_map / pose_tracker bookkeeping

void test_make_live_point() {
    const LivePoint point = make_live_point(17, Vec2d(3.7, -0.2), Vec3d(1.0, 2.0, 3.0));
    require(point.id == 17 && point.uv == Vec2d(3.7, -0.2) && point.uv_ref == point.uv,
            "make_live_point pixel fields are wrong");
    require(point.position_w == Vec3d(1.0, 2.0, 3.0) && point.reference_keyframe == -1,
            "make_live_point geometry fields are wrong");
    require(point.observations == 1 && point.trail.size() == 1 && point.trail[0] == cv::Point(3, -1),
            "make_live_point trail must start at the floored pixel");
}

void test_direct_map_add_observation() {
    DirectMap map;
    for (int id = 0; id < 2; ++id) {
        DirectKeyframe keyframe;
        keyframe.id = id;
        map.add_keyframe(std::move(keyframe));
    }
    DirectPoint observation;
    observation.id = 7;
    map.add_observation(0, observation);
    map.add_observation(1, observation);
    map.add_observation(1, observation); // Duplicate: stored, but indexed once.
    map.add_observation(0, observation); // An older keyframe must also remain indexed once.
    const std::vector<int> *observers = map.observers(7);
    require(observers && *observers == std::vector<int>({0, 1}), "add_observation observer index is wrong");
    require(map.keyframes()[0].points.size() == 2 && map.keyframes()[1].points.size() == 2,
            "add_observation did not store the observation");
    bool out_of_range = false;
    try {
        map.add_observation(5, observation);
    } catch (const std::out_of_range &) {
        out_of_range = true;
    }
    require(out_of_range && map.observers(7)->size() == 2, "add_observation accepted an unknown keyframe");
}

void test_landmark_inverse_prior() {
    DirectLandmark landmark;
    require(!landmark.has_inverse_prior && landmark.prior_keyframe == -1, "a new landmark has a prior");
    landmark.set_inverse_prior(0.25, 1e-4, 3);
    require(landmark.has_inverse_prior && landmark.inverse_depth_mu == 0.25 &&
                landmark.inverse_depth_sigma2 == 1e-4 && landmark.prior_keyframe == 3,
            "set_inverse_prior fields are wrong");
}

void test_tracking_result_for_points() {
    const TrackingResult result = TrackingResult::for_points(3);
    require(!result.accepted && result.inliers == 0, "a sized result must start rejected");
    require(result.tracked_ok == std::vector<char>(3, 0) && result.tracked_uv.size() == 3 &&
                result.tracked_uv[2] == Vec2d::Zero(),
            "a sized result must have cleared slots");
}

// ---------------------------------------------------------------------------------------
// Tracker hooks through DirectSlam

// Largest |z(current camera) - z| over the live points, for a scene at constant depth z.
double max_live_depth_error(const DirectSlam &slam, double z) {
    const Eigen::Isometry3d &Tcw = slam.pose().isometry3d();
    double worst = 0.0;
    for (const LivePoint &point : slam.live_points()) worst = std::max(worst, std::abs((Tcw * point.position_w).z() - z));
    return worst;
}

// Lateral motion over a fronto-parallel plane at depth z: the image shifts 3 px per frame.
// Random texture keeps KLT well conditioned (a checkerboard is ambiguous for KLT, and a
// failure accepted under the permissive test thresholds would hide what is measured).
// Returns false if any frame after the first was not tracked as TRACKING_GOOD.
bool track_lateral_rgbd(DirectSlam &slam, double z, int frames) {
    const int width = 160, height = 120;
    const cv::Mat texture = random_texture(width, height, 5);
    const cv::Mat depth(height, width, CV_32F, cv::Scalar(static_cast<float>(z)));
    bool all_good = true;
    for (int i = 0; i < frames; ++i) {
        cv::Mat image;
        const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 3.0 * i, 0, 1, 0);
        cv::warpAffine(texture, image, shift, texture.size());
        slam.track(image, cv::Mat(), depth, i, static_cast<double>(i));
        all_good = all_good && slam.status() == DirectTrackingStatus::Good;
    }
    return all_good;
}

ConfigDict lateral_rgbd_config(const char *mode) {
    ConfigDict config;
    config["mode"] = mode;
    config["local_ba"] = false;
    config["num_features"] = 150;
    config["gftt_quality"] = 0.001;
    config["gftt_min_distance"] = 6.0;
    config["num_features_tracking"] = 8;
    config["num_features_tracking_bad"] = 3;
    config["num_features_needed_for_keyframe"] = 10;
    return config;
}

void test_hybrid_refreshes_live_depth_without_ba() {
    // Every accepted frame sees the plane at z. Without local BA, hybrid's after_accept
    // re-measures each live point in the current frame, so its depth in the accepted camera
    // equals the sensor sample exactly. KLT keeps the first measurement, which inherits the
    // small pose error of each tracked frame.
    const double z = 2.5;
    auto camera = std::make_shared<PinholeCamera>(make_camera(160, 120, 100.0, 100.0, 80.0, 60.0));
    camera->sensor_type = SensorType::RGBD;

    DirectSlam hybrid(camera, lateral_rgbd_config("hybrid"));
    require(track_lateral_rgbd(hybrid, z, 4) && !hybrid.live_points().empty(),
            "hybrid did not track the lateral RGBD sequence");
    const double hybrid_error = max_live_depth_error(hybrid, z);

    DirectSlam klt(camera, lateral_rgbd_config("klt"));
    require(track_lateral_rgbd(klt, z, 4), "klt did not track the lateral RGBD sequence");
    const double klt_error = max_live_depth_error(klt, z);
    std::cout << "live depth error: hybrid " << hybrid_error << " klt " << klt_error << std::endl;
    require(hybrid_error < 1e-9, "hybrid did not refresh live-point depth from the current frame");
    // KLT's error is genuine pose noise: nonzero, but small for a well-tracked sequence.
    require(klt_error > 1e-9 && klt_error < 0.05, "klt live depth error is not ordinary pose noise");
}

void test_keyframe_images_follow_tracker() {
    // Only semi_direct keeps keyframe images and re-anchors live points to new keyframes.
    const double z = 2.5;
    auto camera = std::make_shared<PinholeCamera>(make_camera(160, 120, 100.0, 100.0, 80.0, 60.0));
    camera->sensor_type = SensorType::RGBD;
    for (const char *mode : {"klt", "hybrid", "semi_direct"}) {
        DirectSlam slam(camera, lateral_rgbd_config(mode));
        require(track_lateral_rgbd(slam, z, 3), std::string(mode) + " did not track the lateral sequence");
        const bool keeps_images = std::string(mode) == "semi_direct";
        const DirectKeyframe *latest = slam.map().latest_keyframe();
        require(latest != nullptr, std::string(mode) + " created no keyframe");
        require(latest->gray.empty() != keeps_images, std::string(mode) + " keyframe image ownership is wrong");
        for (const LivePoint &point : slam.live_points()) {
            if (keeps_images)
                require(point.reference_keyframe >= 0, "semi_direct live point has no reference keyframe");
            else
                require(point.reference_keyframe == -1, std::string(mode) + " re-anchored live points");
        }
    }
}

} // namespace

int main() {
    try {
        test_require_gray();
        test_bilinear_samplers();
        test_build_pyramid();
        test_inside_border();
        test_occupancy_mask();
        test_measure_live_points_rgbd();
        test_measure_live_points_stereo();
        test_measure_live_points_ignores_right_image_without_stereo();
        test_make_live_point();
        test_direct_map_add_observation();
        test_landmark_inverse_prior();
        test_tracking_result_for_points();
        test_hybrid_refreshes_live_depth_without_ba();
        test_keyframe_images_follow_tracker();
    } catch (const std::exception &error) {
        std::cerr << "test_direct_components failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "test_direct_components passed" << std::endl;
    return 0;
}
