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

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "camera_model_math.h"
#include "camera_models.h"

using namespace pyslam;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_pinhole_batch_matches_existing_projection() {
    PinholeCamera camera;
    camera.width = 640;
    camera.height = 480;
    camera.fx = 525.0;
    camera.fy = 520.0;
    camera.cx = 320.0;
    camera.cy = 240.0;
    camera.set_intrinsic_matrices();

    MatNx3d points(3, 3);
    points << 0.2, -0.1, 2.0, 1.0, 0.5, 4.0, -0.3, 0.4, 3.0;
    const auto [uv, depth] = camera.project(points);

    MatNx2d uv_into(3, 2);
    VecNd depth_into(3);
    VecNd valid(3);
    camera.project_into(points, uv_into, depth_into, valid);

    require((uv - uv_into).norm() < 1e-12, "pinhole project_into drifted from project()");
    require((depth - depth_into).norm() < 1e-12, "pinhole depths drifted");
    require((valid.array() > 0.5).all(), "pinhole points should be valid");

    Vec3d point = points.row(1).transpose();
    Vec2d projected;
    Eigen::Matrix<double, 2, 3> jacobian;
    require(camera.project_with_jacobian(point, projected, &jacobian), "pinhole jacobian call");
    const double inv_z = 1.0 / point.z();
    Eigen::Matrix<double, 2, 3> expected;
    expected << camera.fx * inv_z, 0.0, -camera.fx * point.x() * inv_z * inv_z, 0.0,
        camera.fy * inv_z, -camera.fy * point.y() * inv_z * inv_z;
    require((jacobian - expected).norm() < 1e-12, "pinhole class Jacobian drifted from g2o block");

    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 2, 6> error_pose;
    Eigen::Matrix<double, 2, 3> error_point;
    camera.g2o_reprojection_error_jacobians(point, rotation, error_pose, error_point);
    Eigen::Matrix<double, 2, 6> expected_pose;
    Eigen::Matrix<double, 2, 3> expected_point;
    camera_model::g2o_reprojection_error_jacobians(jacobian, point, rotation, expected_pose,
                                                   expected_point);
    require((error_pose - expected_pose).norm() < 1e-12, "g2o pose Jacobian");
    require((error_point - expected_point).norm() < 1e-12, "g2o point Jacobian");
}

void test_model_roundtrip_and_json(Camera &camera, CameraType expected_type) {
    MatNx3d points(2, 3);
    points << 0.3, -0.2, 1.5, 0.1, 0.4, 2.5;
    const auto [uv, depth] = camera.project(points);
    MatNx2d uv_into(2, 2);
    VecNd depth_into(2);
    VecNd valid(2);
    camera.project_into(points, uv_into, depth_into, valid);
    require((uv - uv_into).norm() < 1e-9, "batch project drifted from project()");
    require((valid.array() > 0.5).all(), "sample points should project");

    MatNx3d bearings(2, 3);
    VecNd bearing_valid(2);
    camera.unproject_bearings_into(uv, bearings, bearing_valid);
    for (int row = 0; row < points.rows(); ++row) {
        const Eigen::Vector3d source = points.row(row).transpose();
        const Eigen::Vector3d restored = bearings.row(row).transpose();
        require(source.normalized().dot(restored.normalized()) > 1.0 - 1e-8,
                "unproject(project(X)) is not parallel to X");
    }

    const nlohmann::json payload = nlohmann::json::parse(camera.to_json());
    CameraPtr loaded = create_camera_from_json(payload);
    require(loaded != nullptr, "camera JSON factory returned null");
    require(loaded->type == expected_type, "camera JSON type was not preserved");
    require(loaded->D.size() == camera.D.size(), "distortion parameter count changed");
    const auto [uv_loaded, depth_loaded] = loaded->project(points);
    require((uv - uv_loaded).norm() < 1e-9, "loaded camera projection drifted");
    (void)depth;
    (void)depth_loaded;
}

void test_undistort_is_pinhole_image(Camera &camera) {
    require(camera.projects_raw_keypoints(), "model projection should consume detected pixels");
    Vec3d point;
    point << 0.2, -0.1, 1.5;
    Vec2d detected;
    require(camera.project_with_jacobian(point, detected, nullptr), "test point did not project");
    MatNx2d pixels(1, 2);
    pixels << detected.x(), detected.y();
    const MatNx2d undistorted = camera.undistort_points(pixels);
    const double expected_u = camera.fx * point.x() / point.z() + camera.cx;
    const double expected_v = camera.fy * point.y() / point.z() + camera.cy;
    require(std::abs(undistorted(0, 0) - expected_u) < 1e-6, "undistort missed the pinhole u");
    require(std::abs(undistorted(0, 1) - expected_v) < 1e-6, "undistort missed the pinhole v");
    require(std::abs(undistorted(0, 0) - detected.x()) > 1e-4,
            "undistort returned the detected pixel");

    const MatNx2d normalized = camera.unproject_points(pixels);
    const double xn = (undistorted(0, 0) - camera.cx) / camera.fx;
    const double yn = (undistorted(0, 1) - camera.cy) / camera.fy;
    require(std::abs(normalized(0, 0) - xn) < 1e-6, "raw unproject disagrees with undistort");
    require(std::abs(normalized(0, 1) - yn) < 1e-6, "raw unproject disagrees with undistort");
}

void test_tracking_uses_model_pixels_and_validity() {
    KannalaBrandt4Camera kannala(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001, 0.0002);
    test_undistort_is_pinhole_image(kannala);

    const Eigen::Vector2d inside(100.0, 100.0);
    PinholeCamera pinhole;
    pinhole.width = 640;
    pinhole.height = 480;
    pinhole.u_min = 0.0;
    pinhole.u_max = 640.0;
    pinhole.v_min = 0.0;
    pinhole.v_max = 480.0;
    require(!pinhole.is_in_image<double>(inside, -1.0), "pinhole accepted a negative depth");

    DoubleSphereCamera sphere(640, 480, 250.0, 260.0, 320.0, 240.0, -0.2, 0.6);
    require(sphere.is_in_image<double>(inside, -0.05),
            "double sphere rejected a finite negative depth inside the image");
    Vec3d invalid;
    invalid << 0.2, 0.0, -0.3;
    const auto invalid_projection = sphere.project_point(invalid);
    require(std::isnan(invalid_projection.second), "invalid double-sphere depth was not NaN");
    require(!sphere.is_in_image<double>(inside, invalid_projection.second),
            "an invalid projection stayed visible");

    Vec3d behind;
    behind << 1.0, 0.0, -0.05;
    const auto behind_projection = sphere.project_point(behind);
    require(std::isfinite(behind_projection.second) && behind_projection.second < 0.0,
            "valid negative double-sphere depth was dropped");
}

void test_stereo_is_rejected(Camera &camera) {
    Vec3d point;
    point << 0.1, 0.2, 1.0;
    bool rejected = false;
    try {
        camera.project_point_stereo(point);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected, "fisheye stereo projection should be rejected");
}

void test_bearing_to_plane_pixel_distance() {
    KannalaBrandt4Camera camera(512, 512, 190.0, 190.0, 256.0, 255.0, 0.003, 0.003, -0.002, 0.0004);
    Vec3d normal(0.1, 0.9, -0.2);
    normal.normalize();
    // Rotate a bearing out of the plane by a small known angle; the image distance to the
    // plane's curve is that angle times the pixel rate across the plane.
    Vec3d in_plane = Vec3d(0.6, 0.0, 0.8).cross(normal).normalized();
    for (double angle : {1e-3, 4e-3, -2e-3}) {
        const Vec3d bearing = (std::cos(angle) * in_plane + std::sin(angle) * normal).normalized();
        Vec2d uv_plane, uv_off;
        require(camera.project_with_jacobian(in_plane, uv_plane, nullptr) &&
                    camera.project_with_jacobian(bearing, uv_off, nullptr),
                "test bearings did not project");
        const double expected = (uv_off - uv_plane).norm();
        const double distance = camera.bearing_to_plane_pixel_distance(bearing, normal);
        require(std::abs(distance - expected) < 0.02 * expected,
                "bearing-to-plane pixel distance disagrees with the projected offset");
    }
    require(camera.bearing_to_plane_pixel_distance(in_plane, normal) < 1e-9,
            "a bearing on the plane has nonzero distance");
}

void test_rig_triangulation() {
    KannalaBrandt4Camera left(512, 512, 190.0, 190.0, 256.0, 255.0, 0.003, 0.003, -0.002, 0.0004);
    left.right_camera = std::make_shared<KannalaBrandt4Camera>(512, 512, 191.0, 190.5, 255.0, 257.0,
                                                               0.004, 0.002, -0.001, 0.0003);
    left.T_right_left.topRightCorner<3, 1>() = Vec3d(-0.10, 0.002, 0.001);
    const Eigen::Matrix3d R = left.T_right_left.topLeftCorner<3, 3>();
    const Vec3d t = left.T_right_left.topRightCorner<3, 1>();
    // Includes a ray beyond 90 degrees, which a pinhole rig cannot represent.
    for (const Vec3d &X_true : {Vec3d(0.3, -0.2, 2.0), Vec3d(-1.5, 0.4, 1.0), Vec3d(2.0, 0.1, -0.1)}) {
        Vec2d ul, ur;
        require(left.project_with_jacobian(X_true, ul, nullptr) &&
                    left.right_camera->project_with_jacobian(R * X_true + t, ur, nullptr),
                "rig test point did not project");
        Vec3d X;
        require(left.triangulate_stereo_point(ul, ur, 1.0, X), "rig triangulation failed");
        require((X - X_true).norm() < 1e-6 * X_true.norm(), "rig triangulation is inaccurate");
        require(!left.triangulate_stereo_point(ul, ur + Vec2d(0.0, 6.0), 1.0, X),
                "an off-epipolar right observation was triangulated");
    }
}

} // namespace

int main() {
    try {
        test_pinhole_batch_matches_existing_projection();
        test_tracking_uses_model_pixels_and_validity();
        test_bearing_to_plane_pixel_distance();
        test_rig_triangulation();

        KannalaBrandt4Camera kannala(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001,
                                     0.0002);
        test_model_roundtrip_and_json(kannala, CameraType::KANNALA_BRANDT4);
        test_stereo_is_rejected(kannala);

        PinholeRadtan8Camera radtan(640, 480, 400.0, 410.0, 320.0, 240.0, -0.1, 0.02, 0.001,
                                    -0.0005, 0.003, 0.01, -0.004, 0.001);
        test_model_roundtrip_and_json(radtan, CameraType::PINHOLE_RADTAN8);
        test_undistort_is_pinhole_image(radtan);
        test_stereo_is_rejected(radtan);

        DoubleSphereCamera sphere(640, 480, 250.0, 260.0, 320.0, 240.0, -0.2, 0.6);
        test_model_roundtrip_and_json(sphere, CameraType::DOUBLE_SPHERE);
        test_undistort_is_pinhole_image(sphere);
        test_stereo_is_rejected(sphere);

        std::cout << "camera model class tests passed" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return 1;
    }
}
