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
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "camera_model_math.h"

using namespace pyslam::camera_model;

namespace {

void require(bool cond, const std::string &msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

template <typename Project>
void check_project_jacobian(const Eigen::Vector3d &X, Project project, const std::string &name) {
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 3> J;
    require(project(X, uv, &J), name + ": projection marked invalid");

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 3> numeric;
    for (int col = 0; col < 3; ++col) {
        Eigen::Vector3d Xp = X;
        Eigen::Vector3d Xm = X;
        Xp[col] += eps;
        Xm[col] -= eps;
        Eigen::Vector2d up;
        Eigen::Vector2d um;
        require(project(Xp, up, nullptr), name + ": plus step invalid");
        require(project(Xm, um, nullptr), name + ": minus step invalid");
        numeric.col(col) = (up - um) / (2.0 * eps);
    }
    const double scale = std::max(1.0, numeric.norm());
    const double err = (J - numeric).norm() / scale;
    if (!std::isfinite(err) || err > 1e-6) {
        std::cerr << name << " analytic\n"
                  << J << "\nnumeric\n"
                  << numeric << "\nrel " << err << std::endl;
        throw std::runtime_error(name + ": project Jacobian mismatch");
    }
}

template <typename Unproject>
void check_unproject_jacobian(const Eigen::Vector2d &uv, Unproject unproject, const std::string &name) {
    Eigen::Vector3d bearing;
    Eigen::Matrix<double, 3, 2> J;
    require(unproject(uv, bearing, &J), name + ": unprojection marked invalid");

    const double eps = 1e-6;
    Eigen::Matrix<double, 3, 2> numeric;
    for (int col = 0; col < 2; ++col) {
        Eigen::Vector2d up = uv;
        Eigen::Vector2d um = uv;
        up[col] += eps;
        um[col] -= eps;
        Eigen::Vector3d bp;
        Eigen::Vector3d bm;
        require(unproject(up, bp, nullptr), name + ": plus step invalid");
        require(unproject(um, bm, nullptr), name + ": minus step invalid");
        numeric.col(col) = (bp - bm) / (2.0 * eps);
    }
    const double scale = std::max(1.0, numeric.norm());
    const double err = (J - numeric).norm() / scale;
    if (!std::isfinite(err) || err > 1e-5) {
        std::cerr << name << " analytic\n"
                  << J << "\nnumeric\n"
                  << numeric << "\nrel " << err << std::endl;
        throw std::runtime_error(name + ": unproject Jacobian mismatch");
    }
}

void check_parallel(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const std::string &name) {
    const double c = a.normalized().dot(b.normalized());
    require(c > 1.0 - 1e-8, name + ": unproject(project(X)) is not parallel to X");
}

void test_pinhole() {
    const double fx = 500.0;
    const double fy = 480.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const Eigen::Vector3d X(1.0, -0.5, 4.0);
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 3> J;
    require(project_pinhole(fx, fy, cx, cy, X, uv, &J), "pinhole valid");
    require(std::abs(uv[0] - (fx * X[0] / X[2] + cx)) < 1e-12, "pinhole u");
    require(std::abs(uv[1] - (fy * X[1] / X[2] + cy)) < 1e-12, "pinhole v");

    Eigen::Matrix<double, 2, 3> expected;
    const double inv_z = 1.0 / X[2];
    const double inv_z2 = inv_z * inv_z;
    expected << fx * inv_z, 0.0, -fx * X[0] * inv_z2, 0.0, fy * inv_z, -fy * X[1] * inv_z2;
    require((J - expected).norm() < 1e-12, "pinhole Jacobian differs from the g2o 2x3 block");

    Eigen::Vector3d bearing;
    require(unproject_pinhole_bearing(fx, fy, cx, cy, uv, bearing), "pinhole unproject");
    require(std::abs(bearing[0] - X[0] / X[2]) < 1e-12, "pinhole bearing x");
    require(std::abs(bearing[1] - X[1] / X[2]) < 1e-12, "pinhole bearing y");
    require(std::abs(bearing[2] - 1.0) < 1e-12, "pinhole bearing stays on z = 1");

    Eigen::Matrix3d Rcw = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix<double, 2, 6> J_pose;
    Eigen::Matrix<double, 2, 3> J_point;
    g2o_reprojection_error_jacobians(J, X, Rcw, J_pose, J_point);
    require((J_point + J * Rcw).norm() < 1e-12, "g2o point block");
    Eigen::Matrix<double, 3, 6> dXc;
    g2o_left_se3_jacobian(X, dXc);
    require((J_pose + J * dXc).norm() < 1e-12, "g2o pose block");

    Eigen::Matrix<double, 3, 3> Jstereo;
    pinhole_stereo_jacobian(J, X, 50.0, Jstereo);
    require((Jstereo.topRows<2>() - J).norm() < 1e-12, "stereo reuses the mono Jacobian");
    require(!project_pinhole(fx, fy, cx, cy, Eigen::Vector3d(0, 0, -1), uv),
            "pinhole behind the camera is invalid");
}

void test_kannala_brandt4() {
    const double fx = 300.0;
    const double fy = 310.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double k1 = -0.02;
    const double k2 = 0.005;
    const double k3 = -0.001;
    const double k4 = 0.0002;
    auto project = [&](const Eigen::Vector3d &X, Eigen::Vector2d &uv, Eigen::Matrix<double, 2, 3> *J) {
        return project_kannala_brandt4(fx, fy, cx, cy, k1, k2, k3, k4, X, uv, J);
    };
    auto unproject = [&](const Eigen::Vector2d &uv, Eigen::Vector3d &b, Eigen::Matrix<double, 3, 2> *J) {
        return unproject_kannala_brandt4_bearing(fx, fy, cx, cy, k1, k2, k3, k4, uv, b, J);
    };

    const Eigen::Vector3d axis(0.0, 0.0, 2.0);
    Eigen::Vector2d uv;
    require(project(axis, uv, nullptr), "kb axis");
    require((uv - Eigen::Vector2d(cx, cy)).norm() < 1e-10, "kb optical axis");

    const Eigen::Vector3d X(0.4, -0.2, 1.5);
    check_project_jacobian(X, project, "kb4");
    require(project(X, uv, nullptr), "kb project");
    Eigen::Vector3d bearing;
    require(unproject(uv, bearing, nullptr), "kb unproject");
    check_parallel(bearing, X, "kb4");
    check_unproject_jacobian(uv, unproject, "kb4");

    // Zero distortion, point in the xy plane of the ray (1, 0, 1): theta = pi/4.
    Eigen::Vector2d uv0;
    require(project_kannala_brandt4(fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, Eigen::Vector3d(1, 0, 1), uv0),
            "kb zero");
    const double theta = std::atan2(1.0, 1.0);
    require(std::abs(uv0[0] - (fx * theta + cx)) < 1e-10, "kb zero-distortion u");
    require(std::abs(uv0[1] - cy) < 1e-10, "kb zero-distortion v");
}

void test_radtan8() {
    const double fx = 400.0;
    const double fy = 410.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double k1 = -0.1;
    const double k2 = 0.02;
    const double p1 = 0.001;
    const double p2 = -0.0005;
    const double k3 = 0.003;
    const double k4 = 0.01;
    const double k5 = -0.004;
    const double k6 = 0.001;
    auto project = [&](const Eigen::Vector3d &X, Eigen::Vector2d &uv, Eigen::Matrix<double, 2, 3> *J) {
        return project_pinhole_radtan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, X, uv, J);
    };
    auto unproject = [&](const Eigen::Vector2d &uv, Eigen::Vector3d &b, Eigen::Matrix<double, 3, 2> *J) {
        return unproject_pinhole_radtan8_bearing(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, uv, b,
                                                 J);
    };

    const Eigen::Vector3d X(0.3, -0.2, 2.0);
    check_project_jacobian(X, project, "radtan8");
    Eigen::Vector2d uv;
    require(project(X, uv, nullptr), "radtan project");
    Eigen::Vector3d bearing;
    require(unproject(uv, bearing, nullptr), "radtan unproject");
    check_parallel(bearing, X, "radtan8");
    check_unproject_jacobian(uv, unproject, "radtan8");

    Eigen::Vector2d uv_lin;
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Matrix<double, 2, 3> Jp;
    require(project_pinhole_radtan8(fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, X, uv_lin, &J),
            "radtan zero");
    require(project_pinhole(fx, fy, cx, cy, X, uv, &Jp), "pinhole ref");
    require((uv_lin - uv).norm() < 1e-12, "zero radtan matches pinhole");
    require((J - Jp).norm() < 1e-10, "zero radtan Jacobian matches pinhole");
    require(!project(Eigen::Vector3d(0, 0, -0.2), uv, nullptr), "radtan behind camera");
}

void test_double_sphere() {
    const double fx = 250.0;
    const double fy = 260.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double xi = -0.2;
    const double alpha = 0.6;
    auto project = [&](const Eigen::Vector3d &X, Eigen::Vector2d &uv, Eigen::Matrix<double, 2, 3> *J) {
        return project_double_sphere(fx, fy, cx, cy, xi, alpha, X, uv, J);
    };
    auto unproject = [&](const Eigen::Vector2d &uv, Eigen::Vector3d &b, Eigen::Matrix<double, 3, 2> *J) {
        return unproject_double_sphere_bearing(fx, fy, cx, cy, xi, alpha, uv, b, J);
    };

    const Eigen::Vector3d X(0.25, -0.15, 1.2);
    check_project_jacobian(X, project, "double sphere");
    Eigen::Vector2d uv;
    require(project(X, uv, nullptr), "ds project");
    Eigen::Vector3d bearing;
    require(unproject(uv, bearing, nullptr), "ds unproject");
    check_parallel(bearing, X, "double sphere");
    check_unproject_jacobian(uv, unproject, "double sphere");

    // z is negative, but the point is still inside the double-sphere domain.
    const Eigen::Vector3d behind(1.0, 0.0, -0.05);
    check_project_jacobian(behind, project, "double sphere negative z");
    require(project(behind, uv, nullptr), "ds accepts a ray with negative z");
    require(unproject(uv, bearing, nullptr), "ds unproject of that ray");
    check_parallel(bearing, behind, "double sphere negative z");
}

} // namespace

int main() {
    try {
        test_pinhole();
        test_kannala_brandt4();
        test_radtan8();
        test_double_sphere();
        std::cout << "camera model math tests passed" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}
