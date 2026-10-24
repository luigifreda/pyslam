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

// cpp_core gtsam camera factors. This test does not include g2o.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam_factors/weighted_projection_factors.h>

#include "camera_factors_gtsam.h"
#include "camera_models.h"

using namespace pyslam;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected, double tolerance,
                   const std::string &message) {
    const double scale = std::max(1.0, expected.norm());
    if (!actual.allFinite() || !expected.allFinite() ||
        (actual - expected).norm() / scale > tolerance) {
        throw std::runtime_error(message);
    }
}

void test_gtsam_pinhole_matches_cal3_s2() {
    pyslam::PinholeCamera camera;
    camera.fx = 500.0;
    camera.fy = 480.0;
    camera.cx = 320.0;
    camera.cy = 240.0;
    const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.1, -0.2, 0.05), gtsam::Point3(0.3, -0.1, 0.2));
    const gtsam::Point3 point(1.0, 0.4, 3.0);
    const gtsam::Point2 measured(200.0, 180.0);
    auto noise = gtsam::noiseModel::Unit::Create(2);

    CameraProjectionFactor factor(measured, noise, 0, 1, &camera);
    auto calibration = boost::make_shared<gtsam::Cal3_S2>(camera.fx, camera.fy, 0.0, camera.cx, camera.cy);
    gtsam_factors::WeightedGenericProjectionFactorCal3_S2 reference(measured, noise, 0, 1, calibration);

    gtsam::Matrix pose_jacobian;
    gtsam::Matrix point_jacobian;
    gtsam::Matrix reference_pose;
    gtsam::Matrix reference_point;
    const gtsam::Vector error = factor.evaluateError(pose, point, pose_jacobian, point_jacobian);
    const gtsam::Vector reference_error =
        reference.evaluateError(pose, point, reference_pose, reference_point);
    require_close(error, reference_error, 1e-9, "pinhole gtsam residual");
    require_close(pose_jacobian, reference_pose, 1e-8, "pinhole gtsam pose Jacobian");
    require_close(point_jacobian, reference_point, 1e-8, "pinhole gtsam point Jacobian");

    const gtsam::Pose3 identity;
    const gtsam::Point3 behind(0.0, 0.0, -1.0);
    gtsam::Matrix behind_pose;
    gtsam::Matrix behind_point;
    const gtsam::Vector behind_error = factor.evaluateError(identity, behind, behind_pose, behind_point);
    require_close(behind_error, gtsam::Vector2::Constant(2.0 * camera.fx), 1e-12,
                  "invalid pinhole residual");
    require_close(behind_pose, gtsam::Matrix::Zero(2, 6), 1e-12, "invalid pinhole pose Jacobian");
    require_close(behind_point, gtsam::Matrix::Zero(2, 3), 1e-12, "invalid pinhole point Jacobian");
}

void test_gtsam_kannala_brandt_jacobian() {
    const KannalaBrandt4Camera camera(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001,
                                      0.0002);
    const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(-0.05, 0.15, 0.02), gtsam::Point3(-0.2, 0.1, 0.3));
    const gtsam::Point3 point(0.4, -0.3, 2.0);
    auto noise = gtsam::noiseModel::Unit::Create(2);
    CameraProjectionFactor factor(gtsam::Point2(310.0, 250.0), noise, 0, 1, &camera);
    CameraResectioningFactor resection(noise, 0, &camera, gtsam::Point2(310.0, 250.0), point);

    gtsam::Matrix pose_jacobian;
    gtsam::Matrix point_jacobian;
    factor.evaluateError(pose, point, pose_jacobian, point_jacobian);
    gtsam::Matrix resection_jacobian;
    resection.evaluateError(pose, resection_jacobian);
    require_close(resection_jacobian, pose_jacobian, 1e-12, "resectioning matches the projection factor");

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 6> numeric_pose;
    Eigen::Matrix<double, 2, 3> numeric_point;
    for (int k = 0; k < 6; ++k) {
        gtsam::Vector6 xi = gtsam::Vector6::Zero();
        xi(k) = eps;
        const gtsam::Vector plus = factor.evaluateError(pose.retract(xi), point);
        xi(k) = -eps;
        const gtsam::Vector minus = factor.evaluateError(pose.retract(xi), point);
        numeric_pose.col(k) = (plus - minus) / (2.0 * eps);
    }
    for (int k = 0; k < 3; ++k) {
        gtsam::Point3 plus_point = point;
        gtsam::Point3 minus_point = point;
        plus_point(k) += eps;
        minus_point(k) -= eps;
        const gtsam::Vector plus = factor.evaluateError(pose, plus_point);
        const gtsam::Vector minus = factor.evaluateError(pose, minus_point);
        numeric_point.col(k) = (plus - minus) / (2.0 * eps);
    }
    require_close(pose_jacobian, numeric_pose, 1e-5, "Kannala-Brandt gtsam pose Jacobian");
    require_close(point_jacobian, numeric_point, 1e-5, "Kannala-Brandt gtsam point Jacobian");
}

void benchmark_pinhole_factors() {
    pyslam::PinholeCamera camera;
    camera.fx = 500; camera.fy = 480; camera.cx = 320; camera.cy = 240;
    const gtsam::Pose3 pose_wc =
        gtsam::Pose3(gtsam::Rot3(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix()),
                     gtsam::Point3(0.2, -0.1, 0.4)).inverse();
    const gtsam::Point3 point(0.5, -0.2, 2.0);
    auto noise = gtsam::noiseModel::Unit::Create(2);
    CameraProjectionFactor model(gtsam::Point2(300, 250), noise, 0, 1, &camera);
    gtsam_factors::WeightedGenericProjectionFactorCal3_S2 reference(
        gtsam::Point2(300, 250), noise, 0, 1,
        boost::make_shared<gtsam::Cal3_S2>(500, 480, 0, 320, 240));
    gtsam::Matrix H1, H2;
    volatile double sink = 0;
    auto compare = [&](const char *name, const std::function<void()> &old_call,
                        const std::function<void()> &new_call) {
        std::vector<double> old_samples, new_samples;
        for (int i = 0; i < 3; ++i) { old_call(); new_call(); }
        auto measure = [](const std::function<void()> &call) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 200000; ++i) call();
            return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / 200000;
        };
        for (int sample = 0; sample < 9; ++sample) {
            if (sample % 2 == 0) { old_samples.push_back(measure(old_call)); new_samples.push_back(measure(new_call)); }
            else { new_samples.push_back(measure(new_call)); old_samples.push_back(measure(old_call)); }
        }
        std::sort(old_samples.begin(), old_samples.end());
        std::sort(new_samples.begin(), new_samples.end());
        std::cout << name << ": " << old_samples[4] << " -> " << new_samples[4]
                  << " ns (" << new_samples[4] / old_samples[4] << "x)\n";
    };
    compare("gtsam error", [&] { sink = reference.evaluateError(pose_wc, point).sum(); },
            [&] { sink = model.evaluateError(pose_wc, point).sum(); });
    compare("gtsam linearize", [&] { sink = reference.evaluateError(pose_wc, point, H1, H2).sum(); },
            [&] { sink = model.evaluateError(pose_wc, point, H1, H2).sum(); });
}

} // namespace

int main(int argc, char **argv) {
    try {
        test_gtsam_pinhole_matches_cal3_s2();
        test_gtsam_kannala_brandt_jacobian();
        if (argc > 1 && std::string(argv[1]) == "--benchmark") benchmark_pinhole_factors();
    } catch (const std::exception &exception) {
        std::cerr << "camera gtsam factor tests failed: " << exception.what() << std::endl;
        return 1;
    }
    std::cout << "camera gtsam factor tests passed" << std::endl;
    return 0;
}
