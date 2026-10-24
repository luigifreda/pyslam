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

// cpp_core g2o camera edges. This test does not include gtsam.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <g2o/core/jacobian_workspace.h>

#include "camera_edges_g2o.h"
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

g2o::SE3Quat sample_tcw() {
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix();
    return g2o::SE3Quat(rotation, Eigen::Vector3d(0.2, -0.1, 0.4));
}

void linearize(g2o::OptimizableGraph::Edge &edge) {
    // Edge Jacobian maps reference this memory after the helper returns.
    // Tests copy their matrices before linearizing the next edge.
    static g2o::JacobianWorkspace workspace;
    workspace.updateSize(&edge);
    workspace.allocate();
    edge.linearizeOplus(workspace);
}

void test_pinhole_edges_match_g2o() {
    pyslam::PinholeCamera camera;
    camera.fx = 500.0;
    camera.fy = 480.0;
    camera.cx = 320.0;
    camera.cy = 240.0;

    g2o::VertexSE3Expmap pose;
    g2o::VertexSBAPointXYZ point;
    pose.setEstimate(sample_tcw());
    point.setEstimate(Eigen::Vector3d(0.5, -0.2, 2.0));
    const Eigen::Vector2d measurement(180.0, 220.0);

    EdgeCameraProjectXYZ edge(&camera);
    g2o::EdgeSE3ProjectXYZ reference;
    reference.fx = camera.fx;
    reference.fy = camera.fy;
    reference.cx = camera.cx;
    reference.cy = camera.cy;
    edge.setVertex(0, &point);
    edge.setVertex(1, &pose);
    edge.setMeasurement(measurement);
    reference.setVertex(0, &point);
    reference.setVertex(1, &pose);
    reference.setMeasurement(measurement);
    edge.computeError();
    reference.computeError();
    require_close(edge.error(), reference.error(), 1e-12, "pinhole g2o residual");
    linearize(edge);
    const Eigen::Matrix<double, 2, 3> point_jacobian = edge.jacobianOplusXi();
    const Eigen::Matrix<double, 2, 6> pose_jacobian = edge.jacobianOplusXj();
    linearize(reference);
    require_close(point_jacobian, reference.jacobianOplusXi(), 1e-9, "pinhole g2o point Jacobian");
    require_close(pose_jacobian, reference.jacobianOplusXj(), 1e-9, "pinhole g2o pose Jacobian");
    require(edge.isDepthPositive(), "pinhole point in front is valid");

    g2o::VertexSE3Expmap pose_only_vertex;
    pose_only_vertex.setEstimate(sample_tcw());
    EdgeCameraProjectXYZOnlyPose pose_edge(&camera);
    g2o::EdgeSE3ProjectXYZOnlyPose pose_reference;
    pose_edge.Xw = point.estimate();
    pose_reference.Xw = point.estimate();
    pose_reference.fx = camera.fx;
    pose_reference.fy = camera.fy;
    pose_reference.cx = camera.cx;
    pose_reference.cy = camera.cy;
    pose_edge.setVertex(0, &pose_only_vertex);
    pose_reference.setVertex(0, &pose_only_vertex);
    pose_edge.setMeasurement(measurement);
    pose_reference.setMeasurement(measurement);
    pose_edge.computeError();
    pose_reference.computeError();
    require_close(pose_edge.error(), pose_reference.error(), 1e-12, "pinhole pose-only residual");
    linearize(pose_edge);
    const Eigen::Matrix<double, 2, 6> pose_only_jacobian = pose_edge.jacobianOplusXi();
    linearize(pose_reference);
    require_close(pose_only_jacobian, pose_reference.jacobianOplusXi(), 1e-9,
                  "pinhole pose-only Jacobian");
}

void test_kannala_brandt_edge_jacobian() {
    const KannalaBrandt4Camera camera(640, 480, 300.0, 310.0, 320.0, 240.0, -0.02, 0.005, -0.001,
                                      0.0002);
    g2o::VertexSE3Expmap pose;
    g2o::VertexSBAPointXYZ point;
    pose.setEstimate(sample_tcw());
    point.setEstimate(Eigen::Vector3d(0.4, -0.2, 1.5));

    EdgeCameraProjectXYZ edge(&camera);
    edge.setVertex(0, &point);
    edge.setVertex(1, &pose);
    edge.setMeasurement(Eigen::Vector2d(300.0, 250.0));
    edge.computeError();
    linearize(edge);
    const Eigen::Matrix<double, 2, 6> pose_jacobian = edge.jacobianOplusXj();
    const Eigen::Matrix<double, 2, 3> point_jacobian = edge.jacobianOplusXi();

    const double eps = 1e-6;
    const g2o::SE3Quat pose0 = pose.estimate();
    const Eigen::Vector3d point0 = point.estimate();
    Eigen::Matrix<double, 2, 6> numeric_pose;
    for (int k = 0; k < 6; ++k) {
        double update[6] = {};
        update[k] = eps;
        pose.oplus(update);
        edge.computeError();
        const Eigen::Vector2d plus = edge.error();
        pose.setEstimate(pose0);
        update[k] = -eps;
        pose.oplus(update);
        edge.computeError();
        const Eigen::Vector2d minus = edge.error();
        pose.setEstimate(pose0);
        numeric_pose.col(k) = (plus - minus) / (2.0 * eps);
    }
    Eigen::Matrix<double, 2, 3> numeric_point;
    for (int k = 0; k < 3; ++k) {
        double update[3] = {};
        update[k] = eps;
        point.oplus(update);
        edge.computeError();
        const Eigen::Vector2d plus = edge.error();
        point.setEstimate(point0);
        update[k] = -eps;
        point.oplus(update);
        edge.computeError();
        const Eigen::Vector2d minus = edge.error();
        point.setEstimate(point0);
        numeric_point.col(k) = (plus - minus) / (2.0 * eps);
    }
    require_close(pose_jacobian, numeric_pose, 1e-5, "Kannala-Brandt g2o pose Jacobian");
    require_close(point_jacobian, numeric_point, 1e-5, "Kannala-Brandt g2o point Jacobian");
    require(edge.isDepthPositive(), "Kannala-Brandt sample point is valid");
}

void test_double_sphere_accepts_negative_depth() {
    const DoubleSphereCamera camera(640, 480, 250.0, 260.0, 320.0, 240.0, -0.2, 0.6);
    g2o::VertexSE3Expmap pose;
    g2o::VertexSBAPointXYZ point;
    pose.setEstimate(g2o::SE3Quat());
    point.setEstimate(Eigen::Vector3d(1.0, 0.0, -0.05));
    EdgeCameraProjectXYZ edge(&camera);
    edge.setVertex(0, &point);
    edge.setVertex(1, &pose);
    require(edge.isDepthPositive(), "double sphere keeps a valid ray with negative z");

    pyslam::PinholeCamera pinhole;
    pinhole.fx = 250.0;
    pinhole.fy = 260.0;
    pinhole.cx = 320.0;
    pinhole.cy = 240.0;
    EdgeCameraProjectXYZ pinhole_edge(&pinhole);
    pinhole_edge.setVertex(0, &point);
    pinhole_edge.setVertex(1, &pose);
    require(!pinhole_edge.isDepthPositive(), "pinhole rejects a point behind the camera");
}

void benchmark_pinhole_edges() {
    pyslam::PinholeCamera camera;
    camera.fx = 500; camera.fy = 480; camera.cx = 320; camera.cy = 240;
    g2o::VertexSE3Expmap pose;
    g2o::VertexSBAPointXYZ point;
    pose.setEstimate(sample_tcw());
    point.setEstimate(Eigen::Vector3d(0.5, -0.2, 2.0));
    EdgeCameraProjectXYZ current(&camera);
    g2o::EdgeSE3ProjectXYZ legacy;
    legacy.fx = camera.fx; legacy.fy = camera.fy; legacy.cx = camera.cx; legacy.cy = camera.cy;
    for (g2o::OptimizableGraph::Edge *edge : {static_cast<g2o::OptimizableGraph::Edge *>(&current), static_cast<g2o::OptimizableGraph::Edge *>(&legacy)}) {
        edge->setVertex(0, &point); edge->setVertex(1, &pose);
    }
    current.setMeasurement(Eigen::Vector2d(300, 250));
    legacy.setMeasurement(Eigen::Vector2d(300, 250));
    g2o::JacobianWorkspace ws_old, ws_new;
    ws_old.updateSize(&legacy); ws_old.allocate();
    ws_new.updateSize(&current); ws_new.allocate();
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
    compare("g2o error", [&] { legacy.computeError(); sink = legacy.error().sum(); },
            [&] { current.computeError(); sink = current.error().sum(); });
    compare("g2o linearize", [&] { static_cast<g2o::OptimizableGraph::Edge &>(legacy).linearizeOplus(ws_old); sink = legacy.jacobianOplusXi().sum(); },
            [&] { static_cast<g2o::OptimizableGraph::Edge &>(current).linearizeOplus(ws_new); sink = current.jacobianOplusXi().sum(); });
}

} // namespace

int main(int argc, char **argv) {
    try {
        test_pinhole_edges_match_g2o();
        test_kannala_brandt_edge_jacobian();
        test_double_sphere_accepts_negative_depth();
        if (argc > 1 && std::string(argv[1]) == "--benchmark") benchmark_pinhole_edges();
    } catch (const std::exception &exception) {
        std::cerr << "camera g2o edge tests failed: " << exception.what() << std::endl;
        return 1;
    }
    std::cout << "camera g2o edge tests passed" << std::endl;
    return 0;
}
