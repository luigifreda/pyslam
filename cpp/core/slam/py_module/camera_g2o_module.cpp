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

// g2o edges of the cpp_core cameras. This translation unit does not include gtsam.

#include "py_module/camera_optimizer_bindings.h"

#include "camera_edges_g2o.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

void bind_camera_g2o_edges(py::module &m) {
    // The g2o base classes are registered by the g2o module. Import it first so these
    // edges can be added to a Python SparseOptimizer.
    py::module_::import("g2o");
    using MonoEdge = pyslam::EdgeCameraProjectXYZ;
    using MonoEdgeBase =
        g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>;
    py::class_<MonoEdge, MonoEdgeBase>(m, "EdgeCameraProjectXYZ", py::dynamic_attr())
        .def(py::init([](const std::shared_ptr<pyslam::Camera> &camera) {
                 return std::make_unique<MonoEdge>(camera.get());
             }),
             py::arg("camera"), py::keep_alive<1, 2>())
        .def("is_depth_positive", &MonoEdge::isDepthPositive);

    using PoseEdge = pyslam::EdgeCameraProjectXYZOnlyPose;
    using PoseEdgeBase = g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>;
    py::class_<PoseEdge, PoseEdgeBase>(m, "EdgeCameraProjectXYZOnlyPose", py::dynamic_attr())
        .def(py::init([](const std::shared_ptr<pyslam::Camera> &camera) {
                 return std::make_unique<PoseEdge>(camera.get());
             }),
             py::arg("camera"), py::keep_alive<1, 2>())
        .def("is_depth_positive", &PoseEdge::isDepthPositive)
        .def_readwrite("Xw", &PoseEdge::Xw);

    py::class_<pyslam::EdgeStereoCameraProjectXYZ, g2o::EdgeStereoSE3ProjectXYZ>(m, "EdgeStereoCameraProjectXYZ", py::dynamic_attr())
        .def(py::init([](const pyslam::CameraPtr &camera) {
            return std::make_unique<pyslam::EdgeStereoCameraProjectXYZ>(camera.get());
        }), py::keep_alive<1, 2>())
        .def("is_depth_positive", &pyslam::EdgeStereoCameraProjectXYZ::isDepthPositive);
    py::class_<pyslam::EdgeStereoCameraProjectXYZOnlyPose, g2o::EdgeStereoSE3ProjectXYZOnlyPose>(m, "EdgeStereoCameraProjectXYZOnlyPose", py::dynamic_attr())
        .def(py::init([](const pyslam::CameraPtr &camera) {
            return std::make_unique<pyslam::EdgeStereoCameraProjectXYZOnlyPose>(camera.get());
        }), py::keep_alive<1, 2>())
        .def("is_depth_positive", &pyslam::EdgeStereoCameraProjectXYZOnlyPose::isDepthPositive);
}
