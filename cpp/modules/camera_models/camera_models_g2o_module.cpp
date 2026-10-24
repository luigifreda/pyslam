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

#include "camera_model_g2o_factors.h"
#include "camera_model_python.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(camera_models_g2o, m) {
    m.doc() = "g2o edges for camera_models cameras. This module does not include or load gtsam.";
    py::module_::import("g2o");
    py::module_::import("camera_models");

    using MonoEdge = camera_models::EdgeCameraProjectXYZ;
    using MonoEdgeBase = g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>;
    py::class_<MonoEdge, MonoEdgeBase>(m, "EdgeCameraProjectXYZ", py::dynamic_attr())
        .def(py::init([](py::handle camera) { return std::make_unique<MonoEdge>(owning_camera(camera)); }),
             py::arg("camera"))
        .def("is_depth_positive", &MonoEdge::isDepthPositive);

    using PoseEdge = camera_models::EdgeCameraProjectXYZOnlyPose;
    using PoseEdgeBase = g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>;
    py::class_<PoseEdge, PoseEdgeBase>(m, "EdgeCameraProjectXYZOnlyPose", py::dynamic_attr())
        .def(py::init([](py::handle camera) { return std::make_unique<PoseEdge>(owning_camera(camera)); }),
             py::arg("camera"))
        .def("is_depth_positive", &PoseEdge::isDepthPositive)
        .def_readwrite("Xw", &PoseEdge::Xw);

    py::class_<camera_models::EdgeStereoCameraProjectXYZ, g2o::EdgeStereoSE3ProjectXYZ>(
        m, "EdgeStereoCameraProjectXYZ", py::dynamic_attr())
        .def(py::init([](py::handle camera) {
                 return std::make_unique<camera_models::EdgeStereoCameraProjectXYZ>(owning_camera(camera));
             }),
             py::arg("camera"))
        .def("is_depth_positive", &camera_models::EdgeStereoCameraProjectXYZ::isDepthPositive);
    py::class_<camera_models::EdgeStereoCameraProjectXYZOnlyPose, g2o::EdgeStereoSE3ProjectXYZOnlyPose>(
        m, "EdgeStereoCameraProjectXYZOnlyPose", py::dynamic_attr())
        .def(py::init([](py::handle camera) {
                 return std::make_unique<camera_models::EdgeStereoCameraProjectXYZOnlyPose>(owning_camera(camera));
             }),
             py::arg("camera"))
        .def("is_depth_positive", &camera_models::EdgeStereoCameraProjectXYZOnlyPose::isDepthPositive);
}
