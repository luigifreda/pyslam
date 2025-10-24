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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "casters/dictionary_casters.h" // for Dict, List
#include "casters/opencv_type_casters.h"

#include "dictionary.h"
#include "py_module/camera_module.h"
#include "py_module/ckdtree_module.h"
#include "py_module/feature_shared_info_module.h"
#include "py_module/frame_module.h"
#include "py_module/keyframe_module.h"
#include "py_module/map_module.h"
#include "py_module/map_point_module.h"
#include "py_module/optimizer_g2o_module.h"
#include "py_module/sim3_pose_module.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "PYSLAM C++ Core Module - High-performance SLAM classes";

    // ------------------------------------------------------------
    // Dictionary

    // Expose the containers
    py::bind_vector<pyslam::List>(m, "List");
    py::bind_map<pyslam::Dict>(m, "Dict");

    // Tiny demo function to show automatic conversion
    m.def("echo_dict", [](const pyslam::Dict &d) { return d; });

    // Expose Value directly
    py::class_<pyslam::Value>(m, "Value")
        .def(py::init<>())
        .def_property_readonly("is_null", [](const pyslam::Value &v) {
            return std::holds_alternative<std::monostate>(v.data);
        });

    // ------------------------------------------------------------
    // Camera and CameraPose

    bind_camera(m);

    // ------------------------------------------------------------
    // Sim3Pose class

    bind_sim3_pose(m);

    // ------------------------------------------------------------
    // FeatureSharedInfo

    bind_feature_shared_info(m);

    // ------------------------------------------------------------
    // Frame class
    bind_frame(m);

    // ------------------------------------------------------------
    // KeyFrameGraph class - matches Python KeyFrameGraph
    bind_keyframe(m);

    // ------------------------------------------------------------
    // MapPoint class
    bind_map_point(m);

    // ------------------------------------------------------------
    // Map and LocalMapBase classes

    bind_map(m);

    // ------------------------------------------------------------
    // CKDTree class

    bind_ckdtree(m);

    // ------------------------------------------------------------
    // OptimizerG2o class

    bind_optimizer_g2o(m);

} // PYBIND11_MODULE
