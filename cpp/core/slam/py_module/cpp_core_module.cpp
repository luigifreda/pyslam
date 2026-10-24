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

// CRITICAL: Include opaque types BEFORE stl.h to ensure opaque declarations take effect
#include "casters/slam_opaque_types.h" // for SLAM container opaque types

#include <pybind11/stl.h>

#include "casters/dictionary_casters.h" // for Dict, List
#include "casters/opencv_type_casters.h"

#include "dictionary.h"
#include "py_module/camera_module.h"
#include "py_module/ckdtree_module.h"
#include "py_module/config_parameters_module.h"
#include "py_module/eigen_module.h"
#include "py_module/feature_shared_resources_module.h"
#include "py_module/frame_module.h"
#include "py_module/geometry_matchers_module.h"
#include "py_module/keyframe_module.h"
#include "py_module/local_mapping_core_module.h"
#include "py_module/map_module.h"
#include "py_module/map_point_module.h"
#include "py_module/mutex_wrapper_module.h"
#include "py_module/optimizer_common_module.h"
#include "py_module/optimizer_g2o_module.h"
#include "py_module/optimizer_gtsam_module.h"
#include "py_module/rotation_histogram_module.h"
#include "py_module/sim3_pose_module.h"
#include "py_module/tracking_core_module.h"

#include "py_module/semantic_mapping_shared_resources_module.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "PYSLAM C++ Core Module - High-performance SLAM classes";

    // ------------------------------------------------------------
    // SLAM-specific opaque containers
    //
    // These containers can be optionally declared opaque in casters/slam_opaque_types.h
    // to avoid copying when passing between C++ and Python, and to enable
    // mutations in Python to be reflected in C++. However, it seems enabling opaque breaks smooth
    // interoperation between C++ and Python. At present, we leave it disabled
    // (ENABLE_OPAQUE_TYPES=0).
    //
    // KeyFramePtr and MapPointPtr vectors are heavily used in:
    // - Bundle adjustment and optimization functions
    // - Loop closure operations
    // - Map point projection and matching
    // - Return values from covisibility queries
    //
    // Note: If enabled, opaque types would provide significant performance benefits
    // for large vectors, but may break existing Python code that expects list-like behavior.

    py::bind_vector<std::vector<pyslam::KeyFramePtr>>(m, "KeyFramePtrVector");
    py::bind_vector<std::vector<pyslam::MapPointPtr>>(m, "MapPointPtrVector");
    py::bind_vector<std::vector<pyslam::FramePtr>>(m, "FramePtrVector");
    py::bind_vector<std::vector<int>>(m, "IntVector");
    py::bind_vector<std::vector<bool>>(m, "BoolVector");

    // ------------------------------------------------------------
    // Dictionary

    // Expose the containers (already declared opaque in dictionary_casters.h)
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
    // Eigen

    bind_eigen(m);

    // ------------------------------------------------------------
    // Camera and CameraPose

    bind_camera(m);

    // ------------------------------------------------------------
    // Sim3Pose class

    bind_sim3_pose(m);

    // ------------------------------------------------------------
    // FeatureSharedResources

    bind_feature_shared_resources(m);

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
    // MutexWrapper class

    bind_mutex_wrapper(m);

    // ------------------------------------------------------------
    // OptimizerCommon class

    bind_optimizer_common(m);

    // ------------------------------------------------------------
    // OptimizerG2o class

    bind_optimizer_g2o(m);

    // ------------------------------------------------------------
    // OptimizerGTSAM class

    bind_optimizer_gtsam(m);

    // ------------------------------------------------------------
    // TrackingCore class

    bind_tracking_core(m);

    // ------------------------------------------------------------
    // LocalMappingCore class

    bind_local_mapping_core(m);

    // ------------------------------------------------------------
    // Rotation histogram

    bind_rotation_histogram(m);

    // ------------------------------------------------------------
    // ConfigParameters class

    bind_config_parameters(m);

    // ------------------------------------------------------------
    // GeometryMatchers class

    bind_geometry_matchers(m);

    // ------------------------------------------------------------
    // SemanticMappingSharedResources class

    bind_semantic_mapping_shared_resources(m);

} // PYBIND11_MODULE
