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

#include "camera.h"
#include "camera_pose.h"
#include "feature_shared_info.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "semantic_types.h"

#include "casters/dictionary_casters.h"
#include "casters/opencv_type_casters.h"
#include "utils/dictionary.h"

namespace py = pybind11;

pyslam::NormType convert_cv2_norm_type_to_norm_type(const int &cv2_norm_type) {
    // Convert cv2 norm type to NormType
    switch (cv2_norm_type) {
    case cv::NORM_HAMMING:
        return pyslam::NormType::Hamming;
    case cv::NORM_L2:
        return pyslam::NormType::L2;
    default:
        // Default to L2 for unknown norm types
        return pyslam::NormType::None;
    }
}

int convert_norm_type_to_cv2_norm_type(const pyslam::NormType &norm_type) {
    // Convert NormType to cv2 norm type
    switch (norm_type) {
    case pyslam::NormType::Hamming:
        return cv::NORM_HAMMING;
    case pyslam::NormType::L2:
        return cv::NORM_L2;
    default:
        // Default to L2 for unknown norm types
        return cv::NORM_L2;
    }
}

PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "PYSLAM C++ Core Module - High-performance SLAM classes";

    // ------------------------------------------------------------
    // Dictionary

    // Expose the containers (optional but handy)
    py::bind_vector<pyslam::List>(m, "List");
    py::bind_map<pyslam::Dict>(m, "Dict");

    // Tiny demo function to show automatic conversion
    m.def("echo_dict", [](const pyslam::Dict &d) { return d; });

    // If you want to expose Value directly (optional)
    py::class_<pyslam::Value>(m, "Value")
        .def(py::init<>())
        .def_property_readonly("is_null", [](const pyslam::Value &v) {
            return std::holds_alternative<std::monostate>(v.data);
        });

    // ------------------------------------------------------------
    // SLAM Classes

    // Register enums FIRST, before they are used in class bindings
    py::enum_<pyslam::CameraType>(m, "CameraType")
        .value("NONE", pyslam::CameraType::NONE)
        .value("PINHOLE", pyslam::CameraType::PINHOLE);

    py::enum_<pyslam::SensorType>(m, "SensorType")
        .value("MONOCULAR", pyslam::SensorType::MONOCULAR)
        .value("STEREO", pyslam::SensorType::STEREO)
        .value("RGBD", pyslam::SensorType::RGBD);

    // ------------------------------------------------------------
    // FeatureSharedInfo

    py::class_<pyslam::FeatureSharedInfo>(m, "FeatureSharedInfo")
        .def_readwrite_static("scale_factor", &pyslam::FeatureSharedInfo::scale_factor)
        .def_readwrite_static("inv_scale_factor", &pyslam::FeatureSharedInfo::inv_scale_factor)
        .def_readwrite_static("log_scale_factor", &pyslam::FeatureSharedInfo::log_scale_factor)
        .def_readwrite_static("scale_factors", &pyslam::FeatureSharedInfo::scale_factors)
        .def_readwrite_static("inv_scale_factors", &pyslam::FeatureSharedInfo::inv_scale_factors)
        .def_readwrite_static("level_sigmas", &pyslam::FeatureSharedInfo::level_sigmas)
        .def_readwrite_static("level_sigmas2", &pyslam::FeatureSharedInfo::level_sigmas2)
        .def_readwrite_static("inv_level_sigmas2", &pyslam::FeatureSharedInfo::inv_level_sigmas2)
        .def_readwrite_static("num_levels", &pyslam::FeatureSharedInfo::num_levels)
        .def_readwrite_static("num_features", &pyslam::FeatureSharedInfo::num_features)
        .def_readwrite_static("detector_type", &pyslam::FeatureSharedInfo::detector_type)
        .def_readwrite_static("descriptor_type", &pyslam::FeatureSharedInfo::descriptor_type)
        .def_property_static(
            "norm_type",
            [](py::object) -> int {
                // Getter: convert NormType to cv2 norm type
                return convert_norm_type_to_cv2_norm_type(pyslam::FeatureSharedInfo::norm_type);
            },
            [](py::object, int cv2_norm_type) {
                // Setter: convert cv2 norm type to NormType
                pyslam::FeatureSharedInfo::norm_type =
                    convert_cv2_norm_type_to_norm_type(cv2_norm_type);
            })
        .def_property_static(
            "semantic_feature_type",
            [](py::object) -> int {
                return static_cast<int>(pyslam::FeatureSharedInfo::semantic_feature_type);
            },
            [](py::object, int semantic_feature_type) {
                pyslam::FeatureSharedInfo::semantic_feature_type =
                    static_cast<pyslam::SemanticFeatureType>(semantic_feature_type);
            })
        .def_static("set_feature_detect_and_compute_callback",
                    &pyslam::FeatureSharedInfo::set_feature_detect_and_compute_callback)
        .def_static("set_feature_detect_and_compute_right_callback",
                    &pyslam::FeatureSharedInfo::set_feature_detect_and_compute_right_callback)
        .def_static("set_stereo_matching_callback",
                    &pyslam::FeatureSharedInfo::set_stereo_matching_callback)
        .def_static("clear_callbacks", &pyslam::FeatureSharedInfo::clear_callbacks);

    // ------------------------------------------------------------
    // Camera and CameraPose

    // CameraPose class
    py::class_<pyslam::CameraPose, std::shared_ptr<pyslam::CameraPose>>(m, "CameraPose")
        // Constructors
        .def(py::init<>())
        .def(py::init<const Eigen::Isometry3d &>())
        // Convenience constructor from 4x4 matrix
        .def(py::init([](const Eigen::Matrix4d &Tcw) {
                 return pyslam::CameraPose(Eigen::Isometry3d(Tcw));
             }),
             py::arg("Tcw"))
        // Read-write accessors
        .def_property(
            "Tcw", [](pyslam::CameraPose &self) -> Eigen::Matrix4d & { return self.Tcw(); },
            [](pyslam::CameraPose &self, const Eigen::Matrix4d &value) { self.Tcw() = value; })
        .def_property(
            "Rcw", [](pyslam::CameraPose &self) -> Eigen::Matrix3d & { return self.Rcw(); },
            [](pyslam::CameraPose &self, const Eigen::Matrix3d &value) { self.Rcw() = value; })
        .def_property(
            "tcw", [](pyslam::CameraPose &self) -> Eigen::Vector3d & { return self.tcw(); },
            [](pyslam::CameraPose &self, const Eigen::Vector3d &value) { self.tcw() = value; })
        .def_property(
            "Rwc", [](pyslam::CameraPose &self) -> Eigen::Matrix3d & { return self.Rwc(); },
            [](pyslam::CameraPose &self, const Eigen::Matrix3d &value) { self.Rwc() = value; })
        .def_property(
            "Ow", [](pyslam::CameraPose &self) -> Eigen::Vector3d & { return self.Ow(); },
            [](pyslam::CameraPose &self, const Eigen::Vector3d &value) { self.Ow() = value; })
        .def_property(
            "covariance",
            [](pyslam::CameraPose &self) -> Eigen::Matrix<double, 6, 6> & {
                return self.covariance();
            },
            [](pyslam::CameraPose &self, const Eigen::Matrix<double, 6, 6> &value) {
                self.covariance() = value;
            })
        // Core ops
        .def("copy", &pyslam::CameraPose::copy)
        .def("set", &pyslam::CameraPose::set, py::arg("pose"))
        .def("update", py::overload_cast<const Eigen::Isometry3d &>(&pyslam::CameraPose::update),
             py::arg("pose"))
        .def("update", py::overload_cast<const Eigen::Matrix4d &>(&pyslam::CameraPose::update),
             py::arg("Tcw"))
        .def("set_mat", &pyslam::CameraPose::set_mat, py::arg("Tcw"))
        .def("update_mat", &pyslam::CameraPose::update_mat, py::arg("Tcw"))
        // Properties
        .def_property_readonly("isometry3d", &pyslam::CameraPose::isometry3d,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("quaternion", &pyslam::CameraPose::quaternion)
        .def_property_readonly("orientation", &pyslam::CameraPose::orientation)
        .def_property_readonly("position", &pyslam::CameraPose::position)
        // Utility getters
        .def("get_matrix", &pyslam::CameraPose::get_matrix)
        .def("get_inverse_matrix", &pyslam::CameraPose::get_inverse_matrix)
        .def("get_rotation_matrix", &pyslam::CameraPose::get_rotation_matrix)
        .def("get_inverse_rotation_matrix", &pyslam::CameraPose::get_inverse_rotation_matrix)
        // Setters
        .def("set_from_quaternion_and_position",
             &pyslam::CameraPose::set_from_quaternion_and_position, py::arg("quaternion"),
             py::arg("position"))
        .def("set_from_matrix", &pyslam::CameraPose::set_from_matrix, py::arg("Tcw"))
        .def("set_from_rotation_and_translation",
             &pyslam::CameraPose::set_from_rotation_and_translation, py::arg("Rcw"), py::arg("tcw"))
        .def("set_quaternion", &pyslam::CameraPose::set_quaternion, py::arg("quaternion"))
        .def("set_rotation_matrix", &pyslam::CameraPose::set_rotation_matrix, py::arg("Rcw"))
        .def("set_translation", &pyslam::CameraPose::set_translation, py::arg("tcw"))
        // Comparison
        .def("__eq__", &pyslam::CameraPose::operator==)
        .def("__ne__", &pyslam::CameraPose::operator!=)
        // String / JSON
        .def("to_string", &pyslam::CameraPose::to_string)
        .def("to_json", &pyslam::CameraPose::to_json)
        .def_static("from_json", &pyslam::CameraPose::from_json);

    // ------------------------------------------------------------
    // CameraUtils class
    py::class_<pyslam::CameraUtils>(m, "CameraUtils")
        .def_static(
            "project",
            [](pyslam::MatNx3dRef xcs, pyslam::Mat3dRef K) {
                return pyslam::CameraUtils::project_points(xcs, K);
            },
            py::arg("xcs"), py::arg("K"))
        .def_static(
            "project_stereo",
            [](const pyslam::MatNx3dRef xcs, const Eigen::Matrix3d &K, double bf) {
                return pyslam::CameraUtils::project_points_stereo(xcs, K, bf);
            },
            py::arg("xcs"), py::arg("K"), py::arg("bf"))
        .def_static(
            "unproject_points",
            [](const pyslam::MatNx2dRef uvs, const pyslam::Mat3dRef Kinv) {
                return pyslam::CameraUtils::unproject_points(uvs, Kinv);
            },
            py::arg("uvs"), py::arg("Kinv"))
        .def_static(
            "unproject_points_3d",
            [](const pyslam::MatNx2dRef uvs, const pyslam::VecNdRef depths,
               const pyslam::Mat3dRef Kinv) {
                return pyslam::CameraUtils::unproject_points_3d(uvs, Kinv, depths);
            },
            py::arg("uvs"), py::arg("depths"), py::arg("Kinv"))
        .def_static(
            "backproject_3d", // same as unproject_points_3d (for Python compatibility)
            [](const pyslam::MatNx2dRef uvs, const pyslam::VecNdRef depths,
               const pyslam::Mat3dRef Kinv) {
                return pyslam::CameraUtils::unproject_points_3d(uvs, Kinv, depths);
            },
            py::arg("uvs"), py::arg("depths"), py::arg("Kinv"))
        .def_static(
            "are_in_image",
            [](const pyslam::MatNx2dRef uvs, const pyslam::VecNdRef zs, double u_min, double u_max,
               double v_min, double v_max) {
                return pyslam::CameraUtils::are_in_image(uvs, zs, u_min, u_max, v_min, v_max);
            },
            py::arg("uvs"), py::arg("zs"), py::arg("u_min"), py::arg("u_max"), py::arg("v_min"),
            py::arg("v_max"));

    // ------------------------------------------------------------
    // CameraBase class
    py::class_<pyslam::CameraBase, std::shared_ptr<pyslam::CameraBase>>(m, "CameraBase")
        .def(py::init<>())
        // Intrinsics and size
        .def_readwrite("type", &pyslam::CameraBase::type)
        .def_readwrite("width", &pyslam::CameraBase::width)
        .def_readwrite("height", &pyslam::CameraBase::height)
        .def_readwrite("fx", &pyslam::CameraBase::fx)
        .def_readwrite("fy", &pyslam::CameraBase::fy)
        .def_readwrite("cx", &pyslam::CameraBase::cx)
        .def_readwrite("cy", &pyslam::CameraBase::cy)
        .def_readwrite("D", &pyslam::CameraBase::D)
        .def_readwrite("is_distorted", &pyslam::CameraBase::is_distorted)
        .def_readwrite("fps", &pyslam::CameraBase::fps)
        .def_readwrite("bf", &pyslam::CameraBase::bf)
        .def_readwrite("b", &pyslam::CameraBase::b)
        .def_readwrite("u_min", &pyslam::CameraBase::u_min)
        .def_readwrite("u_max", &pyslam::CameraBase::u_max)
        .def_readwrite("v_min", &pyslam::CameraBase::v_min)
        .def_readwrite("v_max", &pyslam::CameraBase::v_max)
        .def_readwrite("initialized", &pyslam::CameraBase::initialized);

    // Camera class
    py::class_<pyslam::Camera, pyslam::CameraBase, std::shared_ptr<pyslam::Camera>>(m, "Camera")
        // Constructors from Python dict or empty
        .def(py::init([]() {
            return std::unique_ptr<pyslam::Camera>(new pyslam::Camera(pyslam::ConfigDict{}));
        }))
        .def(py::init([](py::dict d) {
                 pyslam::ConfigDict cfg;
                 for (auto item : d) {
                     std::string key = py::cast<std::string>(item.first);
                     pyslam::Value val = py::cast<pyslam::Value>(item.second);
                     cfg.emplace(std::move(key), std::move(val));
                 }
                 return std::unique_ptr<pyslam::Camera>(new pyslam::Camera(cfg));
             }),
             py::arg("config"))
        // Additional parameters
        .def_readwrite("fovx", &pyslam::Camera::fovx)
        .def_readwrite("fovy", &pyslam::Camera::fovy)
        .def_readwrite("sensor_type", &pyslam::Camera::sensor_type)
        .def_readwrite("depth_factor", &pyslam::Camera::depth_factor)
        .def_readwrite("depth_threshold", &pyslam::Camera::depth_threshold)
        .def_readwrite("K", &pyslam::Camera::K)
        .def_readwrite("Kinv", &pyslam::Camera::Kinv)
        // Methods
        .def("is_stereo", &pyslam::Camera::is_stereo)
        .def("to_json", &pyslam::Camera::to_json)
        .def("init_from_json", &pyslam::Camera::init_from_json, py::arg("json_str"))
        .def("is_in_image", &pyslam::Camera::is_in_image, py::arg("uv"), py::arg("z"))
        .def("are_in_image", &pyslam::Camera::are_in_image, py::arg("uvs"), py::arg("zs"))
        .def("get_render_projection_matrix", &pyslam::Camera::get_render_projection_matrix,
             py::arg("znear") = 0.01, py::arg("zfar") = 100.0)
        .def("set_fovx", &pyslam::Camera::set_fovx, py::arg("fovx"))
        .def("set_fovy", &pyslam::Camera::set_fovy, py::arg("fovy"))
        .def("to_json", &pyslam::Camera::to_json)
        .def("init_from_json", &pyslam::Camera::init_from_json, py::arg("json_str"));

    // PinholeCamera class
    py::class_<pyslam::PinholeCamera, pyslam::Camera, std::shared_ptr<pyslam::PinholeCamera>>(
        m, "PinholeCamera")
        .def(py::init([]() {
            return std::unique_ptr<pyslam::PinholeCamera>(
                new pyslam::PinholeCamera(pyslam::ConfigDict{}));
        }))
        .def(py::init([](py::dict d) {
                 pyslam::ConfigDict cfg;
                 for (auto item : d) {
                     std::string key = py::cast<std::string>(item.first);
                     pyslam::Value val = py::cast<pyslam::Value>(item.second);
                     cfg.emplace(std::move(key), std::move(val));
                 }
                 return std::unique_ptr<pyslam::PinholeCamera>(new pyslam::PinholeCamera(cfg));
             }),
             py::arg("config"))
        .def("init", &pyslam::PinholeCamera::init)
        .def(
            "project",
            [](const pyslam::PinholeCamera &self, pyslam::MatNx3dRef xcs) {
                return self.project(xcs);
            },
            py::arg("xcs"))
        .def(
            "project_stereo",
            [](const pyslam::PinholeCamera &self, pyslam::MatNx3dRef xcs) {
                return self.project_stereo(xcs);
            },
            py::arg("xcs"))
        .def(
            "unproject",
            [](const pyslam::PinholeCamera &self, const Eigen::Vector2d &uv) {
                return self.unproject_point(uv(0), uv(1));
            },
            py::arg("uv"))
        .def(
            "unproject_3d",
            [](const pyslam::PinholeCamera &self, const Eigen::Vector2d &uv, const double depth) {
                return self.unproject_point_3d(uv(0), uv(1), depth);
            },
            py::arg("uv"), py::arg("depth"))
        .def(
            "unproject_points",
            [](const pyslam::PinholeCamera &self, pyslam::MatNx2dRef uvs) {
                return self.unproject_points(uvs);
            },
            py::arg("uvs"))
        .def(
            "unproject_points_3d",
            [](const pyslam::PinholeCamera &self, pyslam::MatNx2dRef uvs, pyslam::VecNdRef depths) {
                return self.unproject_points_3d(uvs, depths);
            },
            py::arg("uvs"), py::arg("depths"))
        .def(
            "undistort_points",
            [](const pyslam::PinholeCamera &self, pyslam::MatNx2dRef uvs) {
                return self.undistort_points(uvs);
            },
            py::arg("uvs"))
        .def("undistort_image_bounds", &pyslam::PinholeCamera::undistort_image_bounds)
        .def("to_json", &pyslam::PinholeCamera::to_json)
        .def_static("from_json", &pyslam::PinholeCamera::from_json, py::arg("json_str"));

    // ------------------------------------------------------------
    // ReloadedSessionMapInfo class
    py::class_<pyslam::ReloadedSessionMapInfo, std::shared_ptr<pyslam::ReloadedSessionMapInfo>>(
        m, "ReloadedSessionMapInfo")
        .def(py::init<int, int, int, int, int>(), py::arg("nkf") = 0, py::arg("np") = 0,
             py::arg("mpid") = 0, py::arg("mfid") = 0, py::arg("mkfid") = 0)
        .def_readwrite("num_keyframes", &pyslam::ReloadedSessionMapInfo::num_keyframes)
        .def_readwrite("num_points", &pyslam::ReloadedSessionMapInfo::num_points)
        .def_readwrite("max_point_id", &pyslam::ReloadedSessionMapInfo::max_point_id)
        .def_readwrite("max_frame_id", &pyslam::ReloadedSessionMapInfo::max_frame_id)
        .def_readwrite("max_keyframe_id", &pyslam::ReloadedSessionMapInfo::max_keyframe_id);

    // ------------------------------------------------------------
    // MapPointBase class - matches Python MapPointBase exactly
    py::class_<pyslam::MapPointBase, std::shared_ptr<pyslam::MapPointBase>>(m, "MapPointBase")
        .def(py::init<int>(), py::arg("id") = -1)
        .def_readwrite("id", &pyslam::MapPointBase::id)
        .def_readwrite("map", &pyslam::MapPointBase::map)
        .def_readwrite("_observations", &pyslam::MapPointBase::_observations)
        .def_readwrite("_frame_views", &pyslam::MapPointBase::_frame_views)
        .def_readwrite("_is_bad", &pyslam::MapPointBase::_is_bad)
        .def_readwrite("_num_observations", &pyslam::MapPointBase::_num_observations)
        .def_readwrite("num_times_visible", &pyslam::MapPointBase::num_times_visible)
        .def_readwrite("num_times_found", &pyslam::MapPointBase::num_times_found)
        .def_readwrite("last_frame_id_seen", &pyslam::MapPointBase::last_frame_id_seen)
        .def_readwrite("replacement", &pyslam::MapPointBase::replacement)
        .def_readwrite("corrected_by_kf", &pyslam::MapPointBase::corrected_by_kf)
        .def_readwrite("corrected_reference", &pyslam::MapPointBase::corrected_reference)
        .def_readwrite("kf_ref", &pyslam::MapPointBase::kf_ref)
        .def("observations", &pyslam::MapPointBase::observations)
        .def("observations_iter", &pyslam::MapPointBase::observations_iter)
        .def("keyframes", &pyslam::MapPointBase::keyframes)
        .def("keyframes_iter", &pyslam::MapPointBase::keyframes_iter)
        .def("is_in_keyframe", &pyslam::MapPointBase::is_in_keyframe)
        .def("get_observation_idx", &pyslam::MapPointBase::get_observation_idx)
        .def("add_observation", &pyslam::MapPointBase::add_observation)
        .def("add_observation_if_not_bad", &pyslam::MapPointBase::add_observation_if_not_bad)
        .def("remove_observation", &pyslam::MapPointBase::remove_observation)
        .def("frame_views", &pyslam::MapPointBase::frame_views)
        .def("frame_views_iter", &pyslam::MapPointBase::frame_views_iter)
        .def("frames", &pyslam::MapPointBase::frames)
        .def("frames_iter", &pyslam::MapPointBase::frames_iter)
        .def("is_in_frame", &pyslam::MapPointBase::is_in_frame)
        .def("add_frame_view", &pyslam::MapPointBase::add_frame_view)
        .def("remove_frame_view", &pyslam::MapPointBase::remove_frame_view)
        .def_property_readonly("is_bad",
                               &pyslam::MapPointBase::is_bad) // matches Python MapPointBase.is_bad
        .def("num_observations", &pyslam::MapPointBase::num_observations)
        .def("is_good_with_min_obs", &pyslam::MapPointBase::is_good_with_min_obs)
        .def("is_bad_and_is_good_with_min_obs",
             &pyslam::MapPointBase::is_bad_and_is_good_with_min_obs)
        .def("is_bad_or_is_in_keyframe", &pyslam::MapPointBase::is_bad_or_is_in_keyframe)
        .def("increase_visible", &pyslam::MapPointBase::increase_visible)
        .def("increase_found", &pyslam::MapPointBase::increase_found)
        .def("get_found_ratio", &pyslam::MapPointBase::get_found_ratio)
        .def("observations_string", &pyslam::MapPointBase::observations_string)
        .def("frame_views_string", &pyslam::MapPointBase::frame_views_string)
        .def("to_string", &pyslam::MapPointBase::to_string)
        .def("__eq__", &pyslam::MapPointBase::operator==)
        .def("__lt__", &pyslam::MapPointBase::operator<)
        .def("__le__", &pyslam::MapPointBase::operator<=)
        .def("__hash__", &pyslam::MapPointBase::hash);

    // MapPoint class - matches Python MapPoint exactly
    py::class_<pyslam::MapPoint, pyslam::MapPointBase, std::shared_ptr<pyslam::MapPoint>>(
        m, "MapPoint")
        .def(py::init<const Eigen::Vector3d &, const Eigen::Matrix<unsigned char, 3, 1> &,
                      pyslam::KeyFrame *, int, int>(),
             py::arg("position"), py::arg("color"), py::arg("keyframe") = nullptr,
             py::arg("idxf") = -1, py::arg("id") = -1)
        .def_readwrite("_pt", &pyslam::MapPoint::_pt)
        .def_readwrite("normal", &pyslam::MapPoint::normal)
        .def_readwrite("_min_distance", &pyslam::MapPoint::_min_distance)
        .def_readwrite("_max_distance", &pyslam::MapPoint::_max_distance)
        .def_readwrite("color", &pyslam::MapPoint::color)
        .def_readwrite("semantic_des", &pyslam::MapPoint::semantic_des)
        .def_readwrite("des", &pyslam::MapPoint::des)
        .def_readwrite("first_kid", &pyslam::MapPoint::first_kid)
        .def_readwrite("num_observations_on_last_update_des",
                       &pyslam::MapPoint::num_observations_on_last_update_des)
        .def_readwrite("num_observations_on_last_update_normals",
                       &pyslam::MapPoint::num_observations_on_last_update_normals)
        .def_readwrite("num_observations_on_last_update_semantics",
                       &pyslam::MapPoint::num_observations_on_last_update_semantics)
        .def_readwrite("pt_GBA", &pyslam::MapPoint::pt_GBA)
        .def_readwrite("GBA_kf_id", &pyslam::MapPoint::GBA_kf_id)
        .def_property_readonly("pt", &pyslam::MapPoint::pt) // matches Python MapPoint.pt
        .def("homogeneous", &pyslam::MapPoint::homogeneous)
        .def("update_position", &pyslam::MapPoint::update_position)
        .def("min_distance", &pyslam::MapPoint::min_distance)
        .def("max_distance", &pyslam::MapPoint::max_distance)
        .def("get_all_pos_info", &pyslam::MapPoint::get_all_pos_info)
        .def("get_reference_keyframe", &pyslam::MapPoint::get_reference_keyframe)
        .def("descriptors", &pyslam::MapPoint::descriptors)
        .def("min_des_distance", &pyslam::MapPoint::min_des_distance)
        .def("delete_point", &pyslam::MapPoint::delete_point)
        .def("set_bad", &pyslam::MapPoint::set_bad)
        .def("get_replacement", &pyslam::MapPoint::get_replacement)
        .def("get_normal", &pyslam::MapPoint::get_normal)
        .def("replace_with", &pyslam::MapPoint::replace_with)
        .def("update_normal_and_depth", &pyslam::MapPoint::update_normal_and_depth)
        .def("update_best_descriptor", &pyslam::MapPoint::update_best_descriptor)
        .def("update_semantics", &pyslam::MapPoint::update_semantics)
        .def("update_info", &pyslam::MapPoint::update_info)
        .def("predict_detection_level", &pyslam::MapPoint::predict_detection_level)
        .def("to_json", &pyslam::MapPoint::to_json)
        .def_static("from_json", &pyslam::MapPoint::from_json)
        .def("replace_ids_with_objects", &pyslam::MapPoint::replace_ids_with_objects)
        .def("set_pt_GBA", &pyslam::MapPoint::set_pt_GBA)
        .def("set_GBA_kf_id", &pyslam::MapPoint::set_GBA_kf_id)
        .def(py::pickle(
            // __getstate__
            [](const pyslam::MapPoint &self) { return self.state_tuple(); },
            // __setstate__
            [](py::tuple t) {
                // Create MapPoint with default values - the restore_from_state will overwrite them
                const Eigen::Vector3d default_position(0, 0, 0);
                const Eigen::Matrix<unsigned char, 3, 1> default_color(255, 255, 255);
                auto mp = std::make_shared<pyslam::MapPoint>(default_position, default_color,
                                                             nullptr, -1, -1);
                mp->restore_from_state(t);
                return mp;
            }))
        // (optional) also expose the dunders for debugging/tests:
        .def("__getstate__", &pyslam::MapPoint::state_tuple)
        .def("__setstate__",
             [](pyslam::MapPoint &self, py::tuple t) { self.restore_from_state(t); });

    // ------------------------------------------------------------
    // FrameBase class - matches Python FrameBase exactly
    py::class_<pyslam::FrameBase, std::shared_ptr<pyslam::FrameBase>>(m, "FrameBase")
        .def(py::init<pyslam::Camera *, pyslam::CameraPose *, int, double, int>(),
             py::arg("camera") = nullptr, py::arg("pose") = nullptr, py::arg("id") = -1,
             py::arg("timestamp") = 0.0, py::arg("img_id") = -1)
        .def_readwrite("id", &pyslam::FrameBase::id)
        .def_readwrite("timestamp", &pyslam::FrameBase::timestamp)
        .def_readwrite("img_id", &pyslam::FrameBase::img_id)
        .def_readwrite("camera", &pyslam::FrameBase::camera)
        .def_property(
            "pose",
            [](const pyslam::FrameBase &self) -> const pyslam::CameraPose & { return *self._pose; },
            [](pyslam::FrameBase &self, const pyslam::CameraPose &pose) {
                self._pose = std::make_unique<pyslam::CameraPose>(pose);
            })
        .def("width", &pyslam::FrameBase::width)
        .def("height", &pyslam::FrameBase::height)
        .def("isometry3d", &pyslam::FrameBase::isometry3d)
        .def("Tcw", &pyslam::FrameBase::Tcw)
        .def("Twc", &pyslam::FrameBase::Twc)
        .def("Rcw", &pyslam::FrameBase::Rcw)
        .def("Rwc", &pyslam::FrameBase::Rwc)
        .def("tcw", &pyslam::FrameBase::tcw)
        .def("Ow", &pyslam::FrameBase::Ow)
        .def("quaternion", &pyslam::FrameBase::quaternion)
        .def("orientation", &pyslam::FrameBase::orientation)
        .def("position", &pyslam::FrameBase::position)
        .def("update_pose", &pyslam::FrameBase::update_pose)
        .def("update_translation", &pyslam::FrameBase::update_translation)
        .def("update_rotation_and_translation", &pyslam::FrameBase::update_rotation_and_translation)
        .def("__eq__", &pyslam::FrameBase::operator==)
        .def("__lt__", &pyslam::FrameBase::operator<)
        .def("__le__", &pyslam::FrameBase::operator<=)
        .def("__hash__", &pyslam::FrameBase::hash);

    // Frame class - matches Python Frame exactly
    py::class_<pyslam::Frame, pyslam::FrameBase, std::shared_ptr<pyslam::Frame>>(m, "Frame")
        .def(py::init<pyslam::Camera *, const cv::Mat &, const cv::Mat &, const cv::Mat &,
                      pyslam::CameraPose *, int, double, int, const cv::Mat &,
                      const std::map<std::string, void *> &>(),
             py::arg("camera"), py::arg("img") = cv::Mat(), py::arg("img_right") = cv::Mat(),
             py::arg("depth") = cv::Mat(), py::arg("pose") = nullptr, py::arg("id") = -1,
             py::arg("timestamp") = 0.0, py::arg("img_id") = -1,
             py::arg("semantic_img") = cv::Mat(),
             py::arg("frame_data_dict") = std::map<std::string, void *>{})
        .def_readwrite("is_keyframe", &pyslam::Frame::is_keyframe)
        .def_readwrite("kps", &pyslam::Frame::kps)
        .def_readwrite("kps_r", &pyslam::Frame::kps_r)
        .def_readwrite("kpsu", &pyslam::Frame::kpsu)
        .def_readwrite("kpsn", &pyslam::Frame::kpsn)
        .def_readwrite("kps_sem", &pyslam::Frame::kps_sem)
        .def_readwrite("octaves", &pyslam::Frame::octaves)
        .def_readwrite("octaves_r", &pyslam::Frame::octaves_r)
        .def_readwrite("sizes", &pyslam::Frame::sizes)
        .def_readwrite("angles", &pyslam::Frame::angles)
        .def_readwrite("des", &pyslam::Frame::des)
        .def_readwrite("des_r", &pyslam::Frame::des_r)
        .def_readwrite("depths", &pyslam::Frame::depths)
        .def_readwrite("kps_ur", &pyslam::Frame::kps_ur)
        .def_readwrite("points", &pyslam::Frame::points)
        .def_readwrite("outliers", &pyslam::Frame::outliers)
        .def_readwrite("kf_ref", &pyslam::Frame::kf_ref)
        .def_readwrite("img", &pyslam::Frame::img)
        .def_readwrite("img_right", &pyslam::Frame::img_right)
        .def_readwrite("depth_img", &pyslam::Frame::depth_img)
        .def_readwrite("semantic_img", &pyslam::Frame::semantic_img)
        .def_readwrite("is_blurry", &pyslam::Frame::is_blurry)
        .def_readwrite("laplacian_var", &pyslam::Frame::laplacian_var)
        .def_readwrite("fov_center_c", &pyslam::Frame::fov_center_c)
        .def_readwrite("fov_center_w", &pyslam::Frame::fov_center_w)
        .def("project_point", &pyslam::Frame::project_point)
        .def("project_points", &pyslam::Frame::project_points, py::arg("points"),
             py::arg("do_stereo_project") = false)
        .def("unproject_points_3d", &pyslam::Frame::unproject_points_3d)
        .def("are_visible", &pyslam::Frame::are_visible)
        .def("is_visible", &pyslam::Frame::is_visible)
        .def("get_point_match", &pyslam::Frame::get_point_match)
        .def("set_point_match", &pyslam::Frame::set_point_match)
        .def("remove_point_match", &pyslam::Frame::remove_point_match)
        .def("remove_point", &pyslam::Frame::remove_point)
        .def("get_matched_points", &pyslam::Frame::get_matched_points)
        .def("get_matched_good_points", &pyslam::Frame::get_matched_good_points)
        .def("compute_stereo_from_rgbd", &pyslam::Frame::compute_stereo_from_rgbd)
        .def("compute_stereo_matches", &pyslam::Frame::compute_stereo_matches)
        .def("set_depth_img", &pyslam::Frame::set_depth_img)
        .def("ensure_contiguous_arrays", &pyslam::Frame::ensure_contiguous_arrays)
        .def("__eq__", &pyslam::Frame::operator==)
        .def("__lt__", &pyslam::Frame::operator<)
        .def("__le__", &pyslam::Frame::operator<=)
        .def("__hash__", &pyslam::Frame::hash)
        .def("to_json", &pyslam::Frame::to_json)
        .def_static("from_json", &pyslam::Frame::from_json)
        .def(py::pickle(
            // __getstate__
            [](const pyslam::Frame &self) { return self.state_tuple(); },
            // __setstate__
            [](py::tuple t) {
                auto frame = std::make_shared<pyslam::Frame>(nullptr);
                frame->restore_from_state(t);
                return frame;
            }))
        // (optional) also expose the dunders for debugging/tests:
        .def("__getstate__", &pyslam::Frame::state_tuple)
        .def("__setstate__", [](pyslam::Frame &self, py::tuple t) { self.restore_from_state(t); });

    // ------------------------------------------------------------
    // KeyFrameGraph class - matches Python KeyFrameGraph
    py::class_<pyslam::KeyFrameGraph, std::shared_ptr<pyslam::KeyFrameGraph>>(m, "KeyFrameGraph")
        .def(py::init<>())
        .def_readwrite("init_parent", &pyslam::KeyFrameGraph::init_parent)
        .def_readwrite("parent", &pyslam::KeyFrameGraph::parent)
        .def_readwrite("children", &pyslam::KeyFrameGraph::children)
        .def_readwrite("loop_edges", &pyslam::KeyFrameGraph::loop_edges)
        .def_readwrite("not_to_erase", &pyslam::KeyFrameGraph::not_to_erase)
        .def_readwrite("connected_keyframes_weights",
                       &pyslam::KeyFrameGraph::connected_keyframes_weights)
        .def_readwrite("ordered_keyframes_weights",
                       &pyslam::KeyFrameGraph::ordered_keyframes_weights)
        .def_readwrite("is_first_connection", &pyslam::KeyFrameGraph::is_first_connection)
        .def("reset_covisibility", &pyslam::KeyFrameGraph::reset_covisibility)
        .def("get_connected_keyframes", &pyslam::KeyFrameGraph::get_connected_keyframes)
        .def("get_covisible_keyframes", &pyslam::KeyFrameGraph::get_covisible_keyframes)
        .def("get_best_covisible_keyframes", &pyslam::KeyFrameGraph::get_best_covisible_keyframes)
        .def("get_children", &pyslam::KeyFrameGraph::get_children)
        .def("get_parent", &pyslam::KeyFrameGraph::get_parent)
        .def("add_child", &pyslam::KeyFrameGraph::add_child)
        .def("erase_child", &pyslam::KeyFrameGraph::erase_child)
        .def("set_parent", &pyslam::KeyFrameGraph::set_parent)
        .def("has_child", &pyslam::KeyFrameGraph::has_child)
        .def("get_loop_edges", &pyslam::KeyFrameGraph::get_loop_edges)
        .def("add_connection", &pyslam::KeyFrameGraph::add_connection)
        .def("erase_connection", &pyslam::KeyFrameGraph::erase_connection)
        .def("update_best_covisibles", &pyslam::KeyFrameGraph::update_best_covisibles)
        .def("add_loop_edge", &pyslam::KeyFrameGraph::add_loop_edge)
        .def("get_weight", &pyslam::KeyFrameGraph::get_weight);

    // KeyFrame class - matches Python KeyFrame
    py::class_<pyslam::KeyFrame, pyslam::Frame, pyslam::KeyFrameGraph,
               std::shared_ptr<pyslam::KeyFrame>>(m, "KeyFrame")
        .def(py::init<pyslam::Frame *, const cv::Mat &, const cv::Mat &, const cv::Mat &, int>(),
             py::arg("frame"), py::arg("img") = cv::Mat(), py::arg("img_right") = cv::Mat(),
             py::arg("depth") = cv::Mat(), py::arg("kid") = -1)
        .def_readwrite("kid", &pyslam::KeyFrame::kid)
        .def_readwrite("_is_bad", &pyslam::KeyFrame::_is_bad)
        .def_readwrite("lba_count", &pyslam::KeyFrame::lba_count)
        .def_readwrite("_pose_Tcp", &pyslam::KeyFrame::_pose_Tcp)
        .def_readwrite("g_des", &pyslam::KeyFrame::g_des)
        .def_readwrite("loop_query_id", &pyslam::KeyFrame::loop_query_id)
        .def_readwrite("num_loop_words", &pyslam::KeyFrame::num_loop_words)
        .def_readwrite("loop_score", &pyslam::KeyFrame::loop_score)
        .def_readwrite("reloc_query_id", &pyslam::KeyFrame::reloc_query_id)
        .def_readwrite("num_reloc_words", &pyslam::KeyFrame::num_reloc_words)
        .def_readwrite("reloc_score", &pyslam::KeyFrame::reloc_score)
        .def_readwrite("GBA_kf_id", &pyslam::KeyFrame::GBA_kf_id)
        .def_readwrite("Tcw_GBA", &pyslam::KeyFrame::Tcw_GBA)
        .def_readwrite("Tcw_before_GBA", &pyslam::KeyFrame::Tcw_before_GBA)
        .def_readwrite("map", &pyslam::KeyFrame::map)
        .def("set_bad", &pyslam::KeyFrame::set_bad)
        .def("get_matched_points", &pyslam::KeyFrame::get_matched_points)
        .def("get_matched_good_points", &pyslam::KeyFrame::get_matched_good_points)
        .def("update_connections", &pyslam::KeyFrame::update_connections)
        .def("add_connection", &pyslam::KeyFrame::add_connection)
        .def("__eq__", &pyslam::KeyFrame::operator==)
        .def("__lt__", &pyslam::KeyFrame::operator<)
        .def("__le__", &pyslam::KeyFrame::operator<=)
        .def("__hash__", &pyslam::KeyFrame::hash)
        .def(py::pickle(
            // __getstate__
            [](const pyslam::KeyFrame &self) { return self.state_tuple(); },
            // __setstate__
            [](py::tuple t) {
                auto keyframe = std::make_shared<pyslam::KeyFrame>(nullptr);
                keyframe->restore_from_state(t);
                return keyframe;
            }))
        .def("__getstate__", &pyslam::KeyFrame::state_tuple)
        .def("__setstate__",
             [](pyslam::KeyFrame &self, py::tuple t) { self.restore_from_state(t); });

    // ------------------------------------------------------------
    // Map class - complete interface matching Python Map
    py::class_<pyslam::Map, std::shared_ptr<pyslam::Map>>(m, "Map")
        .def(py::init<>())

        // Core data structures (read/write access)
        .def_readwrite("frames", &pyslam::Map::frames)
        .def_readwrite("keyframes", &pyslam::Map::keyframes)
        .def_readwrite("points", &pyslam::Map::points)
        .def_readwrite("keyframe_origins", &pyslam::Map::keyframe_origins)
        .def_readwrite("keyframes_map", &pyslam::Map::keyframes_map)

        // ID counters
        .def_readwrite("max_point_id", &pyslam::Map::max_point_id)
        .def_readwrite("max_frame_id", &pyslam::Map::max_frame_id)
        .def_readwrite("max_keyframe_id", &pyslam::Map::max_keyframe_id)

        // Session info
        // Property that ties the lifetime of the returned pointer to `Map`
        .def_property(
            "reloaded_session_map_info",
            [](pyslam::Map &m) -> pyslam::ReloadedSessionMapInfo * {
                return m.reloaded_session_map_info.get(); // non-owning
            },
            [](pyslam::Map &m, pyslam::ReloadedSessionMapInfo *v) {
                if (v)
                    m.reloaded_session_map_info = std::make_unique<pyslam::ReloadedSessionMapInfo>(
                        *v); // copy into unique_ptr
                else
                    m.reloaded_session_map_info.reset(); // allow Python None
            },
            py::return_value_policy::reference_internal)

        // Local map
        .def_property_readonly(
            "local_map",
            [](const pyslam::Map &m) -> const pyslam::LocalCovisibilityMap * {
                return m.local_map.get();
            },
            py::return_value_policy::reference_internal)

        // Viewer scale
        .def_readwrite("viewer_scale", &pyslam::Map::viewer_scale)

        // Core operations
        .def("reset", &pyslam::Map::reset)
        .def("reset_session", &pyslam::Map::reset_session)
        .def("delete", &pyslam::Map::delete_map)
        .def("delete_map", &pyslam::Map::delete_map)

        // Point operations
        .def("get_points", &pyslam::Map::get_points)
        .def("num_points", &pyslam::Map::num_points)
        .def("add_point", &pyslam::Map::add_point)
        .def("remove_point", &pyslam::Map::remove_point)

        // Frame operations
        .def("get_frame", &pyslam::Map::get_frame, py::arg("idx"))
        .def("get_frames", &pyslam::Map::get_frames)
        .def("num_frames", &pyslam::Map::num_frames)
        .def("add_frame", &pyslam::Map::add_frame, py::arg("frame"), py::arg("override_id") = false)
        .def("remove_frame", &pyslam::Map::remove_frame)

        // KeyFrame operations
        .def("get_keyframes", &pyslam::Map::get_keyframes)
        .def("get_last_keyframe", &pyslam::Map::get_last_keyframe)
        .def("get_last_keyframes", &pyslam::Map::get_last_keyframes, py::arg("local_window") = 5)
        .def("num_keyframes", &pyslam::Map::num_keyframes)
        .def("num_keyframes_session", &pyslam::Map::num_keyframes_session)
        .def("add_keyframe", &pyslam::Map::add_keyframe)
        .def("remove_keyframe", &pyslam::Map::remove_keyframe)

        // Visualization
        .def("draw_feature_trails", &pyslam::Map::draw_feature_trails)

        // Point management
        .def("add_points", &pyslam::Map::add_points, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("kf1"), py::arg("kf2"), py::arg("idxs1"),
             py::arg("idxs2"), py::arg("img"), py::arg("do_check") = true,
             py::arg("cos_max_parallax") = 0.9998, py::arg("far_points_threshold") = py::none())
        .def("add_stereo_points", &pyslam::Map::add_stereo_points, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("f"), py::arg("kf"), py::arg("idxs"),
             py::arg("img"))

        // Point filtering
        .def("remove_points_with_big_reproj_err", &pyslam::Map::remove_points_with_big_reproj_err)
        .def("compute_mean_reproj_error", &pyslam::Map::compute_mean_reproj_error,
             py::arg("points") = std::vector<pyslam::MapPoint *>{})

        // Optimization
        .def("optimize", &pyslam::Map::optimize, py::arg("num_iterations") = 10)
        .def("locally_optimize", &pyslam::Map::locally_optimize, py::arg("kf_ref"),
             py::arg("num_iterations") = 5)

        // Serialization
        .def("to_json", &pyslam::Map::to_json, py::arg("out_json") = "{}")
        .def("serialize", &pyslam::Map::serialize)
        .def("from_json", &pyslam::Map::from_json)
        .def("deserialize", &pyslam::Map::deserialize)
        .def("save", &pyslam::Map::save)
        .def("load", &pyslam::Map::load)

        // Session management
        .def("is_reloaded", &pyslam::Map::is_reloaded)
        .def("set_reloaded_session_info", &pyslam::Map::set_reloaded_session_info)
        .def("get_reloaded_session_info", &pyslam::Map::get_reloaded_session_info);

    // LocalMapBase class - complete interface matching Python LocalMapBase
    py::class_<pyslam::LocalMapBase>(m, "LocalMapBase")
        .def(py::init<pyslam::Map *>(), py::arg("map") = nullptr)

        // Core data structures
        .def_readwrite("keyframes", &pyslam::LocalMapBase::keyframes)
        .def_readwrite("points", &pyslam::LocalMapBase::points)
        .def_readwrite("ref_keyframes", &pyslam::LocalMapBase::ref_keyframes)
        .def_readwrite("map", &pyslam::LocalMapBase::map)

        // Lock property
        .def_property_readonly(
            "lock", [](pyslam::LocalMapBase &self) -> std::mutex & { return self.lock(); })

        // Core operations
        .def("reset", &pyslam::LocalMapBase::reset)
        .def("reset_session", &pyslam::LocalMapBase::reset_session,
             py::arg("keyframes_to_remove") = std::vector<pyslam::KeyFrame *>{},
             py::arg("points_to_remove") = std::vector<pyslam::MapPoint *>{})

        // Status
        .def("is_empty", &pyslam::LocalMapBase::is_empty)

        // Access methods
        .def("get_points", &pyslam::LocalMapBase::get_points)
        .def("num_points", &pyslam::LocalMapBase::num_points)
        .def("get_keyframes", &pyslam::LocalMapBase::get_keyframes)
        .def("num_keyframes", &pyslam::LocalMapBase::num_keyframes)

        // Update methods
        .def("update_from_keyframes", &pyslam::LocalMapBase::update_from_keyframes)
        .def("get_frame_covisibles", &pyslam::LocalMapBase::get_frame_covisibles);

    // LocalWindowMap class - complete interface matching Python LocalWindowMap
    py::class_<pyslam::LocalWindowMap, pyslam::LocalMapBase>(m, "LocalWindowMap")
        .def(py::init<pyslam::Map *, int>(), py::arg("map") = nullptr, py::arg("local_window") = 5)

        // Local window size
        .def_readwrite("local_window", &pyslam::LocalWindowMap::local_window)

        // Update methods
        .def("update_keyframes", &pyslam::LocalWindowMap::update_keyframes,
             py::arg("kf_ref") = nullptr)
        .def("get_best_neighbors", &pyslam::LocalWindowMap::get_best_neighbors,
             py::arg("kf_ref") = nullptr, py::arg("N") = 20)
        .def("update", &pyslam::LocalWindowMap::update, py::arg("kf_ref") = nullptr);

    // LocalCovisibilityMap class - complete interface matching Python
    py::class_<pyslam::LocalCovisibilityMap, pyslam::LocalMapBase>(m, "LocalCovisibilityMap")
        .def(py::init<pyslam::Map *>(), py::arg("map") = nullptr)

        // Update methods
        .def("update_keyframes", &pyslam::LocalCovisibilityMap::update_keyframes)
        .def("get_best_neighbors", &pyslam::LocalCovisibilityMap::get_best_neighbors,
             py::arg("kf_ref"), py::arg("N") = 20)
        .def("update", &pyslam::LocalCovisibilityMap::update);

} // PYBIND11_MODULE
