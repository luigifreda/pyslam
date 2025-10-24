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
#pragma once

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "camera.h"
#include "camera_pose.h"
#include "casters/nlohmann_json_type_casters.h"
#include "dictionary.h"
#include "utils/numpy_helpers.h"

namespace py = pybind11;

void bind_camera(py::module &m) {

    // Register enums FIRST, before they are used in class bindings
    py::enum_<pyslam::CameraType>(m, "CameraType")
        .value("NONE", pyslam::CameraType::NONE)
        .value("PINHOLE", pyslam::CameraType::PINHOLE);

    py::enum_<pyslam::SensorType>(m, "SensorType")
        .value("MONOCULAR", pyslam::SensorType::MONOCULAR)
        .value("STEREO", pyslam::SensorType::STEREO)
        .value("RGBD", pyslam::SensorType::RGBD);

    // ------------------------------------------------------------
    // Camera and CameraPose

    // CameraPose class
    py::class_<pyslam::CameraPose, std::shared_ptr<pyslam::CameraPose>>(m, "CameraPose")
        // Constructors
        .def(py::init<>())
        .def(py::init<const Eigen::Isometry3d &>())
        .def(py::init<const Eigen::Matrix4d &>())
        .def(py::init<const Eigen::Matrix3d &, const Eigen::Vector3d &>())
        .def(py::init<const Eigen::Quaterniond &, const Eigen::Vector3d &>())

        // readonly properties
        // clang-format off
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.Tcw(), "Tcw")
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.Rcw(), "Rcw")
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.tcw(), "tcw")
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.Rwc(), "Rwc")
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.Ow(), "Ow")
        DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(pyslam::CameraPose, self.covariance(), "covariance")
        // clang-format on                                

        // Core ops (unchanged)
        .def("copy", &pyslam::CameraPose::copy)
        .def("set", &pyslam::CameraPose::set, py::arg("pose"))
        .def("update", py::overload_cast<const Eigen::Isometry3d &>(&pyslam::CameraPose::update),
             py::arg("pose"))
        .def("update", py::overload_cast<const Eigen::Matrix4d &>(&pyslam::CameraPose::update),
             py::arg("Tcw"))
        .def("set_mat", &pyslam::CameraPose::set_mat, py::arg("Tcw"))
        .def("update_mat", &pyslam::CameraPose::update_mat, py::arg("Tcw"))

        // readonly derived quantities
        .def_property_readonly("isometry3d",
                               [](const pyslam::CameraPose &s) { return s.isometry3d(); })
        .def_property_readonly("quaternion",
                               [](const pyslam::CameraPose &s) { return s.quaternion(); })
        .def_property_readonly("orientation",
                               [](const pyslam::CameraPose &s) { return s.orientation(); })
        .def_property_readonly("position", [](const pyslam::CameraPose &s) { return s.position(); })

        // Utility getters (by value is fine)
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

        // Comparison / string / JSON (unchanged)
        .def("__eq__", &pyslam::CameraPose::operator==)
        .def("__ne__", &pyslam::CameraPose::operator!=)
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
                return pyslam::CameraUtils::project_points_stereo<double>(xcs, K, bf);
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
                return pyslam::CameraUtils::unproject_points_3d<double>(uvs, depths, Kinv);
            },
            py::arg("uvs"), py::arg("depths"), py::arg("Kinv"))
        .def_static(
            "backproject_3d", // same as unproject_points_3d (for Python compatibility)
            [](const pyslam::MatNx2dRef uvs, const pyslam::VecNdRef depths,
               const pyslam::Mat3dRef Kinv) {
                return pyslam::CameraUtils::unproject_points_3d<double>(uvs, depths, Kinv);
            },
            py::arg("uvs"), py::arg("depths"), py::arg("Kinv"))
        .def_static(
            "are_in_image",
            [](const pyslam::MatNxMdRef uvs, const pyslam::VecNdRef zs, double u_min, double u_max,
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
        .def_property(
            "K", [](const pyslam::Camera &self) { return self.K; }, // copy out
            [](pyslam::Camera &self, const Eigen::Matrix3d &val) { self.K = val; })
        .def_property(
            "Kinv", [](const pyslam::Camera &self) { return self.Kinv; }, // copy out
            [](pyslam::Camera &self, const Eigen::Matrix3d &val) { self.Kinv = val; })
        // Methods
        .def("set_intrinsic_matrices", &pyslam::Camera::set_intrinsic_matrices)
        .def(
            "project_point",
            [](const pyslam::Camera &self, pyslam::Vec3fRef xcs) {
                return self.project_point(xcs);
            },
            py::arg("xcs"))
        .def(
            "project_point",
            [](const pyslam::Camera &self, pyslam::Vec3dRef xcs) {
                return self.project_point(xcs);
            },
            py::arg("xcs"))
        .def(
            "project",
            [](const pyslam::Camera &self, pyslam::MatNx3fRef xcs) { return self.project(xcs); },
            py::arg("xcs"))
        .def(
            "project",
            [](const pyslam::Camera &self, pyslam::MatNx3dRef xcs) { return self.project(xcs); },
            py::arg("xcs"))
        .def(
            "project_point_stereo",
            [](const pyslam::Camera &self, pyslam::Vec3fRef xcs) {
                return self.project_point_stereo(xcs);
            },
            py::arg("xcs"))
        .def(
            "project_point_stereo",
            [](const pyslam::Camera &self, pyslam::Vec3dRef xcs) {
                return self.project_point_stereo(xcs);
            },
            py::arg("xcs"))
        .def(
            "project_stereo",
            [](const pyslam::Camera &self, pyslam::MatNx3fRef xcs) {
                return self.project_stereo(xcs);
            },
            py::arg("xcs"))
        .def(
            "project_stereo",
            [](const pyslam::Camera &self, pyslam::MatNx3dRef xcs) {
                return self.project_stereo(xcs);
            },
            py::arg("xcs"))
        .def(
            "undistort_points",
            [](pyslam::Camera &self, pyslam::MatNx2fRef uvs) {
                return self.undistort_points(uvs);
            },
            py::arg("uvs"))
        .def(
            "undistort_points",
            [](pyslam::Camera &self, pyslam::MatNx2dRef uvs) {
                return self.undistort_points(uvs);
            },
            py::arg("uvs"))
        .def("is_stereo", &pyslam::Camera::is_stereo)
        .def("to_json", [](const pyslam::PinholeCamera &self) {
            try {
                std::string json_str = self.to_json();
                nlohmann::json j = nlohmann::json::parse(json_str);
                return pyslam::json_to_dict(j);
            } catch (const std::exception &e) {
                throw std::runtime_error("Failed to convert camera to dict: " + std::string(e.what()));
            }
        })
        .def("init_from_json", &pyslam::Camera::init_from_json, py::arg("json_str"))
        .def(
            "is_in_image",
            [](const pyslam::Camera &self, pyslam::Vec2dRef uv, const double z) {
                return self.is_in_image<double>(uv, z);
            },
            py::arg("uv"), py::arg("z"))
        .def(
            "are_in_image",
            [](const pyslam::Camera &self, pyslam::MatNxMfRef uvs, pyslam::VecNfRef zs) {
                return self.are_in_image<float>(uvs, zs);
            },
            py::arg("uvs"), py::arg("zs"))
        .def(
            "are_in_image",
            [](const pyslam::Camera &self, pyslam::MatNxMdRef uvs, pyslam::VecNdRef zs) {
                return self.are_in_image<double>(uvs, zs);
            },
            py::arg("uvs"), py::arg("zs"))
        .def("get_render_projection_matrix", &pyslam::Camera::get_render_projection_matrix,
             py::arg("znear") = 0.01, py::arg("zfar") = 100.0)
        .def("set_fovx", &pyslam::Camera::set_fovx, py::arg("fovx"))
        .def("set_fovy", &pyslam::Camera::set_fovy, py::arg("fovy"))
        .def("to_json", &pyslam::Camera::to_json)
        .def("init_from_json", &pyslam::Camera::init_from_json, py::arg("json_str"))
        .def(py::pickle([](const pyslam::Camera &self) { return self.state_tuple(); },
        [](py::tuple t) {
            auto camera = std::make_shared<pyslam::Camera>();
            camera->restore_from_state(t);
            return camera;
        }))
        //.def("__setstate__", [](pyslam::Camera &self, py::tuple t) { self.restore_from_state(t); })
        .def("__getstate__", &pyslam::Camera::state_tuple);
    

    // PinholeCamera class
    py::class_<pyslam::PinholeCamera, pyslam::Camera, std::shared_ptr<pyslam::PinholeCamera>>(
        m, "PinholeCamera")
        .def(py::init([]() {
            return std::unique_ptr<pyslam::PinholeCamera>(
                new pyslam::PinholeCamera(pyslam::ConfigDict{}));
        }))
        .def(py::init([](py::object obj) {
                 if (obj.is_none()) {
                     return std::unique_ptr<pyslam::PinholeCamera>(
                         new pyslam::PinholeCamera(pyslam::ConfigDict{}));
                 } else if (py::isinstance<py::dict>(obj)) {
                     py::dict d = py::cast<py::dict>(obj);
                     pyslam::ConfigDict cfg;
                     for (auto item : d) {
                         std::string key = py::cast<std::string>(item.first);
                         pyslam::Value val = py::cast<pyslam::Value>(item.second);
                         cfg.emplace(std::move(key), std::move(val));
                     }
                     return std::unique_ptr<pyslam::PinholeCamera>(new pyslam::PinholeCamera(cfg));
                 } else {
                     // Check if it's a Config object by trying to call to_json()
                     try {
                         py::dict config_dict = obj.attr("to_json")();
                         pyslam::ConfigDict cfg;
                         for (auto item : config_dict) {
                             std::string key = py::cast<std::string>(item.first);
                             pyslam::Value val = py::cast<pyslam::Value>(item.second);
                             cfg.emplace(std::move(key), std::move(val));
                         }
                         return std::unique_ptr<pyslam::PinholeCamera>(
                             new pyslam::PinholeCamera(cfg));
                     } catch (const py::error_already_set &) {
                         throw std::runtime_error("PinholeCamera config must be None, a dict, or a "
                                                  "Config object with to_json() method");
                     }
                 }
             }),
             py::arg("config"))
        .def("init", &pyslam::PinholeCamera::init)
        .def("set_intrinsic_matrices", &pyslam::PinholeCamera::set_intrinsic_matrices)
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
            [](pyslam::PinholeCamera &self, pyslam::MatNx2dRef uvs) {
                return self.undistort_points(uvs);
            },
            py::arg("uvs"))
        .def("undistort_image_bounds", &pyslam::PinholeCamera::undistort_image_bounds)
        .def("to_json", [](const pyslam::PinholeCamera &self) {
            try {
                std::string json_str = self.to_json();
                nlohmann::json j = nlohmann::json::parse(json_str);
                return pyslam::json_to_dict(j);
            } catch (const std::exception &e) {
                throw std::runtime_error("Failed to convert camera to dict: " + std::string(e.what()));
            }
        })
        .def_static("from_json", &pyslam::PinholeCamera::from_json, py::arg("json_str"))
        .def(py::pickle([](const pyslam::PinholeCamera &self) { return self.state_tuple(); },
        [](py::tuple t) {
            auto pinhole_camera = std::make_shared<pyslam::PinholeCamera>();
            pinhole_camera->restore_from_state(t);
            return pinhole_camera;
        }))
        //.def("__setstate__", [](pyslam::PinholeCamera &self, py::tuple t) { self.restore_from_state(t); })
        .def("__getstate__", &pyslam::PinholeCamera::state_tuple);

} // bind_camera
