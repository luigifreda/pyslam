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

#include "camera_model_gtsam_factors.h"
#include "camera_model_python.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <gtsam/linear/NoiseModel.h>

#include <memory>
#include <stdexcept>

namespace py = pybind11;

using pyslam::camera_model::python_noise_model;

PYBIND11_MODULE(camera_models_gtsam, m) {
    m.doc() = "gtsam factors for camera_models cameras. This module does not include or load g2o.";
    py::module_::import("gtsam");
    py::module_::import("camera_models");

    py::class_<camera_models::CameraProjectionFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<camera_models::CameraProjectionFactor>>(m, "CameraProjectionFactor")
        .def(py::init([](const gtsam::Point2 &measured, const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                         gtsam::Key pose, gtsam::Key point, py::handle camera) {
            return std::make_shared<camera_models::CameraProjectionFactor>(
                measured, python_noise_model(noise), pose, point, owning_camera(camera));
        }))
        .def("set_weight", &camera_models::CameraProjectionFactor::setWeight)
        .def("get_weight", &camera_models::CameraProjectionFactor::getWeight);

    py::class_<camera_models::CameraResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<camera_models::CameraResectioningFactor>>(m, "CameraResectioningFactor")
        .def(py::init([](const std::shared_ptr<gtsam::noiseModel::Base> &noise, gtsam::Key pose,
                         py::handle camera, const gtsam::Point2 &measured,
                         const gtsam::Point3 &point) {
            return std::make_shared<camera_models::CameraResectioningFactor>(python_noise_model(noise), pose,
                                                                            owning_camera(camera), measured, point);
        }))
        .def("set_weight", &camera_models::CameraResectioningFactor::setWeight)
        .def("get_weight", &camera_models::CameraResectioningFactor::getWeight);

    py::class_<camera_models::StereoCameraProjectionFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<camera_models::StereoCameraProjectionFactor>>(m, "StereoCameraProjectionFactor")
        .def(py::init([](const gtsam::StereoPoint2 &measured, const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                         gtsam::Key pose, gtsam::Key point, py::handle camera) {
            return std::make_shared<camera_models::StereoCameraProjectionFactor>(
                measured, python_noise_model(noise), pose, point, owning_camera(camera));
        }))
        .def("set_weight", &camera_models::StereoCameraProjectionFactor::setWeight)
        .def("get_weight", &camera_models::StereoCameraProjectionFactor::getWeight);

    py::class_<camera_models::StereoCameraResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<camera_models::StereoCameraResectioningFactor>>(m, "StereoCameraResectioningFactor")
        .def(py::init([](const std::shared_ptr<gtsam::noiseModel::Base> &noise, gtsam::Key pose,
                         py::handle camera, const gtsam::StereoPoint2 &measured,
                         const gtsam::Point3 &point) {
            return std::make_shared<camera_models::StereoCameraResectioningFactor>(
                python_noise_model(noise), pose, owning_camera(camera), measured, point);
        }))
        .def("set_weight", &camera_models::StereoCameraResectioningFactor::setWeight)
        .def("get_weight", &camera_models::StereoCameraResectioningFactor::getWeight);
}
