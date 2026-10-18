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

// gtsam factors of the cpp_core cameras. This translation unit does not include g2o.

#include "py_module/camera_optimizer_bindings.h"

#include "camera_factors_gtsam.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;

using pyslam::camera_model::python_noise_model;

void bind_camera_gtsam_factors(py::module &m) {
    py::module_::import("gtsam");
    py::class_<pyslam::StereoCameraProjectionFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<pyslam::StereoCameraProjectionFactor>>(m, "StereoCameraProjectionFactor")
        .def(py::init([](const gtsam::StereoPoint2 &measured,
                        const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                        gtsam::Key pose, gtsam::Key point, pyslam::CameraPtr camera) {
            return std::make_shared<pyslam::StereoCameraProjectionFactor>(measured, python_noise_model(noise), pose, point, camera);
        }))
        .def("set_weight", &pyslam::StereoCameraProjectionFactor::setWeight)
        .def("get_weight", &pyslam::StereoCameraProjectionFactor::getWeight);
    py::class_<pyslam::StereoCameraResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<pyslam::StereoCameraResectioningFactor>>(m, "StereoCameraResectioningFactor")
        .def(py::init([](const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                        gtsam::Key pose, pyslam::CameraPtr camera,
                        const gtsam::StereoPoint2 &measured, const gtsam::Point3 &point) {
            return std::make_shared<pyslam::StereoCameraResectioningFactor>(python_noise_model(noise), pose, camera, measured, point);
        }))
        .def("set_weight", &pyslam::StereoCameraResectioningFactor::setWeight)
        .def("get_weight", &pyslam::StereoCameraResectioningFactor::getWeight);
    py::class_<pyslam::CameraProjectionFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<pyslam::CameraProjectionFactor>>(m, "CameraProjectionFactor")
        .def(py::init([](const gtsam::Point2 &measured,
                        const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                        gtsam::Key pose, gtsam::Key point, pyslam::CameraPtr camera) {
            return std::make_shared<pyslam::CameraProjectionFactor>(measured, python_noise_model(noise), pose, point, camera);
        }))
        .def("set_weight", &pyslam::CameraProjectionFactor::setWeight)
        .def("get_weight", &pyslam::CameraProjectionFactor::getWeight);
    py::class_<pyslam::CameraResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<pyslam::CameraResectioningFactor>>(m, "CameraResectioningFactor")
        .def(py::init([](const std::shared_ptr<gtsam::noiseModel::Base> &noise,
                        gtsam::Key pose, pyslam::CameraPtr camera,
                        const gtsam::Point2 &measured, const gtsam::Point3 &point) {
            return std::make_shared<pyslam::CameraResectioningFactor>(python_noise_model(noise), pose, camera, measured, point);
        }))
        .def("set_weight", &pyslam::CameraResectioningFactor::setWeight)
        .def("get_weight", &pyslam::CameraResectioningFactor::getWeight);
}
