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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include <opencv2/core/core.hpp>

#include "../casters/opencv_type_casters.h"
#include "color_utils.h"

namespace py = pybind11;

PYBIND11_MODULE(color_utils, m) {
    // Optional module docstring
    m.doc() = "pybind11 plugin for color_utils module";

    // Bind IdsColorTable class
    py::class_<pyslam::IdsColorTable, std::shared_ptr<pyslam::IdsColorTable>>(m, "IdsColorTable")
        .def(py::init<>())
        .def("ids_to_rgb", &pyslam::IdsColorTable::ids_to_rgb, py::arg("ids"),
             py::arg("bgr") = false, py::arg("unlabeled_color") = cv::Vec3b(0, 0, 0),
             py::arg("use_hash") = false, "Convert IDs to RGB colors using hash-based color table.")
        .def("ids_to_rgb_float", &pyslam::IdsColorTable::ids_to_rgb_float, py::arg("ids"),
             py::arg("bgr") = false, py::arg("unlabeled_color") = cv::Vec3b(0, 0, 0),
             py::arg("use_hash") = false, "Convert IDs to RGB colors as float32 in [0, 1].")
        .def("get_color_from_id", &pyslam::IdsColorTable::get_color_from_id, py::arg("id_val"),
             "Get the color from the color LUT using an hash ID.")
        .def("get_color_from_id_array", &pyslam::IdsColorTable::get_color_from_id_array,
             py::arg("id_val"), "Get the color from the color LUT using an hash ID as an array.");
}
