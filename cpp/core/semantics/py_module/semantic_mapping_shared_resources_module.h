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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "semantic_colormap.h"
#include "semantic_mapping_shared_resources.h"
#include "utils/descriptor_helpers.h"
#include "utils/messages.h"
#include "utils/pybinding_helpers.h"

namespace py = pybind11;

void bind_semantic_mapping_shared_resources(py::module &m) {

    // ------------------------------------------------------------
    // SemanticColorMap

    py::class_<pyslam::SemanticColorMap, std::shared_ptr<pyslam::SemanticColorMap>>(
        m, "SemanticColorMap")
        .def(py::init<const pyslam::SemanticDatasetType &, int>())
        .def("get_color_map", &pyslam::SemanticColorMap::get_color_map)
        .def("get_labels", &pyslam::SemanticColorMap::get_labels)
        .def("get_information_weights", &pyslam::SemanticColorMap::get_information_weights)
        .def("to_rgb",
             static_cast<cv::Mat (pyslam::SemanticColorMap::*)(const cv::Mat &, bool) const>(
                 &pyslam::SemanticColorMap::to_rgb),
             py::arg("semantics"), py::arg("bgr") = false)
        .def("to_rgb",
             static_cast<cv::Mat (pyslam::SemanticColorMap::*)(
                 const cv::Mat &, bool, const std::vector<int> &, const cv::Mat &) const>(
                 &pyslam::SemanticColorMap::to_rgb),
             py::arg("semantics"), py::arg("bgr") = false, py::arg("ignore_labels"),
             py::arg("rgb_image"))
        .def("to_heatmap", &pyslam::SemanticColorMap::to_heatmap, py::arg("semantics"),
             py::arg("bgr") = false,
             py::arg("colormap") = pyslam::convert_cv2_colormap_type_to_int(cv::COLORMAP_JET),
             py::arg("sim_scale") = 1.0)
        .def("single_label_to_color", &pyslam::SemanticColorMap::single_label_to_color,
             py::arg("label"), py::arg("bgr") = false)
        .def("single_similarity_to_color", &pyslam::SemanticColorMap::single_similarity_to_color,
             py::arg("sim_value"), py::arg("bgr") = false,
             py::arg("colormap") = pyslam::convert_cv2_colormap_type_to_int(cv::COLORMAP_JET),
             py::arg("sim_scale") = 1.0)
        .def("get_dataset_type", &pyslam::SemanticColorMap::get_dataset_type)
        .def("get_num_classes", &pyslam::SemanticColorMap::get_num_classes)
        .def("supports_similarity", &pyslam::SemanticColorMap::supports_similarity);

    // ------------------------------------------------------------
    // SemanticMappingSharedResources

    // clang-format off
    py::class_<pyslam::SemanticMappingSharedResources,
               std::shared_ptr<pyslam::SemanticMappingSharedResources>>(m, "SemanticMappingSharedResources")
        DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(pyslam::SemanticMappingSharedResources,
                                                  semantic_feature_type,
                                                  pyslam::SemanticFeatureType)
        DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(pyslam::SemanticMappingSharedResources,
                                                  semantic_dataset_type,
                                                  pyslam::SemanticDatasetType)
        DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(pyslam::SemanticMappingSharedResources,
                                                  semantic_entity_type, pyslam::SemanticEntityType)
        DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(pyslam::SemanticMappingSharedResources,
                                                  semantic_segmentation_type,
                                                  pyslam::SemanticSegmentationType)
        .def_property_readonly_static(
            "semantic_color_map",
            [](py::object) -> std::shared_ptr<pyslam::SemanticColorMap> {
                return pyslam::SemanticMappingSharedResources::semantic_color_map;
            })
        .def_static("init_color_map", [](py::object semantic_dataset_type_obj,
                                            int num_classes) {
            pyslam::SemanticDatasetType semantic_dataset_type;
            try {
                semantic_dataset_type = static_cast<pyslam::SemanticDatasetType>(semantic_dataset_type_obj.attr("value").cast<int>());
            } catch (const std::exception &e) {
                MSG_ERROR("SemanticMappingSharedResources: Failed to convert semantic dataset type to int");
                throw py::value_error(e.what());
            }
            pyslam::SemanticMappingSharedResources::init_color_map(
                static_cast<pyslam::SemanticDatasetType>(semantic_dataset_type), num_classes);
        })
        .def_static("reset_color_map", &pyslam::SemanticMappingSharedResources::reset_color_map);
    // clang-format on
}
