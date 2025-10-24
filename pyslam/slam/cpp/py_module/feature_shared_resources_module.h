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

#include "feature_shared_resources.h"
#include "utils/descriptor_helpers.h"
#include "utils/messages.h"
#include "utils/pybinding_helpers.h"

namespace py = pybind11;

void bind_feature_shared_resources(py::module &m) {

    // ------------------------------------------------------------
    // FeatureSharedResources

    py::class_<pyslam::FeatureSharedResources>(m, "FeatureSharedResources")
        .def_readwrite_static("scale_factor", &pyslam::FeatureSharedResources::scale_factor)
        .def_readwrite_static("inv_scale_factor", &pyslam::FeatureSharedResources::inv_scale_factor)
        .def_readwrite_static("log_scale_factor", &pyslam::FeatureSharedResources::log_scale_factor)
        .def_readwrite_static("scale_factors", &pyslam::FeatureSharedResources::scale_factors)
        .def_readwrite_static("inv_scale_factors",
                              &pyslam::FeatureSharedResources::inv_scale_factors)
        .def_readwrite_static("level_sigmas", &pyslam::FeatureSharedResources::level_sigmas)
        .def_readwrite_static("level_sigmas2", &pyslam::FeatureSharedResources::level_sigmas2)
        .def_readwrite_static("inv_level_sigmas2",
                              &pyslam::FeatureSharedResources::inv_level_sigmas2)
        .def_readwrite_static("num_levels", &pyslam::FeatureSharedResources::num_levels)
        .def_readwrite_static("num_features", &pyslam::FeatureSharedResources::num_features)
        .def_readwrite_static("detector_type", &pyslam::FeatureSharedResources::detector_type)
        .def_readwrite_static("descriptor_type", &pyslam::FeatureSharedResources::descriptor_type)
        .def_readwrite_static("feature_match_ratio_test",
                              &pyslam::FeatureSharedResources::feature_match_ratio_test)
        .def_property_static(
            "norm_type",
            [](py::object) -> int {
                // Getter: convert NormType to cv2 norm type
                return pyslam::convert_norm_type_to_cv2_norm_type(
                    pyslam::FeatureSharedResources::norm_type);
            },
            [](py::object, int cv2_norm_type) {
                // Setter: convert cv2 norm type to NormType
                pyslam::FeatureSharedResources::norm_type =
                    pyslam::convert_cv2_norm_type_to_norm_type(cv2_norm_type);
            })
        // .def_property_static(
        //     "semantic_feature_type",
        //     [](py::object) -> int {
        //         return static_cast<int>(pyslam::FeatureSharedResources::semantic_feature_type);
        //     },
        //     [](py::object, int semantic_feature_type) {
        //         pyslam::FeatureSharedResources::semantic_feature_type =
        //             static_cast<pyslam::SemanticFeatureType>(semantic_feature_type);
        //     })
        DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(
            pyslam::FeatureSharedResources, semantic_feature_type, pyslam::SemanticFeatureType)
        .def_static(
            "set_feature_detect_and_compute_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedResources::set_feature_detect_and_compute_callback({});
                } else {
                    pyslam::FeatureSharedResources::set_feature_detect_and_compute_callback(
                        py::cast<pyslam::FeatureDetectAndComputeCallback>(cb));
                }
            },
            R"doc(Set/clear the left-image detect+compute callback. Pass None to clear.)doc")

        .def_static(
            "set_feature_detect_and_compute_right_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedResources::set_feature_detect_and_compute_right_callback(
                        {});
                } else {
                    pyslam::FeatureSharedResources::set_feature_detect_and_compute_right_callback(
                        py::cast<pyslam::FeatureDetectAndComputeCallback>(cb));
                }
            },
            R"doc(Set/clear the right-image detect+compute callback. Pass None to clear.)doc")

        .def_static(
            "set_stereo_matching_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedResources::set_stereo_matching_callback({});
                } else {

                    pyslam::FeatureSharedResources::set_stereo_matching_callback(
                        py::cast<pyslam::FeatureMatchingCallback>(cb));
                }
            },
            R"doc(Set/clear the stereo matching callback. Pass None to clear.)doc")

        .def_static(
            "set_feature_matching_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedResources::set_feature_matching_callback({});
                } else {
                    pyslam::FeatureSharedResources::set_feature_matching_callback(
                        py::cast<pyslam::FeatureMatchingCallback>(cb));
                }
            },
            R"doc(Set/clear the feature matching callback. Pass None to clear.)doc")

        .def_static("clear_callbacks", &pyslam::FeatureSharedResources::clear_callbacks);
}
