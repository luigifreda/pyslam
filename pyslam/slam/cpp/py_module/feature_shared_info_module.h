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

#include "feature_shared_info.h"
#include "utils/descriptor_helpers.h"

namespace py = pybind11;

void bind_feature_shared_info(py::module &m) {

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
                return pyslam::convert_norm_type_to_cv2_norm_type(
                    pyslam::FeatureSharedInfo::norm_type);
            },
            [](py::object, int cv2_norm_type) {
                // Setter: convert cv2 norm type to NormType
                pyslam::FeatureSharedInfo::norm_type =
                    pyslam::convert_cv2_norm_type_to_norm_type(cv2_norm_type);
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
        .def_static(
            "set_feature_detect_and_compute_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedInfo::set_feature_detect_and_compute_callback({});
                } else {
                    pyslam::FeatureSharedInfo::set_feature_detect_and_compute_callback(
                        py::cast<pyslam::FeatureDetectAndComputeCallback>(cb));
                }
            },
            R"doc(Set/clear the left-image detect+compute callback. Pass None to clear.)doc")

        .def_static(
            "set_feature_detect_and_compute_right_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedInfo::set_feature_detect_and_compute_right_callback({});
                } else {
                    pyslam::FeatureSharedInfo::set_feature_detect_and_compute_right_callback(
                        py::cast<pyslam::FeatureDetectAndComputeCallback>(cb));
                }
            },
            R"doc(Set/clear the right-image detect+compute callback. Pass None to clear.)doc")

        .def_static(
            "set_stereo_matching_callback",
            [](py::object cb) {
                if (cb.is_none()) {
                    pyslam::FeatureSharedInfo::set_stereo_matching_callback({});
                } else {

                    pyslam::FeatureSharedInfo::set_stereo_matching_callback(
                        py::cast<pyslam::StereoMatchingCallback>(cb));
                }
            },
            R"doc(Set/clear the stereo matching callback. Pass None to clear.)doc")
        .def_static("clear_callbacks", &pyslam::FeatureSharedInfo::clear_callbacks);
}
