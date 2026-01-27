/**
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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "tbb_utils.h"
#include "voxel_block_grid.h"
#include "voxel_block_semantic_grid.h"
#include "voxel_grid.h"
#include "voxel_semantic_grid.h"

#include "image_utils.h"
#include "opencv_type_casters.h"

namespace py = pybind11;

// ----------------------------------------
// Helper functions for pybind11 bindings
// ----------------------------------------

void bind_image_utils(py::module &m) {
    m.def("check_image_size", &volumetric::check_image_size);
    m.def("convert_image_type_if_needed", &volumetric::convert_image_type_if_needed);
    m.def(
        "remap_instance_ids",
        [](const cv::Mat &instance_ids, volumetric::MapInstanceIdToObjectId &map,
           const int invalid_instance_id = -1) {
            if (instance_ids.empty()) {
                return instance_ids;
            }
            if (map.empty()) {
                return instance_ids;
            }

            // Check that it's single-channel
            if (instance_ids.channels() != 1) {
                throw std::runtime_error("Instance ids must be single-channel");
            }

            // Check the depth of the instance_ids and call the appropriate remap_instance_ids
            // function
            const int mat_depth = CV_MAT_DEPTH(instance_ids.type());
            if (mat_depth == CV_32S) {
                return volumetric::remap_instance_ids<volumetric::MapInstanceIdToObjectId, int32_t>(
                    instance_ids, map, invalid_instance_id);
            } else if (mat_depth == CV_8S) {
                return volumetric::remap_instance_ids<volumetric::MapInstanceIdToObjectId, int8_t>(
                    instance_ids, map, invalid_instance_id);
            } else if (mat_depth == CV_8U) {
                return volumetric::remap_instance_ids<volumetric::MapInstanceIdToObjectId, uint8_t>(
                    instance_ids, map, invalid_instance_id);
            } else if (mat_depth == CV_16S) {
                return volumetric::remap_instance_ids<volumetric::MapInstanceIdToObjectId, int16_t>(
                    instance_ids, map, invalid_instance_id);
            } else if (mat_depth == CV_16U) {
                return volumetric::remap_instance_ids<volumetric::MapInstanceIdToObjectId,
                                                      uint16_t>(instance_ids, map,
                                                                invalid_instance_id);
            } else {
                throw std::runtime_error("Unsupported instance id type");
            }
        },
        py::arg("instance_ids"), py::arg("map"), py::arg("invalid_instance_id") = -1);
}
