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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// CRITICAL: Include opaque bindings BEFORE other stl bindings
#include "volumetric_opaque_bindings.h"

#include "bounding_boxes_module.h"
#include "camera_frustrum_module.h"
#include "eigen_module.h"
#include "image_utils_module.h"
#include "volumetric_grid_module.h"
#include "voxel_grid_data_module.h"

PYBIND11_MODULE(volumetric, m) {
    m.doc() = "PYSLAM Volumetric Module - Volumetric Mapping";

    bind_volumetric_opaque_containers(m);

    // ----------------------------------------
    // TBB Utils
    // ----------------------------------------

    py::class_<volumetric::TBBUtils>(m, "TBBUtils")
        .def_static(
            "set_max_threads", &volumetric::TBBUtils::set_max_threads,
            "Set the maximum number of threads for TBB parallel operations (global setting)",
            py::arg("num_threads"))
        .def_static("get_max_threads", &volumetric::TBBUtils::get_max_threads,
                    "Get the current maximum number of "
                    "threads for TBB");

    bind_eigen(m);

    bind_bounding_boxes(m);

    bind_camera_frustrum(m);

    bind_voxel_grid_data(m);

    bind_volumetric_grid(m);

    bind_image_utils(m);
}