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

#include "config_parameters.h"

namespace py = pybind11;

void bind_config_parameters(py::module &m) {

    // ------------------------------------------------------------
    // FeatureSharedResources - dynamic config parameters

    py::class_<pyslam::Parameters>(m, "Parameters")
        .def_readwrite_static("kFeatureMatchDefaultRatioTest",
                              &pyslam::Parameters::kFeatureMatchDefaultRatioTest)
        .def_readwrite_static("kMaxDescriptorDistance",
                              &pyslam::Parameters::kMaxDescriptorDistance);
}
