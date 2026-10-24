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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using DoubleArrayNoCopy = py::array_t<double, py::array::c_style>;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using FloatArrayNoCopy = py::array_t<float, py::array::c_style>;

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using IntArrayNoCopy = py::array_t<int, py::array::c_style>;

using UByteArray = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;
using UByteArrayNoCopy = py::array_t<unsigned char, py::array::c_style>;
