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

#include "utils/messages.h"

namespace py = pybind11;

// Macro for static properties that need to extract value from Python object with error handling
#define DEF_STATIC_PROPERTY_WITH_ENUM_TO_INT_EXTRACTION(class_type, prop_name, enum_type)          \
    .def_property_static(                                                                          \
        #prop_name, [](py::object) -> int { return static_cast<int>(class_type::prop_name); },     \
        [](py::object, py::object prop_obj) {                                                      \
            int prop_value;                                                                        \
            try {                                                                                  \
                prop_value = prop_obj.attr("value").cast<int>();                                   \
            } catch (const py::cast_error &e) {                                                    \
                MSG_ERROR_STREAM(#class_type                                                       \
                                 << ": " << #prop_name                                             \
                                 << " cannot be casted to an integer: " << std::string(e.what())); \
                return;                                                                            \
            }                                                                                      \
            class_type::prop_name = static_cast<enum_type>(prop_value);                            \
        })
