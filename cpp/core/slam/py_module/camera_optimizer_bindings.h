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

// Optimizer terms of the cpp_core cameras. Each binding is compiled in its own translation
// unit, so g2o headers and gtsam headers are never included together.

#include <pybind11/pybind11.h>

void bind_camera_g2o_edges(pybind11::module &m);    // py_module/camera_g2o_module.cpp
void bind_camera_gtsam_factors(pybind11::module &m); // py_module/camera_gtsam_module.cpp
