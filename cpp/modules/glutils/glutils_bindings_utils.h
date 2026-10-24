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

#include "glutils_bindings_common.h"

#include <array>
#include <vector>

namespace glutils_bindings_detail {

template <typename PointT, int Flags>
inline void ValidatePointsArray(const py::array_t<PointT, Flags> &points) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }
}

template <typename ColorT, int Flags>
inline void ValidateColorsArray(const py::array_t<ColorT, Flags> &colors) {
    auto info = colors.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("colors must be an Nx3 array");
    }
}

template <typename IndexT, int Flags>
inline void ValidateTrianglesArray(const py::array_t<IndexT, Flags> &triangles) {
    auto info = triangles.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("triangles must be an Nx3 array");
    }
}

template <typename ScalarT>
inline const ScalarT *GetPackedVectorData(const std::vector<std::array<ScalarT, 3>> &values) {
    if (values.empty()) {
        return nullptr;
    }
    return values.front().data();
}

template <typename ScalarOutT, typename ScalarT>
inline const ScalarOutT *
GetPackedVectorDataConverted(const std::vector<std::array<ScalarT, 3>> &values,
                             std::vector<std::array<ScalarOutT, 3>> &converted_values) {
    if (values.empty()) {
        converted_values.clear();
        return nullptr;
    }
    converted_values.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        converted_values[i] = {static_cast<ScalarOutT>(values[i][0]),
                               static_cast<ScalarOutT>(values[i][1]),
                               static_cast<ScalarOutT>(values[i][2])};
    }
    return converted_values.front().data();
}

} // namespace glutils_bindings_detail
