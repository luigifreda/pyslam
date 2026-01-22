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

#include "glutils_gl_includes.h"

namespace glutils {

template <typename T, int Flags>
std::array<double, kMatrixElementCount> ExtractPoseMatrix(const py::array_t<T, Flags> &pose) {
    auto info = pose.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::runtime_error("pose must be a 4x4 array");
    }

    const auto *src = static_cast<const T *>(info.ptr);
    std::array<double, kMatrixElementCount> matrix{};
    for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
        matrix[i] = static_cast<double>(src[i]);
    }
    return matrix;
}

inline std::size_t ComputeAlignment(const py::buffer_info &info) {
    const ssize_t row_stride = info.strides[0] >= 0 ? info.strides[0] : -info.strides[0];
    for (int align : {8, 4, 2, 1}) {
        if (row_stride % align == 0) {
            return static_cast<std::size_t>(align);
        }
    }
    return 1;
}

} // namespace glutils
