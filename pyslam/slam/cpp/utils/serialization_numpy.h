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

#include <cstring>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyslam {

static inline py::array cvmat_to_numpy(const cv::Mat &m) {
    if (m.empty())
        return py::array();
    CV_Assert(m.isContinuous());
    int ndim = (m.channels() == 1) ? 2 : 3;
    std::vector<ssize_t> shape, strides;
    if (ndim == 2) {
        shape = {m.rows, m.cols};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1]};
    } else {
        shape = {m.rows, m.cols, m.channels()};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1], (ssize_t)m.elemSize1()};
    }

    switch (m.depth()) {
    case CV_8U:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<uint8_t>::format().c_str(), ndim,
                                         shape, strides));
    case CV_8S:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<int8_t>::format().c_str(), ndim,
                                         shape, strides));
    case CV_16U:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<uint16_t>::format().c_str(), ndim,
                                         shape, strides));
    case CV_16S:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<int16_t>::format().c_str(), ndim,
                                         shape, strides));
    case CV_32S:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<int32_t>::format().c_str(), ndim,
                                         shape, strides));
    case CV_32F:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<float>::format().c_str(), ndim,
                                         shape, strides));
    case CV_64F:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<double>::format().c_str(), ndim,
                                         shape, strides));
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F:
        return py::array(py::buffer_info((void *)m.data, m.elemSize1(),
                                         py::format_descriptor<uint16_t>::format().c_str(), ndim,
                                         shape, strides));
#endif
    default:
        throw std::runtime_error("Unsupported cv::Mat depth: " + std::to_string(m.depth()));
    }
}

static inline cv::Mat numpy_to_cvmat(const py::array &a, int depth_hint) {
    if (!a || a.ndim() == 0)
        return {};
    auto info = a.request();

    // Handle empty arrays (zero-sized)
    if (info.size == 0) {
        return {};
    }

    int channels = (info.ndim == 3) ? (int)info.shape[2] : 1;
    int type = CV_MAKETYPE(depth_hint, channels);

    // Validate dimensions
    if (info.shape[0] < 0 || info.shape[1] < 0) {
        throw std::runtime_error(
            "Invalid numpy array dimensions: " + std::to_string(info.shape[0]) + "x" +
            std::to_string(info.shape[1]));
    }

    // Create a Mat with proper step calculation instead of using numpy strides
    cv::Mat m((int)info.shape[0], (int)info.shape[1], type);

    // Calculate total bytes to copy
    size_t total_bytes = info.shape[0] * info.shape[1] * channels * m.elemSize1();

    // Copy data from numpy array to cv::Mat
    switch (depth_hint) {
    case CV_8U:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_8S:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_16U:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_16S:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_32S:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_32F:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
    case CV_64F:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
#if CV_VERSION_MAJOR >= 4 && defined(CV_16F)
    case CV_16F:
        std::memcpy(m.data, info.ptr, total_bytes);
        break;
#endif
    default:
        throw std::runtime_error("Unsupported depth hint: " + std::to_string(depth_hint));
    }

    return m; // already owns memory
}

// Helper function to automatically detect depth from numpy dtype
static inline int numpy_dtype_to_cv_depth(const py::array &a) {
    if (!a || a.ndim() == 0)
        return CV_8U; // fallback

    auto info = a.request();
    int dtype_num = a.dtype().num();
    std::string format = py::str(a.dtype());
    size_t itemsize = a.dtype().itemsize();

    // Debug output (remove in production)
    std::cout << "Debug: dtype_num=" << dtype_num << ", format=" << format
              << ", itemsize=" << itemsize << std::endl;

    // Primary matching by dtype_num
    switch (dtype_num) {
    case py::detail::npy_api::NPY_UINT8_:
        return CV_8U;
    case py::detail::npy_api::NPY_INT8_:
        return CV_8S;
    case py::detail::npy_api::NPY_UINT16_:
        return CV_16U;
    case py::detail::npy_api::NPY_INT16_:
        return CV_16S;
    case py::detail::npy_api::NPY_INT32_:
        return CV_32S;
    case py::detail::npy_api::NPY_FLOAT_:
        return CV_32F;
    case py::detail::npy_api::NPY_DOUBLE_:
        return CV_64F;
    default:
        break; // Fall through to format-based matching
    }

    // Fallback: match by format character and itemsize
    if (format == "d" && itemsize == 8) { // double
        return CV_64F;
    } else if (format == "f" && itemsize == 4) { // float
        return CV_32F;
    } else if (format == "i" && itemsize == 4) { // int32
        return CV_32S;
    } else if (format == "h" && itemsize == 2) { // int16
        return CV_16S;
    } else if (format == "H" && itemsize == 2) { // uint16
        // Note: CV_16F is also stored as uint16, so we default to CV_16U
        // For CV_16F, explicit depth hint should be used
        return CV_16U;
    } else if (format == "b" && itemsize == 1) { // int8
        return CV_8S;
    } else if (format == "B" && itemsize == 1) { // uint8
        return CV_8U;
    }

    throw std::runtime_error("Unsupported numpy dtype: " + std::to_string(dtype_num) +
                             " format: " + format + " itemsize: " + std::to_string(itemsize));
}

// Convenience function that auto-detects depth
static inline cv::Mat numpy_to_cvmat_auto(const py::array &a) {
    int depth_hint = numpy_dtype_to_cv_depth(a);
    return numpy_to_cvmat(a, depth_hint);
}

} // namespace pyslam