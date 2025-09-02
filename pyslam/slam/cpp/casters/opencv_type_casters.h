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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;
using namespace pybind11::literals;

// void declareCvTypes(py::module & m)   // just experiemental
// {
//     // N.B. this produces orbslam2_features.KeyPoint which are identical to cv2.KeyPoint but
//     cannot be used as cv2.KeyPoint;
//     //      at the present time, we use the converter in opencv_type_casters.h!
//     py::class_<cv::KeyPoint>(m, "KeyPoint")
//         .def(py::init<cv::Point2f, float, float, float, int,
//         int>(),"_pt"_a,"_size"_a,"_angle"_a=-1,"_response"_a=0,"_octave"_a=0,"_class_id"_a=-1)
//         .def(py::init<float, float,float, float, float, int,
//         int>(),"x"_a,"y"_a,"_size"_a,"_angle"_a=-1,"_response"_a=0,"_octave"_a=0,"_class_id"_a=-1)
//         .def_readwrite("pt", &cv::KeyPoint::pt)
//         .def_readwrite("size", &cv::KeyPoint::size)
//         .def_readwrite("angle", &cv::KeyPoint::angle)
//         .def_readwrite("response", &cv::KeyPoint::response)
//         .def_readwrite("octave", &cv::KeyPoint::octave)
//         .def_readwrite("class_id", &cv::KeyPoint::class_id);

// }
namespace pybind11 {
namespace detail {

// -----------------------------
// Helpers: NumPy dtype <-> cv depth
// -----------------------------
inline int numpy_to_cv_depth(const py::dtype &dt) {
    if (dt.is(py::dtype::of<uint8_t>()))
        return CV_8U;
    if (dt.is(py::dtype::of<int8_t>()))
        return CV_8S;
    if (dt.is(py::dtype::of<uint16_t>()))
        return CV_16U;
    if (dt.is(py::dtype::of<int16_t>()))
        return CV_16S;
    if (dt.is(py::dtype::of<int32_t>()))
        return CV_32S;
    if (dt.is(py::dtype::of<float>()))
        return CV_32F;
    if (dt.is(py::dtype::of<double>()))
        return CV_64F;
    throw std::logic_error(
        "Unsupported NumPy dtype: expected one of [uint8,int8,uint16,int16,int32,float32,float64]");
}

inline py::dtype cv_depth_to_numpy_dtype(int depth) {
    switch (depth) {
    case CV_8U:
        return py::dtype::of<uint8_t>();
    case CV_8S:
        return py::dtype::of<int8_t>();
    case CV_16U:
        return py::dtype::of<uint16_t>();
    case CV_16S:
        return py::dtype::of<int16_t>();
    case CV_32S:
        return py::dtype::of<int32_t>();
    case CV_32F:
        return py::dtype::of<float>();
    case CV_64F:
        return py::dtype::of<double>();
    default:
        throw std::logic_error("Unsupported cv::Mat depth.");
    }
}

// -----------------------------
// cv::Mat <-> numpy.ndarray
// -----------------------------

template <> struct type_caster<cv::Mat> {
  public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    // NumPy -> cv::Mat
    bool load(handle src, bool) {
        // Force C-order and cast dtype if needed
        py::array arr = py::array::ensure(src, py::array::c_style | py::array::forcecast);
        if (!arr)
            return false;

        py::buffer_info info = arr.request();
        if (info.ndim < 2 || info.ndim > 3)
            throw std::logic_error("Expected 2D or 3D array shaped (H,W[,C]).");

        const int H = static_cast<int>(info.shape[0]);
        const int W = static_cast<int>(info.shape[1]);
        int C = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
        if (C <= 0)
            C = 1;

        const int depth = numpy_to_cv_depth(py::dtype(info));
        const int type = CV_MAKETYPE(depth, C);

        // With ensure(c_style), strides are standard C-order.
        value = cv::Mat(H, W, type, info.ptr);
        // IMPORTANT: If you need to store this Mat, clone() it on the C++ side.
        return true;
    }

    // cv::Mat -> NumPy
    static handle cast(const cv::Mat &mat, return_value_policy, handle) {
        if (mat.empty())
            return py::array(py::dtype::of<uint8_t>(), {0, 0}).release();

        // Ensure contiguous buffer
        cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();

        const int depth = contiguous.depth();
        const int channels = contiguous.channels();
        py::dtype dt = cv_depth_to_numpy_dtype(depth);

        std::vector<ssize_t> shape, strides;
        const ssize_t elem = static_cast<ssize_t>(dt.itemsize());

        if (channels == 1) {
            shape = {contiguous.rows, contiguous.cols};
            strides = {static_cast<ssize_t>(contiguous.cols) * elem, elem};
        } else {
            shape = {contiguous.rows, contiguous.cols, channels};
            strides = {static_cast<ssize_t>(contiguous.cols) * channels * elem,
                       static_cast<ssize_t>(channels) * elem, elem};
        }

        // Own the copied buffer via a capsule
        cv::Mat *owner = new cv::Mat(std::move(contiguous));
        auto capsule = py::capsule(owner, [](void *p) { delete reinterpret_cast<cv::Mat *>(p); });
        return py::array(dt, shape, strides, owner->data, capsule).release();
    }
};

// -----------------------------
// cv::Vec<T,N> <-> numpy.ndarray (1D)
// -----------------------------

template <typename T, int N> struct type_caster<cv::Vec<T, N>> {
    using Vec = cv::Vec<T, N>;
    PYBIND11_TYPE_CASTER(Vec, _("numpy.ndarray"));

    bool load(handle src, bool) {
        py::array arr = py::array::ensure(src, py::array::c_style | py::array::forcecast);
        if (!arr)
            return false;
        py::buffer_info info = arr.request();
        if (info.ndim != 1 || info.shape[0] != N)
            return false;
        if (!py::dtype(info).is(py::dtype::of<T>()))
            return false;
        T *ptr = static_cast<T *>(info.ptr);
        for (int i = 0; i < N; ++i)
            value[i] = ptr[i];
        return true;
    }

    static handle cast(const Vec &v, return_value_policy, handle) {
        auto dt = py::dtype::of<T>();
        return py::array(dt, {N}, {}, v.val).release();
    }
};

// -----------------------------
// std::vector<cv::Mat> <-> Python list
// -----------------------------

template <> struct type_caster<std::vector<cv::Mat>> {
    PYBIND11_TYPE_CASTER(std::vector<cv::Mat>, _("List[numpy.ndarray]"));

    bool load(handle src, bool) {
        if (!py::isinstance<py::sequence>(src))
            return false;
        py::sequence seq = py::reinterpret_borrow<py::sequence>(src);
        value.clear();
        value.reserve(seq.size());
        for (auto item : seq) {
            type_caster<cv::Mat> sub;
            if (!sub.load(item, true))
                return false;
            cv::Mat m = cast_op<cv::Mat &>(sub);
            value.emplace_back(std::move(m));
        }
        return true;
    }

    static handle cast(const std::vector<cv::Mat> &vec, return_value_policy p, handle parent) {
        py::list out;
        for (const auto &m : vec)
            out.append(type_caster<cv::Mat>::cast(m, p, parent));
        return out.release();
    }
};

// -----------------------------
// cv::Point (int) <-> tuple(x, y)
// -----------------------------

template <> struct type_caster<cv::Point> {
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xi_yi"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj))
            throw std::logic_error("Point(x,y) should be a tuple!");
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
            throw std::logic_error("Point(x,y) tuple should be size of 2");
        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point &pt, return_value_policy, handle) {
        return py::make_tuple(pt.x, pt.y).release();
    }
};

// -----------------------------
// cv::Point2f <-> tuple(x, y)
// -----------------------------

template <> struct type_caster<cv::Point2f> {
    PYBIND11_TYPE_CASTER(cv::Point2f, _("tuple_xf_yf"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj))
            throw std::logic_error("Point2f(x,y) should be a tuple!");
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
            throw std::logic_error("Point2f(x,y) tuple should be size of 2");
        value = cv::Point2f(pt[0].cast<float>(), pt[1].cast<float>());
        return true;
    }

    static handle cast(const cv::Point2f &pt, return_value_policy, handle) {
        return py::make_tuple(pt.x, pt.y).release();
    }
};

// -----------------------------
// cv::Rect_<T> <-> tuple(x, y, w, h) for int(float)
// -----------------------------

template <typename T> struct type_caster<cv::Rect_<T>> {
    using RectT = cv::Rect_<T>;
    PYBIND11_TYPE_CASTER(RectT, _("tuple_x_y_w_h"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj))
            throw std::logic_error("Rect should be a tuple!");
        py::tuple rect = reinterpret_borrow<py::tuple>(obj);
        if (rect.size() != 4)
            throw std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
        value = RectT(rect[0].cast<T>(), rect[1].cast<T>(), rect[2].cast<T>(), rect[3].cast<T>());
        return true;
    }

    static handle cast(const RectT &rect, return_value_policy, handle) {
        return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
    }
};

// Convenience aliases to preserve prior behavior
// (cv::Rect defaults to int)

// -----------------------------
// cv::DMatch <-> (queryIdx, trainIdx, imgIdx, distance)
// -----------------------------

template <> struct type_caster<cv::DMatch> {
    PYBIND11_TYPE_CASTER(cv::DMatch, _("DMatch"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj))
            throw std::logic_error(
                "DMatch should be a tuple (queryIdx, trainIdx, imgIdx, distance)");
        py::tuple tup = py::reinterpret_borrow<py::tuple>(obj);
        if (tup.size() != 4)
            throw std::logic_error(
                "DMatch tuple should have 4 elements: (queryIdx, trainIdx, imgIdx, distance)");
        value.queryIdx = tup[0].cast<int>();
        value.trainIdx = tup[1].cast<int>();
        value.imgIdx = tup[2].cast<int>();
        value.distance = tup[3].cast<float>();
        return true;
    }

    static handle cast(const cv::DMatch &m, return_value_policy, handle) {
        return py::make_tuple(m.queryIdx, m.trainIdx, m.imgIdx, m.distance).release();
    }
};

// -----------------------------
// cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave)
// -----------------------------

template <> struct type_caster<cv::KeyPoint> {
    PYBIND11_TYPE_CASTER(cv::KeyPoint, _("tuple_x_y_size_angle_response_octave"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj))
            throw std::logic_error("KeyPoint should be a tuple!");
        py::tuple kp = reinterpret_borrow<py::tuple>(obj);
        if (kp.size() != 6) {
            throw std::logic_error("KeyPoint tuple should have 6 elements: (pt.x, pt.y, size, "
                                   "angle, response, octave)");
        }
        value = cv::KeyPoint(kp[0].cast<float>(), kp[1].cast<float>(), kp[2].cast<float>(),
                             kp[3].cast<float>(), kp[4].cast<float>(), kp[5].cast<int>());
        return true;
    }

    static handle cast(const cv::KeyPoint &kp, return_value_policy, handle) {
        return py::make_tuple(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave)
            .release();
    }
};

} // namespace detail
} // namespace pybind11

// OLD VERSION
// template <>
// // cv::Mat <-> numpy array
// struct type_caster<cv::Mat> {
//   public:
//     PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

//     //! 1. cast numpy.ndarray to cv::Mat
//     bool load(handle obj, bool) {
//         array b = reinterpret_borrow<array>(obj);
//         buffer_info info = b.request();

//         // const int ndims = (int)info.ndim;
//         int nh = 1;
//         int nw = 1;
//         int nc = 1;
//         int ndims = info.ndim;
//         if (ndims == 2) {
//             nh = info.shape[0];
//             nw = info.shape[1];
//         } else if (ndims == 3) {
//             nh = info.shape[0];
//             nw = info.shape[1];
//             nc = info.shape[2];
//         } else {
//             char msg[64];
//             std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
//             throw std::logic_error(msg);
//             return false;
//         }

//         int dtype;
//         if (info.format == format_descriptor<unsigned char>::format()) {
//             dtype = CV_8UC(nc);
//         } else if (info.format == format_descriptor<int>::format()) {
//             dtype = CV_32SC(nc);
//         } else if (info.format == format_descriptor<float>::format()) {
//             dtype = CV_32FC(nc);
//         } else {
//             throw std::logic_error("Unsupported type, only support uchar, int32, float");
//             return false;
//         }

//         value = cv::Mat(nh, nw, dtype, info.ptr);
//         return true;
//     }

//     //! 2. cast cv::Mat to numpy.ndarray
//     static handle cast(const cv::Mat &mat, return_value_policy, handle defval) {
//         // UNUSED(defval);

//         std::string format = format_descriptor<unsigned char>::format();
//         size_t elemsize = sizeof(unsigned char);
//         int nw = mat.cols;
//         int nh = mat.rows;
//         int nc = mat.channels();
//         int depth = mat.depth();
//         int type = mat.type();
//         int dim = (depth == type) ? 2 : 3;

//         if (depth == CV_8U) {
//             format = format_descriptor<unsigned char>::format();
//             elemsize = sizeof(unsigned char);
//         } else if (depth == CV_32S) {
//             format = format_descriptor<int>::format();
//             elemsize = sizeof(int);
//         } else if (depth == CV_32F) {
//             format = format_descriptor<float>::format();
//             elemsize = sizeof(float);
//         } else {
//             throw std::logic_error("Unsupport type, only support uchar, int32, float");
//         }

//         std::vector<size_t> bufferdim;
//         std::vector<size_t> strides;
//         if (dim == 2) {
//             bufferdim = {(size_t)nh, (size_t)nw};
//             strides = {elemsize * (size_t)nw, elemsize};
//         } else if (dim == 3) {
//             bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
//             strides = {(size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize};
//         }
//         return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
//     }
// };
