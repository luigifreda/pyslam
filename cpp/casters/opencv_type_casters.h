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

// NOTE: The actual path of this file is cpp/casters/opencv_type_casters.h
//       We actually created a symlink to this file in other places of the project
// TESTED with cpp/test_opencv_casters/run_tests.sh

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <opencv2/core.hpp>

#ifndef PYSLAM_CV_MAT_IMPORT_ALWAYS_COPY
// Import policy for Python ndarray -> cv::Mat.
// 0 enables the fast path where cv::Mat directly views NumPy-owned memory.
// 1 forces a defensive deep copy for every import.
#define PYSLAM_CV_MAT_IMPORT_ALWAYS_COPY 0 // 1: always copy, 0: zero-copy view
#endif

namespace py = pybind11;
using namespace pybind11::literals;

namespace pybind11 {
namespace detail {

// Small helper to stringify dtype safely across pybind11 versions
inline std::string dtype_to_string(const py::dtype &dt) {
    return py::cast<std::string>(dt.attr("name"));
}

// -----------------------------
// Helpers: NumPy dtype <-> cv depth
// -----------------------------
inline int numpy_to_cv_depth(const py::dtype &dt) {
    // Fast path: standard dtypes recognized by pybind
    if (dt.is(py::dtype::of<bool>()))
        return CV_8U; // masks -> 0/1 u8
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

    // Robust fallback: rely on dtype metadata (kind + itemsize) to catch NumPy variants
    const std::string dtype_name = dtype_to_string(dt);
    const char kind = py::cast<char>(dt.attr("kind"));               // 'b','i','u','f',...
    const ssize_t itemsize = py::cast<ssize_t>(dt.attr("itemsize")); // bytes per element

    // int32 variants (e.g., NumPy 2.0 "int32" not matching dtype::of<int32_t>())
    if ((dtype_name == "int32" || dtype_name == "i4") || (kind == 'i' && itemsize == 4)) {
        return CV_32S;
    }
    // int16 variants
    if (kind == 'i' && itemsize == 2)
        return CV_16S;
    // int8 variants
    if (kind == 'i' && itemsize == 1)
        return CV_8S;
    // uint16 variants
    if (kind == 'u' && itemsize == 2)
        return CV_16U;
    // uint8 variants (includes bool-like u1)
    if (kind == 'u' && itemsize == 1)
        return CV_8U;
    // float32/64 variants
    if (kind == 'f' && itemsize == 4)
        return CV_32F;
    if (kind == 'f' && itemsize == 8)
        return CV_64F;

    throw py::value_error("Unsupported NumPy dtype: " + dtype_name +
                          " (expected: bool,uint8,int8,uint16,int16,int32,float32,float64)");
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
        throw py::value_error("Unsupported cv::Mat depth");
    }
}

struct cv_mat_buffer_view {
    // The Python array object that ultimately owns the memory. We keep this around so the
    // zero-copy path can transfer ownership into cv::Mat via UMatData::userdata.
    py::array array;
    py::buffer_info info;
    int rows = 0;
    int cols = 0;
    int channels = 1;
    int type = 0;
    size_t step = 0;
    size_t row_bytes = 0;
};

class py_owned_mat_allocator final : public cv::MatAllocator {
  public:
    // This allocator never creates fresh buffers. It only exists so cv::Mat can release a
    // borrowed NumPy buffer using OpenCV's normal reference-counted lifetime hooks.
    cv::UMatData *allocate(int, const int *, int, void *, size_t *, cv::AccessFlag,
                           cv::UMatUsageFlags) const override {
        throw std::runtime_error("py_owned_mat_allocator does not allocate fresh buffers");
    }

    bool allocate(cv::UMatData *, cv::AccessFlag, cv::UMatUsageFlags) const override {
        return false;
    }

    void deallocate(cv::UMatData *u) const override {
        if (!u)
            return;

        // userdata stores the borrowed NumPy owner with an extra INCREF. Dropping that reference
        // here is what keeps the aliased buffer valid after the pybind11 type_caster temporary
        // has already gone out of scope.
        auto *owner = reinterpret_cast<PyObject *>(u->userdata);
        if (owner) {
            if (Py_IsInitialized()) {
                py::gil_scoped_acquire acquire;
                Py_DECREF(owner);
            }
            u->userdata = nullptr;
        }

        delete u;
    }
};

inline const cv::MatAllocator *get_py_owned_mat_allocator() {
    static py_owned_mat_allocator allocator;
    return &allocator;
}

inline size_t cv_mat_total_bytes(const cv_mat_buffer_view &view) {
    if (view.rows <= 0)
        return 0;
    // For strided views the live byte span is not rows * row_bytes.
    // OpenCV expects the allocation size to cover the full last row reachable through step[0].
    return view.step * static_cast<size_t>(view.rows - 1) + view.row_bytes;
}

inline cv::Mat make_py_owned_cv_mat(const cv_mat_buffer_view &view) {
    // Build a Mat header that aliases NumPy memory.
    cv::Mat mat(view.rows, view.cols, view.type, view.info.ptr, view.step);

    // Attach an OpenCV-owned lifetime record so every Mat clone/header copy participates in
    // reference counting, while the underlying bytes remain owned by Python.
    auto *u = new cv::UMatData(get_py_owned_mat_allocator());
    u->data = u->origdata = static_cast<uchar *>(view.info.ptr);
    u->size = cv_mat_total_bytes(view);
    u->flags |= cv::UMatData::USER_ALLOCATED;
    PyObject *owner = view.array.ptr();
    Py_INCREF(owner);
    u->userdata = owner;
    u->refcount = 1;
    u->urefcount = 0;

    mat.u = u;
    mat.allocator = const_cast<cv::MatAllocator *>(get_py_owned_mat_allocator());
    return mat;
}

inline void normalize_cv_mat_array_dtype(py::array &arr) {
    const py::dtype dt = arr.dtype();
    if (dt.is(py::dtype::of<std::int64_t>()) || dt.is(py::dtype::of<std::uint64_t>())) {
        // OpenCV has no CV_64S/CV_64U depth. Converting to float64 preserves range better than
        // truncating down to 32-bit integer types.
        py::object converted_obj = arr.attr("astype")(py::dtype::of<double>());
        py::array converted = py::array::ensure(converted_obj);
        if (!converted)
            throw py::value_error("Failed to convert int64/uint64 array to float64 for cv::Mat");
        arr = std::move(converted);
    }
}

inline void parse_cv_mat_shape(const py::buffer_info &info, int &rows, int &cols, int &channels) {
    if (info.ndim < 1 || info.ndim > 3)
        throw py::value_error("cv::Mat expects 1D, 2D, or 3D array shaped (N,), (H,W), or (H,W,C)");

    if (info.ndim == 1) {
        rows = static_cast<int>(info.shape[0]);
        cols = 1;
        channels = 1;
    } else if (info.ndim == 2) {
        rows = static_cast<int>(info.shape[0]);
        cols = static_cast<int>(info.shape[1]);
        channels = 1;
    } else {
        rows = static_cast<int>(info.shape[0]);
        cols = static_cast<int>(info.shape[1]);
        channels = static_cast<int>(info.shape[2]);
        if (channels <= 0)
            channels = 1;
    }

    if (channels > CV_CN_MAX)
        throw py::value_error("cv::Mat import has too many channels");
}

inline bool is_cv_mat_stride_layout_compatible(const py::buffer_info &info) {
    if (info.itemsize <= 0)
        return false;
    for (ssize_t d = 0; d < info.ndim; ++d) {
        // Negative strides imply reversed views. cv::Mat can represent positive row steps, but not
        // arbitrary Python slicing semantics, so those inputs must be normalized first.
        if (info.strides[d] < 0)
            return false;
    }
    if (info.ndim == 1) {
        return info.shape[0] <= 1 || info.strides[0] >= info.itemsize;
    }
    if (info.ndim == 2) {
        return info.strides[1] == info.itemsize &&
               (info.shape[0] <= 1 || info.strides[0] >= info.shape[1] * info.itemsize);
    }
    return info.strides[2] == info.itemsize && info.strides[1] == info.shape[2] * info.itemsize &&
           (info.shape[0] <= 1 || info.strides[0] >= info.shape[1] * info.shape[2] * info.itemsize);
}

inline cv_mat_buffer_view make_cv_mat_buffer_view(handle src) {
    py::array source_arr = py::array::ensure(src);
    if (!source_arr)
        throw py::type_error("cv::Mat expects an array-like input");

    py::array arr = source_arr;
    normalize_cv_mat_array_dtype(arr);

    cv_mat_buffer_view view;
    view.array = arr;
    view.info = view.array.request();
    parse_cv_mat_shape(view.info, view.rows, view.cols, view.channels);

    if (view.info.size == 0 || view.rows == 0 || view.cols == 0) {
        return view;
    }

    if (!is_cv_mat_stride_layout_compatible(view.info)) {
        // Normalize exotic views once so the rest of the importer can treat the buffer as a
        // conventional HxW or HxWxC matrix. The normalized array is still NumPy-owned, so the
        // same lifetime-managed zero-copy path can alias it safely.
        view.array = py::array::ensure(view.array, py::array::c_style);
        if (!view.array)
            throw py::value_error("Failed to convert array to C-contiguous layout for cv::Mat");
        view.info = view.array.request();
        parse_cv_mat_shape(view.info, view.rows, view.cols, view.channels);
    }

    const int depth = numpy_to_cv_depth(view.array.dtype());
    view.type = CV_MAKETYPE(depth, view.channels);
    view.step = static_cast<size_t>(view.info.strides[0]);
    view.row_bytes = static_cast<size_t>(view.cols) * static_cast<size_t>(view.channels) *
                     static_cast<size_t>(view.info.itemsize);
    return view;
}

// -----------------------------
// cv::Mat <-> numpy.ndarray
// -----------------------------
template <> struct type_caster<cv::Mat> {
  public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        try {
            cv_mat_buffer_view view = make_cv_mat_buffer_view(src);
            if (view.info.size == 0 || view.rows == 0 || view.cols == 0) {
                // Match OpenCV conventions: zero-sized inputs become empty Mats instead of
                // degenerate 0xN headers.
                value = cv::Mat();
                return true;
            }

            const size_t expected =
                static_cast<size_t>(view.rows) * static_cast<size_t>(view.row_bytes);
            const size_t actual =
                static_cast<size_t>(view.info.size) * static_cast<size_t>(view.info.itemsize);
            if (expected != actual)
                throw py::value_error("Unexpected buffer size for cv::Mat import");

#if PYSLAM_CV_MAT_IMPORT_ALWAYS_COPY
            // Fully detached import mode for debugging or call sites that prefer simpler lifetime
            // behavior over aliasing performance.
            cv::Mat tmp(view.rows, view.cols, view.type);
            const auto *src_ptr = static_cast<const std::uint8_t *>(view.info.ptr);
            if (view.step == view.row_bytes) {
                std::memcpy(tmp.data, src_ptr, expected);
            } else {
                for (int r = 0; r < view.rows; ++r) {
                    std::memcpy(tmp.ptr(r), src_ptr + static_cast<size_t>(r) * view.step,
                                view.row_bytes);
                }
            }
            value = std::move(tmp);
#else
            // Alias NumPy memory and transfer the owning Python reference into the Mat's
            // UMatData record so the view stays valid after the caster returns.
            value = make_py_owned_cv_mat(view);
#endif
            return true;
        } catch (const py::error_already_set &) {
            throw;
        } catch (const py::builtin_exception &) {
            throw;
        } catch (...) {
            return false;
        }
    }

    static handle cast(const cv::Mat &mat, return_value_policy, handle) {
        if (mat.empty())
            return py::array(py::dtype::of<uint8_t>(), {0, 0}).release();

        // Python expects a standard ndarray view. If the source Mat is not continuous we clone so
        // the exported array has a simple ownership story and valid contiguous strides.
        cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();
        const int depth = contiguous.depth();
        const int channels = contiguous.channels();
        py::dtype dt = cv_depth_to_numpy_dtype(depth);

        std::vector<ssize_t> shape, strides;
        const ssize_t elem = static_cast<ssize_t>(dt.itemsize());

        if (channels == 1) {
            shape = {contiguous.rows, contiguous.cols};
            strides = {static_cast<ssize_t>(contiguous.step[0]), elem};
        } else {
            shape = {contiguous.rows, contiguous.cols, channels};
            strides = {static_cast<ssize_t>(contiguous.step[0]),
                       static_cast<ssize_t>(channels) * elem, elem};
        }

        // The capsule owns a heap-allocated cv::Mat header, which in turn owns the underlying
        // buffer through OpenCV's reference counting. This makes C++ -> Python zero-copy as well.
        auto *owner = new cv::Mat(std::move(contiguous));
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
        // Accept array-likes (force cast to C-array of T)
        py::array arr = py::array::ensure(src, py::array::c_style | py::array::forcecast);
        if (!arr)
            return false;
        py::buffer_info info = arr.request();
        if (info.ndim != 1 || info.shape[0] != N)
            return false;
        if (!arr.dtype().is(py::dtype::of<T>()))
            return false; // fixed: use arr.dtype()
        T *ptr = static_cast<T *>(info.ptr);
        for (int i = 0; i < N; ++i)
            value[i] = ptr[i];
        return true;
    }

    static handle cast(const Vec &v, return_value_policy, handle) {
        // Copy out (avoid dangling pointer to stack)
        auto dt = py::dtype::of<T>();
        py::array out(dt, std::vector<ssize_t>{N});
        std::memcpy(out.mutable_data(), v.val, sizeof(T) * N);
        return out.release();
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
        value.reserve(py::len(seq));

        for (auto item : seq) {
            // Reuse the exact same import policy as the single-Mat caster so list inputs behave
            // consistently with standalone ndarray inputs.
            cv_mat_buffer_view view = make_cv_mat_buffer_view(item);
            if (view.info.size == 0 || view.rows == 0 || view.cols == 0) {
                value.emplace_back(); // empty Mat
                continue;
            }

            const size_t expected = static_cast<size_t>(view.rows) * view.row_bytes;
            const size_t actual =
                static_cast<size_t>(view.info.size) * static_cast<size_t>(view.info.itemsize);
            if (expected != actual)
                throw py::value_error("Unexpected buffer size for cv::Mat in list");

#if PYSLAM_CV_MAT_IMPORT_ALWAYS_COPY
            cv::Mat tmp(view.rows, view.cols, view.type);
            const auto *src_ptr = static_cast<const std::uint8_t *>(view.info.ptr);
            if (view.step == view.row_bytes) {
                std::memcpy(tmp.data, src_ptr, expected);
            } else {
                for (int r = 0; r < view.rows; ++r) {
                    std::memcpy(tmp.ptr(r), src_ptr + static_cast<size_t>(r) * view.step,
                                view.row_bytes);
                }
            }
            value.emplace_back(std::move(tmp));
#else
            // Each element gets its own UMatData owner so the vector can outlive the pybind11
            // argument conversion frame safely, even when the source item had to be normalized
            // into a temporary contiguous NumPy array first.
            value.emplace_back(make_py_owned_cv_mat(view));
#endif
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
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::sequence>(obj))
            throw py::value_error("Point(x,y) should be a 2-sequence");
        py::sequence pt = reinterpret_borrow<py::sequence>(obj);
        if (py::len(pt) != 2)
            throw py::value_error("Point(x,y) must have length 2");
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
    PYBIND11_TYPE_CASTER(cv::Point2f, _("tuple"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::sequence>(obj))
            throw py::value_error("Point2f(x,y) should be a 2-sequence");
        py::sequence pt = reinterpret_borrow<py::sequence>(obj);
        if (py::len(pt) != 2)
            throw py::value_error("Point2f(x,y) must have length 2");
        value = cv::Point2f(pt[0].cast<float>(), pt[1].cast<float>());
        return true;
    }

    static handle cast(const cv::Point2f &pt, return_value_policy, handle) {
        return py::make_tuple(pt.x, pt.y).release();
    }
};

// -----------------------------
// cv::Rect_<T> <-> tuple(x, y, w, h)
// -----------------------------
template <typename T> struct type_caster<cv::Rect_<T>> {
    using RectT = cv::Rect_<T>;
    PYBIND11_TYPE_CASTER(RectT, _("tuple"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::sequence>(obj))
            throw py::value_error("Rect should be a 4-sequence (x,y,w,h)");
        py::sequence rect = reinterpret_borrow<py::sequence>(obj);
        if (py::len(rect) != 4)
            throw py::value_error("Rect (x,y,w,h) must have length 4");
        value = RectT(rect[0].cast<T>(), rect[1].cast<T>(), rect[2].cast<T>(), rect[3].cast<T>());
        return true;
    }

    static handle cast(const RectT &rect, return_value_policy, handle) {
        return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
    }
};

// -----------------------------
// cv::DMatch <-> (queryIdx, trainIdx, imgIdx, distance)
// -----------------------------
template <> struct type_caster<cv::DMatch> {
    PYBIND11_TYPE_CASTER(cv::DMatch, _("tuple"));

    bool load(handle obj, bool) {
        // Accept either a 4-tuple (legacy path) or a cv2.DMatch-like object
        // with attributes queryIdx, trainIdx, imgIdx, distance.
        if (py::isinstance<py::sequence>(obj)) {
            py::sequence tup = py::reinterpret_borrow<py::sequence>(obj);
            if (py::len(tup) != 4)
                throw py::value_error("DMatch must have length 4");
            value.queryIdx = tup[0].cast<int>();
            value.trainIdx = tup[1].cast<int>();
            value.imgIdx = tup[2].cast<int>();
            value.distance = tup[3].cast<float>();
            return true;
        }

        // Fallback: try to read attributes from a cv2.DMatch instance
        if (py::hasattr(obj, "queryIdx") && py::hasattr(obj, "trainIdx") &&
            py::hasattr(obj, "imgIdx") && py::hasattr(obj, "distance")) {
            auto o = py::reinterpret_borrow<py::object>(obj);
            value.queryIdx = o.attr("queryIdx").cast<int>();
            value.trainIdx = o.attr("trainIdx").cast<int>();
            value.imgIdx = o.attr("imgIdx").cast<int>();
            value.distance = o.attr("distance").cast<float>();
            return true;
        }

        throw py::value_error("DMatch should be a 4-sequence or an object with "
                              "queryIdx/trainIdx/imgIdx/distance attributes");
    }

    static handle cast(const cv::DMatch &m, return_value_policy, handle) {
        return py::make_tuple(m.queryIdx, m.trainIdx, m.imgIdx, m.distance).release();
    }
};

// -----------------------------
// cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave[, class_id])
// -----------------------------
template <> struct type_caster<cv::KeyPoint> {
    PYBIND11_TYPE_CASTER(cv::KeyPoint, _("tuple"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::sequence>(obj))
            throw py::value_error("KeyPoint should be a sequence");
        py::sequence kp = reinterpret_borrow<py::sequence>(obj);
        const auto n = static_cast<int>(py::len(kp));
        if (n != 6 && n != 7)
            throw py::value_error("KeyPoint must be (x,y,size,angle,response,octave[,class_id])");
        value = cv::KeyPoint(kp[0].cast<float>(), kp[1].cast<float>(), kp[2].cast<float>(),
                             kp[3].cast<float>(), kp[4].cast<float>(), kp[5].cast<int>());
        if (n == 7)
            value.class_id = kp[6].cast<int>();
        else
            value.class_id = -1;
        return true;
    }

    static handle cast(const cv::KeyPoint &kp, return_value_policy, handle) {
        return py::make_tuple(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave,
                              kp.class_id)
            .release();
    }
};

// -----------------------------
// Scalar <-> length-4 sequence
// -----------------------------
template <> struct type_caster<cv::Scalar> {
    PYBIND11_TYPE_CASTER(cv::Scalar, _("tuple"));
    bool load(handle src, bool) {
        py::sequence s = py::reinterpret_borrow<py::sequence>(src);
        if (py::len(s) != 4)
            return false;
        value = cv::Scalar(s[0].cast<double>(), s[1].cast<double>(), s[2].cast<double>(),
                           s[3].cast<double>());
        return true;
    }
    static handle cast(const cv::Scalar &v, return_value_policy, handle) {
        return py::make_tuple(v[0], v[1], v[2], v[3]).release();
    }
};

// -----------------------------
// Matx<T,m,n> <-> ndarray(m,n)
// -----------------------------
template <typename T, int M, int N> struct type_caster<cv::Matx<T, M, N>> {
    using Matx = cv::Matx<T, M, N>;
    PYBIND11_TYPE_CASTER(Matx, _("numpy.ndarray"));
    bool load(handle src, bool) {
        py::array arr = py::array::ensure(src, py::array::c_style | py::array::forcecast);
        if (!arr || arr.ndim() != 2 || arr.shape(0) != M || arr.shape(1) != N)
            return false;
        if (!arr.dtype().is(py::dtype::of<T>()))
            return false;
        auto *p = static_cast<const T *>(arr.request().ptr);
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c)
                value(r, c) = p[r * N + c];
        return true;
    }
    static handle cast(const Matx &m, return_value_policy, handle) {
        auto dt = py::dtype::of<T>();
        py::array a(dt, {M, N});
        auto *p = static_cast<T *>(a.mutable_data());
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c)
                p[r * N + c] = m(r, c);
        return a.release();
    }
};

} // namespace detail
} // namespace pybind11
