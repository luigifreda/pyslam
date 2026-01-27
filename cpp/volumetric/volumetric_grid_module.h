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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tbb_utils.h"
#include "voxel_block_grid.h"
#include "voxel_block_semantic_grid.h"
#include "voxel_grid.h"
#include "voxel_semantic_grid.h"

#include "opencv_type_casters.h"

namespace py = pybind11;

// ----------------------------------------
// Helper functions for pybind11 bindings
// ----------------------------------------
namespace detail {

// Helper functions for safe dtype checking
inline bool is_uint8_dtype(const py::dtype &dt) {
    if (dt.is(py::dtype::of<uint8_t>())) {
        return true;
    }
    // Fallback: check itemsize and kind for compatibility
    const ssize_t itemsize = dt.itemsize();
    const char kind = py::cast<char>(dt.attr("kind"));
    return (kind == 'u' && itemsize == 1);
}

inline bool is_float32_dtype(const py::dtype &dt) {
    if (dt.is(py::dtype::of<float>())) {
        return true;
    }
    // Fallback: check itemsize and kind for compatibility
    const ssize_t itemsize = dt.itemsize();
    const char kind = py::cast<char>(dt.attr("kind"));
    return (kind == 'f' && itemsize == 4);
}

inline bool is_float64_dtype(const py::dtype &dt) {
    if (dt.is(py::dtype::of<double>())) {
        return true;
    }
    // Fallback: check itemsize and kind for compatibility
    const ssize_t itemsize = dt.itemsize();
    const char kind = py::cast<char>(dt.attr("kind"));
    return (kind == 'f' && itemsize == 8);
}

inline bool is_int32_dtype(const py::dtype &dt) {
    if (dt.is(py::dtype::of<int>())) {
        return true;
    }
    // Fallback: check itemsize and kind for compatibility
    // int32 is typically 4 bytes with signed integer kind 'i'
    const ssize_t itemsize = dt.itemsize();
    const char kind = py::cast<char>(dt.attr("kind"));
    return (kind == 'i' && itemsize == 4);
}

// Helper function to get dtype description for error messages
inline std::string dtype_description(const py::dtype &dt) {
    const ssize_t itemsize = dt.itemsize();
    const char kind = py::cast<char>(dt.attr("kind"));
    return "kind=" + std::string(1, kind) + " itemsize=" + std::to_string(itemsize);
}

// Unified helper for integrate_segment bindings: validates shape and contiguity.
template <typename CLASS_TYPE, typename Tpos, typename Tcol>
inline void integrate_segment_with_arrays(CLASS_TYPE &self, py::array_t<Tpos> points,
                                          py::array_t<Tcol> colors, const int class_id,
                                          const int object_id) {
    py::array_t<Tpos, py::array::c_style | py::array::forcecast> points_c = points;
    py::array_t<Tcol, py::array::c_style | py::array::forcecast> colors_c = colors;
    auto pts_info = points_c.request();
    auto cols_info = colors_c.request();

    if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
        throw std::runtime_error("points must be a contiguous Nx3 array");
    }

    if (cols_info.ndim != 2 || cols_info.shape[1] != 3) {
        throw std::runtime_error("colors must be a contiguous Nx3 array");
    }

    if (cols_info.shape[0] != pts_info.shape[0]) {
        throw std::runtime_error("points and colors must have the same size");
    }

    {
        py::gil_scoped_release release;
        self.template integrate_segment_raw<Tpos, Tcol>(static_cast<const Tpos *>(pts_info.ptr),
                                               pts_info.shape[0],
                                               static_cast<const Tcol *>(cols_info.ptr), class_id,
                                               object_id);
    }
}

// Unified helper function for integration with py::array inputs
// Similar to VoxelBlockGridT::integrate but for Python bindings
// Handles all cases: with/without colors, semantics, instance_ids, depths
template <typename CLASS_TYPE, typename Tpos>
inline void integrate_with_arrays(CLASS_TYPE &self, py::array_t<Tpos> points,
                                  py::object colors = py::none(), py::object class_ids = py::none(),
                                  py::object instance_ids = py::none(),
                                  py::object depths = py::none()) {
    // Force contiguous views to avoid misinterpreting strided arrays
    py::array_t<Tpos, py::array::c_style | py::array::forcecast> points_c = points;
    auto pts_info = points_c.request();

    if (pts_info.ndim != 2 || pts_info.shape[1] != 3) {
        throw std::runtime_error("points must be a contiguous Nx3 array");
    }

    const size_t num_points = pts_info.shape[0];

    if (num_points == 0) {
        return;
    }

    // Check if we have colors
    const void *colors_ptr = nullptr;
    bool has_colors = false;
    bool is_uint8_colors = false;
    if (!colors.is_none()) {
        py::array colors_array = py::array::ensure(colors, py::array::c_style);
        auto cols_info = colors_array.request();
        py::dtype dt = colors_array.dtype();

        if (cols_info.ndim != 2 || cols_info.shape[1] != 3) {
            throw std::runtime_error("colors must be a contiguous Nx3 array");
        }

        if (cols_info.shape[0] != num_points) {
            throw std::runtime_error("points and colors must have the same size");
        }

        if (is_uint8_dtype(dt)) {
            is_uint8_colors = true;
        } else if (is_float32_dtype(dt)) {
            is_uint8_colors = false;
        } else {
            throw std::runtime_error("Colors must be uint8 or float32, got dtype with " +
                                     dtype_description(dt));
        }

        colors_ptr = cols_info.ptr;
        has_colors = true;
    }

    // Check if we have depths
    const void *depths_ptr = nullptr;
    bool has_depths = false;
    bool is_float32_depths = false;
    if (!depths.is_none()) {
        py::array depths_array = py::array::ensure(depths, py::array::c_style);
        auto depths_info = depths_array.request();
        py::dtype dt = depths_array.dtype();

        if (depths_info.ndim != 1) {
            throw std::runtime_error("depths must be a contiguous 1D array");
        }

        if (depths_info.shape[0] != num_points) {
            throw std::runtime_error("points and depths must have the same size");
        }

        if (is_float32_dtype(dt)) {
            is_float32_depths = true;
        } else if (is_float64_dtype(dt)) {
            is_float32_depths = false;
        } else {
            throw std::runtime_error("Depths must be float32 or float64, got dtype with " +
                                     dtype_description(dt));
        }

        depths_ptr = depths_info.ptr;
        has_depths = true;
    }

    // Check if we have semantic data
    const void *class_ids_ptr = nullptr;
    bool has_semantics = false;
    if (!class_ids.is_none()) {
        py::array class_ids_array = py::array::ensure(class_ids, py::array::c_style);
        auto class_ids_info = class_ids_array.request();
        py::dtype dt = class_ids_array.dtype();

        if (class_ids_info.ndim != 1) {
            throw std::runtime_error("class_ids must be a contiguous 1D array");
        }

        if (!is_int32_dtype(dt)) {
            throw std::runtime_error("class_ids must be int32, got dtype with " +
                                     dtype_description(dt));
        }

        if (class_ids_info.shape[0] != num_points) {
            throw std::runtime_error("points and class_ids must have the same size");
        }

        class_ids_ptr = class_ids_info.ptr;
        has_semantics = true;
    }

    // Check if we have instance_ids
    const void *instance_ids_ptr = nullptr;
    bool has_instance_ids = false;
    if (!instance_ids.is_none()) {
        if (!has_semantics) {
            throw std::runtime_error("instance_ids but no class_ids is not supported");
        }

        py::array instance_ids_array = py::array::ensure(instance_ids, py::array::c_style);
        auto instance_ids_info = instance_ids_array.request();
        py::dtype dt = instance_ids_array.dtype();

        if (instance_ids_info.ndim != 1) {
            throw std::runtime_error("instance_ids must be a contiguous 1D array");
        }

        if (!is_int32_dtype(dt)) {
            throw std::runtime_error("instance_ids must be int32, got dtype with " +
                                     dtype_description(dt));
        }

        if (instance_ids_info.shape[0] != num_points) {
            throw std::runtime_error("points and instance_ids must have the same size");
        }

        instance_ids_ptr = instance_ids_info.ptr;
        has_instance_ids = true;
    }

    // Release GIL and call integrate_raw with appropriate template parameters
    {
        py::gil_scoped_release release;
        const Tpos *pts_ptr = static_cast<const Tpos *>(pts_info.ptr);

        if (has_colors) {
            // we have colors
            if (has_semantics) {
                // we have semantics
                if (has_instance_ids) {
                    // we have instance ids
                    if (has_depths) {
                        if (is_uint8_colors) {
                            if (is_float32_depths) {
                                self.template integrate_raw<Tpos, uint8_t, int, int, float>(
                                    pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr),
                                    static_cast<const int *>(instance_ids_ptr),
                                    static_cast<const float *>(depths_ptr));
                            } else {
                                self.template integrate_raw<Tpos, uint8_t, int, int, double>(
                                    pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr),
                                    static_cast<const int *>(instance_ids_ptr),
                                    static_cast<const double *>(depths_ptr));
                            }
                        } else {
                            if (is_float32_depths) {
                                self.template integrate_raw<Tpos, float, int, int, float>(
                                    pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr),
                                    static_cast<const int *>(instance_ids_ptr),
                                    static_cast<const float *>(depths_ptr));
                            } else {
                                self.template integrate_raw<Tpos, float, int, int, double>(
                                    pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr),
                                    static_cast<const int *>(instance_ids_ptr),
                                    static_cast<const double *>(depths_ptr));
                            }
                        }
                    } else {
                        if (is_uint8_colors) {
                            self.template integrate_raw<Tpos, uint8_t, int, int>(
                                pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                static_cast<const int *>(class_ids_ptr),
                                static_cast<const int *>(instance_ids_ptr));
                        } else {
                            self.template integrate_raw<Tpos, float, int, int>(
                                pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                static_cast<const int *>(class_ids_ptr),
                                static_cast<const int *>(instance_ids_ptr));
                        }
                    }
                } else {
                    // we do not have instance ids
                    if (has_depths) {
                        if (is_uint8_colors) {
                            if (is_float32_depths) {
                                self.template integrate_raw<Tpos, uint8_t, std::nullptr_t, int,
                                                            float>(
                                    pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr), nullptr,
                                    static_cast<const float *>(depths_ptr));
                            } else {
                                self.template integrate_raw<Tpos, uint8_t, std::nullptr_t, int,
                                                            double>(
                                    pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr), nullptr,
                                    static_cast<const double *>(depths_ptr));
                            }
                        } else {
                            if (is_float32_depths) {
                                self.template integrate_raw<Tpos, float, std::nullptr_t, int,
                                                            float>(
                                    pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr), nullptr,
                                    static_cast<const float *>(depths_ptr));
                            } else {
                                self.template integrate_raw<Tpos, float, std::nullptr_t, int,
                                                            double>(
                                    pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                    static_cast<const int *>(class_ids_ptr), nullptr,
                                    static_cast<const double *>(depths_ptr));
                            }
                        }
                    } else {
                        if (is_uint8_colors) {
                            self.template integrate_raw<Tpos, uint8_t, std::nullptr_t, int>(
                                pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                static_cast<const int *>(class_ids_ptr), nullptr);
                        } else {
                            self.template integrate_raw<Tpos, float, std::nullptr_t, int>(
                                pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                static_cast<const int *>(class_ids_ptr), nullptr);
                        }
                    }
                }
            } else {
                // with colors, no semantics
                if (has_depths) {
                    if (is_uint8_colors) {
                        if (is_float32_depths) {
                            self.template integrate_raw<Tpos, uint8_t, std::nullptr_t,
                                                        std::nullptr_t, float>(
                                pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                nullptr, nullptr, static_cast<const float *>(depths_ptr));
                        } else {
                            self.template integrate_raw<Tpos, uint8_t, std::nullptr_t,
                                                        std::nullptr_t, double>(
                                pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr),
                                nullptr, nullptr, static_cast<const double *>(depths_ptr));
                        }
                    } else {
                        if (is_float32_depths) {
                            self.template integrate_raw<Tpos, float, std::nullptr_t, std::nullptr_t,
                                                        float>(
                                pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                nullptr, nullptr, static_cast<const float *>(depths_ptr));
                        } else {
                            self.template integrate_raw<Tpos, float, std::nullptr_t, std::nullptr_t,
                                                        double>(
                                pts_ptr, num_points, static_cast<const float *>(colors_ptr),
                                nullptr, nullptr, static_cast<const double *>(depths_ptr));
                        }
                    }
                } else {
                    if (is_uint8_colors) {
                        self.template integrate_raw<Tpos, uint8_t>(
                            pts_ptr, num_points, static_cast<const uint8_t *>(colors_ptr));
                    } else {
                        self.template integrate_raw<Tpos, float>(
                            pts_ptr, num_points, static_cast<const float *>(colors_ptr));
                    }
                }
            }
        } else {
            // No colors
            if (has_semantics) {
                // we have semantics
                if (has_instance_ids) {
                    // we have instance ids
                    if (has_depths) {
                        if (is_float32_depths) {
                            self.template integrate_raw<Tpos, std::nullptr_t, int, int, float>(
                                pts_ptr, num_points, nullptr,
                                static_cast<const int *>(class_ids_ptr),
                                static_cast<const int *>(instance_ids_ptr),
                                static_cast<const float *>(depths_ptr));
                        } else {
                            self.template integrate_raw<Tpos, std::nullptr_t, int, int, double>(
                                pts_ptr, num_points, nullptr,
                                static_cast<const int *>(class_ids_ptr),
                                static_cast<const int *>(instance_ids_ptr),
                                static_cast<const double *>(depths_ptr));
                        }
                    } else {
                        self.template integrate_raw<Tpos, std::nullptr_t, int, int>(
                            pts_ptr, num_points, nullptr, static_cast<const int *>(class_ids_ptr),
                            static_cast<const int *>(instance_ids_ptr));
                    }
                } else {
                    // we do not have instance ids
                    if (has_depths) {
                        if (is_float32_depths) {
                            self.template integrate_raw<Tpos, std::nullptr_t, std::nullptr_t, int,
                                                        float>(
                                pts_ptr, num_points, nullptr,
                                static_cast<const int *>(class_ids_ptr), nullptr,
                                static_cast<const float *>(depths_ptr));
                        } else {
                            self.template integrate_raw<Tpos, std::nullptr_t, std::nullptr_t, int,
                                                        double>(
                                pts_ptr, num_points, nullptr,
                                static_cast<const int *>(class_ids_ptr), nullptr,
                                static_cast<const double *>(depths_ptr));
                        }
                    } else {
                        self.template integrate_raw<Tpos, std::nullptr_t, std::nullptr_t, int>(
                            pts_ptr, num_points, nullptr, static_cast<const int *>(class_ids_ptr),
                            nullptr);
                    }
                }
            } else {
                // no colors, no semantics
                if (has_depths) {
                    if (is_float32_depths) {
                        self.template integrate_raw<Tpos, std::nullptr_t, std::nullptr_t,
                                                    std::nullptr_t, float>(
                            pts_ptr, num_points, nullptr, nullptr, nullptr,
                            static_cast<const float *>(depths_ptr));
                    } else {
                        self.template integrate_raw<Tpos, std::nullptr_t, std::nullptr_t,
                                                    std::nullptr_t, double>(
                            pts_ptr, num_points, nullptr, nullptr, nullptr,
                            static_cast<const double *>(depths_ptr));
                    }
                } else {
                    self.template integrate_raw<Tpos>(pts_ptr, num_points);
                }
            }
        }
    }
}
} // namespace detail

// ----------------------------------------
// Unified macro for VoxelSemanticGrid bindings
// Parameters:
//   - CLASS_TYPE: The C++ class type
//   - PYTHON_NAME: The Python class name
//   - CLASS_DEF: Class definition (e.g., CLASS_TYPE or CLASS_TYPE,
//   std::shared_ptr<CLASS_TYPE>)
//   - CONSTRUCTOR: Constructor definition (e.g., py::init<float>() or
//   py::init<float, int>())
//   - CONSTRUCTOR_ARGS: Constructor arguments (e.g.,
//   py::arg("voxel_size") or py::arg("voxel_size"),
//   py::arg("block_size") = 8)
// ----------------------------------------
// Helper to expand class definition properly (handles both parenthesized and non-parenthesized
// forms)
#define EXPAND_CLASS_DEF(TYPE, PTR_TYPE) TYPE, PTR_TYPE
#define DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, TYPE, PTR_TYPE,       \
                                                    CONSTRUCTOR, ...)                              \
    py::class_<EXPAND_CLASS_DEF(TYPE, PTR_TYPE)>(m, PYTHON_NAME)                                   \
        .def(CONSTRUCTOR, __VA_ARGS__)                                                             \
        .def(                                                                                      \
            "assign_object_ids_to_instance_ids",                                                   \
            [](CLASS_TYPE &self, const volumetric::CameraFrustrum &camera_frustrum,                \
               const cv::Mat &class_ids_image, const cv::Mat &semantic_instances_image,            \
               const cv::Mat &depth_image, const float depth_threshold = 0.1f,                     \
               bool do_carving = false, const float min_vote_ratio = 0.5f,                         \
               const int min_votes = 3) {                                                          \
                volumetric::MapInstanceIdToObjectId result;                                        \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    result = self.assign_object_ids_to_instance_ids(                               \
                        camera_frustrum, class_ids_image, semantic_instances_image, depth_image,   \
                        depth_threshold, do_carving, min_vote_ratio, min_votes);                   \
                }                                                                                  \
                return result;                                                                     \
            },                                                                                     \
            "Assign object IDs to instance IDs using semantic instances image and depth image",    \
            py::arg("camera_frustrum"), py::arg("class_ids_image"),                                \
            py::arg("semantic_instances_image"), py::arg("depth_image"),                           \
            py::arg("depth_threshold") = 0.1f, py::arg("do_carving") = false,                      \
            py::arg("min_vote_ratio") = 0.5f, py::arg("min_votes") = 3)                            \
        .def("set_depth_threshold", &CLASS_TYPE::set_depth_threshold,                              \
             py::arg("depth_threshold") = 5.0)                                                     \
        .def("set_depth_decay_rate", &CLASS_TYPE::set_depth_decay_rate,                            \
             py::arg("depth_decay_rate") = 0.07)                                                   \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<uint8_t> colors,          \
               py::object class_ids = py::none(), py::object instance_ids = py::none(),            \
               py::object depths = py::none()) {                                                   \
                detail::integrate_with_arrays<CLASS_TYPE, double>(self, points, colors, class_ids, \
                                                                  instance_ids, depths);           \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional "                             \
            "class_ids, instance_ids and depths)",                                                 \
            py::arg("points"), py::arg("colors"), py::arg("class_ids") = py::none(),               \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<float> colors,            \
               py::object class_ids = py::none(), py::object instance_ids = py::none(),            \
               py::object depths = py::none()) {                                                   \
                detail::integrate_with_arrays<CLASS_TYPE, double>(self, points, colors, class_ids, \
                                                                  instance_ids, depths);           \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional "                             \
            "class_ids, instance_ids and depths)",                                                 \
            py::arg("points"), py::arg("colors"), py::arg("class_ids") = py::none(),               \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<uint8_t> colors,           \
               py::object class_ids = py::none(), py::object instance_ids = py::none(),            \
               py::object depths = py::none()) {                                                   \
                detail::integrate_with_arrays<CLASS_TYPE, float>(self, points, colors, class_ids,  \
                                                                 instance_ids, depths);            \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional "                             \
            "class_ids, instance_ids and depths)",                                                 \
            py::arg("points"), py::arg("colors"), py::arg("class_ids") = py::none(),               \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<float> colors,             \
               py::object class_ids = py::none(), py::object instance_ids = py::none(),            \
               py::object depths = py::none()) {                                                   \
                detail::integrate_with_arrays<CLASS_TYPE, float>(self, points, colors, class_ids,  \
                                                                 instance_ids, depths);            \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional "                             \
            "class_ids, instance_ids and depths)",                                                 \
            py::arg("points"), py::arg("colors"), py::arg("class_ids") = py::none(),               \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<uint8_t> colors,          \
               const int class_id, const int object_id) {                                          \
                detail::integrate_segment_with_arrays<CLASS_TYPE, double, uint8_t>(                \
                    self, points, colors, class_id, object_id);                                    \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("class_id"), py::arg("object_id"))                          \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<float> colors,            \
               const int class_id, const int object_id) {                                          \
                detail::integrate_segment_with_arrays<CLASS_TYPE, double, float>(                  \
                    self, points, colors, class_id, object_id);                                    \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("class_id"), py::arg("object_id"))                          \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<uint8_t> colors,           \
               const int class_id, const int object_id) {                                          \
                detail::integrate_segment_with_arrays<CLASS_TYPE, float, uint8_t>(                 \
                    self, points, colors, class_id, object_id);                                    \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("class_id"), py::arg("object_id"))                          \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<float> colors,             \
               const int class_id, const int object_id) {                                          \
                detail::integrate_segment_with_arrays<CLASS_TYPE, float, float>(                   \
                    self, points, colors, class_id, object_id);                                    \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("class_id"), py::arg("object_id"))                          \
        .def(                                                                                      \
            "merge_segments",                                                                      \
            [](CLASS_TYPE &self, const int instance_id1, const int instance_id2) {                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.merge_segments(instance_id1, instance_id2);                               \
                }                                                                                  \
            },                                                                                     \
            "Merge two segments of voxels into a single segment of "                               \
            "voxels with the same instance "                                                       \
            "ID",                                                                                  \
            py::arg("instance_id1"), py::arg("instance_id2"))                                      \
        .def(                                                                                      \
            "remove_segment",                                                                      \
            [](CLASS_TYPE &self, const int object_id) {                                            \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_segment(object_id);                                                \
                }                                                                                  \
            },                                                                                     \
            "Remove a segment of voxels from the voxel grid", py::arg("object_id"))                \
        .def(                                                                                      \
            "remove_low_confidence_segments",                                                      \
            [](CLASS_TYPE &self, const int min_confidence) {                                       \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_low_confidence_segments(min_confidence);                           \
                }                                                                                  \
            },                                                                                     \
            "Remove segments of voxels with low confidence counter "                               \
            "from the voxel grid",                                                                 \
            py::arg("min_confidence"))                                                             \
        .def(                                                                                      \
            "get_voxels",                                                                          \
            [](CLASS_TYPE &self, int min_count = 1, float min_confidence = 0.0) {                  \
                typename CLASS_TYPE::VoxelGridDataType result;                                     \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    result = self.get_voxels(min_count, min_confidence);                           \
                }                                                                                  \
                return result;                                                                     \
            },                                                                                     \
            "Returns the voxels", py::arg("min_count") = 1, py::arg("min_confidence") = 0.0)       \
        .def(                                                                                      \
            "get_points",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<typename CLASS_TYPE::PosScalar, 3>> points;                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    points = self.get_points();                                                    \
                }                                                                                  \
                return points;                                                                     \
            },                                                                                     \
            "Returns the points")                                                                  \
        .def(                                                                                      \
            "get_colors",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<typename CLASS_TYPE::ColorScalar, 3>> colors;               \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    colors = self.get_colors();                                                    \
                }                                                                                  \
                return colors;                                                                     \
            },                                                                                     \
            "Returns the colors")                                                                  \
        .def(                                                                                      \
            "get_object_segments",                                                                 \
            [](CLASS_TYPE &self, int min_count = 1, float min_confidence = 0.0) {                  \
                std::shared_ptr<volumetric::ObjectDataGroup> object_data_group;                    \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    object_data_group = self.get_object_segments(min_count, min_confidence);       \
                }                                                                                  \
                return object_data_group;                                                          \
            },                                                                                     \
            "Returns a dictionary of segments of voxels based on instance IDs",                    \
            py::arg("min_count") = 1, py::arg("min_confidence") = 0.0)                             \
        .def(                                                                                      \
            "get_class_segments",                                                                  \
            [](CLASS_TYPE &self, int min_count = 1, float min_confidence = 0.0) {                  \
                std::shared_ptr<volumetric::ClassDataGroup> class_data_group;                      \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    class_data_group = self.get_class_segments(min_count, min_confidence);         \
                }                                                                                  \
                return class_data_group;                                                           \
            },                                                                                     \
            "Returns a dictionary of segments of voxels based on class IDs",                       \
            py::arg("min_count") = 1, py::arg("min_confidence") = 0.0)                             \
        .def(                                                                                      \
            "get_ids",                                                                             \
            [](CLASS_TYPE &self) {                                                                 \
                std::pair<std::vector<int>, std::vector<int>> ids;                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    ids = self.get_ids();                                                          \
                }                                                                                  \
                auto &[class_ids, instance_ids] = ids;                                             \
                return py::make_tuple(std::move(class_ids), std::move(instance_ids));              \
            },                                                                                     \
            "Returns a tuple (class_ids, instance_ids)")                                           \
        .def(                                                                                      \
            "clear",                                                                               \
            [](CLASS_TYPE &self) {                                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.clear();                                                                  \
                }                                                                                  \
            },                                                                                     \
            "Clear the voxel grid")                                                                \
        .def(                                                                                      \
            "reset",                                                                               \
            [](CLASS_TYPE &self) {                                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.clear();                                                                  \
                }                                                                                  \
            },                                                                                     \
            "Reset the voxel grid, same as clear()")                                               \
        .def("size", &CLASS_TYPE::size, "Returns the size of the voxel grid")                      \
        .def("empty", &CLASS_TYPE::empty, "Returns True if the voxel grid is empty")

// ----------------------------------------
// Unified macro for VoxelGrid and VoxelBlockGrid bindings
// Parameters:
//   - CLASS_TYPE: The C++ class type
//   - PYTHON_NAME: The Python class name
//   - CLASS_DEF: Class definition (e.g., CLASS_TYPE or CLASS_TYPE,
//   std::shared_ptr<CLASS_TYPE>)
//   - CONSTRUCTOR: Constructor definition (e.g., py::init<float>() or
//   py::init<float, int>())
//   - CONSTRUCTOR_ARGS: Constructor arguments
// ----------------------------------------
#define DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, TYPE, PTR_TYPE, CONSTRUCTOR,   \
                                           ...)                                                    \
    py::class_<EXPAND_CLASS_DEF(TYPE, PTR_TYPE)>(m, PYTHON_NAME)                                   \
        .def(CONSTRUCTOR, __VA_ARGS__)                                                             \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::object colors) {                  \
                detail::integrate_with_arrays<CLASS_TYPE, double>(self, points, colors);           \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"),                         \
            py::arg("colors") = py::none())                                                        \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::object colors) {                   \
                detail::integrate_with_arrays<CLASS_TYPE, float>(self, points, colors);            \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"),                         \
            py::arg("colors") = py::none())                                                        \
        .def(                                                                                      \
            "get_voxels",                                                                          \
            [](CLASS_TYPE &self, int min_count = 1, float min_confidence = 0.0) {                  \
                typename CLASS_TYPE::VoxelGridDataType result;                                     \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    result = self.get_voxels(min_count, min_confidence);                           \
                }                                                                                  \
                return result;                                                                     \
            },                                                                                     \
            "Returns the voxels", py::arg("min_count") = 1, py::arg("min_confidence") = 0.0)       \
        .def(                                                                                      \
            "get_points",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<typename CLASS_TYPE::PosScalar, 3>> points;                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    points = self.get_points();                                                    \
                }                                                                                  \
                return points;                                                                     \
            },                                                                                     \
            "Returns the points")                                                                  \
        .def(                                                                                      \
            "get_colors",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<typename CLASS_TYPE::ColorScalar, 3>> colors;               \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    colors = self.get_colors();                                                    \
                }                                                                                  \
                return colors;                                                                     \
            },                                                                                     \
            "Returns the colors")                                                                  \
        .def(                                                                                      \
            "clear",                                                                               \
            [](CLASS_TYPE &self) {                                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.clear();                                                                  \
                }                                                                                  \
            },                                                                                     \
            "Clear the voxel grid")                                                                \
        .def(                                                                                      \
            "reset",                                                                               \
            [](CLASS_TYPE &self) {                                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.clear();                                                                  \
                }                                                                                  \
            },                                                                                     \
            "Reset the voxel grid, same as clear()")                                               \
        .def("size", &CLASS_TYPE::size, "Returns the size of the voxel grid")                      \
        .def("empty", &CLASS_TYPE::empty, "Returns True if the voxel grid is empty")

// ----------------------------------------
// Convenience wrapper macros for easier usage
// ----------------------------------------
// Helper macro to add block-specific methods (num_blocks, get_block_size, get_total_voxel_count)
#define ADD_VOXEL_BLOCK_SPECIFIC_METHODS(CLASS_TYPE)                                               \
    .def("num_blocks", &CLASS_TYPE::num_blocks, "Returns the number of blocks")                    \
        .def("get_block_size", &CLASS_TYPE::get_block_size, "Returns the block size")              \
        .def("get_total_voxel_count", &CLASS_TYPE::get_total_voxel_count,                          \
             "Returns the total voxel count")

// For VoxelGrid (non-block)
#define DEFINE_VOXEL_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                                        \
    DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, CLASS_TYPE,                        \
                                       std::shared_ptr<CLASS_TYPE>, py::init<float>(),             \
                                       py::arg("voxel_size"))

// For VoxelBlockGrid (block-based)
#define DEFINE_VOXEL_BLOCK_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                                  \
    DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, CLASS_TYPE,                        \
                                       std::shared_ptr<CLASS_TYPE>, py::init<float, int>(),        \
                                       py::arg("voxel_size"), py::arg("block_size") = 8)           \
    ADD_VOXEL_BLOCK_SPECIFIC_METHODS(CLASS_TYPE)

// For block-based semantic grids (VoxelBlockSemanticGrid,
// VoxelBlockSemanticProbabilisticGrid)
#define DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                         \
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(                                                   \
        CLASS_TYPE, PYTHON_NAME, CLASS_TYPE, std::shared_ptr<CLASS_TYPE>, py::init<float, int>(),  \
        py::arg("voxel_size"), py::arg("block_size") = 8)                                          \
    ADD_VOXEL_BLOCK_SPECIFIC_METHODS(CLASS_TYPE)

// For non-block semantic grids (VoxelSemanticGrid)
// Note: We need to pass CLASS_TYPE, std::shared_ptr<CLASS_TYPE> as a
// single parameter We use parentheses to group the comma-separated
// types
#define DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                               \
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, CLASS_TYPE,               \
                                                std::shared_ptr<CLASS_TYPE>, py::init<float>(),    \
                                                py::arg("voxel_size"))

// Unified macro to add get_voxels_in_bb bindings for both semantic and
// non-semantic grids IS_SEMANTIC: true if the grid type supports
// semantic data
#define DEFINE_MORE_VOXELS_OPS_BINDINGS(CLASS_TYPE, IS_SEMANTIC)                                   \
    .def(                                                                                          \
        "get_voxels_in_bb",                                                                        \
        [](CLASS_TYPE &self, const volumetric::BoundingBox3D &bbox, int min_count = 1,             \
           float min_confidence = 0.0, bool include_semantics = false) {                           \
            typename CLASS_TYPE::VoxelGridDataType result;                                         \
            {                                                                                      \
                py::gil_scoped_release release;                                                    \
                if constexpr (IS_SEMANTIC) {                                                       \
                    if (include_semantics) {                                                       \
                        result =                                                                   \
                            self.template get_voxels_in_bb<true>(bbox, min_count, min_confidence); \
                    } else {                                                                       \
                        result = self.template get_voxels_in_bb<false>(bbox, min_count,            \
                                                                       min_confidence);            \
                    }                                                                              \
                } else {                                                                           \
                    if (include_semantics) {                                                       \
                        throw std::runtime_error("include_semantics=True is not supported "        \
                                                 "for non-semantic grids");                        \
                    }                                                                              \
                    result =                                                                       \
                        self.template get_voxels_in_bb<false>(bbox, min_count, min_confidence);    \
                }                                                                                  \
            }                                                                                      \
            return result;                                                                         \
        },                                                                                         \
        "Get voxels within a spatial interval (bounding box)", py::arg("bbox"),                    \
        py::arg("min_count") = 1, py::arg("min_confidence") = 0.0,                                 \
        py::arg("include_semantics") = false)                                                      \
        .def(                                                                                      \
            "get_voxels_in_camera_frustrum",                                                       \
            [](CLASS_TYPE &self, const volumetric::CameraFrustrum &camera_frustrum,                \
               int min_count = 1, float min_confidence = 0.0, bool include_semantics = false) {    \
                typename CLASS_TYPE::VoxelGridDataType result;                                     \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    if constexpr (IS_SEMANTIC) {                                                   \
                        if (include_semantics) {                                                   \
                            result = self.template get_voxels_in_camera_frustrum<true>(            \
                                camera_frustrum, min_count, min_confidence);                       \
                        } else {                                                                   \
                            result = self.template get_voxels_in_camera_frustrum<false>(           \
                                camera_frustrum, min_count, min_confidence);                       \
                        }                                                                          \
                    } else {                                                                       \
                        if (include_semantics) {                                                   \
                            throw std::runtime_error("include_semantics=True is not "              \
                                                     "supported for non-semantic grids");          \
                        }                                                                          \
                        result = self.template get_voxels_in_camera_frustrum<false>(               \
                            camera_frustrum, min_count, min_confidence);                           \
                    }                                                                              \
                }                                                                                  \
                return result;                                                                     \
            },                                                                                     \
            "Get voxels within a camera frustrum", py::arg("camera_frustrum"),                     \
            py::arg("min_count") = 1, py::arg("min_confidence") = 0.0,                             \
            py::arg("include_semantics") = false)                                                  \
        .def(                                                                                      \
            "carve",                                                                               \
            [](CLASS_TYPE &self, const volumetric::CameraFrustrum &camera_frustrum,                \
               cv::Mat &depth_image, const float depth_threshold = 1e-2) {                         \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.carve(camera_frustrum, depth_image, depth_threshold);                     \
                }                                                                                  \
            },                                                                                     \
            "Carve the voxel grid using the camera frustrum and "                                  \
            "camera depths",                                                                       \
            py::arg("camera_frustrum"), py::arg("depth_image"), py::arg("depth_threshold") = 1e-2) \
        .def(                                                                                      \
            "remove_low_count_voxels",                                                             \
            [](CLASS_TYPE &self, const int min_count) {                                            \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_low_count_voxels(min_count);                                       \
                }                                                                                  \
            },                                                                                     \
            "Remove all voxels with low count", py::arg("min_count"))                              \
        .def(                                                                                      \
            "remove_low_confidence_voxels",                                                        \
            [](CLASS_TYPE &self, const float min_confidence) {                                     \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_low_confidence_voxels(min_confidence);                             \
                }                                                                                  \
            },                                                                                     \
            "Remove all voxels with low confidence", py::arg("min_confidence"))

// ----------------------------------------

void bind_volumetric_grid(py::module &m) {

    // ----------------------------------------
    // VoxelData
    // ----------------------------------------
    py::class_<volumetric::VoxelData, std::shared_ptr<volumetric::VoxelData>>(m, "VoxelData")
        .def(py::init<>())
        .def("get_position", &volumetric::VoxelData::get_position, "Returns the position")
        .def("get_color", &volumetric::VoxelData::get_color, "Returns the color")
        .def_readwrite("count", &volumetric::VoxelData::count, "Returns the count");

    // ----------------------------------------
    // VoxelSemanticData
    // ----------------------------------------
    py::class_<volumetric::VoxelSemanticData, std::shared_ptr<volumetric::VoxelSemanticData>>(
        m, "VoxelSemanticData")
        .def(py::init<>())
        .def("get_position", &volumetric::VoxelSemanticData::get_position, "Returns the position")
        .def("get_color", &volumetric::VoxelSemanticData::get_color, "Returns the color")
        .def_readwrite("count", &volumetric::VoxelSemanticData::count, "Returns the count")
        .def("get_object_id", &volumetric::VoxelSemanticData::get_object_id,
             "Returns the object ID")
        .def("get_class_id", &volumetric::VoxelSemanticData::get_class_id, "Returns the class ID")
        .def("get_confidence", &volumetric::VoxelSemanticData::get_confidence,
             "Returns the confidence (normalized joint probability of the most likely object and "
             "class IDs)")
        .def("get_confidence_counter", &volumetric::VoxelSemanticData::get_confidence_counter,
             "Returns the confidence counter (count of the most likely object and class IDs)");

    // ----------------------------------------
    // VoxelGrid
    // ----------------------------------------
    DEFINE_VOXEL_GRID_BINDINGS(volumetric::VoxelGrid, "VoxelGrid")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelGrid, false);

    // ----------------------------------------
    // VoxelBlockGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_GRID_BINDINGS(volumetric::VoxelBlockGrid, "VoxelBlockGrid")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelBlockGrid, false);

    // ----------------------------------------
    // VoxelSemanticGrid
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGrid, "VoxelSemanticGrid")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelSemanticGrid, true);

    // ----------------------------------------
    // VoxelSemanticGrid2
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGrid2, "VoxelSemanticGrid2")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelSemanticGrid2, true);

    // ----------------------------------------
    // VoxelSemanticGridProbabilistic
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGridProbabilistic,
                                        "VoxelSemanticGridProbabilistic")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelSemanticGridProbabilistic, true);

    // ----------------------------------------
    // VoxelSemanticGridProbabilistic2
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGridProbabilistic2,
                                        "VoxelSemanticGridProbabilistic2")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelSemanticGridProbabilistic2, true);

    // ----------------------------------------
    // VoxelBlockSemanticGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticGrid,
                                              "VoxelBlockSemanticGrid")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelBlockSemanticGrid, true);

    // ----------------------------------------
    // VoxelBlockSemanticGrid2
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticGrid2,
                                              "VoxelBlockSemanticGrid2")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelBlockSemanticGrid2, true);

    // ----------------------------------------
    // VoxelBlockSemanticProbabilisticGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticProbabilisticGrid,
                                              "VoxelBlockSemanticProbabilisticGrid")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelBlockSemanticProbabilisticGrid, true);

    // ----------------------------------------
    // VoxelBlockSemanticProbabilisticGrid2
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticProbabilisticGrid2,
                                              "VoxelBlockSemanticProbabilisticGrid2")
    DEFINE_MORE_VOXELS_OPS_BINDINGS(volumetric::VoxelBlockSemanticProbabilisticGrid2, true);
}
