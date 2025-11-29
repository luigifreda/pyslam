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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "tbb_utils.h"
#include "voxel_block_grid.h"
#include "voxel_block_semantic_grid.h"
#include "voxel_grid.h"
#include "voxel_semantic_grid.h"

namespace py = pybind11;

// ----------------------------------------
// Unified macro for VoxelSemanticGrid bindings
// Parameters:
//   - CLASS_TYPE: The C++ class type
//   - PYTHON_NAME: The Python class name
//   - CLASS_DEF: Class definition (e.g., CLASS_TYPE or CLASS_TYPE, std::shared_ptr<CLASS_TYPE>)
//   - CONSTRUCTOR: Constructor definition (e.g., py::init<float>() or py::init<float, int>())
//   - CONSTRUCTOR_ARGS: Constructor arguments (e.g., py::arg("voxel_size") or
//   py::arg("voxel_size"), py::arg("block_size") = 8)
// ----------------------------------------
// Helper to remove parentheses from CLASS_DEF if present
#define REMOVE_PARENS(...) __VA_ARGS__
// Helper to expand class definition properly
#define EXPAND_CLASS_DEF(x) REMOVE_PARENS x
#define DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, CLASS_DEF,            \
                                                    CONSTRUCTOR, ...)                              \
    py::class_<EXPAND_CLASS_DEF(CLASS_DEF)>(m, PYTHON_NAME)                                        \
        .def(CONSTRUCTOR, __VA_ARGS__)                                                             \
        .def("set_depth_threshold", &CLASS_TYPE::set_depth_threshold,                              \
             py::arg("depth_threshold") = 5.0)                                                     \
        .def("set_depth_decay_rate", &CLASS_TYPE::set_depth_decay_rate,                            \
             py::arg("depth_decay_rate") = 0.07)                                                   \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<uint8_t> colors,          \
               py::array_t<int> class_ids, py::object instance_ids = py::none(),                   \
               py::object depths = py::none()) {                                                   \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                auto class_ids_info = class_ids.request();                                         \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    if (depths.is_none()) {                                                        \
                        if (instance_ids.is_none()) {                                              \
                            self.integrate_raw<double, uint8_t, std::nullptr_t, int>(              \
                                static_cast<const double *>(pts_info.ptr), pts_info.shape[0],      \
                                static_cast<const uint8_t *>(cols_info.ptr), nullptr,              \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        } else {                                                                   \
                            auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();         \
                            auto instance_ids_info = instance_ids_arr.request();                   \
                            self.integrate_raw<double, uint8_t, int, int>(                         \
                                static_cast<const double *>(pts_info.ptr), pts_info.shape[0],      \
                                static_cast<const uint8_t *>(cols_info.ptr),                       \
                                static_cast<const int *>(instance_ids_info.ptr),                   \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        }                                                                          \
                    } else {                                                                       \
                        try {                                                                      \
                            auto depths_arr = depths.cast<py::array_t<float>>();                   \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<double, uint8_t, std::nullptr_t, int, float>(   \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const uint8_t *>(cols_info.ptr), nullptr,          \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<double, uint8_t, int, int, float>(              \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const uint8_t *>(cols_info.ptr),                   \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            }                                                                      \
                        } catch (const py::cast_error &) {                                         \
                            auto depths_arr = depths.cast<py::array_t<double>>();                  \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<double, uint8_t, std::nullptr_t, int, double>(  \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const uint8_t *>(cols_info.ptr), nullptr,          \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<double, uint8_t, int, int, double>(             \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const uint8_t *>(cols_info.ptr),                   \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional instance_ids and depths)",    \
            py::arg("points"), py::arg("colors"), py::arg("class_ids"),                            \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<float> colors,            \
               py::array_t<int> class_ids, py::object instance_ids = py::none(),                   \
               py::object depths = py::none()) {                                                   \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                auto class_ids_info = class_ids.request();                                         \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    if (depths.is_none()) {                                                        \
                        if (instance_ids.is_none()) {                                              \
                            self.integrate_raw<double, float, std::nullptr_t, int>(                \
                                static_cast<const double *>(pts_info.ptr), pts_info.shape[0],      \
                                static_cast<const float *>(cols_info.ptr), nullptr,                \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        } else {                                                                   \
                            auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();         \
                            auto instance_ids_info = instance_ids_arr.request();                   \
                            self.integrate_raw<double, float, int, int>(                           \
                                static_cast<const double *>(pts_info.ptr), pts_info.shape[0],      \
                                static_cast<const float *>(cols_info.ptr),                         \
                                static_cast<const int *>(instance_ids_info.ptr),                   \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        }                                                                          \
                    } else {                                                                       \
                        try {                                                                      \
                            auto depths_arr = depths.cast<py::array_t<float>>();                   \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<double, float, std::nullptr_t, int, float>(     \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const float *>(cols_info.ptr), nullptr,            \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<double, float, int, int, float>(                \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const float *>(cols_info.ptr),                     \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            }                                                                      \
                        } catch (const py::cast_error &) {                                         \
                            auto depths_arr = depths.cast<py::array_t<double>>();                  \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<double, float, std::nullptr_t, int, double>(    \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const float *>(cols_info.ptr), nullptr,            \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<double, float, int, int, double>(               \
                                    static_cast<const double *>(pts_info.ptr), pts_info.shape[0],  \
                                    static_cast<const float *>(cols_info.ptr),                     \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional instance_ids and depths)",    \
            py::arg("points"), py::arg("colors"), py::arg("class_ids"),                            \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<uint8_t> colors,           \
               py::array_t<int> class_ids, py::object instance_ids = py::none(),                   \
               py::object depths = py::none()) {                                                   \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                auto class_ids_info = class_ids.request();                                         \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    if (depths.is_none()) {                                                        \
                        if (instance_ids.is_none()) {                                              \
                            self.integrate_raw<float, uint8_t, std::nullptr_t, int>(               \
                                static_cast<const float *>(pts_info.ptr), pts_info.shape[0],       \
                                static_cast<const uint8_t *>(cols_info.ptr), nullptr,              \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        } else {                                                                   \
                            auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();         \
                            auto instance_ids_info = instance_ids_arr.request();                   \
                            self.integrate_raw<float, uint8_t, int, int>(                          \
                                static_cast<const float *>(pts_info.ptr), pts_info.shape[0],       \
                                static_cast<const uint8_t *>(cols_info.ptr),                       \
                                static_cast<const int *>(instance_ids_info.ptr),                   \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        }                                                                          \
                    } else {                                                                       \
                        try {                                                                      \
                            auto depths_arr = depths.cast<py::array_t<float>>();                   \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<float, uint8_t, std::nullptr_t, int, float>(    \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const uint8_t *>(cols_info.ptr), nullptr,          \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<float, uint8_t, int, int, float>(               \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const uint8_t *>(cols_info.ptr),                   \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            }                                                                      \
                        } catch (const py::cast_error &) {                                         \
                            auto depths_arr = depths.cast<py::array_t<double>>();                  \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<float, uint8_t, std::nullptr_t, int, double>(   \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const uint8_t *>(cols_info.ptr), nullptr,          \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<float, uint8_t, int, int, double>(              \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const uint8_t *>(cols_info.ptr),                   \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional instance_ids and depths)",    \
            py::arg("points"), py::arg("colors"), py::arg("class_ids"),                            \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<float> colors,             \
               py::array_t<int> class_ids, py::object instance_ids = py::none(),                   \
               py::object depths = py::none()) {                                                   \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                auto class_ids_info = class_ids.request();                                         \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    if (depths.is_none()) {                                                        \
                        if (instance_ids.is_none()) {                                              \
                            self.integrate_raw<float, float, std::nullptr_t, int>(                 \
                                static_cast<const float *>(pts_info.ptr), pts_info.shape[0],       \
                                static_cast<const float *>(cols_info.ptr), nullptr,                \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        } else {                                                                   \
                            auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();         \
                            auto instance_ids_info = instance_ids_arr.request();                   \
                            self.integrate_raw<float, float, int, int>(                            \
                                static_cast<const float *>(pts_info.ptr), pts_info.shape[0],       \
                                static_cast<const float *>(cols_info.ptr),                         \
                                static_cast<const int *>(instance_ids_info.ptr),                   \
                                static_cast<const int *>(class_ids_info.ptr));                     \
                        }                                                                          \
                    } else {                                                                       \
                        try {                                                                      \
                            auto depths_arr = depths.cast<py::array_t<float>>();                   \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<float, float, std::nullptr_t, int, float>(      \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const float *>(cols_info.ptr), nullptr,            \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<float, float, int, int, float>(                 \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const float *>(cols_info.ptr),                     \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const float *>(depths_info.ptr));                  \
                            }                                                                      \
                        } catch (const py::cast_error &) {                                         \
                            auto depths_arr = depths.cast<py::array_t<double>>();                  \
                            auto depths_info = depths_arr.request();                               \
                            if (instance_ids.is_none()) {                                          \
                                self.integrate_raw<float, float, std::nullptr_t, int, double>(     \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const float *>(cols_info.ptr), nullptr,            \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            } else {                                                               \
                                auto instance_ids_arr = instance_ids.cast<py::array_t<int>>();     \
                                auto instance_ids_info = instance_ids_arr.request();               \
                                self.integrate_raw<float, float, int, int, double>(                \
                                    static_cast<const float *>(pts_info.ptr), pts_info.shape[0],   \
                                    static_cast<const float *>(cols_info.ptr),                     \
                                    static_cast<const int *>(instance_ids_info.ptr),               \
                                    static_cast<const int *>(class_ids_info.ptr),                  \
                                    static_cast<const double *>(depths_info.ptr));                 \
                            }                                                                      \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid (with optional instance_ids and depths)",    \
            py::arg("points"), py::arg("colors"), py::arg("class_ids"),                            \
            py::arg("instance_ids") = py::none(), py::arg("depths") = py::none())                  \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<uint8_t> colors,          \
               const int instance_id, const int class_id) {                                        \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_segment_raw<double, uint8_t>(                                   \
                        static_cast<const double *>(pts_info.ptr), pts_info.shape[0],              \
                        static_cast<const uint8_t *>(cols_info.ptr), instance_id, class_id);       \
                }                                                                                  \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("instance_id"), py::arg("class_id"))                        \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<float> colors,            \
               const int instance_id, const int class_id) {                                        \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_segment_raw<double, float>(                                     \
                        static_cast<const double *>(pts_info.ptr), pts_info.shape[0],              \
                        static_cast<const float *>(cols_info.ptr), instance_id, class_id);         \
                }                                                                                  \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("instance_id"), py::arg("class_id"))                        \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<uint8_t> colors,           \
               const int instance_id, const int class_id) {                                        \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_segment_raw<float, uint8_t>(                                    \
                        static_cast<const float *>(pts_info.ptr), pts_info.shape[0],               \
                        static_cast<const uint8_t *>(cols_info.ptr), instance_id, class_id);       \
                }                                                                                  \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("instance_id"), py::arg("class_id"))                        \
        .def(                                                                                      \
            "integrate_segment",                                                                   \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<float> colors,             \
               const int instance_id, const int class_id) {                                        \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_segment_raw<float, float>(                                      \
                        static_cast<const float *>(pts_info.ptr), pts_info.shape[0],               \
                        static_cast<const float *>(cols_info.ptr), instance_id, class_id);         \
                }                                                                                  \
            },                                                                                     \
            "Insert a segment of points into the voxel grid", py::arg("points"),                   \
            py::arg("colors"), py::arg("instance_id"), py::arg("class_id"))                        \
        .def(                                                                                      \
            "merge_segments",                                                                      \
            [](CLASS_TYPE &self, const int instance_id1, const int instance_id2) {                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.merge_segments(instance_id1, instance_id2);                               \
                }                                                                                  \
            },                                                                                     \
            "Merge two segments of voxels into a single segment of voxels with the same instance " \
            "ID",                                                                                  \
            py::arg("instance_id1"), py::arg("instance_id2"))                                      \
        .def(                                                                                      \
            "remove_segment",                                                                      \
            [](CLASS_TYPE &self, const int instance_id) {                                          \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_segment(instance_id);                                              \
                }                                                                                  \
            },                                                                                     \
            "Remove a segment of voxels from the voxel grid", py::arg("instance_id"))              \
        .def(                                                                                      \
            "remove_low_confidence_segments",                                                      \
            [](CLASS_TYPE &self, const int min_confidence_counter) {                               \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.remove_low_confidence_segments(min_confidence_counter);                   \
                }                                                                                  \
            },                                                                                     \
            "Remove segments of voxels with low confidence counter from the voxel grid",           \
            py::arg("min_confidence_counter"))                                                     \
        .def(                                                                                      \
            "get_voxel_data",                                                                      \
            [](CLASS_TYPE &self, int min_count = 1) {                                              \
                std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>,  \
                           std::vector<int>, std::vector<int>>                                     \
                    result;                                                                        \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    result = self.get_voxel_data(min_count);                                       \
                }                                                                                  \
                auto &[points, colors, class_ids, instance_ids] = result;                          \
                return py::make_tuple(std::move(points), std::move(colors), std::move(class_ids),  \
                                      std::move(instance_ids));                                    \
            },                                                                                     \
            "Returns a tuple (points, colors, class_ids, instance_ids)", py::arg("min_count"))     \
        .def(                                                                                      \
            "get_points",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<double, 3>> points;                                         \
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
                std::vector<std::array<float, 3>> colors;                                          \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    colors = self.get_colors();                                                    \
                }                                                                                  \
                return colors;                                                                     \
            },                                                                                     \
            "Returns the colors")                                                                  \
        .def(                                                                                      \
            "get_segments",                                                                        \
            [](CLASS_TYPE &self) {                                                                 \
                std::unordered_map<int, std::vector<std::array<double, 3>>> segments;              \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    segments = self.get_segments();                                                \
                }                                                                                  \
                return segments;                                                                   \
            },                                                                                     \
            "Returns a dictionary of segments of voxels")                                          \
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
//   - CLASS_DEF: Class definition (e.g., CLASS_TYPE or CLASS_TYPE, std::shared_ptr<CLASS_TYPE>)
//   - CONSTRUCTOR: Constructor definition (e.g., py::init<float>() or py::init<float, int>())
//   - CONSTRUCTOR_ARGS: Constructor arguments
// ----------------------------------------
#define DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME, CLASS_DEF, CONSTRUCTOR, ...)   \
    py::class_<EXPAND_CLASS_DEF(CLASS_DEF)>(m, PYTHON_NAME)                                        \
        .def(CONSTRUCTOR, __VA_ARGS__)                                                             \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<uint8_t> colors) {        \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<double, uint8_t>(                                           \
                        static_cast<const double *>(pts_info.ptr), pts_info.shape[0],              \
                        static_cast<const uint8_t *>(cols_info.ptr));                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"), py::arg("colors"))      \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<double> points, py::array_t<float> colors) {          \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<double, float>(static_cast<const double *>(pts_info.ptr),   \
                                                      pts_info.shape[0],                           \
                                                      static_cast<const float *>(cols_info.ptr));  \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"), py::arg("colors"))      \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<uint8_t> colors) {         \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<float, uint8_t>(                                            \
                        static_cast<const float *>(pts_info.ptr), pts_info.shape[0],               \
                        static_cast<const uint8_t *>(cols_info.ptr));                              \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"), py::arg("colors"))      \
        .def(                                                                                      \
            "integrate",                                                                           \
            [](CLASS_TYPE &self, py::array_t<float> points, py::array_t<float> colors) {           \
                auto pts_info = points.request();                                                  \
                auto cols_info = colors.request();                                                 \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<float, float>(static_cast<const float *>(pts_info.ptr),     \
                                                     pts_info.shape[0],                            \
                                                     static_cast<const float *>(cols_info.ptr));   \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"), py::arg("colors"))      \
        .def(                                                                                      \
            "integrate_points",                                                                    \
            [](CLASS_TYPE &self, py::array_t<double> points) {                                     \
                auto pts_info = points.request();                                                  \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<double>(static_cast<const double *>(pts_info.ptr),          \
                                               pts_info.shape[0]);                                 \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"))                         \
        .def(                                                                                      \
            "integrate_points",                                                                    \
            [](CLASS_TYPE &self, py::array_t<float> points) {                                      \
                auto pts_info = points.request();                                                  \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    self.integrate_raw<float>(static_cast<const float *>(pts_info.ptr),            \
                                              pts_info.shape[0]);                                  \
                }                                                                                  \
            },                                                                                     \
            "Insert a point cloud into the voxel grid", py::arg("points"))                         \
        .def(                                                                                      \
            "get_voxel_data",                                                                      \
            [](CLASS_TYPE &self, int min_count = 1) {                                              \
                std::pair<std::vector<std::array<double, 3>>, std::vector<std::array<float, 3>>>   \
                    result;                                                                        \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    result = self.get_voxel_data(min_count);                                       \
                }                                                                                  \
                auto &[points, colors] = result;                                                   \
                return py::make_tuple(std::move(points), std::move(colors));                       \
            },                                                                                     \
            "Returns a tuple (points, colors)", py::arg("min_count"))                              \
        .def(                                                                                      \
            "get_points",                                                                          \
            [](CLASS_TYPE &self) {                                                                 \
                std::vector<std::array<double, 3>> points;                                         \
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
                std::vector<std::array<float, 3>> colors;                                          \
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
// For VoxelGrid (non-block)
#define DEFINE_VOXEL_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                                        \
    DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME,                                    \
                                       (CLASS_TYPE, std::shared_ptr<CLASS_TYPE>),                  \
                                       py::init<float>(), py::arg("voxel_size"))

// For VoxelBlockGrid (block-based)
#define DEFINE_VOXEL_BLOCK_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                                  \
    DEFINE_VOXEL_GRID_BINDINGS_UNIFIED(                                                            \
        CLASS_TYPE, PYTHON_NAME, (CLASS_TYPE, std::shared_ptr<CLASS_TYPE>),                        \
        py::init<float, int>(), py::arg("voxel_size"), py::arg("block_size") = 8)                  \
        .def("num_blocks", &CLASS_TYPE::num_blocks, "Returns the number of blocks")                \
        .def("get_block_size", &CLASS_TYPE::get_block_size, "Returns the block size")              \
        .def("get_total_voxel_count", &CLASS_TYPE::get_total_voxel_count,                          \
             "Returns the total voxel count")

// For block-based semantic grids (VoxelBlockSemanticGrid, VoxelBlockSemanticProbabilisticGrid)
#define DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                         \
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(                                                   \
        CLASS_TYPE, PYTHON_NAME, (CLASS_TYPE, std::shared_ptr<CLASS_TYPE>),                        \
        py::init<float, int>(), py::arg("voxel_size"), py::arg("block_size") = 8)

// For non-block semantic grids (VoxelSemanticGrid)
// Note: We need to pass CLASS_TYPE, std::shared_ptr<CLASS_TYPE> as a single parameter
// We use parentheses to group the comma-separated types
#define DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(CLASS_TYPE, PYTHON_NAME)                               \
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS_UNIFIED(CLASS_TYPE, PYTHON_NAME,                           \
                                                (CLASS_TYPE, std::shared_ptr<CLASS_TYPE>),         \
                                                py::init<float>(), py::arg("voxel_size"))

// ----------------------------------------

PYBIND11_MODULE(volumetric_grid, m) {

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
        .def_readwrite("instance_id", &volumetric::VoxelSemanticData::instance_id,
                       "Returns the instance ID")
        .def_readwrite("class_id", &volumetric::VoxelSemanticData::class_id, "Returns the class ID")
        .def_readwrite("confidence_counter", &volumetric::VoxelSemanticData::confidence_counter,
                       "Returns the confidence counter");

    // ----------------------------------------
    // TBB Utils
    // ----------------------------------------

    py::class_<volumetric::TBBUtils>(m, "TBBUtils")
        .def_static(
            "set_max_threads", &volumetric::TBBUtils::set_max_threads,
            "Set the maximum number of threads for TBB parallel operations (global setting)",
            py::arg("num_threads"))
        .def_static("get_max_threads", &volumetric::TBBUtils::get_max_threads,
                    "Get the current maximum number of "
                    "threads for TBB");

    // ----------------------------------------
    // VoxelGrid
    // ----------------------------------------
    DEFINE_VOXEL_GRID_BINDINGS(volumetric::VoxelGrid, "VoxelGrid");

    // ----------------------------------------
    // VoxelBlockGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_GRID_BINDINGS(volumetric::VoxelBlockGrid, "VoxelBlockGrid");

    // ----------------------------------------
    // VoxelSemanticGrid
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGrid, "VoxelSemanticGrid");

    // ----------------------------------------
    // VoxelSemanticGridProbabilistic
    // ----------------------------------------
    DEFINE_VOXEL_SEMANTIC_GRID_BINDINGS(volumetric::VoxelSemanticGridProbabilistic,
                                        "VoxelSemanticGridProbabilistic");

    // ----------------------------------------
    // VoxelBlockSemanticGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticGrid,
                                              "VoxelBlockSemanticGrid");

    // ----------------------------------------
    // VoxelBlockSemanticProbabilisticGrid
    // ----------------------------------------
    DEFINE_VOXEL_BLOCK_SEMANTIC_GRID_BINDINGS(volumetric::VoxelBlockSemanticProbabilisticGrid,
                                              "VoxelBlockSemanticProbabilisticGrid");
}
