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

#include "eigen_aliases.h"
#include "map.h"
#include "optimizer_g2o_bind_helpers.h"
#include "py_wrappers.h"
#include "utils/map_helpers.h"

namespace py = pybind11;

namespace pyslam {

using PyMapMutexWrapper = pyslam::PyMutexWrapperT<pyslam::MapMutex>;

// Wrapper for map->optimize to return double (mean_squared_error)
inline std::pair<double, py::dict> map_optimize_wrapper(MapPtr map, int local_window_size,
                                                        bool verbose, int rounds,
                                                        bool use_robust_kernel, bool do_cull_points,
                                                        py::object abort_flag) {

    PyG2oAbortFlag abort_flag_wrapper(abort_flag);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();
    double mean_squared_error = 0.0;
    {

        py::gil_scoped_release gil_release;

        mean_squared_error = map->optimize(local_window_size, verbose, rounds, use_robust_kernel,
                                           do_cull_points, abort_flag_value_ptr);

        abort_flag_wrapper.stop_monitoring();
    }

    return std::make_pair(mean_squared_error, py::dict());
}

// Wrapper for map->locally_optimize to return double (mean_squared_error)
inline double map_locally_optimize_wrapper(MapPtr map, KeyFramePtr kf_ref, bool verbose, int rounds,
                                           py::object abort_flag,
                                           py::object mp_abort_flag = py::none()) {

    py::object abort_flag_object = abort_flag.is_none() ? mp_abort_flag : abort_flag;
    PyG2oAbortFlag abort_flag_wrapper(abort_flag_object);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();

    double mean_squared_error = 0.0;
    {
        py::gil_scoped_release gil_release;

        mean_squared_error = map->locally_optimize(kf_ref, verbose, rounds, abort_flag_value_ptr);

        abort_flag_wrapper.stop_monitoring();
    }

    return mean_squared_error;
}

// Wrapper for Map::add_points to release the GIL during heavy C++ work
inline std::tuple<int, std::vector<bool>, std::vector<MapPointPtr>>
map_add_points_wrapper(MapPtr map, const std::vector<Eigen::Vector3d> &points3d,
                       const std::optional<std::vector<bool>> &mask_pts3d, KeyFramePtr &kf1,
                       KeyFramePtr &kf2, const std::vector<int> &idxs1,
                       const std::vector<int> &idxs2, const cv::Mat &img, bool do_check = true,
                       double cos_max_parallax = Parameters::kCosMaxParallax,
                       std::optional<double> far_points_threshold = std::nullopt) {
    py::gil_scoped_release gil_release;
    return map->add_points(points3d, mask_pts3d, kf1, kf2, idxs1, idxs2, img, do_check,
                           cos_max_parallax, far_points_threshold);
}

// Wrapper for Map::add_stereo_points to release the GIL during heavy C++ work
inline int map_add_stereo_points_wrapper(MapPtr map, const std::vector<Eigen::Vector3d> &points3d,
                                         const std::optional<std::vector<bool>> &mask_pts3d,
                                         FramePtr &f, KeyFramePtr &kf, const std::vector<int> &idxs,
                                         const cv::Mat &img) {
    py::gil_scoped_release gil_release;
    return map->add_stereo_points(points3d, mask_pts3d, f, kf, idxs, img);
}

// Wrapper for Map::remove_points_with_big_reproj_err to release the GIL
inline void map_remove_points_with_big_reproj_err_wrapper(MapPtr map,
                                                          const std::vector<MapPointPtr> &points) {
    py::gil_scoped_release gil_release;
    map->remove_points_with_big_reproj_err(points);
}

// Wrapper for Map::compute_mean_reproj_error to release the GIL
inline float map_compute_mean_reproj_error_wrapper(MapPtr map,
                                                   const std::vector<MapPointPtr> &points = {}) {
    py::gil_scoped_release gil_release;
    return map->compute_mean_reproj_error(points);
}

// Wrapper for Map serialization/deserialization to release the GIL
inline std::string map_to_json_wrapper(MapPtr map, const std::string &out_json = "{}") {
    py::gil_scoped_release gil_release;
    return map->to_json(out_json);
}

inline std::string map_serialize_wrapper(MapPtr map) {
    py::gil_scoped_release gil_release;
    return map->serialize();
}

inline void map_from_json_wrapper(MapPtr map, const std::string &loaded_json) {
    py::gil_scoped_release gil_release;
    map->from_json(loaded_json);
}

inline void map_deserialize_wrapper(MapPtr map, const std::string &s) {
    py::gil_scoped_release gil_release;
    map->deserialize(s);
}

inline void map_save_wrapper(MapPtr map, const std::string &filename) {
    py::gil_scoped_release gil_release;
    map->save(filename);
}

inline void map_load_wrapper(MapPtr map, const std::string &filename) {
    py::gil_scoped_release gil_release;
    map->load(filename);
}

// Wrapper for LocalMapBase::update_from_keyframes template method
inline std::tuple<pyslam::KeyFrameIdSet, std::unordered_set<pyslam::MapPointPtr>,
                  pyslam::KeyFrameIdSet>
local_map_update_from_keyframes_wrapper(pyslam::LocalMapBase &self,
                                        const pyslam::KeyFrameIdSet &local_keyframes) {
    py::gil_scoped_release gil_release;
    return self.update_from_keyframes(local_keyframes);
}

inline std::tuple<KeyFramePtr, std::vector<KeyFramePtr>, std::vector<MapPointPtr>>
local_map_get_frame_covisibles_wrapper(pyslam::LocalMapBase &self, pyslam::FramePtr frame) {
    py::gil_scoped_release gil_release;
    return self.get_frame_covisibles(frame);
}

// Wrappers for LocalWindowMap/LocalCovisibilityMap to release the GIL
inline KeyFrameIdSet
local_window_map_update_keyframes_wrapper(pyslam::LocalWindowMap &self,
                                          const KeyFramePtr &kf_ref = nullptr) {
    py::gil_scoped_release gil_release;
    return self.update_keyframes(kf_ref);
}

inline std::vector<KeyFramePtr>
local_window_map_get_best_neighbors_wrapper(pyslam::LocalWindowMap &self,
                                            const KeyFramePtr &kf_ref = nullptr, int N = 20) {
    py::gil_scoped_release gil_release;
    return self.get_best_neighbors(kf_ref, N);
}

inline std::tuple<KeyFrameIdSet, std::unordered_set<MapPointPtr>, KeyFrameIdSet>
local_window_map_update_wrapper(pyslam::LocalWindowMap &self, const KeyFramePtr &kf_ref = nullptr) {
    py::gil_scoped_release gil_release;
    return self.update(kf_ref);
}

inline KeyFrameIdSet
local_covisibility_map_update_keyframes_wrapper(pyslam::LocalCovisibilityMap &self,
                                                const KeyFramePtr &kf_ref) {
    py::gil_scoped_release gil_release;
    return self.update_keyframes(kf_ref);
}

inline std::vector<KeyFramePtr>
local_covisibility_map_get_best_neighbors_wrapper(pyslam::LocalCovisibilityMap &self,
                                                  const KeyFramePtr &kf_ref, int N = 20) {
    py::gil_scoped_release gil_release;
    return self.get_best_neighbors(kf_ref, N);
}

inline std::tuple<KeyFrameIdSet, std::unordered_set<MapPointPtr>, KeyFrameIdSet>
local_covisibility_map_update_wrapper(pyslam::LocalCovisibilityMap &self,
                                      const KeyFramePtr &kf_ref) {
    py::gil_scoped_release gil_release;
    return self.update(kf_ref);
}

} // namespace pyslam

void bind_map(pybind11::module &m) {

    // ------------------------------------------------------------
    // ReloadedSessionMapInfo class
    py::class_<pyslam::ReloadedSessionMapInfo, std::shared_ptr<pyslam::ReloadedSessionMapInfo>>(
        m, "ReloadedSessionMapInfo")
        .def(py::init<int, int, int, int, int>(), py::arg("num_keyframes") = 0,
             py::arg("num_points") = 0, py::arg("max_point_id") = 0, py::arg("max_frame_id") = 0,
             py::arg("max_keyframe_id") = 0)
        .def_readwrite("num_keyframes", &pyslam::ReloadedSessionMapInfo::num_keyframes)
        .def_readwrite("num_points", &pyslam::ReloadedSessionMapInfo::num_points)
        .def_readwrite("max_point_id", &pyslam::ReloadedSessionMapInfo::max_point_id)
        .def_readwrite("max_frame_id", &pyslam::ReloadedSessionMapInfo::max_frame_id)
        .def_readwrite("max_keyframe_id", &pyslam::ReloadedSessionMapInfo::max_keyframe_id);

    // ------------------------------------------------------------
    // VectorProxy bindings for common types used in MapStateData
    // Expose VectorProxy<Mat4d> for poses and allows appending and other list-like operations
    py::class_<pyslam::VectorProxy<pyslam::Mat4d>>(m, "VectorProxyMat4d")
        .def("__getitem__", &pyslam::VectorProxy<pyslam::Mat4d>::getitem,
             py::return_value_policy::reference_internal)
        .def("__setitem__", &pyslam::VectorProxy<pyslam::Mat4d>::setitem)
        .def("append", &pyslam::VectorProxy<pyslam::Mat4d>::append, "Append an item (zero-copy)")
        .def("extend", &pyslam::VectorProxy<pyslam::Mat4d>::extend, "Extend with multiple items")
        .def("__len__", &pyslam::VectorProxy<pyslam::Mat4d>::size)
        .def("size", &pyslam::VectorProxy<pyslam::Mat4d>::size)
        .def("empty", &pyslam::VectorProxy<pyslam::Mat4d>::empty)
        .def("clear", &pyslam::VectorProxy<pyslam::Mat4d>::clear)
        .def("to_list", &pyslam::VectorProxy<pyslam::Mat4d>::to_list)
        .def(py::pickle(
            [](const pyslam::VectorProxy<pyslam::Mat4d> &self) { return self.__getstate__(); },
            [](py::tuple t) {
                // Unpickling: create a new vector from the list
                // Note: This creates a standalone vector, not connected to MapStateData
                // In practice, VectorProxy should be created from MapStateData properties
                auto vec = std::make_shared<std::vector<pyslam::Mat4d>>();
                py::list items = t[0].cast<py::list>();
                vec->reserve(py::len(items));
                for (auto item : items) {
                    vec->push_back(py::cast<pyslam::Mat4d>(item));
                }
                return pyslam::VectorProxy<pyslam::Mat4d>(std::move(vec));
            }))
        .def("__getstate__", &pyslam::VectorProxy<pyslam::Mat4d>::__getstate__);

    // Expose VectorProxy<double> for timestamps and allows appending and other list-like operations
    py::class_<pyslam::VectorProxy<double>>(m, "VectorProxyDouble")
        .def("__getitem__", &pyslam::VectorProxy<double>::getitem,
             py::return_value_policy::reference_internal)
        .def("__setitem__", &pyslam::VectorProxy<double>::setitem)
        .def("append", &pyslam::VectorProxy<double>::append, "Append an item (zero-copy)")
        .def("extend", &pyslam::VectorProxy<double>::extend, "Extend with multiple items")
        .def("__len__", &pyslam::VectorProxy<double>::size)
        .def("size", &pyslam::VectorProxy<double>::size)
        .def("empty", &pyslam::VectorProxy<double>::empty)
        .def("clear", &pyslam::VectorProxy<double>::clear)
        .def("to_list", &pyslam::VectorProxy<double>::to_list)
        .def(py::pickle([](const pyslam::VectorProxy<double> &self) { return self.__getstate__(); },
                        [](py::tuple t) {
                            // Unpickling: create a new vector from the list
                            // Note: This creates a standalone vector, not connected to MapStateData
                            // In practice, VectorProxy should be created from MapStateData
                            // properties
                            auto vec = std::make_shared<std::vector<double>>();
                            py::list items = t[0].cast<py::list>();
                            vec->reserve(py::len(items));
                            for (auto item : items) {
                                vec->push_back(py::cast<double>(item));
                            }
                            return pyslam::VectorProxy<double>(std::move(vec));
                        }))
        .def("__getstate__", &pyslam::VectorProxy<double>::__getstate__);

#define USE_VECTOR_PROXY 1
    // ------------------------------------------------------------
    // MapStateData class - complete interface matching Python MapStateData
    // Use VectorProxy for vectors to enable .append() with zero-copy access
    py::class_<pyslam::MapStateData, std::shared_ptr<pyslam::MapStateData>>(m, "MapStateData")
        .def(py::init<>())
#if USE_VECTOR_PROXY
        // Use VectorProxy for poses to enable .append() with zero-copy
        // Return by value instead of unique_ptr to avoid heap allocation overhead
        .def_property(
            "poses",
            [](pyslam::MapStateData &self) {
                return pyslam::VectorProxy<pyslam::Mat4d>(self.poses);
            },
            [](pyslam::MapStateData &self, py::object value) {
                // Allow assignment from list/iterable
                pyslam::VectorProxy<pyslam::Mat4d> proxy(self.poses);
                proxy.clear();
                proxy.extend(value);
            },
            py::return_value_policy::reference_internal)
        // Use VectorProxy for pose_timestamps
        // Return by value instead of unique_ptr to avoid heap allocation overhead
        .def_property(
            "pose_timestamps",
            [](pyslam::MapStateData &self) {
                return pyslam::VectorProxy<double>(self.pose_timestamps);
            },
            [](pyslam::MapStateData &self, py::object value) {
                pyslam::VectorProxy<double> proxy(self.pose_timestamps);
                proxy.clear();
                proxy.extend(value);
            },
            py::return_value_policy::reference_internal)
    // Keep other vectors as readwrite for now (can be converted to VectorProxy if needed)
#else
        .def_readwrite("poses", &pyslam::MapStateData::poses)
        .def_readwrite("pose_timestamps", &pyslam::MapStateData::pose_timestamps)
#endif
        .def_readwrite("fov_centers", &pyslam::MapStateData::fov_centers)
        .def_readwrite("fov_centers_colors", &pyslam::MapStateData::fov_centers_colors)
        .def_readwrite("points", &pyslam::MapStateData::points)
        .def_readwrite("colors", &pyslam::MapStateData::colors)
        .def_readwrite("semantic_colors", &pyslam::MapStateData::semantic_colors)
        .def_readwrite("covisibility_graph", &pyslam::MapStateData::covisibility_graph)
        .def_readwrite("spanning_tree", &pyslam::MapStateData::spanning_tree)
        .def_readwrite("loops", &pyslam::MapStateData::loops)
        .def(py::pickle([](const pyslam::MapStateData &self) { return self.state_tuple(); },
                        [](py::tuple t) {
                            auto map_state =
                                std::make_shared<pyslam::MapStateData>(); // Remove nullptr
                            map_state->restore_from_state(t);
                            return map_state;
                        }))
        //.def("__setstate__", [](pyslam::MapStateData &self, py::tuple t) {
        // self.restore_from_state(t);
        //})
        .def("__getstate__", &pyslam::MapStateData::state_tuple);

    // ------------------------------------------------------------
    // Map class - complete interface matching Python Map
    py::class_<pyslam::Map, std::shared_ptr<pyslam::Map>>(m, "Map")
        .def(py::init<>())

        // Core data structures (read/write access)
        .def_readwrite("frames", &pyslam::Map::frames)
        .def_readwrite("keyframes", &pyslam::Map::keyframes)
        .def_readwrite("points", &pyslam::Map::points)
        .def_readwrite("keyframe_origins", &pyslam::Map::keyframe_origins)
        .def_readwrite("keyframes_map", &pyslam::Map::keyframes_map)

        // ID counters
        .def_readwrite("max_point_id", &pyslam::Map::max_point_id)
        .def_readwrite("max_frame_id", &pyslam::Map::max_frame_id)
        .def_readwrite("max_keyframe_id", &pyslam::Map::max_keyframe_id)

        // Session info
        // Property that ties the lifetime of the returned pointer to `Map`
        .def_property(
            "reloaded_session_map_info",
            [](pyslam::Map &m) -> pyslam::ReloadedSessionMapInfo * {
                return m.reloaded_session_map_info.get(); // non-owning
            },
            [](pyslam::Map &m, pyslam::ReloadedSessionMapInfo *v) {
                if (v)
                    m.reloaded_session_map_info = std::make_unique<pyslam::ReloadedSessionMapInfo>(
                        *v); // copy into unique_ptr
                else
                    m.reloaded_session_map_info.reset(); // allow Python None
            },
            py::return_value_policy::reference_internal)

        // Local map
        .def_property_readonly("local_map",
                               [](const pyslam::Map &m) -> const pyslam::LocalCovisibilityMap * {
                                   return m.local_map.get();
                               })

        // Viewer scale
        .def_readwrite("viewer_scale", &pyslam::Map::viewer_scale)

        // Core operations
        .def("reset", &pyslam::Map::reset)
        .def("reset_session", &pyslam::Map::reset_session)
        .def("delete", &pyslam::Map::delete_map)
        .def("delete_map", &pyslam::Map::delete_map)

        // Point operations
        .def("get_points", &pyslam::Map::get_points)
        .def("num_points", &pyslam::Map::num_points)
        .def("add_point", &pyslam::Map::add_point)
        .def("remove_point", &pyslam::Map::remove_point)
        .def("remove_point_no_lock", &pyslam::Map::remove_point_no_lock)

        // Frame operations
        .def("get_frame", &pyslam::Map::get_frame, py::arg("idx"))
        .def("get_frames", &pyslam::Map::get_frames)
        .def("num_frames", &pyslam::Map::num_frames)
        .def("add_frame", &pyslam::Map::add_frame, py::arg("frame"), py::arg("override_id") = false)
        .def("remove_frame", &pyslam::Map::remove_frame)

        // KeyFrame operations
        .def("get_keyframes",
             &pyslam::Map::get_keyframes_vector) // NOTE: returns a vector of KeyFramePtrs
        .def("get_first_keyframe", &pyslam::Map::get_first_keyframe)
        .def("get_last_keyframe", &pyslam::Map::get_last_keyframe)
        .def("get_last_keyframes", &pyslam::Map::get_last_keyframes,
             py::arg("local_window_size") = 5)
        .def("num_keyframes", &pyslam::Map::num_keyframes)
        .def("num_keyframes_session", &pyslam::Map::num_keyframes_session)
        .def("add_keyframe", &pyslam::Map::add_keyframe)
        .def("remove_keyframe", &pyslam::Map::remove_keyframe)

        // Visualization
        .def(
            "draw_feature_trails",
            [](pyslam::Map &self, cv::Mat &img, const bool with_level_radius,
               int trail_max_length) {
                py::gil_scoped_release gil_release;
                return self.draw_feature_trails(img, with_level_radius, trail_max_length);
            },
            py::arg("img"), py::arg("with_level_radius") = false, py::arg("trail_max_length") = 16)
        .def(
            "get_data_arrays_for_drawing",
            [](pyslam::Map &self, std::size_t max_points_to_visualize,
               std::size_t min_weight_for_drawing_covisibility_edge) {
                py::gil_scoped_release gil_release;
                return self.get_data_arrays_for_drawing(max_points_to_visualize,
                                                        min_weight_for_drawing_covisibility_edge);
            },
            py::arg("max_points_to_visualize") = pyslam::Parameters::kMaxSparseMapPointsToVisualize,
            py::arg("min_weight_for_drawing_covisibility_edge") =
                pyslam::Parameters::kMinWeightForDrawingCovisibilityEdge)

        // Point management
        .def("add_points", &pyslam::map_add_points_wrapper, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("kf1"), py::arg("kf2"), py::arg("idxs1"),
             py::arg("idxs2"), py::arg("img"), py::arg("do_check") = true,
             py::arg("cos_max_parallax") = 0.9998, py::arg("far_points_threshold") = py::none())
        .def("add_stereo_points", &pyslam::map_add_stereo_points_wrapper, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("f"), py::arg("kf"), py::arg("idxs"),
             py::arg("img"))

        // Point filtering
        .def("remove_points_with_big_reproj_err",
             &pyslam::map_remove_points_with_big_reproj_err_wrapper)
        .def("compute_mean_reproj_error", &pyslam::map_compute_mean_reproj_error_wrapper,
             py::arg("points") = std::vector<pyslam::MapPointPtr>{})

        // Optimization
        .def(
            "optimize",
            [](pyslam::MapPtr &self, int local_window_size, bool verbose, int rounds,
               bool use_robust_kernel, bool do_cull_points, py::object abort_flag) {
                return pyslam::map_optimize_wrapper(self, local_window_size, verbose, rounds,
                                                    use_robust_kernel, do_cull_points, abort_flag);
            },
            py::arg("local_window_size") = pyslam::Parameters::kLargeBAWindowSize,
            py::arg("verbose") = false, py::arg("rounds") = 10,
            py::arg("use_robust_kernel") = false, py::arg("do_cull_points") = false,
            py::arg("abort_flag") = nullptr)
        .def("locally_optimize", pyslam::map_locally_optimize_wrapper, py::arg("kf_ref"),
             py::arg("verbose") = false, py::arg("rounds") = 10, py::arg("abort_flag") = nullptr,
             py::arg("mp_abort_flag") = py::none())

        // Serialization
        .def("to_json", &pyslam::map_to_json_wrapper, py::arg("out_json") = "{}")
        .def("serialize", &pyslam::map_serialize_wrapper)
        .def("from_json", &pyslam::map_from_json_wrapper)
        .def("deserialize", &pyslam::map_deserialize_wrapper)
        .def("save", &pyslam::map_save_wrapper)
        .def("load", &pyslam::map_load_wrapper)

        // Session management
        .def("is_reloaded", &pyslam::Map::is_reloaded)
        .def("set_reloaded_session_info", &pyslam::Map::set_reloaded_session_info)
        .def("get_reloaded_session_info", &pyslam::Map::get_reloaded_session_info)
        // Lock properties
        .def_property_readonly(
            "lock",
            [](pyslam::Map &self) {
                return std::make_shared<pyslam::PyMapMutexWrapper>(self.lock());
            },
            py::keep_alive<0, 1>() // keep Map (arg #1) alive as long as the returned wrapper lives
            )
        .def_property_readonly(
            "update_lock",
            [](pyslam::Map &self) {
                return std::make_shared<pyslam::PyMapMutexWrapper>(self.update_lock());
            },
            py::keep_alive<0, 1>() // keep Map (arg #1) alive as long as the returned wrapper lives
        );

    // LocalMapBase class - complete interface matching Python LocalMapBase
    py::class_<pyslam::LocalMapBase, std::shared_ptr<pyslam::LocalMapBase>>(m, "LocalMapBase")
        .def(py::init([](pyslam::Map *map) { return std::make_shared<pyslam::LocalMapBase>(map); }),
             py::arg("map") = nullptr,
             py::keep_alive<0, 1>()) // Keep map alive as long as LocalMapBase is alive

        // Core data structures
        .def_readwrite("keyframes", &pyslam::LocalMapBase::keyframes)
        .def_readwrite("points", &pyslam::LocalMapBase::points)
        .def_readwrite("ref_keyframes", &pyslam::LocalMapBase::ref_keyframes)
        .def_readwrite("map", &pyslam::LocalMapBase::map)

        // Lock property
        .def_property_readonly(
            "lock",
            [](pyslam::LocalMapBase &self) {
                return std::make_shared<pyslam::PyMapMutexWrapper>(self.lock());
            },
            py::keep_alive<0, 1>() // keep LocalMapBase (arg #1) alive as long as the returned
                                   // wrapper lives
            )
        // Core operations
        .def("reset", &pyslam::LocalMapBase::reset)
        .def("reset_session", &pyslam::LocalMapBase::reset_session,
             py::arg("keyframes_to_remove") = std::vector<pyslam::KeyFrame *>{},
             py::arg("points_to_remove") = std::vector<pyslam::MapPoint *>{})

        // Status
        .def("is_empty", &pyslam::LocalMapBase::is_empty)

        // Access methods
        .def("get_points", &pyslam::LocalMapBase::get_points)
        .def("num_points", &pyslam::LocalMapBase::num_points)
        .def("get_keyframes", &pyslam::LocalMapBase::get_keyframes)
        .def("num_keyframes", &pyslam::LocalMapBase::num_keyframes)

        // Update methods
        .def("update_from_keyframes", &pyslam::local_map_update_from_keyframes_wrapper,
             py::arg("local_keyframes"))
        .def("get_frame_covisibles", &pyslam::local_map_get_frame_covisibles_wrapper,
             py::arg("frame"));

    // LocalWindowMap class - complete interface matching Python LocalWindowMap
    py::class_<pyslam::LocalWindowMap, pyslam::LocalMapBase,
               std::shared_ptr<pyslam::LocalWindowMap>>(m, "LocalWindowMap")
        .def(py::init([](pyslam::Map *map, int local_window_size) {
                 return std::make_shared<pyslam::LocalWindowMap>(map, local_window_size);
             }),
             py::arg("map") = nullptr, py::arg("local_window_size") = 5)

        // Local window size
        .def_readwrite("local_window_size", &pyslam::LocalWindowMap::local_window_size)

        // Update methods
        .def("update_keyframes", &pyslam::local_window_map_update_keyframes_wrapper,
             py::arg("kf_ref") = nullptr)
        .def("get_best_neighbors", &pyslam::local_window_map_get_best_neighbors_wrapper,
             py::arg("kf_ref") = nullptr, py::arg("N") = 20)
        .def("update", &pyslam::local_window_map_update_wrapper, py::arg("kf_ref") = nullptr);

    // LocalCovisibilityMap class - complete interface matching Python
    py::class_<pyslam::LocalCovisibilityMap, pyslam::LocalMapBase,
               std::shared_ptr<pyslam::LocalCovisibilityMap>>(m, "LocalCovisibilityMap")
        .def(py::init([](pyslam::Map *map) {
                 return std::make_shared<pyslam::LocalCovisibilityMap>(map);
             }),
             py::arg("map") = nullptr)

        // Update methods
        .def("update_keyframes", &pyslam::local_covisibility_map_update_keyframes_wrapper)
        .def("get_best_neighbors", &pyslam::local_covisibility_map_get_best_neighbors_wrapper,
             py::arg("kf_ref"), py::arg("N") = 20)
        .def("update", &pyslam::local_covisibility_map_update_wrapper);
}
