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

        py::gil_scoped_release release;

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
        py::gil_scoped_release release;

        mean_squared_error = map->locally_optimize(kf_ref, verbose, rounds, abort_flag_value_ptr);

        abort_flag_wrapper.stop_monitoring();
    }

    return mean_squared_error;
}

// Wrapper for LocalMapBase::update_from_keyframes template method
inline std::tuple<pyslam::KeyFrameIdSet, std::unordered_set<pyslam::MapPointPtr>,
                  pyslam::KeyFrameIdSet>
local_map_update_from_keyframes_wrapper(std::shared_ptr<pyslam::LocalMapBase> local_map,
                                        const pyslam::KeyFrameIdSet &local_keyframes) {
    return local_map->update_from_keyframes(local_keyframes);
}

std::tuple<pyslam::KeyFramePtr, std::vector<pyslam::KeyFramePtr>, std::vector<pyslam::MapPointPtr>>
get_frame_covisibles_wrapper(pyslam::LocalMapBase &self, pyslam::FramePtr frame) {
    auto [kf_ref, local_keyframes, local_points_set] = self.get_frame_covisibles(frame);

    // Use the safer validation function to filter valid MapPoints
    std::vector<pyslam::MapPointPtr> local_points_vector = filter_valid_mappoints(local_points_set);

    return std::make_tuple(kf_ref, local_keyframes, local_points_vector);
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
    // MapState class - complete interface matching Python MapState
    py::class_<pyslam::MapState, std::shared_ptr<pyslam::MapState>>(m, "MapState")
        .def(py::init<>())
        .def_readwrite("poses", &pyslam::MapState::poses)
        .def_readwrite("pose_timestamps", &pyslam::MapState::pose_timestamps)
        .def_readwrite("fov_centers", &pyslam::MapState::fov_centers)
        .def_readwrite("fov_centers_colors", &pyslam::MapState::fov_centers_colors)
        .def_readwrite("points", &pyslam::MapState::points)
        .def_readwrite("colors", &pyslam::MapState::colors)
        .def_readwrite("semantic_colors", &pyslam::MapState::semantic_colors)
        .def_readwrite("covisibility_graph", &pyslam::MapState::covisibility_graph)
        .def_readwrite("spanning_tree", &pyslam::MapState::spanning_tree)
        .def_readwrite("loops", &pyslam::MapState::loops)
        .def(py::pickle([](const pyslam::MapState &self) { return self.state_tuple(); },
                        [](py::tuple t) {
                            auto map_state = std::make_shared<pyslam::MapState>(); // Remove nullptr
                            map_state->restore_from_state(t);
                            return map_state;
                        }))
        //.def("__setstate__", [](pyslam::MapState &self, py::tuple t) { self.restore_from_state(t);
        //})
        .def("__getstate__", &pyslam::MapState::state_tuple);

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
        .def("get_last_keyframe", &pyslam::Map::get_last_keyframe)
        .def("get_last_keyframes", &pyslam::Map::get_last_keyframes,
             py::arg("local_window_size") = 5)
        .def("num_keyframes", &pyslam::Map::num_keyframes)
        .def("num_keyframes_session", &pyslam::Map::num_keyframes_session)
        .def("add_keyframe", &pyslam::Map::add_keyframe)
        .def("remove_keyframe", &pyslam::Map::remove_keyframe)

        // Visualization
        .def("draw_feature_trails", &pyslam::Map::draw_feature_trails, py::arg("img"),
             py::arg("with_level_radius") = false)
        .def("get_data_arrays_for_drawing", &pyslam::Map::get_data_arrays_for_drawing,
             py::arg("max_points_to_visualize") =
                 pyslam::Parameters::kMaxSparseMapPointsToVisualize,
             py::arg("min_weight_for_drawing_covisibility_edge") =
                 pyslam::Parameters::kMinWeightForDrawingCovisibilityEdge)

        // Point management
        .def("add_points", &pyslam::Map::add_points, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("kf1"), py::arg("kf2"), py::arg("idxs1"),
             py::arg("idxs2"), py::arg("img"), py::arg("do_check") = true,
             py::arg("cos_max_parallax") = 0.9998, py::arg("far_points_threshold") = py::none())
        .def("add_stereo_points", &pyslam::Map::add_stereo_points, py::arg("points3d"),
             py::arg("mask_pts3d") = py::none(), py::arg("f"), py::arg("kf"), py::arg("idxs"),
             py::arg("img"))

        // Point filtering
        .def("remove_points_with_big_reproj_err", &pyslam::Map::remove_points_with_big_reproj_err)
        .def("compute_mean_reproj_error", &pyslam::Map::compute_mean_reproj_error,
             py::arg("points") = std::vector<pyslam::MapPoint *>{})

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
        .def("to_json", &pyslam::Map::to_json, py::arg("out_json") = "{}")
        .def("serialize", &pyslam::Map::serialize)
        .def("from_json", &pyslam::Map::from_json)
        .def("deserialize", &pyslam::Map::deserialize)
        .def("save", &pyslam::Map::save)
        .def("load", &pyslam::Map::load)

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
        .def("get_frame_covisibles", &pyslam::LocalMapBase::get_frame_covisibles, py::arg("frame"));

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
        .def("update_keyframes", &pyslam::LocalWindowMap::update_keyframes,
             py::arg("kf_ref") = nullptr)
        .def("get_best_neighbors", &pyslam::LocalWindowMap::get_best_neighbors,
             py::arg("kf_ref") = nullptr, py::arg("N") = 20)
        .def("update", &pyslam::LocalWindowMap::update, py::arg("kf_ref") = nullptr);

    // LocalCovisibilityMap class - complete interface matching Python
    py::class_<pyslam::LocalCovisibilityMap, pyslam::LocalMapBase,
               std::shared_ptr<pyslam::LocalCovisibilityMap>>(m, "LocalCovisibilityMap")
        .def(py::init([](pyslam::Map *map) {
                 return std::make_shared<pyslam::LocalCovisibilityMap>(map);
             }),
             py::arg("map") = nullptr)

        // Update methods
        .def("update_keyframes", &pyslam::LocalCovisibilityMap::update_keyframes)
        .def("get_best_neighbors", &pyslam::LocalCovisibilityMap::get_best_neighbors,
             py::arg("kf_ref"), py::arg("N") = 20)
        .def("update", &pyslam::LocalCovisibilityMap::update);
}
