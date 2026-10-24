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

#include "map_point.h"
#include "utils/map_helpers.h"

namespace py = pybind11;

void bind_map_point(py::module &m) {

    // ------------------------------------------------------------
    // MapPointBase
    py::class_<pyslam::MapPointBase, std::shared_ptr<pyslam::MapPointBase>>(m, "MapPointBase")
        .def(py::init<int>(), py::arg("id") = -1)
        .def_readwrite("id", &pyslam::MapPointBase::id)
        .def_readwrite("map", &pyslam::MapPointBase::map)
        .def_readwrite("_observations", &pyslam::MapPointBase::_observations)
        .def_readwrite("_frame_views", &pyslam::MapPointBase::_frame_views)
        .def_property(
            "_is_bad", [](const pyslam::MapPointBase &self) { return self._is_bad.load(); },
            [](pyslam::MapPointBase &self, bool value) { self._is_bad.store(value); })
        .def_readwrite("_num_observations", &pyslam::MapPointBase::_num_observations)
        .def_readwrite("num_times_visible", &pyslam::MapPointBase::num_times_visible)
        .def_readwrite("num_times_found", &pyslam::MapPointBase::num_times_found)
        .def_readwrite("last_frame_id_seen", &pyslam::MapPointBase::last_frame_id_seen)
        .def_readwrite("replacement", &pyslam::MapPointBase::replacement)
        .def_readwrite("corrected_by_kf", &pyslam::MapPointBase::corrected_by_kf)
        .def_readwrite("corrected_reference", &pyslam::MapPointBase::corrected_reference)
        .def_readwrite("kf_ref", &pyslam::MapPointBase::kf_ref)
        .def("observations", &pyslam::MapPointBase::observations)
        .def("observations_iter", &pyslam::MapPointBase::observations_iter)
        .def("keyframes", &pyslam::MapPointBase::keyframes)
        .def("keyframes_iter", &pyslam::MapPointBase::keyframes_iter)
        .def("is_in_keyframe", &pyslam::MapPointBase::is_in_keyframe)
        .def("get_observation_idx", &pyslam::MapPointBase::get_observation_idx)
        .def("add_observation", &pyslam::MapPointBase::add_observation)
        .def("remove_observation", &pyslam::MapPointBase::remove_observation, py::arg("keyframe"),
             py::arg("idx") = -1, py::arg("map_no_lock") = false)
        .def("frame_views", &pyslam::MapPointBase::frame_views)
        .def("frame_views_iter", &pyslam::MapPointBase::frame_views_iter)
        .def("frames", &pyslam::MapPointBase::frames)
        .def("frames_iter", &pyslam::MapPointBase::frames_iter)
        .def("is_in_frame", &pyslam::MapPointBase::is_in_frame)
        .def("add_frame_view", &pyslam::MapPointBase::add_frame_view)
        .def("remove_frame_view", &pyslam::MapPointBase::remove_frame_view)
        //.def("is_bad", &pyslam::MapPointBase::is_bad)
        .def("is_bad", &pyslam::MapPointBase::is_bad, py::call_guard<py::gil_scoped_release>())
        .def("num_observations", &pyslam::MapPointBase::num_observations)
        .def("is_good_with_min_obs", &pyslam::MapPointBase::is_good_with_min_obs)
        .def("is_bad_and_is_good_with_min_obs",
             &pyslam::MapPointBase::is_bad_and_is_good_with_min_obs)
        .def("is_bad_or_is_in_keyframe", &pyslam::MapPointBase::is_bad_or_is_in_keyframe)
        .def("increase_visible", &pyslam::MapPointBase::increase_visible, py::arg("num_times") = 1)
        .def("increase_found", &pyslam::MapPointBase::increase_found, py::arg("num_times") = 1)
        .def("get_found_ratio", &pyslam::MapPointBase::get_found_ratio)
        .def("observations_string", &pyslam::MapPointBase::observations_string)
        .def("frame_views_string", &pyslam::MapPointBase::frame_views_string)
        .def("to_string", &pyslam::MapPointBase::to_string)
        .def("__eq__", &pyslam::MapPointBase::operator==)
        .def("__lt__", &pyslam::MapPointBase::operator<)
        .def("__le__", &pyslam::MapPointBase::operator<=)
        .def("__hash__", &pyslam::MapPointBase::hash)
        .def_static("next_id", &pyslam::MapPointBase::next_id)
        .def_static("set_id", &pyslam::MapPointBase::set_id);

    // ------------------------------------------------------------
    // MapPoint
    py::class_<pyslam::MapPoint, pyslam::MapPointBase, std::shared_ptr<pyslam::MapPoint>>(
        m, "MapPoint")
        .def(py::init([](const Eigen::Vector3d &position,
                         const Eigen::Matrix<unsigned char, 3, 1> &color) {
                 return std::make_shared<pyslam::MapPoint>(position, color);
             }),
             py::arg("position"), py::arg("color"))
        .def(py::init([](const Eigen::Vector3d &position,
                         const Eigen::Matrix<unsigned char, 3, 1> &color,
                         pyslam::KeyFramePtr keyframe, int idxf, int id) {
                 return std::make_shared<pyslam::MapPoint>(position, color, keyframe, idxf, id);
             }),
             py::arg("position"), py::arg("color"), py::arg("keyframe"), py::arg("idxf") = -1,
             py::arg("id") = -1)
        .def(py::init([](const Eigen::Vector3d &position,
                         const Eigen::Matrix<unsigned char, 3, 1> &color, pyslam::FramePtr frame,
                         int idxf, int id) {
                 return std::make_shared<pyslam::MapPoint>(position, color, frame, idxf, id);
             }),
             py::arg("position"), py::arg("color"), py::arg("frame"), py::arg("idxf") = -1,
             py::arg("id") = -1)

        // Safer Eigen exposure: copy out, set explicitly
        .def_property(
            "_pt", [](const pyslam::MapPoint &self) { return self._pt; }, // copy
            [](pyslam::MapPoint &self, const Eigen::Vector3d &v) { self._pt = v; })
        .def_property(
            "normal", [](const pyslam::MapPoint &self) { return self.normal; }, // copy
            [](pyslam::MapPoint &self, const Eigen::Vector3d &n) { self.normal = n; })
        .def_readwrite("_min_distance", &pyslam::MapPoint::_min_distance)
        .def_readwrite("_max_distance", &pyslam::MapPoint::_max_distance)
        .def_readwrite("color", &pyslam::MapPoint::color)
        .def_readwrite("semantic_des", &pyslam::MapPoint::semantic_des)
        .def_readwrite("des", &pyslam::MapPoint::des)
        .def_readwrite("first_kid", &pyslam::MapPoint::first_kid)
        .def_readwrite("num_observations_on_last_update_des",
                       &pyslam::MapPoint::num_observations_on_last_update_des)
        .def_readwrite("num_observations_on_last_update_normals",
                       &pyslam::MapPoint::num_observations_on_last_update_normals)
        .def_readwrite("num_observations_on_last_update_semantics",
                       &pyslam::MapPoint::num_observations_on_last_update_semantics)
        .def_property(
            "pt_GBA", [](const pyslam::MapPoint &self) { return self.pt_GBA; }, // copy
            [](pyslam::MapPoint &self, const Eigen::Vector3d &v) { self.pt_GBA = v; })
        .def_readwrite("GBA_kf_id", &pyslam::MapPoint::GBA_kf_id)
        .def_readwrite("is_pt_GBA_valid", &pyslam::MapPoint::is_pt_GBA_valid)

        // Derived getters: copy out to avoid dangling refs
        .def("pt", [](const pyslam::MapPoint &self) { return self.pt(); })                   // copy
        .def("homogeneous", [](const pyslam::MapPoint &self) { return self.homogeneous(); }) // copy

        // Mutators / queries
        .def("update_position", &pyslam::MapPoint::update_position)
        .def("min_distance", &pyslam::MapPoint::min_distance)
        .def("max_distance", &pyslam::MapPoint::max_distance)
        .def("get_all_pos_info", &pyslam::MapPoint::get_all_pos_info)
        .def("get_reference_keyframe", &pyslam::MapPoint::get_reference_keyframe)
        .def("descriptors", &pyslam::MapPoint::descriptors)
        .def("min_des_distance", &pyslam::MapPoint::min_des_distance)
        .def("delete", &pyslam::MapPoint::delete_point)
        .def("set_bad", &pyslam::MapPoint::set_bad, py::arg("map_no_lock") = false)
        .def("get_replacement", &pyslam::MapPoint::get_replacement)
        .def("get_normal", [](const pyslam::MapPoint &self) { return self.get_normal(); }) // copy
        .def("replace_with", &pyslam::MapPoint::replace_with)
        .def("update_normal_and_depth", &pyslam::MapPoint::update_normal_and_depth,
             py::arg("force") = false)
        .def("update_best_descriptor", &pyslam::MapPoint::update_best_descriptor,
             py::arg("force") = false)
        .def(
            "update_semantics",
            [](pyslam::MapPoint &self, py::object semantic_fusion_method, bool force) {
                // NOTE: we are not using the semantic_fusion_method from Python here,
                // C++ reimplements the semantic fusion method
                self.update_semantics(nullptr, force);
            },
            py::arg("semantic_fusion_method") = py::none(), py::arg("force") = false)
        .def("update_info", &pyslam::MapPoint::update_info)
        .def("predict_detection_level", &pyslam::MapPoint::predict_detection_level)
        .def("to_json", &pyslam::MapPoint::to_json)
        .def_static("from_json", &pyslam::MapPoint::from_json)
        .def("replace_ids_with_objects", &pyslam::MapPoint::replace_ids_with_objects)
        .def("set_pt_GBA", &pyslam::MapPoint::set_pt_GBA)
        .def("set_GBA_kf_id", &pyslam::MapPoint::set_GBA_kf_id)
        .def_static("predict_detection_levels", &pyslam::MapPoint::predict_detection_levels)

        // Pickle
        .def(py::pickle([](const pyslam::MapPoint &self) { return self.state_tuple(); },
                        [](py::tuple t) {
                            const Eigen::Vector3d default_position(0, 0, 0);
                            const Eigen::Matrix<unsigned char, 3, 1> default_color(255, 255, 255);
                            auto mp =
                                std::make_shared<pyslam::MapPoint>(default_position, default_color);
                            mp->restore_from_state(t);
                            return mp;
                        }))
        //.def("__setstate__", [](pyslam::MapPoint &self, py::tuple t) { self.restore_from_state(t);
        //})
        .def("__getstate__", &pyslam::MapPoint::state_tuple)
        .def("__del__",
             [](pyslam::MapPoint &self) {
                 // Ensure cleanup happens before destruction
                 self.clear_references();
             })
        // Add a method to check if the MapPoint is valid for pybind11 operations
        .def("is_valid_for_python",
             [](const pyslam::MapPoint &self) { return !self.is_bad() && self.id >= 0; });
} // bind_map_point
