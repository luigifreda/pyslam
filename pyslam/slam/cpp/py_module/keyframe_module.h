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

#include "keyframe.h"

namespace py = pybind11;

void bind_keyframe(py::module &m) {

    // ------------------------------------------------------------
    // KeyFrameGraph class - matches Python KeyFrameGraph
    py::class_<pyslam::KeyFrameGraph, std::shared_ptr<pyslam::KeyFrameGraph>>(m, "KeyFrameGraph")
        .def(py::init<>())
        .def_readwrite("init_parent", &pyslam::KeyFrameGraph::init_parent)
        .def_readwrite("parent", &pyslam::KeyFrameGraph::parent)
        .def_readwrite("children", &pyslam::KeyFrameGraph::children)
        .def_readwrite("loop_edges", &pyslam::KeyFrameGraph::loop_edges)
        .def_readwrite("not_to_erase", &pyslam::KeyFrameGraph::not_to_erase)
        .def_readwrite("connected_keyframes_weights",
                       &pyslam::KeyFrameGraph::connected_keyframes_weights)
        .def_readwrite("ordered_keyframes_weights",
                       &pyslam::KeyFrameGraph::ordered_keyframes_weights)
        .def_readwrite("is_first_connection", &pyslam::KeyFrameGraph::is_first_connection)
        .def("reset_covisibility", &pyslam::KeyFrameGraph::reset_covisibility)
        .def("get_connected_keyframes", &pyslam::KeyFrameGraph::get_connected_keyframes)
        .def("get_covisible_keyframes", &pyslam::KeyFrameGraph::get_covisible_keyframes)
        .def("get_covisible_by_weight", &pyslam::KeyFrameGraph::get_covisible_by_weight)
        .def("get_best_covisible_keyframes", &pyslam::KeyFrameGraph::get_best_covisible_keyframes)
        .def("get_children", &pyslam::KeyFrameGraph::get_children)
        .def("get_parent", &pyslam::KeyFrameGraph::get_parent)
        .def("add_child", &pyslam::KeyFrameGraph::add_child)
        .def("erase_child", &pyslam::KeyFrameGraph::erase_child)
        .def("set_parent", &pyslam::KeyFrameGraph::set_parent)
        .def("has_child", &pyslam::KeyFrameGraph::has_child)
        .def("get_loop_edges", &pyslam::KeyFrameGraph::get_loop_edges)
        .def("add_connection", &pyslam::KeyFrameGraph::add_connection)
        .def("erase_connection", &pyslam::KeyFrameGraph::erase_connection)
        .def("add_loop_edge", &pyslam::KeyFrameGraph::add_loop_edge)
        .def("get_weight", &pyslam::KeyFrameGraph::get_weight)
        .def("get_connected_keyframes_weights",
             &pyslam::KeyFrameGraph::get_connected_keyframes_weights);

    // KeyFrame class - matches Python KeyFrame
    py::class_<pyslam::KeyFrame, pyslam::Frame, pyslam::KeyFrameGraph,
               std::shared_ptr<pyslam::KeyFrame>>(m, "KeyFrame")
        .def(py::init([](pyslam::FramePtr frame, py::object img_obj, py::object img_right_obj,
                         py::object depth_obj, py::object kid_obj) {
                 cv::Mat img = img_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(img_obj);
                 cv::Mat img_right =
                     img_right_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(img_right_obj);
                 cv::Mat depth = depth_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(depth_obj);
                 int kid = kid_obj.is_none() ? -1 : py::cast<int>(kid_obj);
                 return std::make_shared<pyslam::KeyFrame>(frame, img, img_right, depth, kid);
             }),
             py::arg("frame"),                  // 1
             py::arg("img") = py::none(),       // 2
             py::arg("img_right") = py::none(), // 3
             py::arg("depth") = py::none(),     // 4
             py::arg("kid") = py::none(),       // 5
             py::keep_alive<0, 1>(),            // frame
             py::keep_alive<0, 2>(),            // img
             py::keep_alive<0, 3>(),            // img_right
             py::keep_alive<0, 4>(),            // depth
             py::keep_alive<0, 5>()             // kid
             )
        .def_readwrite("kid", &pyslam::KeyFrame::kid)
        .def_readwrite("_is_bad", &pyslam::KeyFrame::_is_bad)
        .def_readwrite("lba_count", &pyslam::KeyFrame::lba_count)
        .def_readwrite("_pose_Tcp", &pyslam::KeyFrame::_pose_Tcp)
        .def_readwrite("g_des", &pyslam::KeyFrame::g_des)
        .def_readwrite("loop_query_id", &pyslam::KeyFrame::loop_query_id)
        .def_readwrite("num_loop_words", &pyslam::KeyFrame::num_loop_words)
        .def_readwrite("loop_score", &pyslam::KeyFrame::loop_score)
        .def_readwrite("reloc_query_id", &pyslam::KeyFrame::reloc_query_id)
        .def_readwrite("num_reloc_words", &pyslam::KeyFrame::num_reloc_words)
        .def_readwrite("reloc_score", &pyslam::KeyFrame::reloc_score)
        .def_readwrite("GBA_kf_id", &pyslam::KeyFrame::GBA_kf_id)
        .def_readwrite("Tcw_GBA", &pyslam::KeyFrame::Tcw_GBA)
        .def_readwrite("is_Tcw_GBA_valid", &pyslam::KeyFrame::is_Tcw_GBA_valid)
        .def_readwrite("Tcw_before_GBA", &pyslam::KeyFrame::Tcw_before_GBA)
        .def_readwrite("map", &pyslam::KeyFrame::map)
        .def("Tcp", &pyslam::KeyFrame::Tcp)
        .def("is_bad", &pyslam::KeyFrame::is_bad)
        .def("set_not_erase", &pyslam::KeyFrame::set_not_erase)
        .def("set_erase", &pyslam::KeyFrame::set_erase)
        .def("set_bad", &pyslam::KeyFrame::set_bad)
        .def("get_matched_points", &pyslam::KeyFrame::get_matched_points)
        .def("get_matched_good_points", &pyslam::KeyFrame::get_matched_good_points)
        .def("init_observations", &pyslam::KeyFrame::init_observations)
        .def("update_connections", &pyslam::KeyFrame::update_connections)
        .def("add_connection", &pyslam::KeyFrame::add_connection)
        .def("__eq__", &pyslam::KeyFrame::operator==)
        .def("__lt__", &pyslam::KeyFrame::operator<)
        .def("__le__", &pyslam::KeyFrame::operator<=)
        .def("__hash__", &pyslam::KeyFrame::hash)
        .def(py::pickle(
            // __getstate__
            [](const pyslam::KeyFrame &self) { return self.state_tuple(); },
            // __setstate__
            [](py::tuple t) {
                auto frame_ptr = std::make_shared<pyslam::Frame>(nullptr);
                auto keyframe = std::make_shared<pyslam::KeyFrame>(frame_ptr);
                keyframe->restore_from_state(t);
                return keyframe;
            }))
        //.def("__setstate__", [](pyslam::KeyFrame &self, py::tuple t) { self.restore_from_state(t);
        //})
        .def("__getstate__", &pyslam::KeyFrame::state_tuple)
        .def("__del__", [](pyslam::KeyFrame &self) {
            // Ensure cleanup happens before destruction
            self.clear_references();
        });
} // bind_keyframe
