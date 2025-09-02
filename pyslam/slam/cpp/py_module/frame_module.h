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

#include "frame.h"
#include "map_point.h"

#include "smart_pointers.h"
#include "utils/numpy_helpers.h"

namespace py = pybind11;

void bind_frame(py::module &m) {

    // ------------------------------------------------------------
    // FrameBase
    py::class_<pyslam::FrameBase, std::shared_ptr<pyslam::FrameBase>>(m, "FrameBase")
        .def(py::init([](py::object camera_obj, py::object pose_obj, int id, double timestamp,
                         int img_id) {
                 // Camera: simple cast with default nullptr
                 pyslam::CameraPtr camera_shared_ptr =
                     camera_obj.is_none() ? nullptr : py::cast<pyslam::CameraPtr>(camera_obj);

                 // Pose: simple cast with default CameraPose()
                 pyslam::CameraPose pose = pose_obj.is_none()
                                               ? pyslam::CameraPose()
                                               : py::cast<pyslam::CameraPose>(pose_obj);

                 return std::make_shared<pyslam::FrameBase>(camera_shared_ptr, pose, id, timestamp,
                                                            img_id);
             }),
             py::arg("camera") = nullptr, py::arg("pose") = pyslam::CameraPose(),
             py::arg("id") = -1, py::arg("timestamp") = 0.0, py::arg("img_id") = -1,
             py::keep_alive<0, 1>(), // keep camera alive
             py::keep_alive<0, 2>()  // keep pose alive
             )
        .def("__del__",
             [](pyslam::FrameBase &self) {
                 // Ensure cleanup happens before destruction
                 self.reset_camera();
             })
        .def_readwrite("id", &pyslam::FrameBase::id)
        .def_static("next_id", &pyslam::FrameBase::next_id)
        .def_readwrite("timestamp", &pyslam::FrameBase::timestamp)
        .def_readwrite("img_id", &pyslam::FrameBase::img_id)
        .def_property(
            "camera",
            [](const pyslam::FrameBase &self) -> py::object {
                return self.camera ? py::cast(self.camera) : py::none();
            },
            [](pyslam::FrameBase &self, py::object camera_obj) {
                if (camera_obj.is_none()) {
                    self.camera = nullptr;
                    self.reset_camera();
                } else {
                    self.camera = py::cast<pyslam::CameraPtr>(camera_obj);
                    self.set_camera(self.camera);
                }
            })
        .def_property_readonly("width", &pyslam::FrameBase::width)
        .def_property_readonly("height", &pyslam::FrameBase::height)
        .def("pose", &pyslam::FrameBase::pose)
        .def("isometry3d", &pyslam::FrameBase::isometry3d)
        .def("Tcw", &pyslam::FrameBase::Tcw)
        .def("Twc", &pyslam::FrameBase::Twc)
        .def("Rcw", &pyslam::FrameBase::Rcw)
        .def("Rwc", &pyslam::FrameBase::Rwc)
        .def("tcw", &pyslam::FrameBase::tcw)
        .def("Ow", &pyslam::FrameBase::Ow)
        .def("quaternion", &pyslam::FrameBase::quaternion)
        .def("orientation", &pyslam::FrameBase::orientation,
             py::call_guard<py::gil_scoped_release>())
        .def("position", &pyslam::FrameBase::position)
        .def("update_pose", [](pyslam::FrameBase &self,
                               const pyslam::CameraPose &pose) { self.update_pose(pose); })
        .def("update_pose",
             [](pyslam::FrameBase &self, const Eigen::Matrix4d &Tcw) { self.update_pose(Tcw); })
        .def("update_translation", &pyslam::FrameBase::update_translation,
             py::call_guard<py::gil_scoped_release>())
        .def("update_rotation_and_translation", &pyslam::FrameBase::update_rotation_and_translation,
             py::call_guard<py::gil_scoped_release>())
        .def("project_point", &pyslam::FrameBase::project_point<double>,
             py::call_guard<py::gil_scoped_release>())
        .def("project_map_point", &pyslam::FrameBase::project_map_point<double>,
             py::call_guard<py::gil_scoped_release>())
        .def("is_visible", &pyslam::FrameBase::is_visible<double>,
             py::call_guard<py::gil_scoped_release>())
        .def("are_visible", &pyslam::FrameBase::are_visible<double>,
             py::call_guard<py::gil_scoped_release>())
        .def("project_points", &pyslam::FrameBase::project_points<double>, py::arg("points"),
             py::arg("do_stereo_project") = false)
        .def("project_map_points", &pyslam::FrameBase::project_map_points<double>,
             py::arg("map_points"), py::arg("do_stereo_project") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("__eq__", &pyslam::FrameBase::operator==)
        .def("__lt__", &pyslam::FrameBase::operator<)
        .def("__le__", &pyslam::FrameBase::operator<=)
        .def("__hash__", &pyslam::FrameBase::hash);

    // ------------------------------------------------------------
    // Frame
    py::class_<pyslam::Frame, pyslam::FrameBase, std::shared_ptr<pyslam::Frame>>(m, "Frame")
        .def(py::init([](py::object camera_obj, py::object img_obj, py::object img_right_obj,
                         py::object depth_obj, pyslam::CameraPose pose, int id,
                         py::object timestamp_obj, int img_id, py::object semantic_img_obj,
                         const pyslam::FrameDataDict &frame_data_dict) {
                 // Camera: simple cast with default nullptr
                 pyslam::CameraPtr camera_shared_ptr =
                     camera_obj.is_none() ? nullptr : py::cast<pyslam::CameraPtr>(camera_obj);

                 // Timestamp: simple cast with default value
                 double timestamp = timestamp_obj.is_none() ? std::numeric_limits<double>::lowest()
                                                            : py::cast<double>(timestamp_obj);

                 // Images: simple casts with default empty Mat
                 cv::Mat img = img_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(img_obj);
                 cv::Mat img_right =
                     img_right_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(img_right_obj);
                 cv::Mat depth = depth_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(depth_obj);
                 cv::Mat semantic_img =
                     semantic_img_obj.is_none() ? cv::Mat() : py::cast<cv::Mat>(semantic_img_obj);

                 return std::make_shared<pyslam::Frame>(camera_shared_ptr, img, img_right, depth,
                                                        pose, id, timestamp, img_id, semantic_img,
                                                        frame_data_dict);
             }),
             py::arg("camera"),                                    // 1
             py::arg("img") = py::none(),                          // 2
             py::arg("img_right") = py::none(),                    // 3
             py::arg("depth") = py::none(),                        // 4
             py::arg("pose") = pyslam::CameraPose(),               // 5
             py::arg("id") = -1,                                   // 6
             py::arg("timestamp") = py::none(),                    // 7
             py::arg("img_id") = -1,                               // 8
             py::arg("semantic_img") = py::none(),                 // 9
             py::arg("frame_data_dict") = pyslam::FrameDataDict{}, // 10
             py::keep_alive<0, 1>(),                               // camera
             py::keep_alive<0, 2>(),                               // img
             py::keep_alive<0, 3>(),                               // img_right
             py::keep_alive<0, 4>(),                               // depth
             py::keep_alive<0, 5>(),                               // pose
             py::keep_alive<0, 9>(),                               // semantic_img
             py::keep_alive<0, 10>()                               // frame_data_dict (optional)
             )
        .def("__del__",
             [](pyslam::Frame &self) {
                 // Ensure cleanup happens before destruction
                 self.clear_references();
             })
        .def_readwrite("is_keyframe", &pyslam::Frame::is_keyframe)
        .def_readwrite("kps", &pyslam::Frame::kps)
        .def_readwrite("kps_r", &pyslam::Frame::kps_r)

        // Eigen matrix with validation and zero-copy
        DEFINE_EIGEN_ZERO_COPY_PROPERTY(pyslam::Frame, kpsu, pyslam::MatNx2f, "kpsu")

        .def_readwrite("kpsn", &pyslam::Frame::kpsn)
        .def_readwrite("kps_sem", &pyslam::Frame::kps_sem)
        .def_readwrite("octaves", &pyslam::Frame::octaves)
        .def_readwrite("octaves_r", &pyslam::Frame::octaves_r)
        .def_readwrite("sizes", &pyslam::Frame::sizes)

        // vector with zero-copy
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, angles, float, "angles")

        .def_readwrite("des", &pyslam::Frame::des)
        .def_readwrite("des_r", &pyslam::Frame::des_r)
        .def_readwrite("depths", &pyslam::Frame::depths)

        // vector with zero-copy
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, kps_ur, float, "kps_ur")

        .def_property(
            "points",
            [](pyslam::Frame &self) {
                py::list out;
                for (auto &p : self.points) {
                    out.append(p ? py::cast(p) : py::none());
                }
                return out;
            },
            [](pyslam::Frame &self, py::object points_obj) {
                self.points = points_obj.is_none()
                                  ? std::vector<pyslam::MapPointPtr>()
                                  : py::cast<std::vector<pyslam::MapPointPtr>>(points_obj);
            },
            py::keep_alive<0, 1>() // keep points_obj alive
            )
        .def_readwrite("outliers", &pyslam::Frame::outliers)
        .def_readwrite("kf_ref", &pyslam::Frame::kf_ref)
        .def_readwrite("img", &pyslam::Frame::img)
        .def_readwrite("img_right", &pyslam::Frame::img_right)
        .def_readwrite("depth_img", &pyslam::Frame::depth_img)
        .def_readwrite("semantic_img", &pyslam::Frame::semantic_img)
        .def_readwrite("is_blurry", &pyslam::Frame::is_blurry)
        .def_readwrite("laplacian_var", &pyslam::Frame::laplacian_var)
        .def_readwrite("fov_center_c", &pyslam::Frame::fov_center_c)
        .def_readwrite("fov_center_w", &pyslam::Frame::fov_center_w)
        .def("unproject_points_3d", &pyslam::Frame::unproject_points_3d<double>, py::arg("idxs"),
             py::arg("transform_in_world") = false)
        .def_property_readonly("kd", &pyslam::Frame::kd)
        .def("get_point_match", &pyslam::Frame::get_point_match)
        .def("set_point_match", &pyslam::Frame::set_point_match)
        .def("remove_point_match", &pyslam::Frame::remove_point_match)
        .def("remove_point", &pyslam::Frame::remove_point)
        .def("reset_points", &pyslam::Frame::reset_points)
        .def("get_points", &pyslam::Frame::get_points)
        .def("get_matched_points", &pyslam::Frame::get_matched_points)
        .def("get_matched_good_points", &pyslam::Frame::get_matched_good_points)
        .def("compute_stereo_from_rgbd", &pyslam::Frame::compute_stereo_from_rgbd)
        .def("compute_stereo_matches", &pyslam::Frame::compute_stereo_matches)
        .def("compute_points_median_depth", &pyslam::Frame::compute_points_median_depth<double>,
             py::arg("points3d") = pyslam::MatNx3d(), py::arg("percentile") = 0.5)
        .def("draw_feature_trails", &pyslam::Frame::draw_feature_trails, py::arg("img"),
             py::arg("kps_idxs"), py::arg("trail_max_length") = 9)
        .def("draw_all_feature_trails", &pyslam::Frame::draw_all_feature_trails)
        .def("set_depth_img", &pyslam::Frame::set_depth_img)
        .def("ensure_contiguous_arrays", &pyslam::Frame::ensure_contiguous_arrays)
        .def("__eq__", &pyslam::Frame::operator==)
        .def("__lt__", &pyslam::Frame::operator<)
        .def("__le__", &pyslam::Frame::operator<=)
        .def("__hash__", &pyslam::Frame::hash)
        .def("to_json", &pyslam::Frame::to_json)
        .def_static("from_json", &pyslam::Frame::from_json)
        .def(py::pickle([](const pyslam::Frame &self) { return self.state_tuple(); },
                        [](py::tuple t) {
                            auto frame = std::make_shared<pyslam::Frame>(nullptr);
                            frame->restore_from_state(t);
                            return frame;
                        }))
        //.def("__setstate__", [](pyslam::Frame &self, py::tuple t) { self.restore_from_state(t); })
        .def("__getstate__", &pyslam::Frame::state_tuple);
} // bind_frame
