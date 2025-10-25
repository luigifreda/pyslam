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
#include "utils/map_helpers.h"

#include "config_parameters.h"
#include "smart_pointers.h"
#include "utils/messages.h"
#include "utils/numpy_helpers.h"

namespace py = pybind11;

int update_map_points_statistics_wrapper(pyslam::Frame &self, int sensor_type_value) {
    pyslam::SensorType cpp_sensor_type = static_cast<pyslam::SensorType>(sensor_type_value);
    return self.update_map_points_statistics(cpp_sensor_type);
}

pyslam::MapPointPtr cast_to_mappoint(py::handle item) {
    // Handle None objects gracefully
    if (item.is_none()) {
        return nullptr;
    }

    // First try direct cast to MapPoint
    try {
        auto mp = item.cast<pyslam::MapPointPtr>();
        // Validate the MapPoint if it's not null
        if (!is_valid_mappoint(mp)) {
            return nullptr; // Return nullptr for invalid MapPoints
        }
        return mp;
    } catch (const py::cast_error &) {
        // If that fails, try casting to MapPointBase and then dynamic_pointer_cast
        try {
            auto base_ptr = item.cast<std::shared_ptr<pyslam::MapPointBase>>();
            if (!base_ptr) {
                return nullptr; // Return nullptr instead of throwing
            }
            auto derived_ptr = std::dynamic_pointer_cast<pyslam::MapPoint>(base_ptr);
            if (derived_ptr && is_valid_mappoint(derived_ptr)) {
                return derived_ptr;
            }
            // If dynamic_cast fails or MapPoint is invalid, return nullptr
            return nullptr;
        } catch (const py::cast_error &) {
            // If all casting fails, return nullptr instead of throwing
            return nullptr;
        }
    }
}

// Wrapper function for Frame::get_points() which can contain null/invalid MapPoint objects
// This preserves the original structure but safely handles invalid MapPoint objects
py::list get_points_wrapper(const pyslam::Frame &self) {
    auto points = self.get_points();
    py::list result;
    for (const auto &point : points) {
        if (is_valid_mappoint(point)) {
            result.append(py::cast(point));
        } else {
            result.append(py::none());
        }
    }
    return result;
}

std::vector<pyslam::MapPointPtr> points_vector_wrapper(py::object points_obj) {
    std::vector<pyslam::MapPointPtr> points_vector;
    if (py::isinstance<py::list>(points_obj) || py::isinstance<py::tuple>(points_obj)) {
        points_vector.reserve(py::len(points_obj));
        for (auto item : points_obj) {
            auto mp = cast_to_mappoint(item);
            if (mp) { // Only add non-null points
                points_vector.push_back(mp);
            } else {
                MSG_WARN("points_vector_wrapper() - point is invalid");
                points_vector.push_back(nullptr);
            }
        }
    } else {
        try {
            points_vector = py::cast<std::vector<pyslam::MapPointPtr>>(points_obj);
        } catch (const py::cast_error &e) {
            throw std::runtime_error(
                "points_vector_wrapper() - cannot convert object to MapPoint vector: " +
                std::string(e.what()));
        }
    }
    return points_vector;
}

// Wrapper function for Frame::are_visible() which can contain null/invalid MapPoint objects
// This preserves the original structure but safely handles invalid MapPoint objects
py::tuple are_visible_wrapper(pyslam::Frame &self, py::object points_obj, bool do_stereo_project) {
    std::vector<pyslam::MapPointPtr> points_vector = points_vector_wrapper(points_obj);

    // Call the C++ method with valid points only
    auto [visibility_flags, projections, depths, dists] =
        self.are_visible<double>(points_vector, do_stereo_project);

    // Return the result directly - let Python handle the size mismatch
    return py::make_tuple(py::cast(visibility_flags), // visibility flags
                          py::cast(projections),      // projections
                          py::cast(depths),           // depths
                          py::cast(dists)             // distances
    );
}

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
        .def("orientation", &pyslam::FrameBase::orientation)
        .def("position", &pyslam::FrameBase::position)
        .def("update_pose",
             [](pyslam::FrameBase &self, pyslam::CameraPose &pose) { self.update_pose(pose); })
        .def("update_pose",
             [](pyslam::FrameBase &self, const Eigen::Matrix4d &Tcw) { self.update_pose(Tcw); })
        .def("update_pose",
             [](pyslam::FrameBase &self, const Eigen::Isometry3d &isometry3d) {
                 self.update_pose(isometry3d);
             })
        .def("update_translation", &pyslam::FrameBase::update_translation)
        .def("update_rotation_and_translation", &pyslam::FrameBase::update_rotation_and_translation)
        .def("transform_point", &pyslam::FrameBase::transform_point<double>)
        .def("transform_points", &pyslam::FrameBase::transform_points<double>)
        .def("project_point", &pyslam::FrameBase::project_point<double>, py::arg("pw"),
             py::arg("do_stereo_project") = false)
        .def("project_map_point", &pyslam::FrameBase::project_map_point<double>,
             py::arg("map_point"), py::arg("do_stereo_project") = false)
        .def("is_in_image", &pyslam::FrameBase::is_in_image<double>)
        .def("are_in_image", &pyslam::FrameBase::are_in_image<double>)
        .def("is_visible", &pyslam::FrameBase::is_visible<double>)
        // In the Frame binding section
        .def("are_visible", &are_visible_wrapper, py::arg("map_points"),
             py::arg("do_stereo_project") = false)
        .def("project_points", &pyslam::FrameBase::project_points<double>, py::arg("points"),
             py::arg("do_stereo_project") = false)
        .def("project_map_points", &pyslam::FrameBase::project_map_points<double>,
             py::arg("map_points"), py::arg("do_stereo_project") = false)
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
        .def_readwrite_static("is_store_imgs", &pyslam::Frame::is_store_imgs)
        .def_readwrite_static("is_compute_median_depth", &pyslam::Frame::is_compute_median_depth)
        .def_readwrite("is_keyframe", &pyslam::Frame::is_keyframe)
        //.def_readwrite("kps", &pyslam::Frame::kps)
        DEFINE_EIGEN_ZERO_COPY_PROPERTY(pyslam::Frame, kps, pyslam::MatNx2f, "kps")
        //.def_readwrite("kps_r", &pyslam::Frame::kps_r)
        DEFINE_EIGEN_ZERO_COPY_PROPERTY(pyslam::Frame, kps_r, pyslam::MatNx2f, "kps_r")

        // Eigen matrix with validation and zero-copy
        DEFINE_EIGEN_ZERO_COPY_PROPERTY(pyslam::Frame, kpsu, pyslam::MatNx2f, "kpsu")

        //.def_readwrite("kpsn", &pyslam::Frame::kpsn)
        DEFINE_EIGEN_ZERO_COPY_PROPERTY(pyslam::Frame, kpsn, pyslam::MatNx2f, "kpsn")

        .def_readwrite("kps_sem", &pyslam::Frame::kps_sem)
        //.def_readwrite("octaves", &pyslam::Frame::octaves)
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, octaves, int, "octaves")
        //.def_readwrite("octaves_r", &pyslam::Frame::octaves_r)
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, octaves_r, int, "octaves_r")
        //.def_readwrite("sizes", &pyslam::Frame::sizes)
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, sizes, int, "sizes")

        // vector with zero-copy
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, angles, float, "angles")

        .def_readwrite("des", &pyslam::Frame::des)
        .def_readwrite("des_r", &pyslam::Frame::des_r)

        //.def_readwrite("depths", &pyslam::Frame::depths)
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, depths, float, "depths")
        // vector with zero-copy
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, kps_ur, float, "kps_ur")

        .def_property(
            "points",
            [](pyslam::Frame &self) {
                // Create object array that can handle shared_ptr objects with validation
                py::list result;
                for (size_t i = 0; i < self.points.size(); ++i) {
                    if (self.points[i] && is_valid_mappoint(self.points[i])) {
                        result.append(py::cast(self.points[i]));
                    } else {
                        result.append(py::none());
                    }
                }
                return py::cast<py::array>(result);
            },
            [](pyslam::Frame &self, py::object points_obj) {
                self.points = points_obj.is_none()
                                  ? std::vector<pyslam::MapPointPtr>()
                                  : py::cast<std::vector<pyslam::MapPointPtr>>(points_obj);
            },
            py::keep_alive<0, 1>())
        //.def_readwrite("outliers", &pyslam::Frame::outliers)
        DEFINE_VECTOR_PROPERTY_ZERO_COPY(pyslam::Frame, outliers, bool, "outliers")

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
        .def("get_matched_inlier_points", &pyslam::Frame::get_matched_inlier_points)
        .def("set_point_match", &pyslam::Frame::set_point_match)
        .def("remove_point_match", &pyslam::Frame::remove_point_match)
        .def("remove_point", &pyslam::Frame::remove_point)
        .def("remove_frame_views", &pyslam::Frame::remove_frame_views)
        .def("reset_points", &pyslam::Frame::reset_points)
        .def("get_points", &get_points_wrapper)
        .def("get_matched_points", &pyslam::Frame::get_matched_points)
        .def("get_matched_good_points", &pyslam::Frame::get_matched_good_points)
        .def("num_tracked_points", &pyslam::Frame::num_tracked_points)
        .def("num_matched_inlier_map_points", &pyslam::Frame::num_matched_inlier_map_points)
        .def(
            "update_map_points_statistics",
            [](pyslam::Frame &self, py::object sensor_type_obj) {
                // Try to extract .value attribute first, fallback to direct cast
                int sensor_type_value;
                try {
                    sensor_type_value = sensor_type_obj.attr("value").cast<int>();
                } catch (const py::cast_error &) {
                    // Fallback: try direct cast to int
                    sensor_type_value = sensor_type_obj.cast<int>();
                }
                pyslam::SensorType cpp_sensor_type =
                    static_cast<pyslam::SensorType>(sensor_type_value);
                return self.update_map_points_statistics(cpp_sensor_type);
            },
            py::arg("sensor_type"))
        .def("clean_outlier_map_points", &pyslam::Frame::clean_outlier_map_points)
        .def("clean_bad_map_points", &pyslam::Frame::clean_bad_map_points)
        .def("clean_vo_matches", &pyslam::Frame::clean_vo_matches)
        .def("check_replaced_map_points", &pyslam::Frame::check_replaced_map_points)
        .def("compute_stereo_from_rgbd", &pyslam::Frame::compute_stereo_from_rgbd)
        .def("compute_stereo_matches", &pyslam::Frame::compute_stereo_matches)
        .def("compute_points_median_depth", &pyslam::Frame::compute_points_median_depth<double>,
             py::arg("points3d") = pyslam::MatNx3d(), py::arg("percentile") = 0.5)
        .def("draw_feature_trails", &pyslam::Frame::draw_feature_trails, py::arg("img"),
             py::arg("kps_idxs"), py::arg("with_level_radius") = false,
             py::arg("trail_max_length") = 9)
        .def("draw_all_feature_trails", &pyslam::Frame::draw_all_feature_trails, py::arg("img"),
             py::arg("with_level_radius") = false)
        .def("set_img_right", &pyslam::Frame::set_img_right)
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
