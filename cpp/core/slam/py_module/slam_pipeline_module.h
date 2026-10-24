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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "initializer.h"
#include "motion_model.h"
#include "slam.h"
#include "slam_dynamic_config.h"

namespace py = pybind11;

namespace {

pyslam::SensorType sensor_type_from_python(const py::handle &obj) {
    if (py::isinstance<pyslam::SensorType>(obj)) {
        return obj.cast<pyslam::SensorType>();
    }
    if (py::hasattr(obj, "value")) {
        return static_cast<pyslam::SensorType>(obj.attr("value").cast<int>());
    }
    return static_cast<pyslam::SensorType>(obj.cast<int>());
}

pyslam::SlamMode slam_mode_from_python(const py::handle &obj) {
    if (obj.is_none()) {
        return pyslam::SlamMode::SLAM;
    }
    if (py::hasattr(obj, "value")) {
        return static_cast<pyslam::SlamMode>(obj.attr("value").cast<int>());
    }
    return static_cast<pyslam::SlamMode>(obj.cast<int>());
}

py::object python_slam_state(pyslam::SlamState state) {
    return py::module_::import("pyslam.slam.slam_commons").attr("SlamState")(static_cast<int>(state));
}

pyslam::SlamState slam_state_from_python(const py::handle &obj) {
    if (py::hasattr(obj, "value")) {
        return static_cast<pyslam::SlamState>(obj.attr("value").cast<int>());
    }
    return static_cast<pyslam::SlamState>(obj.cast<int>());
}

cv::Mat mat_from_object(const py::object &obj) {
    if (obj.is_none()) {
        return cv::Mat();
    }
    return obj.cast<cv::Mat>();
}

} // namespace

inline void bind_motion_model(py::module &m) {
    py::class_<pyslam::MotionModelBase, std::shared_ptr<pyslam::MotionModelBase>>(m, "MotionModelBase")
        .def_readwrite("is_ok", &pyslam::MotionModelBase::is_ok)
        .def_readwrite("initialized", &pyslam::MotionModelBase::initialized)
        .def("reset", &pyslam::MotionModelBase::reset)
        .def("current_pose", &pyslam::MotionModelBase::current_pose)
        .def("predict_pose",
             [](pyslam::MotionModelBase &self, py::object timestamp, py::object position,
                py::object orientation) {
                 std::optional<double> stamp;
                 if (!timestamp.is_none()) {
                     stamp = timestamp.cast<double>();
                 }
                 std::optional<Eigen::Vector3d> pos;
                 std::optional<Eigen::Quaterniond> quat;
                 if (!position.is_none()) {
                     pos = position.cast<Eigen::Vector3d>();
                 }
                 if (!orientation.is_none()) {
                     quat = orientation.cast<Eigen::Quaterniond>();
                 }
                 auto predicted =
                     self.predict_pose(stamp, pos ? &*pos : nullptr, quat ? &*quat : nullptr);
                 return py::make_tuple(predicted.first, py::none());
             }, py::arg("timestamp"), py::arg("position") = py::none(),
             py::arg("orientation") = py::none())
        .def("update_pose",
             [](pyslam::MotionModelBase &self, py::object timestamp, const Eigen::Vector3d &position,
                const Eigen::Quaterniond &orientation) {
                 std::optional<double> stamp;
                 if (!timestamp.is_none()) {
                     stamp = timestamp.cast<double>();
                 }
                 self.update_pose(stamp, position, orientation);
             })
        .def("apply_correction", [](pyslam::MotionModelBase &self, const Eigen::Matrix4d &correction) {
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.matrix() = correction;
            self.apply_correction(pose);
        });

    py::class_<pyslam::MotionModel, pyslam::MotionModelBase, std::shared_ptr<pyslam::MotionModel>>(
        m, "MotionModel")
        .def(py::init<>());

    py::class_<pyslam::MotionModelDamping, pyslam::MotionModelBase,
               std::shared_ptr<pyslam::MotionModelDamping>>(m, "MotionModelDamping")
        .def(py::init<double>(), py::arg("damping") = 0.95)
        .def("reset", &pyslam::MotionModelDamping::reset);
}

inline void bind_slam_dynamic_config(py::module &m) {
    py::class_<pyslam::SLAMDynamicConfig, std::shared_ptr<pyslam::SLAMDynamicConfig>>(
        m, "SLAMDynamicConfig")
        .def(py::init<float>(), py::arg("init_max_descriptor_distance"))
        .def_readwrite("descriptor_distance_sigma", &pyslam::SLAMDynamicConfig::descriptor_distance_sigma)
        .def("update_descriptor_stats", &pyslam::SLAMDynamicConfig::update_descriptor_stats)
        .def("update_reproj_err_map_stats", &pyslam::SLAMDynamicConfig::update_reproj_err_map_stats);
}

inline void bind_initializer(py::module &m) {
    py::class_<pyslam::InitializerOutput>(m, "InitializerOutput")
        .def(py::init<>())
        .def_readwrite("pts", &pyslam::InitializerOutput::pts)
        .def_readwrite("kf_cur", &pyslam::InitializerOutput::kf_cur)
        .def_readwrite("kf_ref", &pyslam::InitializerOutput::kf_ref)
        .def_readwrite("idxs_cur", &pyslam::InitializerOutput::idxs_cur)
        .def_readwrite("idxs_ref", &pyslam::InitializerOutput::idxs_ref);

    py::class_<pyslam::Initializer, std::shared_ptr<pyslam::Initializer>>(m, "Initializer")
        .def(py::init([](py::object sensor_type) {
                 return std::make_shared<pyslam::Initializer>(sensor_type_from_python(sensor_type));
             }),
             py::arg("sensor_type") = py::int_(0))
        .def("reset", &pyslam::Initializer::reset)
        .def("init", &pyslam::Initializer::init, py::arg("f_cur"), py::arg("img_cur"),
             py::arg("img_right_cur") = cv::Mat(), py::arg("depth_img_cur") = cv::Mat())
        .def("initialize", &pyslam::Initializer::initialize, py::arg("f_cur"), py::arg("img_cur"),
             py::arg("img_right_cur") = cv::Mat(), py::arg("depth_img_cur") = cv::Mat());
}

inline void bind_tracking(py::module &m) {
    py::class_<pyslam::Tracking, std::shared_ptr<pyslam::Tracking>>(m, "Tracking")
        .def(py::init<pyslam::Slam *>(), py::arg("slam"), py::keep_alive<1, 2>())
        .def_readonly("initializer", &pyslam::Tracking::initializer)
        .def_readonly("motion_model", &pyslam::Tracking::motion_model)
        .def_readonly("dyn_config", &pyslam::Tracking::dyn_config)
        .def("track_previous_frame", &pyslam::Tracking::track_previous_frame,
             py::call_guard<py::gil_scoped_release>())
        .def("track_keyframe", &pyslam::Tracking::track_keyframe,
             py::arg("kf_ref"), py::arg("f_cur"), py::arg("name") = "",
             py::call_guard<py::gil_scoped_release>())
        .def("track_local_map", &pyslam::Tracking::track_local_map,
             py::call_guard<py::gil_scoped_release>())
        .def("need_new_keyframe", &pyslam::Tracking::need_new_keyframe,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_for_local_mapping", &pyslam::Tracking::wait_for_local_mapping,
             py::arg("timeout") = 1.0, py::arg("check_on_exit") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("relocalize", &pyslam::Tracking::relocalize,
             py::call_guard<py::gil_scoped_release>())
        .def_property(
            "state", [](pyslam::Tracking &self) { return python_slam_state(self.state); },
            [](pyslam::Tracking &self, py::object state) { self.state = slam_state_from_python(state); })
        .def_property_readonly("cur_R",
                               [](pyslam::Tracking &self) -> py::object {
                                   if (!self.snapshot().has_current_pose) {
                                       return py::none();
                                   }
                                   return py::cast(self.snapshot().cur_R);
                               })
        .def_property_readonly("cur_t",
                               [](pyslam::Tracking &self) -> py::object {
                                   if (!self.snapshot().has_current_pose) {
                                       return py::none();
                                   }
                                   return py::cast(self.snapshot().cur_t);
                               })
        .def_property("descriptor_distance_sigma",
                      [](const pyslam::Tracking &self) { return self.descriptor_distance_sigma.load(); },
                      [](pyslam::Tracking &self, float value) { self.descriptor_distance_sigma = value; })
        .def_property_readonly("camera", [](pyslam::Tracking &self) { return self.camera(); })
        .def_property_readonly("f_cur", [](pyslam::Tracking &self) { return self.snapshot().f_cur; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("kf_ref", [](pyslam::Tracking &self) { return self.snapshot().kf_ref; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("predicted_pose",
                               [](pyslam::Tracking &self) -> py::object {
                                   const auto snapshot = self.snapshot();
                                   if (!snapshot.predicted_pose) {
                                       return py::none();
                                   }
                                   return py::cast(*snapshot.predicted_pose);
                               })
        .def_property_readonly("num_matched_kps", [](const pyslam::Tracking &self) { return self.snapshot().num_matched_kps; })
        .def_property_readonly("num_inliers", [](const pyslam::Tracking &self) { return self.snapshot().num_inliers; })
        .def_property_readonly("num_matched_map_points", [](const pyslam::Tracking &self) { return self.snapshot().num_matched_map_points; })
        .def_property_readonly("num_kf_ref_tracked_points", [](const pyslam::Tracking &self) { return self.snapshot().num_kf_ref_tracked_points; })
        .def_property_readonly("last_num_static_stereo_map_points", [](const pyslam::Tracking &self) { return self.snapshot().last_num_static_stereo_map_points; })
        .def_property_readonly("total_num_static_stereo_map_points", [](const pyslam::Tracking &self) { return self.snapshot().total_num_static_stereo_map_points; })
        .def_property_readonly("mean_pose_opt_chi2_error", [](const pyslam::Tracking &self) { return self.snapshot().mean_pose_opt_chi2_error; })
        .def_property_readonly("time_track", [](const pyslam::Tracking &self) { return self.snapshot().time_track; })
        .def("reset", &pyslam::Tracking::reset)
        .def("track",
             [](pyslam::Tracking &self, py::object img, py::object img_right, py::object depth, int img_id,
                py::object timestamp, py::object mask, py::object mask_right) {
                 const cv::Mat image = mat_from_object(img);
                 const cv::Mat image_right = mat_from_object(img_right);
                 const cv::Mat depth_image = mat_from_object(depth);
                 const cv::Mat mask_image = mat_from_object(mask);
                 const cv::Mat mask_right_image = mat_from_object(mask_right);
                 std::optional<double> stamp;
                 if (!timestamp.is_none()) {
                     stamp = timestamp.cast<double>();
                 }
                 py::gil_scoped_release release;
                 self.track(image, image_right, depth_image, img_id, stamp, mask_image, mask_right_image);
             },
             py::arg("img"), py::arg("img_right") = py::none(), py::arg("depth") = py::none(),
             py::arg("img_id") = -1, py::arg("timestamp") = py::none(), py::arg("mask") = py::none(),
             py::arg("mask_right") = py::none());
}

inline void bind_local_mapping(py::module &m) {
    py::class_<pyslam::LocalMapping, std::shared_ptr<pyslam::LocalMapping>>(m, "LocalMapping")
        .def(py::init<pyslam::Slam *>(), py::arg("slam"), py::keep_alive<1, 2>())
        .def("do_local_mapping", &pyslam::LocalMapping::do_local_mapping,
             py::call_guard<py::gil_scoped_release>())
        .def("push_keyframe", [](pyslam::LocalMapping &self, const pyslam::KeyFramePtr &kf,
                                 py::object img, py::object img_right, py::object depth) {
            const cv::Mat image = mat_from_object(img);
            const cv::Mat right = mat_from_object(img_right);
            const cv::Mat depth_image = mat_from_object(depth);
            py::gil_scoped_release release;
            self.push_keyframe(kf, image, right, depth_image);
        }, py::arg("keyframe"), py::arg("img") = py::none(),
           py::arg("img_right") = py::none(), py::arg("depth") = py::none())
        .def("start", &pyslam::LocalMapping::start)
        .def("quit", &pyslam::LocalMapping::quit)
        .def("step", &pyslam::LocalMapping::step, py::call_guard<py::gil_scoped_release>())
        .def("create_new_map_points", &pyslam::LocalMapping::create_new_map_points,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("last_num_triangulated_points", [](const pyslam::LocalMapping &self) { return self.last_num_triangulated_points.load(); })
        .def_property_readonly("total_num_triangulated_points", [](const pyslam::LocalMapping &self) { return self.total_num_triangulated_points.load(); })
        .def_property_readonly("last_num_fused_points", [](const pyslam::LocalMapping &self) { return self.last_num_fused_points.load(); })
        .def_property_readonly("total_num_fused_points", [](const pyslam::LocalMapping &self) { return self.total_num_fused_points.load(); })
        .def_property_readonly("last_num_culled_points", [](const pyslam::LocalMapping &self) { return self.last_num_culled_points.load(); })
        .def_property_readonly("total_num_culled_points", [](const pyslam::LocalMapping &self) { return self.total_num_culled_points.load(); })
        .def_property_readonly("last_num_culled_keyframes", [](const pyslam::LocalMapping &self) { return self.last_num_culled_keyframes.load(); })
        .def_property_readonly("total_num_culled_keyframes", [](const pyslam::LocalMapping &self) { return self.total_num_culled_keyframes.load(); })
        .def_property_readonly("mean_ba_chi2_error", [](const pyslam::LocalMapping &self) { return self.mean_ba_chi2_error.load(); })
        .def_property_readonly("time_local_mapping", [](const pyslam::LocalMapping &self) { return self.time_local_mapping.load(); })
        .def("queue_size", &pyslam::LocalMapping::queue_size)
        .def("is_idle", &pyslam::LocalMapping::is_idle)
        .def("is_stopped", &pyslam::LocalMapping::is_stopped)
        .def("is_stop_requested", &pyslam::LocalMapping::is_stop_requested)
        .def("is_running", [](pyslam::LocalMapping &self) { return self.is_running.load(); })
        .def("interrupt_optimization", &pyslam::LocalMapping::interrupt_optimization)
        .def("request_stop", &pyslam::LocalMapping::request_stop)
        .def("request_reset", &pyslam::LocalMapping::request_reset)
        .def("release", &pyslam::LocalMapping::release)
        .def("set_do_not_stop", &pyslam::LocalMapping::set_do_not_stop)
        .def("wait_idle",
             [](pyslam::LocalMapping &self, py::object, py::object timeout) {
                 std::optional<double> seconds;
                 if (!timeout.is_none()) {
                     seconds = timeout.cast<double>();
                 }
                 self.wait_idle(seconds);
             },
             py::arg("print") = py::none(), py::arg("timeout") = py::none())
        .def_property_readonly("descriptor_distance_sigma",
                               &pyslam::LocalMapping::descriptor_distance_sigma);
}

inline void bind_slam(py::module &m) {
    py::class_<pyslam::Slam, std::shared_ptr<pyslam::Slam>>(m, "Slam")
        .def(py::init([](const pyslam::CameraPtr &camera, py::object feature_tracker_config,
                         py::object loop_detector_config, py::object semantic_mapping_config,
                         py::object sensor_type, py::object environment_type, py::object slam_mode,
                         py::object config, bool headless, py::object viewer3d) {
                 auto slam = std::make_shared<pyslam::Slam>();
                 if (environment_type.is_none()) {
                     environment_type = py::module_::import("pyslam.io.dataset_types")
                                            .attr("DatasetEnvironmentType")
                                            .attr("OUTDOOR");
                 }
                 slam->init(camera, feature_tracker_config, loop_detector_config, semantic_mapping_config,
                            sensor_type_from_python(sensor_type), environment_type,
                            slam_mode_from_python(slam_mode), config, headless, viewer3d);
                 return slam;
             }),
             py::arg("camera"), py::arg("feature_tracker_config"),
             py::arg("loop_detector_config") = py::none(),
             py::arg("semantic_mapping_config") = py::none(), py::arg("sensor_type") = py::int_(0),
             py::arg("environment_type") = py::none(), py::arg("slam_mode") = py::int_(0),
             py::arg("config") = py::none(), py::arg("headless") = false,
             py::arg("viewer3d") = py::none())
        .def_readonly("camera", &pyslam::Slam::camera)
        .def_readonly("map", &pyslam::Slam::map)
        .def_property_readonly("tracking", py::cpp_function(
            [](pyslam::Slam &self) { return self.tracking; }, py::keep_alive<0, 1>()))
        .def_property_readonly("local_mapping", py::cpp_function(
            [](pyslam::Slam &self) { return self.local_mapping; }, py::keep_alive<0, 1>()))
        .def_property_readonly("sensor_type",
                               [](const pyslam::Slam &self) -> py::object { return self.sensor_type_py; })
        .def_property_readonly("slam_mode",
                               [](const pyslam::Slam &self) {
                                   return py::module_::import("pyslam.slam.slam").attr("SlamMode")(
                                       static_cast<int>(self.slam_mode));
                               })
        .def_readwrite("environment_type", &pyslam::Slam::environment_type)
        .def_readwrite("feature_tracker_config", &pyslam::Slam::feature_tracker_config)
        .def_readwrite("loop_detector_config", &pyslam::Slam::loop_detector_config)
        .def_readwrite("semantic_mapping_config", &pyslam::Slam::semantic_mapping_config)
        .def_readwrite("config", &pyslam::Slam::config)
        .def_readwrite("headless", &pyslam::Slam::headless)
        .def_readwrite("has_quit", &pyslam::Slam::has_quit)
        .def_readwrite("feature_tracker", &pyslam::Slam::feature_tracker)
        .def_readwrite("loop_closing", &pyslam::Slam::loop_closing)
        .def_readwrite("semantic_mapping", &pyslam::Slam::semantic_mapping)
        .def_readwrite("volumetric_integrator", &pyslam::Slam::volumetric_integrator)
        .def_readwrite("viewer3d", &pyslam::Slam::viewer3d)
        .def_readwrite("GBA", &pyslam::Slam::GBA)
        .def_readwrite("GBA_on_demand", &pyslam::Slam::GBA_on_demand)
        .def_property("reset_requested",
                      [](const pyslam::Slam &self) { return self.reset_requested.load(); },
                      [](pyslam::Slam &self, bool value) { self.reset_requested = value; })
        .def("get_dense_map",
             [](pyslam::Slam &self) -> py::object {
                 if (!self.volumetric_integrator || self.volumetric_integrator.is_none()) {
                     return py::none();
                 }
                 py::object output_queue = self.volumetric_integrator.attr("q_out");
                 if (output_queue.attr("qsize")().cast<int>() <= 0) {
                     return py::none();
                 }
                 return self.volumetric_integrator.attr("pop_output")();
             })
        .def("track",
             [](pyslam::Slam &self, py::object img, py::object img_right, py::object depth, int img_id,
                py::object timestamp, py::object mask, py::object mask_right) {
                 const cv::Mat image = mat_from_object(img);
                 const cv::Mat image_right = mat_from_object(img_right);
                 const cv::Mat depth_image = mat_from_object(depth);
                 const cv::Mat mask_image = mat_from_object(mask);
                 const cv::Mat mask_right_image = mat_from_object(mask_right);
                 std::optional<double> stamp;
                 if (!timestamp.is_none()) {
                     stamp = timestamp.cast<double>();
                 }
                 py::gil_scoped_release release;
                 self.track(image, image_right, depth_image, img_id, stamp, mask_image, mask_right_image);
             },
             py::arg("img"), py::arg("img_right") = py::none(), py::arg("depth") = py::none(),
             py::arg("img_id") = -1, py::arg("timestamp") = py::none(), py::arg("mask") = py::none(),
             py::arg("mask_right") = py::none())
        .def("reset", &pyslam::Slam::reset)
        .def("reset_session", &pyslam::Slam::reset_session)
        .def("quit", &pyslam::Slam::quit)
        .def("request_reset", &pyslam::Slam::request_reset)
        .def("set_tracking_state",
             [](pyslam::Slam &self, py::object state) { self.set_tracking_state(slam_state_from_python(state)); })
        .def("set_viewer_scale", &pyslam::Slam::set_viewer_scale)
        .def("viewer_scale", &pyslam::Slam::viewer_scale)
        .def("bundle_adjust", &pyslam::Slam::bundle_adjust)
        .def("get_final_trajectory", &pyslam::Slam::get_final_trajectory)
        .def("save_system_state", &pyslam::Slam::save_system_state)
        .def("load_system_state", &pyslam::Slam::load_system_state);
}
