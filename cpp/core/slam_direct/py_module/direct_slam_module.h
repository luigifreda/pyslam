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

#include <optional>
#include <stdexcept>

#include "casters/opencv_type_casters.h"
#include "common/dictionary.h"
#include "slam_direct/direct_frontend.h"

namespace py = pybind11;

namespace {

// None represents an absent optional stream; clone arrays before releasing the Python GIL.
cv::Mat direct_mat_from_object(const py::object &obj) {
    if (obj.is_none()) {
        return cv::Mat();
    }
    cv::Mat image = obj.cast<cv::Mat>();
    return image.empty() ? cv::Mat() : image.clone();
}

// Copy scalar/nested settings only; camera calibration arrays are already held by Camera.
pyslam::ConfigDict direct_config_from_object(const py::handle &obj) {
    pyslam::ConfigDict out;
    if (!obj || obj.is_none()) {
        return out;
    }
    if (!py::isinstance<py::dict>(obj)) {
        throw std::invalid_argument("DirectSlam settings must be a dict");
    }
    const py::dict dict = py::reinterpret_borrow<py::dict>(obj);
    for (const auto &item : dict) {
        const std::string key = py::str(item.first);
        const py::handle value = item.second;
        if (value.is_none()) {
            continue;
        }
        if (py::isinstance<py::dict>(value)) {
            out[key] = pyslam::Value(static_cast<pyslam::Dict>(direct_config_from_object(value)));
            continue;
        }
        if (py::isinstance<py::bool_>(value)) {
            out[key] = pyslam::Value(value.cast<bool>());
        } else if (py::isinstance<py::int_>(value)) {
            out[key] = pyslam::Value(value.cast<int64_t>());
        } else if (py::isinstance<py::float_>(value)) {
            out[key] = pyslam::Value(value.cast<double>());
        } else if (py::isinstance<py::str>(value)) {
            out[key] = pyslam::Value(value.cast<std::string>());
        }
    }
    return out;
}

} // namespace

inline void bind_direct_slam(py::module &m) {
    py::enum_<pyslam::DirectTrackingMode>(m, "DirectTrackingMode")
        .value("KLT", pyslam::DirectTrackingMode::Klt)
        .value("HYBRID", pyslam::DirectTrackingMode::Hybrid)
        .value("SEMI_DIRECT", pyslam::DirectTrackingMode::SemiDirect);

    py::enum_<pyslam::DirectTrackingStatus>(m, "DirectTrackingStatus")
        .value("INIT", pyslam::DirectTrackingStatus::Init)
        .value("TRACKING_GOOD", pyslam::DirectTrackingStatus::Good)
        .value("TRACKING_BAD", pyslam::DirectTrackingStatus::Bad)
        .value("LOST", pyslam::DirectTrackingStatus::Lost);

    py::class_<pyslam::DirectBAResult>(m, "DirectBAResult")
        .def_readonly("applied", &pyslam::DirectBAResult::applied)
        .def_readonly("keyframes", &pyslam::DirectBAResult::keyframes)
        .def_readonly("landmarks", &pyslam::DirectBAResult::landmarks)
        .def_readonly("observations", &pyslam::DirectBAResult::observations)
        .def_readonly("outliers", &pyslam::DirectBAResult::outliers)
        .def_readonly("initial_cost", &pyslam::DirectBAResult::initial_cost)
        .def_readonly("final_cost", &pyslam::DirectBAResult::final_cost);

    py::class_<pyslam::DirectSlam, std::shared_ptr<pyslam::DirectSlam>>(m, "DirectSlam")
        .def(py::init([](const pyslam::CameraPtr &camera, py::object settings) {
                 return std::make_shared<pyslam::DirectSlam>(camera, direct_config_from_object(settings));
             }),
             py::arg("camera"), py::arg("settings") = py::none())
        .def(
            "track",
            [](pyslam::DirectSlam &self, py::object img, py::object img_right, py::object depth, int img_id,
               py::object timestamp) {
                const cv::Mat image = direct_mat_from_object(img);
                const cv::Mat image_right = direct_mat_from_object(img_right);
                const cv::Mat depth_image = direct_mat_from_object(depth);
                std::optional<double> stamp;
                if (!timestamp.is_none()) {
                    stamp = timestamp.cast<double>();
                }
                // All Python conversions finish before the native tracking/BA call.
                py::gil_scoped_release release;
                self.track(image, image_right, depth_image, img_id, stamp);
            },
            py::arg("img"), py::arg("img_right") = py::none(), py::arg("depth") = py::none(),
            py::arg("img_id") = 0, py::arg("timestamp") = py::none())
        .def("reset", &pyslam::DirectSlam::reset)
        .def("bundle_adjustment", &pyslam::DirectSlam::bundle_adjustment, py::call_guard<py::gil_scoped_release>())
        .def("apply_keyframe_poses", &pyslam::DirectSlam::apply_keyframe_poses, py::arg("poses"),
             py::call_guard<py::gil_scoped_release>())
        .def("last_ba", &pyslam::DirectSlam::last_ba)
        .def("draw_feature_trails", [](const pyslam::DirectSlam &self, py::object image,
                                       bool with_level_radius, int trail_max_length) {
            return self.draw_feature_trails(direct_mat_from_object(image), with_level_radius, trail_max_length);
        }, py::arg("img"), py::arg("with_level_radius") = false, py::arg("trail_max_length") = 16)
        // Return owned snapshots so viewer data remains valid across tracking and reset().
        .def("map_points", [](const pyslam::DirectSlam &self) {
            std::vector<pyslam::Vec3d> points;
            for (const auto &[id, point] : self.map().landmarks())
                if (!point.bad) points.push_back(point.position_w);
            Eigen::MatrixXd snapshot(points.size(), 3);
            for (size_t i = 0; i < points.size(); ++i) snapshot.row(i) = points[i].transpose();
            return snapshot;
        })
        .def("status", &pyslam::DirectSlam::status)
        .def("status_name", &pyslam::DirectSlam::status_name)
        .def("pose", &pyslam::DirectSlam::pose)
        .def("last_inliers", &pyslam::DirectSlam::last_inliers)
        .def("num_keyframes", [](const pyslam::DirectSlam &self) { return self.map().keyframes().size(); })
        .def("trajectory", [](const pyslam::DirectSlam &self) {
            std::vector<Eigen::Matrix4d> poses;
            poses.reserve(self.map().trajectory().size());
            for (const pyslam::DirectTrajectoryPose &pose : self.map().trajectory()) {
                poses.push_back(pose.Tcw);
            }
            return poses;
        })
        // Timestamps aligned index-by-index with trajectory().
        .def("trajectory_timestamps", [](const pyslam::DirectSlam &self) {
            std::vector<double> timestamps;
            timestamps.reserve(self.map().trajectory().size());
            for (const pyslam::DirectTrajectoryPose &pose : self.map().trajectory()) {
                timestamps.push_back(pose.timestamp);
            }
            return timestamps;
        })
        .def("tracked_features", [](const pyslam::DirectSlam &self) {
            std::vector<int64_t> ids;
            Eigen::MatrixXd pixels(self.live_points().size(), 2);
            Eigen::MatrixXd points(self.live_points().size(), 3);
            for (size_t i = 0; i < self.live_points().size(); ++i) {
                const auto &point = self.live_points()[i];
                ids.push_back(point.id);
                pixels.row(i) = point.uv.transpose();
                points.row(i) = point.position_w.transpose();
            }
            return py::make_tuple(ids, pixels, points);
        })
        // Viewer camera poses use Twc, whereas trajectory() deliberately retains core Tcw.
        .def("keyframes", [](const pyslam::DirectSlam &self) {
            std::vector<Eigen::Matrix4d> poses;
            std::vector<double> timestamps;
            for (const auto &kf : self.map().keyframes()) {
                poses.push_back(kf.pose.get_inverse_matrix());
                timestamps.push_back(kf.timestamp);
            }
            return py::make_tuple(poses, timestamps);
        })
        .def_property_readonly("mode", [](const pyslam::DirectSlam &self) { return self.params().mode; });
}
