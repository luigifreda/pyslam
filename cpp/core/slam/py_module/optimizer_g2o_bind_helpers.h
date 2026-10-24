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

#include "optimizer_g2o.h"
#include "sim3_pose.h"
#include "utils/messages.h"

#include "py_wrappers.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace py = pybind11;

// Wrapper functions to match Python signatures
namespace pyslam {

// ------------------------------------------------------------
// Wrapper functions to match Python signatures
// ------------------------------------------------------------

// Wrapper for bundle_adjustment to return tuple (mean_squared_error, result_dict)
inline std::pair<double, py::dict> bundle_adjustment_wrapper_g2o(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    std::optional<int> local_window_size = std::nullopt, bool fixed_points = false, int rounds = 10,
    int loop_kf_id = 0, bool use_robust_kernel = false, py::object abort_flag = py::none(),
    py::object mp_abort_flag = py::none(), py::dict result_dict = py::none(), bool verbose = false,
    py::object print_func = py::none()) {

    const bool fill_result_dict = !result_dict.is_none();

    py::object abort_flag_object = abort_flag.is_none() ? mp_abort_flag : abort_flag;
    PyG2oAbortFlag abort_flag_wrapper(abort_flag_object);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();

    BundleAdjustmentResult result;
    {
        py::gil_scoped_release release;
        result = OptimizerG2o::bundle_adjustment(keyframes, points, local_window_size, fixed_points,
                                                 rounds, loop_kf_id, use_robust_kernel,
                                                 abort_flag_value_ptr, fill_result_dict, verbose);
        abort_flag_wrapper.stop_monitoring();
    }

    py::dict out_result_dict = fill_result_dict ? result_dict : py::dict();
    if (fill_result_dict) {
        out_result_dict["keyframe_updates"] = std::move(result.keyframe_updates);
        out_result_dict["point_updates"] = std::move(result.point_updates);
    }

    return std::make_pair(result.mean_squared_error, out_result_dict);
}

// Wrapper for global_bundle_adjustment to return tuple (mean_squared_error, result_dict)
inline std::pair<double, py::dict> global_bundle_adjustment_wrapper_g2o(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    int rounds = 10, int loop_kf_id = 0, bool use_robust_kernel = false,
    py::object abort_flag = py::none(), py::object mp_abort_flag = py::none(),
    py::dict result_dict = py::none(), bool verbose = false, py::object print_func = py::none()) {

    const bool fill_result_dict = !result_dict.is_none();

    py::object abort_flag_object = abort_flag.is_none() ? mp_abort_flag : abort_flag;
    PyG2oAbortFlag abort_flag_wrapper(abort_flag_object);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();

    BundleAdjustmentResult result;
    {
        py::gil_scoped_release release;
        result = OptimizerG2o::global_bundle_adjustment(keyframes, points, rounds, loop_kf_id,
                                                        use_robust_kernel, abort_flag_value_ptr,
                                                        fill_result_dict, verbose);
        abort_flag_wrapper.stop_monitoring();
    }

    py::dict out_result_dict = fill_result_dict ? result_dict : py::dict();
    if (fill_result_dict) {
        out_result_dict["keyframe_updates"] = std::move(result.keyframe_updates);
        out_result_dict["point_updates"] = std::move(result.point_updates);
    }

    return std::make_pair(result.mean_squared_error, out_result_dict);
}

// Wrapper for global_bundle_adjustment_map to return tuple (mean_squared_error, result_dict)
inline std::pair<double, py::dict> global_bundle_adjustment_map_wrapper_g2o(
    MapPtr map, int rounds = 10, int loop_kf_id = 0, bool use_robust_kernel = false,
    py::object abort_flag = py::none(), py::object mp_abort_flag = py::none(),
    py::dict result_dict = py::none(), bool verbose = false, py::object print_func = py::none()) {

    const bool fill_result_dict = !result_dict.is_none();

    py::object abort_flag_object = abort_flag.is_none() ? mp_abort_flag : abort_flag;
    PyG2oAbortFlag abort_flag_wrapper(abort_flag_object);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();

    BundleAdjustmentResult result;
    {
        py::gil_scoped_release release;
        result = OptimizerG2o::global_bundle_adjustment_map(map, rounds, loop_kf_id,
                                                            use_robust_kernel, abort_flag_value_ptr,
                                                            fill_result_dict, verbose);
        abort_flag_wrapper.stop_monitoring();
    }

    py::dict out_result_dict = fill_result_dict ? result_dict : py::dict();
    if (fill_result_dict) {
        out_result_dict["keyframe_updates"] = std::move(result.keyframe_updates);
        out_result_dict["point_updates"] = std::move(result.point_updates);
    }

    return std::make_pair(result.mean_squared_error, out_result_dict);
}

// Wrapper for pose_optimization to return tuple (mean_squared_error, is_ok, num_valid_points)
inline std::tuple<double, bool, int>
pose_optimization_wrapper_g2o(FramePtr frame, bool verbose = false, int rounds = 10) {

    py::gil_scoped_release release;
    const auto result = OptimizerG2o::pose_optimization(frame, verbose, rounds);

    return std::make_tuple(result.mean_squared_error, result.is_ok, result.num_valid_points);
}

// Wrapper for local_bundle_adjustment to return tuple (mean_squared_error, ratio_bad_observations)
inline std::pair<double, double> local_bundle_adjustment_wrapper_g2o(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref = {}, bool fixed_points = false,
    bool verbose = false, int rounds = 10, py::object abort_flag = py::none(),
    py::object map_lock = py::none()) {

    PyG2oAbortFlag abort_flag_wrapper(abort_flag);
    bool *abort_flag_value_ptr = abort_flag_wrapper.get_value_ptr();

    std::pair<double, double> result;
    {
        std::unique_ptr<pyslam::PyLock> lock;
        if (!map_lock.is_none()) {
            lock = std::make_unique<pyslam::PyLock>(map_lock);
        }
        PyLock *lock_ptr = lock ? lock.get() : nullptr;

        py::gil_scoped_release release;
        result = OptimizerG2o::local_bundle_adjustment<PyLock>(keyframes, points, keyframes_ref,
                                                               fixed_points, verbose, rounds,
                                                               abort_flag_value_ptr, lock_ptr);
        abort_flag_wrapper.stop_monitoring();
    }

    return result;
}

// Wrapper for optimize_sim3 to return tuple (num_inliers, R, t, scale, delta_error)
inline std::tuple<int, Eigen::Matrix3d, Eigen::Vector3d, double, double> optimize_sim3_wrapper_g2o(
    KeyFramePtr kf1, KeyFramePtr kf2, const std::vector<MapPointPtr> &map_points1,
    const std::vector<MapPointPtr> &map_point_matches12, const Eigen::Matrix3d &R12,
    const Eigen::Vector3d &t12, double s12, double th2, bool fix_scale, bool verbose = false) {

    py::gil_scoped_release release;
    const auto result = OptimizerG2o::optimize_sim3(kf1, kf2, map_points1, map_point_matches12, R12,
                                                    t12, s12, th2, fix_scale, verbose);

    return std::make_tuple(result.num_inliers, result.R, result.t, result.scale,
                           result.delta_error);
}

// Wrapper for optimize_essential_graph to return double (mean_squared_error)
inline double optimize_essential_graph_wrapper_g2o(
    MapPtr map_object, KeyFramePtr loop_keyframe, KeyFramePtr current_keyframe,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &non_corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, std::vector<KeyFramePtr>> &loop_connections,
    bool fix_scale, py::object print_fun = py::none(), bool verbose = false) {

    py::gil_scoped_release release;
    return OptimizerG2o::optimize_essential_graph(map_object, loop_keyframe, current_keyframe,
                                                  non_corrected_sim3_map, corrected_sim3_map,
                                                  loop_connections, fix_scale, verbose);
}

} // namespace pyslam
