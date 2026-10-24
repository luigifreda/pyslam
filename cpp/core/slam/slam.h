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

#include "local_mapping.h"
#include "map.h"
#include "tracking.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace pyslam {

enum class SlamMode { SLAM = 0, MAP_BROWSER = 1 };

// Same fields as pyslam/slam/slam.py. Unported collaborators stay Python objects.
class Slam : public std::enable_shared_from_this<Slam> {
  public:
    Slam() = default;
    ~Slam();

#ifdef USE_PYTHON
    void init(const CameraPtr &camera_in, const py::object &feature_tracker_config_in,
              const py::object &loop_detector_config_in, const py::object &semantic_mapping_config_in,
              SensorType sensor_type_in, const py::object &environment_type_in, SlamMode slam_mode_in,
              const py::object &config_in, bool headless_in, const py::object &viewer3d_in);
#endif

    void track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
               std::optional<double> timestamp, const cv::Mat &mask, const cv::Mat &mask_right);
    void reset();
    void reset_session();
    void quit();
    void request_reset();
    void set_tracking_state(SlamState state);
    void set_viewer_scale(float scale);
    float viewer_scale() const;
    void bundle_adjust();
#ifdef USE_PYTHON
    py::tuple get_final_trajectory();
    void save_system_state(const std::string &path);
    void load_system_state(const std::string &path);
#endif

    void set_feature_tracker_num_features(bool doubled);
    float feature_max_descriptor_distance() const;

    bool has_loop_closing() const;
    bool loop_closing_is_closing();
    void loop_closing_wait_if_closing();
    bool loop_closing_is_correcting();
    void loop_closing_add_keyframe(const KeyFramePtr &kf, const cv::Mat &img);
    bool loop_closing_relocalize(const FramePtr &frame, const cv::Mat &img);
    void loop_closing_quit();
    void loop_closing_request_reset();
    void loop_closing_save(const std::string &path);
    void loop_closing_load(const std::string &path);

    bool has_semantic() const;
    void semantic_push_keyframe(const KeyFramePtr &kf, const cv::Mat &img, const cv::Mat &img_right,
                                const cv::Mat &depth);
    void semantic_step();
    void semantic_request_reset();
    void semantic_quit();

    bool has_volumetric() const;
    void volumetric_add_keyframe(const KeyFramePtr &kf, const cv::Mat &img, const cv::Mat &img_right,
                                 const cv::Mat &depth);
    void volumetric_request_reset();
    void volumetric_quit();
    void volumetric_rebuild();

    CameraPtr camera;
    MapPtr map;
    TrackingPtr tracking;
    LocalMappingPtr local_mapping;
    SensorType sensor_type = SensorType::MONOCULAR;
#ifdef USE_PYTHON
    py::object sensor_type_py = py::none();
#endif
    SlamMode slam_mode = SlamMode::SLAM;
    bool headless = false;
    std::atomic<bool> reset_requested{false};
    bool has_quit = false;

    std::optional<double> far_points_threshold;
    bool use_fov_centers_based_kf_generation = false;
    double max_fov_centers_distance = -1.0;

#ifdef USE_PYTHON
    py::object feature_tracker_config = py::none();
    py::object loop_detector_config = py::none();
    py::object semantic_mapping_config = py::none();
    py::object environment_type = py::none();
    py::object config = py::none();
    py::object feature_tracker = py::none();
    py::object loop_closing = py::none();
    py::object semantic_mapping = py::none();
    py::object volumetric_integrator = py::none();
    py::object viewer3d = py::none();
    py::object GBA = py::none();
    py::object GBA_on_demand = py::none();
#endif
};

} // namespace pyslam
