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
#include "slam.h"

#include "config_parameters.h"
#include "frame.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#ifdef USE_PYTHON
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
// cv::Mat arguments passed into Python (loop closing, semantics, volumetric
// integration) need this caster in this translation unit. The binding module
// includes it, but that does not cover calls made from here.
#include "casters/opencv_type_casters.h"
#endif

namespace pyslam {

namespace {

#ifdef USE_PYTHON
bool alive(const py::object &obj) { return obj && !obj.is_none(); }

py::object call_kwargs(const py::object &function, const py::dict &kwargs) {
    return function(**kwargs);
}

void wait_until_ready(const py::object &obj, const char *name) {
    if (!alive(obj)) {
        return;
    }
    py::module_::import("pyslam.utilities.waiting")
        .attr("wait_for_ready")(obj.attr("is_ready"), name);
}
#endif

} // namespace

Slam::~Slam() {
    try {
        quit();
    } catch (const std::exception &error) {
        std::cerr << "Slam destructor: " << error.what() << std::endl;
    }
}

#ifdef USE_PYTHON
void Slam::init(const CameraPtr &camera_in, const py::object &feature_tracker_config_in,
                const py::object &loop_detector_config_in, const py::object &semantic_mapping_config_in,
                SensorType sensor_type_in, const py::object &environment_type_in, SlamMode slam_mode_in,
                const py::object &config_in, bool headless_in, const py::object &viewer3d_in) {
    Parameters::sync_runtime_from_python();
    camera = camera_in;
    sensor_type = sensor_type_in;
    environment_type = environment_type_in;
    slam_mode = slam_mode_in;
    headless = headless_in;
    viewer3d = viewer3d_in;
    feature_tracker_config = feature_tracker_config_in;
    loop_detector_config = loop_detector_config_in;
    semantic_mapping_config = semantic_mapping_config_in;
    config = config_in;
    sensor_type_py = py::module_::import("pyslam.io.dataset_types").attr("SensorType")(
        static_cast<int>(sensor_type));

    py::module_ tracker_mod = py::module_::import("pyslam.local_features.feature_tracker");
    py::dict tracker_kwargs = py::cast<py::dict>(feature_tracker_config);
    feature_tracker = call_kwargs(tracker_mod.attr("feature_tracker_factory"), tracker_kwargs);
    py::module_::import("pyslam.slam.feature_tracker_shared")
        .attr("FeatureTrackerShared")
        .attr("set_feature_tracker")(feature_tracker, py::arg("force") = true);
    if (sensor_type == SensorType::STEREO) {
        py::object tracker_right =
            call_kwargs(tracker_mod.attr("feature_tracker_factory"), tracker_kwargs);
        py::module_::import("pyslam.slam.feature_tracker_shared")
            .attr("FeatureTrackerShared")
            .attr("set_feature_tracker_right")(tracker_right, py::arg("force") = true);
    }
    if (Parameters::kUseEssentialMatrixFitting) {
        feature_tracker.attr("matcher").attr("ratio_test") = 0.8;
    }
    const py::object lk = tracker_mod.attr("FeatureTrackerTypes").attr("LK");
    const py::object tracker_type = feature_tracker.attr("tracker_type");
    if (tracker_type.equal(lk)) {
        throw std::runtime_error(
            "SLAM: At present time, you cannot use Lukas-Kanade feature_tracker in this SLAM framework!");
    }

    map = std::make_shared<Map>();
    local_mapping = std::make_shared<LocalMapping>(this);

    if (slam_mode == SlamMode::SLAM) {
        if (Parameters::kDoSparseSemanticMappingAndSegmentation && alive(semantic_mapping_config)) {
            py::dict semantic_kwargs(semantic_mapping_config.attr("copy")());
            semantic_kwargs["slam"] = shared_from_this();
            semantic_kwargs["headless"] = headless;
            semantic_kwargs["image_size"] = py::make_tuple(camera->width, camera->height);
            semantic_mapping = call_kwargs(py::module_::import("pyslam.semantics.semantic_mapping_factory")
                                               .attr("semantic_mapping_factory"),
                                           semantic_kwargs);
            py::module_::import("pyslam.semantics.semantic_mapping_shared")
                .attr("SemanticMappingShared")
                .attr("set_semantic_mapping")(semantic_mapping, py::arg("force") = true);
            if (Parameters::kSemanticMappingOnSeparateThread) {
                semantic_mapping.attr("start")();
            }
            wait_until_ready(semantic_mapping, "SemanticMapping");
        }
        if (Parameters::kDoVolumetricIntegration) {
            py::object integrator_type =
                py::module_::import("pyslam.dense.volumetric_integrator_types")
                    .attr("VolumetricIntegratorType")
                    .attr("from_string")(py::module_::import("pyslam.config_parameters")
                                             .attr("Parameters")
                                             .attr("kVolumetricIntegrationType"));
            py::object qdense = py::none();
            if (alive(viewer3d) && py::hasattr(viewer3d, "qdense")) {
                qdense = viewer3d.attr("qdense");
            }
            volumetric_integrator =
                py::module_::import("pyslam.dense.volumetric_integrator_factory")
                    .attr("volumetric_integrator_factory")(integrator_type, camera, environment_type,
                                                          sensor_type_py, qdense);
            wait_until_ready(volumetric_integrator, "VolumetricIntegrator");
        }
        if (Parameters::kUseLoopClosing && alive(loop_detector_config)) {
            loop_closing = py::module_::import("pyslam.loop_closing.loop_closing")
                               .attr("LoopClosing")(shared_from_this(), loop_detector_config,
                                                    py::arg("headless") = headless);
            GBA = loop_closing.attr("GBA");
            loop_closing.attr("start")();
            wait_until_ready(loop_closing, "LoopClosing");
        }
    }

    if (Parameters::kLocalMappingOnSeparateThread) {
        local_mapping->start();
    }
    tracking = std::make_shared<Tracking>(this);

    if (alive(config)) {
        if (py::hasattr(config, "far_points_threshold") && !config.attr("far_points_threshold").is_none()) {
            far_points_threshold = py::cast<double>(config.attr("far_points_threshold"));
        }
        if (py::hasattr(config, "use_fov_centers_based_kf_generation")) {
            use_fov_centers_based_kf_generation =
                py::cast<bool>(config.attr("use_fov_centers_based_kf_generation"));
        }
        if (py::hasattr(config, "max_fov_centers_distance") &&
            !config.attr("max_fov_centers_distance").is_none()) {
            max_fov_centers_distance = py::cast<double>(config.attr("max_fov_centers_distance"));
        }
        Frame::is_compute_median_depth = use_fov_centers_based_kf_generation;
        tracking->far_points_threshold = far_points_threshold;
        tracking->use_fov_centers_based_kf_generation = use_fov_centers_based_kf_generation;
        tracking->max_fov_centers_distance = max_fov_centers_distance;
        local_mapping->far_points_threshold = far_points_threshold;
        local_mapping->use_fov_centers_based_kf_generation = use_fov_centers_based_kf_generation;
        local_mapping->max_fov_centers_distance = static_cast<float>(max_fov_centers_distance);
    }
}
#endif

void Slam::track(const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth, int img_id,
                 std::optional<double> timestamp, const cv::Mat &mask, const cv::Mat &mask_right) {
    if (tracking) {
        tracking->track(img, img_right, depth, img_id, timestamp, mask, mask_right);
    }
}

void Slam::request_reset() { reset_requested = true; }

void Slam::reset() {
    if (local_mapping) {
        local_mapping->request_reset();
    }
    semantic_request_reset();
    loop_closing_request_reset();
    volumetric_request_reset();
    if (tracking) {
        tracking->reset();
    }
    if (map) {
        map->reset();
    }
    reset_requested = false;
}

void Slam::reset_session() {
    if (local_mapping) {
        local_mapping->request_reset();
    }
    semantic_request_reset();
    volumetric_request_reset();
    if (tracking) {
        tracking->reset();
    }
    if (map) {
        map->reset_session();
    }
    reset_requested = false;
}

void Slam::quit() {
    if (has_quit) {
        return;
    }
    has_quit = true;
    if (local_mapping) {
        local_mapping->quit();
    }
    semantic_quit();
    loop_closing_quit();
    volumetric_quit();
#ifdef USE_PYTHON
    {
        py::gil_scoped_acquire gil;
        if (alive(GBA_on_demand)) {
            GBA_on_demand.attr("quit")();
        }
        // These Python collaborators retain Slam. pybind cannot trace those C++
        // references for cyclic GC, so explicit shutdown must break the cycles.
        GBA_on_demand = py::none();
        GBA = py::none();
        loop_closing = py::none();
        semantic_mapping = py::none();
        volumetric_integrator = py::none();
    }
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void Slam::set_tracking_state(SlamState state) {
    if (tracking) {
        tracking->state = state;
    }
}

void Slam::set_viewer_scale(float scale) {
    if (map) {
        map->viewer_scale = scale;
    }
}

float Slam::viewer_scale() const { return map ? map->viewer_scale : 0.f; }

void Slam::set_feature_tracker_num_features(bool doubled) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(feature_tracker)) {
        return;
    }
    if (doubled) {
        feature_tracker.attr("set_double_num_features")();
    } else {
        feature_tracker.attr("set_normal_num_features")();
    }
#else
    (void)doubled;
#endif
}

float Slam::feature_max_descriptor_distance() const {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (alive(feature_tracker) && py::hasattr(feature_tracker, "feature_manager")) {
        return py::cast<float>(feature_tracker.attr("feature_manager").attr("max_descriptor_distance"));
    }
#endif
    return Parameters::kMaxDescriptorDistance;
}

bool Slam::has_loop_closing() const {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    return alive(loop_closing);
#else
    return false;
#endif
}

bool Slam::loop_closing_is_closing() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return false;
    }
    return py::cast<bool>(loop_closing.attr("is_closing")());
#else
    return false;
#endif
}

void Slam::loop_closing_wait_if_closing() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return;
    }
    loop_closing.attr("wait_if_closing")();
#endif
}

bool Slam::loop_closing_is_correcting() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return false;
    }
    return py::cast<bool>(loop_closing.attr("is_correcting")());
#else
    return false;
#endif
}

void Slam::loop_closing_add_keyframe(const KeyFramePtr &kf, const cv::Mat &img) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing) || !kf) {
        return;
    }
    loop_closing.attr("add_keyframe")(kf, img);
#else
    (void)kf;
    (void)img;
#endif
}

bool Slam::loop_closing_relocalize(const FramePtr &frame, const cv::Mat &img) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing) || !frame) {
        return false;
    }
    return py::cast<bool>(loop_closing.attr("relocalize")(frame, img));
#else
    (void)frame;
    (void)img;
    return false;
#endif
}

void Slam::loop_closing_quit() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return;
    }
    loop_closing.attr("quit")();
#endif
}

void Slam::loop_closing_request_reset() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return;
    }
    loop_closing.attr("request_reset")();
#endif
}

void Slam::loop_closing_save(const std::string &path) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return;
    }
    loop_closing.attr("save")(path);
#else
    (void)path;
#endif
}

void Slam::loop_closing_load(const std::string &path) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(loop_closing)) {
        return;
    }
    loop_closing.attr("load")(path);
#else
    (void)path;
#endif
}

bool Slam::has_semantic() const {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    return alive(semantic_mapping);
#else
    return false;
#endif
}

void Slam::semantic_push_keyframe(const KeyFramePtr &kf, const cv::Mat &img, const cv::Mat &img_right,
                                  const cv::Mat &depth) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(semantic_mapping) || !kf) {
        return;
    }
    semantic_mapping.attr("push_keyframe")(kf, img, img_right, depth);
#else
    (void)kf;
    (void)img;
    (void)img_right;
    (void)depth;
#endif
}

void Slam::semantic_step() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(semantic_mapping)) {
        return;
    }
    semantic_mapping.attr("step")();
#endif
}

void Slam::semantic_request_reset() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(semantic_mapping)) {
        return;
    }
    semantic_mapping.attr("request_reset")();
#endif
}

void Slam::semantic_quit() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(semantic_mapping)) {
        return;
    }
    semantic_mapping.attr("quit")();
#endif
}

bool Slam::has_volumetric() const {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    return alive(volumetric_integrator);
#else
    return false;
#endif
}

void Slam::volumetric_add_keyframe(const KeyFramePtr &kf, const cv::Mat &img, const cv::Mat &img_right,
                                   const cv::Mat &depth) {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(volumetric_integrator) || !kf) {
        return;
    }
    volumetric_integrator.attr("add_keyframe")(kf, img, img_right, depth);
#else
    (void)kf;
    (void)img;
    (void)img_right;
    (void)depth;
#endif
}

void Slam::volumetric_request_reset() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(volumetric_integrator)) {
        return;
    }
    volumetric_integrator.attr("request_reset")();
#endif
}

void Slam::volumetric_quit() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(volumetric_integrator)) {
        return;
    }
    volumetric_integrator.attr("quit")();
#endif
}

void Slam::volumetric_rebuild() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    if (!alive(volumetric_integrator) || !map) {
        return;
    }
    volumetric_integrator.attr("rebuild")(map);
#endif
}

void Slam::bundle_adjust() {
#ifdef USE_PYTHON
    py::gil_scoped_acquire gil;
    const bool use_multiprocessing = !py::cast<bool>(py::module_::import("pyslam.utilities.multi_processing")
                                                         .attr("MultiprocessingManager")
                                                         .attr("is_start_method_spawn")());
    if (!alive(GBA_on_demand)) {
        GBA_on_demand = py::module_::import("pyslam.slam.global_bundle_adjustment")
                            .attr("GlobalBundleAdjustment")(shared_from_this(),
                                                            py::arg("use_multiprocessing") = use_multiprocessing);
    }
    if (py::cast<bool>(GBA_on_demand.attr("is_running")())) {
        return;
    }
    GBA_on_demand.attr("quit")();
    GBA_on_demand.attr("start")(py::arg("loop_kf_id") = 0);
    while (!py::cast<bool>(GBA_on_demand.attr("has_finished")())) {
        py::gil_scoped_release release;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    GBA_on_demand.attr("correct_after_GBA")();
    volumetric_rebuild();
#endif
}

#ifdef USE_PYTHON
py::tuple Slam::get_final_trajectory() {
    py::list poses;
    py::list timestamps;
    py::list ids;
    if (!map || !tracking) {
        return py::make_tuple(poses, timestamps, ids);
    }
    KeyFramePtr first = map->get_first_keyframe();
    if (!first) {
        return py::make_tuple(poses, timestamps, ids);
    }
    const Eigen::Matrix4d Two = first->Twc();
    const auto &history = tracking->tracking_history;
    const size_t n = history.slam_states.size();
    for (size_t i = 0; i < n; ++i) {
        if (history.slam_states[i] != SlamState::OK || i >= history.kf_references.size() ||
            i >= history.relative_frame_poses.size()) {
            continue;
        }
        KeyFramePtr keyframe = history.kf_references[i];
        Eigen::Matrix4d Tcr = Eigen::Matrix4d::Identity();
        while (keyframe && keyframe->is_bad()) {
            Tcr = Tcr * keyframe->Tcp();
            keyframe = keyframe->get_parent();
        }
        if (!keyframe) {
            continue;
        }
        const Eigen::Matrix4d Trw = Tcr * keyframe->Tcw() * Two;
        const Eigen::Matrix4d Tcw = history.relative_frame_poses[i].matrix() * Trw;
        poses.append(Tcw.inverse());
        timestamps.append(i < history.timestamps.size() ? history.timestamps[i] : 0.0);
        ids.append(i < history.ids.size() ? history.ids[i] : -1);
    }
    return py::make_tuple(poses, timestamps, ids);
}

void Slam::save_system_state(const std::string &path) {
    py::gil_scoped_acquire gil;
    if (tracking) {
        tracking->wait_for_local_mapping(1.0, true);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::filesystem::create_directories(path);
    py::module_ ser = py::module_::import("pyslam.utilities.serialization");
    py::object serialize = ser.attr("SerializationJSON").attr("serialize");
    py::dict map_out;
    map_out["USE_CPP_CORE"] = true;
    map_out["sensor_type"] = serialize(sensor_type_py);
    map_out["environment_type"] = serialize(environment_type);
    map_out["map"] = map ? map->to_json() : std::string("{}");
    map_out["feature_tracker_config"] = serialize(feature_tracker_config);
    map_out["loop_detector_config"] = serialize(loop_detector_config);
    map_out["semantic_mapping_config"] = serialize(semantic_mapping_config);
    py::object formatted =
        ser.attr("format_floats_for_json")(map_out);
    py::object dumped = py::module_::import("json").attr("dumps")(formatted);
    std::ofstream map_file(path + "/map.json");
    map_file << py::cast<std::string>(dumped);
    loop_closing_save(path);
    if (alive(volumetric_integrator)) {
        volumetric_integrator.attr("save")(path);
    }
}

void Slam::load_system_state(const std::string &path) {
    py::gil_scoped_acquire gil;
    const std::string map_file_path = path + "/map.json";
    if (!std::filesystem::exists(map_file_path)) {
        throw std::runtime_error("SLAM state file does not exist: " + map_file_path);
    }
    if (local_mapping) {
        local_mapping->quit();
    }
    loop_closing_quit();
    loop_closing = py::none();
    GBA = py::none();
    std::ifstream input(map_file_path);
    std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    py::object loaded = py::module_::import("json").attr("loads")(contents);
    py::module_ ser = py::module_::import("pyslam.utilities.serialization");
    py::object deserialize = ser.attr("SerializationJSON").attr("deserialize");
    sensor_type_py = deserialize(loaded["sensor_type"]);
    sensor_type = static_cast<SensorType>(py::cast<int>(sensor_type_py.attr("value")));
    environment_type = deserialize(loaded["environment_type"]);
    feature_tracker_config = deserialize(loaded["feature_tracker_config"]);
    loop_detector_config = deserialize(loaded["loop_detector_config"]);
    semantic_mapping_config = deserialize(loaded["semantic_mapping_config"]);

    py::module_ tracker_mod = py::module_::import("pyslam.local_features.feature_tracker");
    feature_tracker =
        call_kwargs(tracker_mod.attr("feature_tracker_factory"), py::cast<py::dict>(feature_tracker_config));
    py::module_::import("pyslam.slam.feature_tracker_shared")
        .attr("FeatureTrackerShared")
        .attr("set_feature_tracker")(feature_tracker, py::arg("force") = true);

    if (sensor_type == SensorType::STEREO) {
        py::object tracker_right = call_kwargs(
            tracker_mod.attr("feature_tracker_factory"), py::cast<py::dict>(feature_tracker_config));
        py::module_::import("pyslam.slam.feature_tracker_shared")
            .attr("FeatureTrackerShared")
            .attr("set_feature_tracker_right")(tracker_right, py::arg("force") = true);
    }
    if (Parameters::kUseEssentialMatrixFitting) {
        feature_tracker.attr("matcher").attr("ratio_test") = 0.8;
    }

    if (Parameters::kUseLoopClosing && alive(loop_detector_config)) {
        loop_closing = py::module_::import("pyslam.loop_closing.loop_closing")
                           .attr("LoopClosing")(shared_from_this(), loop_detector_config,
                                                py::arg("headless") = headless);
        GBA = loop_closing.attr("GBA");
        loop_closing.attr("start")();
        wait_until_ready(loop_closing, "LoopClosing");
        loop_closing.attr("load")(path);
    }
    py::object map_json = loaded["map"];
    py::object map_json_clean = ser.attr("convert_inf_nan")(map_json);
    std::string map_json_str = py::isinstance<py::dict>(map_json_clean)
                                   ? py::cast<std::string>(py::module_::import("json").attr("dumps")(map_json_clean))
                                   : py::cast<std::string>(map_json_clean);
    if (map) {
        map->from_json(map_json_str);
    }
    if (Parameters::kLocalMappingOnSeparateThread && local_mapping) {
        local_mapping->start();
    }
}
#endif

} // namespace pyslam
