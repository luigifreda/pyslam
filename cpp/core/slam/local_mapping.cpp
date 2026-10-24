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
#include "local_mapping.h"

#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "frontend_helpers.h"
#include "geometry_matchers.h"
#include "slam.h"
#include "tracking.h"

#include <chrono>
#include <iostream>
#include <future>
#include <memory>
#include <thread>
#include <utility>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace pyslam {

namespace {

constexpr double kLocalMappingSleepTime = 5e-3;

struct PythonGilRelease {
#ifdef USE_PYTHON
    std::unique_ptr<py::gil_scoped_release> guard;
    PythonGilRelease() {
        if (Py_IsInitialized() && PyGILState_Check()) {
            guard = std::make_unique<py::gil_scoped_release>();
        }
    }
#endif
};

std::vector<KeyFramePtr> filter_keyframes_geometrically(const KeyFramePtr &target,
                                                        const std::vector<KeyFramePtr> &others,
                                                        bool is_monocular) {
    std::vector<KeyFramePtr> filtered;
    const int num_frames = static_cast<int>(others.size());
    const Eigen::Vector3d origin = target ? target->Ow() : Eigen::Vector3d::Zero();
    for (const auto &kf : others) {
        if (!kf || kf == target || kf->is_bad() || kf->des.empty()) {
            continue;
        }
        if (is_monocular && num_frames > 10) {
            const double baseline = (origin - kf->Ow()).norm();
            MatNx3d no_points;
            double median_depth = kf->compute_points_median_depth<double>(no_points, 0.75);
            if (median_depth < 0.0 && target) {
                median_depth = target->compute_points_median_depth<double>(no_points, 0.75);
            }
            if (median_depth > 0.0 && baseline / median_depth < Parameters::kMinRatioBaselineDepth) {
                continue;
            }
        }
        filtered.push_back(kf);
    }
    return filtered;
}

} // namespace

LocalMapping::LocalMapping(Slam *slam_in)
    : slam(slam_in),
      local_mapping_core(slam_in && slam_in->map ? slam_in->map.get() : nullptr,
                         slam_in ? slam_in->sensor_type : SensorType::MONOCULAR) {}

LocalMapping::~LocalMapping() { quit(); }

void LocalMapping::start() {
    if (is_running) {
        return;
    }
    if (work_thread_.joinable()) {
        quit();
    }
    is_running = true;
    work_thread_ = std::thread([this]() { run(); });
}

void LocalMapping::quit() {
    is_running = false;
    set_opt_abort_flag(true);
    queue_condition_.notify_all();
    idle_condition_.notify_all();
    PythonGilRelease release_gil;
    if (work_thread_.joinable()) {
        work_thread_.join();
    }
    if (large_ba_thread_.joinable()) {
        large_ba_thread_.join();
    }
}

void LocalMapping::run() {
    while (is_running) {
        step();
    }
    std::deque<LocalMappingKeyframeItem> discarded;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        discarded.swap(queue_);
    }
    set_idle(true);
}

void LocalMapping::push_keyframe(const KeyFramePtr &keyframe, const cv::Mat &img,
                                 const cv::Mat &img_right, const cv::Mat &depth) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        LocalMappingKeyframeItem item;
        item.keyframe = keyframe;
        item.img = img.empty() && keyframe ? keyframe->img : img;
        item.img_right = img_right.empty() && keyframe ? keyframe->img_right : img_right;
        item.depth = depth.empty() && keyframe ? keyframe->depth_img : depth;
        queue_.push_back(std::move(item));
        set_opt_abort_flag(true);
    }
    queue_condition_.notify_all();
}

std::optional<LocalMappingKeyframeItem> LocalMapping::pop_keyframe(double timeout) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (queue_.empty()) {
        queue_condition_.wait_for(lock, std::chrono::duration<double>(timeout), [this]() {
            return !queue_.empty() || stop_requested || reset_requested || !is_running;
        });
    }
    if (queue_.empty() || reset_requested) {
        return std::nullopt;
    }
    LocalMappingKeyframeItem item = std::move(queue_.front());
    set_idle(false);
    queue_.pop_front();
    return item;
}

int LocalMapping::queue_size() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(queue_.size());
}

bool LocalMapping::is_idle() {
    std::lock_guard<std::mutex> lock(idle_mutex_);
    return is_idle_;
}

void LocalMapping::set_idle(bool flag) {
    {
        std::lock_guard<std::mutex> lock(idle_mutex_);
        is_idle_ = flag;
    }
    idle_condition_.notify_all();
}

void LocalMapping::wait_idle(std::optional<double> timeout) {
    if (!is_running) {
        return;
    }
    PythonGilRelease release_gil;
    std::unique_lock<std::mutex> lock(idle_mutex_);
    const auto ready = [this]() { return is_idle_ || !is_running; };
    if (!timeout) {
        idle_condition_.wait(lock, ready);
        return;
    }
    idle_condition_.wait_for(lock, std::chrono::duration<double>(*timeout), ready);
}

void LocalMapping::interrupt_optimization() { set_opt_abort_flag(true); }

void LocalMapping::request_stop() {
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        stop_requested = true;
    }
    set_opt_abort_flag(true);
    queue_condition_.notify_all();
}

bool LocalMapping::is_stop_requested() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    return stop_requested;
}

bool LocalMapping::stop_if_requested() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (stop_requested && !do_not_stop) {
        stopped = true;
        return true;
    }
    return false;
}

bool LocalMapping::is_stopped() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    return stopped;
}

bool LocalMapping::set_do_not_stop(bool value) {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    if (value && stopped) {
        return false;
    }
    do_not_stop = value;
    return true;
}

void LocalMapping::release() {
    if (!is_running) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        stop_requested = false;
        stopped = false;
        do_not_stop = false;
    }
    std::deque<LocalMappingKeyframeItem> discarded;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        discarded.swap(queue_);
    }
    set_idle(true);
}

void LocalMapping::request_reset() {
    PythonGilRelease release_gil;
    {
        std::lock_guard<std::mutex> lock(reset_mutex_);
        if (reset_requested) {
            return;
        }
        reset_requested = true;
    }
    set_opt_abort_flag(true);
    queue_condition_.notify_all();
    if (work_thread_.joinable() && is_running) {
        while (true) {
            queue_condition_.notify_all();
            {
                std::lock_guard<std::mutex> lock(reset_mutex_);
                if (!reset_requested || !is_running) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else {
        reset_if_requested();
    }
}

void LocalMapping::reset_if_requested() {
    // No optimizer may retain the old map or current keyframe after reset is acknowledged.
    PythonGilRelease release_gil;
    std::deque<LocalMappingKeyframeItem> discarded;
    std::lock_guard<std::mutex> lock(reset_mutex_);
    if (!reset_requested) {
        return;
    }
    if (large_ba_thread_.joinable()) {
        large_ba_thread_.join();
    }
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        discarded.swap(queue_);
    }
    reset_requested = false;
    total_num_triangulated_points = 0;
    total_num_fused_points = 0;
    total_num_culled_points = 0;
    total_num_culled_keyframes = 0;
    last_num_triangulated_points = 0;
    local_mapping_core.reset();
    local_mapping_core.set_kf_cur(nullptr);
    kf_cur.reset();
    img_cur.release();
    img_cur_right.release();
    depth_cur.release();
}

void LocalMapping::set_opt_abort_flag(bool value) { local_mapping_core.set_opt_abort_flag(value); }

float LocalMapping::descriptor_distance_sigma() const {
    if (slam && slam->tracking) {
        return slam->tracking->descriptor_distance_sigma.load();
    }
    return Parameters::kMaxDescriptorDistance;
}

void LocalMapping::step() {
    if (slam && slam->map && slam->map->num_keyframes() > 0) {
        // Drain queued keyframes before acknowledging stop (GBA waits for an empty queue).
        if (!is_stop_requested() || queue_size() > 0) {
            auto item = pop_keyframe(Parameters::kLocalMappingTimeoutPopKeyframe);
            if (item && item->keyframe) {
                kf_cur = item->keyframe;
                img_cur = item->img;
                img_cur_right = item->img_right;
                depth_cur = item->depth;
                set_idle(false);
                try {
                    do_local_mapping();
                } catch (const std::exception &error) {
                    std::cerr << "LocalMapping: " << error.what() << std::endl;
                }
                set_idle(true);
            } else {
                set_idle(true);
            }
        } else if (stop_if_requested()) {
            set_idle(true);
            while (is_stopped() && is_running && !reset_requested) {
                std::this_thread::sleep_for(std::chrono::duration<double>(kLocalMappingSleepTime));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::duration<double>(kLocalMappingSleepTime));
        }
    } else {
        std::this_thread::sleep_for(std::chrono::duration<double>(kLocalMappingSleepTime));
    }
    reset_if_requested();
}

void LocalMapping::do_local_mapping() {
    const auto time_start = std::chrono::steady_clock::now();
    if (!kf_cur) {
        return;
    }
    if (large_ba_thread_.joinable()) {
        large_ba_thread_.join();
    }
    local_mapping_core.set_kf_cur(kf_cur);
    const bool is_single_thread = !Parameters::kLocalMappingOnSeparateThread;
    local_mapping_core.process_new_keyframe();
    const int num_culled_points = local_mapping_core.cull_map_points();
    last_num_culled_points = num_culled_points;
    total_num_culled_points += num_culled_points;

    const int total_new_pts = create_new_map_points();
    last_num_triangulated_points = total_new_pts;
    total_num_triangulated_points += total_new_pts;

    if (queue_size() == 0 || is_single_thread) {
        const int fused = local_mapping_core.fuse_map_points(descriptor_distance_sigma());
        last_num_fused_points = fused;
        total_num_fused_points += fused;
    }
    set_opt_abort_flag(false);
    if ((queue_size() == 0 && !is_stop_requested()) || is_single_thread) {
        if (large_ba_thread_.joinable()) {
            large_ba_thread_.join();
        }
        if (!slam->loop_closing_is_correcting()) {
            auto ba = local_mapping_core.local_BA();
            mean_ba_chi2_error = ba.first;
        }
        if (Parameters::kUseLargeWindowBA && kf_cur &&
            kf_cur->kid >= Parameters::kEveryNumFramesLargeWindowBA &&
            local_mapping_core.get_kid_last_BA() != kf_cur->kid &&
            kf_cur->kid % Parameters::kEveryNumFramesLargeWindowBA == 0) {
            large_ba_thread_ = std::thread([this]() {
                try {
                    local_mapping_core.large_window_BA();
                } catch (const std::exception &error) {
                    std::cerr << "LocalMapping large BA: " << error.what() << std::endl;
                }
            });
        }
        const int culled_kfs = local_mapping_core.cull_keyframes(use_fov_centers_based_kf_generation,
                                                                 max_fov_centers_distance);
        last_num_culled_keyframes = culled_kfs;
        total_num_culled_keyframes += culled_kfs;
    }
    if (slam && kf_cur) {
        slam->semantic_push_keyframe(kf_cur, img_cur, img_cur_right, depth_cur);
        if (!Parameters::kSemanticMappingOnSeparateThread) {
            slam->semantic_step();
        }
        slam->loop_closing_add_keyframe(kf_cur, img_cur);
        slam->volumetric_add_keyframe(kf_cur, img_cur, img_cur_right, depth_cur);
    }
    time_local_mapping =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time_start).count();
}

int LocalMapping::create_new_map_points() {
    if (!kf_cur || !slam || !slam->map || !slam->map->local_map) {
        return 0;
    }
    int total_new_pts = 0;
    const bool is_monocular = slam->sensor_type == SensorType::MONOCULAR;
    const int num_neighbors = is_monocular ? Parameters::kLocalMappingNumNeighborKeyFramesMonocular
                                           : Parameters::kLocalMappingNumNeighborKeyFramesStereo;
    const std::vector<KeyFramePtr> neighbors =
        slam->map->local_map->get_best_neighbors(kf_cur, num_neighbors);
    const std::vector<KeyFramePtr> candidates =
        filter_keyframes_geometrically(kf_cur, neighbors, is_monocular);
    const float ratio = FeatureSharedResources::feature_match_ratio_test > 0.f
                            ? FeatureSharedResources::feature_match_ratio_test
                            : Parameters::kFeatureMatchDefaultRatioTest;
    // Python compute_frame_matches precomputes descriptor matches with a bounded
    // worker pool. Preserve that scheduling option: matching callbacks commonly
    // release the GIL in OpenCV/Torch and benefit from concurrent execution.
    std::vector<FeatureMatchingOutput> precomputed_matches(candidates.size());
    const bool parallel = Parameters::kLocalMappingParallelKpsMatching && candidates.size() > 1;
    if (parallel) {
        std::atomic<size_t> next{0};
        const size_t worker_count = std::min(
            candidates.size(), static_cast<size_t>(std::max(1, Parameters::kLocalMappingParallelKpsMatchingNumWorkers)));
        std::vector<std::future<void>> workers;
        workers.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            workers.push_back(std::async(std::launch::async, [&]() {
                for (size_t i = next.fetch_add(1); i < candidates.size(); i = next.fetch_add(1)) {
                    const auto &kf = candidates[i];
                    precomputed_matches[i] = match_descriptors(
                        kf_cur->img, kf->img, kf_cur->des, kf->des, kf_cur->kps, kf->kps, ratio);
                }
            }));
        }
        for (auto &worker : workers) {
            worker.get();
        }
    }
    int neighbor_index = 0;
    for (const auto &kf : candidates) {
        if (neighbor_index > 0 && queue_size() > 0) {
            return total_new_pts;
        }
        const auto matches = parallel
            ? std::move(precomputed_matches[static_cast<size_t>(neighbor_index)])
            : match_descriptors(kf_cur->img, kf->img, kf_cur->des, kf->des, kf_cur->kps, kf->kps, ratio);
        ++neighbor_index;
        if (matches.first.empty() || matches.second.empty() || matches.first.size() != matches.second.size()) {
            continue;
        }
        auto searched = EpipolarMatcher::search_frame_for_triangulation(
            kf_cur, kf, matches.first, matches.second, 0.5f * descriptor_distance_sigma(), is_monocular);
        const std::vector<int> &idxs_cur = std::get<0>(searched);
        const std::vector<int> &idxs = std::get<1>(searched);
        const int num_found = std::get<2>(searched);
        if (num_found == 0 || idxs_cur.empty()) {
            continue;
        }
        const MatNx2f kpn_cur = keypoint_rows(kf_cur->kpsn, idxs_cur);
        const MatNx2f kpn_kf = keypoint_rows(kf->kpsn, idxs);
        auto triangulated = triangulate_normalized_points(kf_cur->pose(), kf->pose(), kpn_cur, kpn_kf);
        KeyFramePtr kf_cur_mut = kf_cur;
        KeyFramePtr kf_mut = kf;
        auto added = slam->map->add_points(triangulated.first, triangulated.second, kf_cur_mut, kf_mut,
                                           idxs_cur, idxs, kf_cur->img, true, Parameters::kCosMaxParallax,
                                           far_points_threshold);
        total_new_pts += std::get<0>(added);
        local_mapping_core.add_points(std::get<2>(added));
    }
    return total_new_pts;
}

} // namespace pyslam
