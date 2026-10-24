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

#include "keyframe.h"
#include "local_mapping_core.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <opencv2/core.hpp>
#include <optional>
#include <thread>

namespace pyslam {

class Slam;

struct LocalMappingKeyframeItem {
    KeyFramePtr keyframe;
    cv::Mat img;
    cv::Mat img_right;
    cv::Mat depth;
};

// Queue, idle flag, and thread around LocalMappingCore. Triangulation stays here.
class LocalMapping {
  public:
    explicit LocalMapping(Slam *slam);
    ~LocalMapping();

    void start();
    void quit();
    void run();
    void step();
    void do_local_mapping();
    int create_new_map_points();

    void push_keyframe(const KeyFramePtr &keyframe, const cv::Mat &img, const cv::Mat &img_right,
                       const cv::Mat &depth);
    std::optional<LocalMappingKeyframeItem> pop_keyframe(double timeout);
    int queue_size();
    bool is_idle();
    void set_idle(bool flag);
    void wait_idle(std::optional<double> timeout);
    void interrupt_optimization();
    void request_stop();
    bool is_stop_requested();
    bool stop_if_requested();
    bool is_stopped();
    bool set_do_not_stop(bool value);
    void release();
    void request_reset();
    void reset_if_requested();
    void set_opt_abort_flag(bool value);
    float descriptor_distance_sigma() const;

    Slam *slam = nullptr;
    LocalMappingCore local_mapping_core;
    KeyFramePtr kf_cur;
    cv::Mat img_cur;
    cv::Mat img_cur_right;
    cv::Mat depth_cur;
    std::atomic<bool> is_running{false};
    std::optional<double> far_points_threshold;
    bool use_fov_centers_based_kf_generation = false;
    float max_fov_centers_distance = -1.f;
    std::atomic<int> last_num_triangulated_points{0};
    std::atomic<int> total_num_triangulated_points{0};
    std::atomic<int> last_num_fused_points{0};
    std::atomic<int> total_num_fused_points{0};
    std::atomic<int> last_num_culled_points{0};
    std::atomic<int> total_num_culled_points{0};
    std::atomic<int> last_num_culled_keyframes{0};
    std::atomic<int> total_num_culled_keyframes{0};
    std::atomic<double> mean_ba_chi2_error{0.0};
    std::atomic<double> time_local_mapping{0.0};

  private:
    std::deque<LocalMappingKeyframeItem> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::mutex idle_mutex_;
    std::condition_variable idle_condition_;
    bool is_idle_ = true;
    std::mutex stop_mutex_;
    std::atomic<bool> stop_requested{false};
    bool stopped = false;
    bool do_not_stop = false;
    std::mutex reset_mutex_;
    std::atomic<bool> reset_requested{false};
    std::thread work_thread_;
    std::thread large_ba_thread_;
};

} // namespace pyslam
