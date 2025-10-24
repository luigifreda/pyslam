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

#include "map.h"
#include "smart_pointers.h"
#include <set>
#include <vector>

namespace pyslam {

class LocalMappingCore {
  public:
    LocalMappingCore(Map *map, const SensorType sensor_type);
    ~LocalMappingCore() = default;

    // ==========================================
    // State Management
    // ==========================================
    void reset();

    void set_kf_cur(KeyFramePtr kf);
    KeyFramePtr &get_kf_cur();

    void set_kid_last_BA(int kid);
    int &get_kid_last_BA();

    void set_opt_abort_flag(bool value);
    bool &get_opt_abort_flag();

    void add_points(const std::vector<MapPointPtr> &points);
    void remove_points(const std::vector<MapPointPtr> &points);
    void clear_recent_points();
    size_t num_recent_points() const;
    std::vector<MapPointPtr> get_recently_added_points() const;

    // ==========================================
    // Computation Methods
    // ==========================================
    void process_new_keyframe();

    int cull_map_points();

    int cull_keyframes(bool use_fov_centers_based_kf_generation, float max_fov_centers_distance);

    std::pair<float, int> local_BA();

    float large_window_BA();

    int fuse_map_points(float descriptor_distance_sigma);

    // ==========================================
    // Static Utility Methods
    // ==========================================
    static bool
    check_remaining_fov_centers_max_distance(const std::vector<KeyFramePtr> &covisible_kfs,
                                             const KeyFramePtr &kf_to_remove, float dist);

  private:
    std::unordered_set<MapPointPtr> recently_added_points_;
    Map *map_ = nullptr;
    SensorType sensor_type_;

    KeyFramePtr kf_cur_ = nullptr;
    int kid_last_BA_ = -1;

    bool opt_abort_flag_ = false;
};

} // namespace pyslam