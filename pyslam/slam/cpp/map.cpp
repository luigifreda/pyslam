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

#include "map.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace pyslam {

// Map implementation
Map::Map() : max_point_id(0), max_frame_id(0), max_keyframe_id(0), viewer_scale(-1.0f) {

    // Initialize local map
    initialize_local_map();
}

void Map::reset() {
    std::lock_guard<std::mutex> lock_map(_lock);
    std::lock_guard<std::mutex> lock_update(_update_lock);

    // Clear all data structures
    frames.clear();
    keyframes.clear();
    points.clear();
    keyframe_origins.clear();
    keyframes_map.clear();

    // Reset counters
    max_point_id = 0;
    max_frame_id = 0;
    max_keyframe_id = 0;

    // Clear session info
    reloaded_session_map_info.reset();

    // Reset local map
    if (local_map) {
        local_map->reset();
    }

    viewer_scale = -1.0f;
}

void Map::reset_session() {
    std::lock_guard<std::mutex> lock_map(_lock);
    std::lock_guard<std::mutex> lock_update(_update_lock);

    // Reset session-specific data
    reloaded_session_map_info.reset();

    // Reset local map
    if (local_map) {
        local_map->reset();
    }
}

void Map::delete_map() {
    std::lock_guard<std::mutex> lock(_lock);
    for (Frame *frame : frames) {
        frame->reset_points();
    }
    for (KeyFrame *keyframe : keyframes) {
        keyframe->reset_points();
    }
}

// Point operations
std::set<MapPoint *> Map::get_points() const {
    std::lock_guard<std::mutex> lock(_lock);
    return points;
}

int Map::num_points() const {
    std::lock_guard<std::mutex> lock(_lock);
    return static_cast<int>(points.size());
}

int Map::add_point(MapPoint *point) {
    if (point == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(_lock);
    const int ret = max_point_id;
    point->id = ret;
    point->map = this;
    max_point_id++;
    points.insert(point);
    return ret;
}

void Map::remove_point(MapPoint *point) {
    if (point == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_lock);
    points.erase(point);
    point->map = nullptr;
}

// Frame operations
Frame *Map::get_frame(int idx) const {
    std::lock_guard<std::mutex> lock(_lock);
    for (Frame *frame : frames) {
        if (frame->id == idx) {
            return frame;
        }
    }
    return nullptr;
}

std::vector<Frame *> Map::get_frames() const {
    std::lock_guard<std::mutex> lock(_lock);
    return std::vector<Frame *>(frames.begin(), frames.end());
}

int Map::num_frames() const {
    std::lock_guard<std::mutex> lock(_lock);
    return static_cast<int>(frames.size());
}

void Map::add_frame(Frame *frame, bool override_id) {
    if (frame == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_lock);

    // Add to frames deque (with size limit)
    constexpr size_t kMaxLenFrameDeque = 20;
    if (frames.size() >= kMaxLenFrameDeque) {
        frames.pop_front();
    }
    frames.push_back(frame);

    // Update max frame ID
    if (override_id) {
        frame->id = max_frame_id++;
    } else {
        max_frame_id = std::max(max_frame_id, frame->id);
    }
}

void Map::remove_frame(Frame *frame) {
    if (frame == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_lock);

    // Remove from frames deque
    auto it = std::find(frames.begin(), frames.end(), frame);
    if (it != frames.end()) {
        frames.erase(it);
    }
}

// KeyFrame operations
std::set<KeyFrame *> Map::get_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock);
    return keyframes;
}

KeyFrame *Map::get_last_keyframe() const {
    std::lock_guard<std::mutex> lock(_lock);
    if (keyframes.empty()) {
        return nullptr;
    }
    return *keyframes.rbegin(); // Last element in set
}

std::vector<KeyFrame *> Map::get_last_keyframes(int local_window) const {
    std::lock_guard<std::mutex> lock(_lock);

    std::vector<KeyFrame *> result;
    auto it = keyframes.rbegin();
    for (int i = 0; i < local_window && it != keyframes.rend(); ++i, ++it) {
        result.push_back(*it);
    }
    return result;
}

int Map::num_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock);
    return static_cast<int>(keyframes.size());
}

int Map::num_keyframes_session() const {
    std::lock_guard<std::mutex> lock(_lock);
    if (reloaded_session_map_info) {
        return reloaded_session_map_info->num_keyframes;
    }
    return static_cast<int>(keyframes.size());
}

void Map::add_keyframe(KeyFrame *keyframe) {
    if (keyframe == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_lock);
    assert(keyframe->is_keyframe);
    keyframe->kid = max_keyframe_id;
    keyframes.insert(keyframe);
    keyframes_map[keyframe->id] = keyframe;
    max_keyframe_id++;
}

void Map::remove_keyframe(KeyFrame *keyframe) {
    if (keyframe == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(_lock);
    keyframes.erase(keyframe);
    keyframes_map.erase(keyframe->id);
    keyframe_origins.erase(keyframe);
}

// Visualization
cv::Mat Map::draw_feature_trails(cv::Mat &img) {
    if (frames.empty()) {
        return img;
    }
    return frames.back()->draw_all_feature_trails(img);
}

// Optimization
void Map::optimize(int num_iterations) {
    // This is a simplified implementation
    // The actual implementation would perform global bundle adjustment
}

void Map::locally_optimize(KeyFrame *kf_ref, int num_iterations) {
    // This is a simplified implementation
    // The actual implementation would perform local bundle adjustment
}

// Session management
bool Map::is_reloaded() const { return reloaded_session_map_info != nullptr; }

void Map::set_reloaded_session_info(const ReloadedSessionMapInfo &info) {
    reloaded_session_map_info = std::make_unique<ReloadedSessionMapInfo>(info);
}

const ReloadedSessionMapInfo *Map::get_reloaded_session_info() const {
    return reloaded_session_map_info.get();
}

// Helper methods
void Map::initialize_local_map() { local_map = std::make_unique<LocalCovisibilityMap>(this); }

void Map::update_keyframes_map() {
    std::lock_guard<std::mutex> lock(_lock);
    keyframes_map.clear();
    for (KeyFrame *kf : keyframes) {
        keyframes_map[kf->id] = kf;
    }
}

void Map::cleanup_isolated_elements() {
    // This is a simplified implementation
    // The actual implementation would remove isolated keyframes and map points
}

void Map::update_map_statistics() {
    // This is a simplified implementation
    // The actual implementation would update map statistics
}

// LocalMapBase implementation
LocalMapBase::LocalMapBase(Map *map) : map(map) {}

void LocalMapBase::reset() {
    std::lock_guard<std::mutex> lock(_lock);
    keyframes.clear();
    points.clear();
    ref_keyframes.clear();
}

void LocalMapBase::reset_session(const std::vector<KeyFrame *> &keyframes_to_remove,
                                 const std::vector<MapPoint *> &points_to_remove) {
    std::lock_guard<std::mutex> lock(_lock);

    if (keyframes_to_remove.empty() && points_to_remove.empty()) {
        reset();
    } else {
        for (KeyFrame *kf : keyframes_to_remove) {
            keyframes.erase(kf);
            ref_keyframes.erase(kf);
        }
        for (MapPoint *p : points_to_remove) {
            points.erase(p);
        }
    }
}

bool LocalMapBase::is_empty() const {
    std::lock_guard<std::mutex> lock(_lock);
    return keyframes.empty();
}

std::set<MapPoint *> LocalMapBase::get_points() const {
    std::lock_guard<std::mutex> lock(_lock);
    return points;
}

int LocalMapBase::num_points() const {
    std::lock_guard<std::mutex> lock(_lock);
    return static_cast<int>(points.size());
}

std::set<KeyFrame *> LocalMapBase::get_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock);
    return keyframes;
}

int LocalMapBase::num_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock);
    return static_cast<int>(keyframes.size());
}

std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
LocalMapBase::update_from_keyframes(const std::set<KeyFrame *> &local_keyframes) {
    std::set<KeyFrame *> good_keyframes;
    std::set<MapPoint *> good_points;
    std::set<KeyFrame *> ref_keyframes;

    // Filter out bad keyframes
    for (KeyFrame *kf : local_keyframes) {
        if (!kf->is_bad()) {
            good_keyframes.insert(kf);
        }
    }

    // Get all good points from local keyframes
    for (KeyFrame *kf : good_keyframes) {
        std::vector<MapPoint *> matched_points = kf->get_matched_good_points();
        for (MapPoint *p : matched_points) {
            good_points.insert(p);
        }
    }

    // Get reference keyframes
    for (MapPoint *p : good_points) {
        std::vector<KeyFrame *> point_keyframes = p->keyframes();
        for (KeyFrame *kf : point_keyframes) {
            if (!kf->is_bad() && good_keyframes.find(kf) == good_keyframes.end()) {
                ref_keyframes.insert(kf);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(_lock);
        this->keyframes = good_keyframes;
        this->points = good_points;
        this->ref_keyframes = ref_keyframes;
    }

    return std::make_tuple(good_keyframes, good_points, ref_keyframes);
}

std::tuple<KeyFrame *, std::vector<KeyFrame *>, std::set<MapPoint *>>
LocalMapBase::get_frame_covisibles(Frame *frame) {
    std::vector<MapPoint *> frame_points = frame->get_matched_good_points();

    if (frame_points.empty()) {
        return std::make_tuple(nullptr, std::vector<KeyFrame *>(), std::set<MapPoint *>());
    }

    // Count keyframes viewing the points
    std::unordered_map<KeyFrame *, int> viewing_keyframes;
    for (MapPoint *p : frame_points) {
        std::vector<KeyFrame *> point_keyframes = p->keyframes();
        for (KeyFrame *kf : point_keyframes) {
            if (!kf->is_bad()) {
                viewing_keyframes[kf]++;
            }
        }
    }

    if (viewing_keyframes.empty()) {
        return std::make_tuple(nullptr, std::vector<KeyFrame *>(), std::set<MapPoint *>());
    }

    // Find reference keyframe (most common)
    KeyFrame *kf_ref = nullptr;
    int max_count = 0;
    for (const auto &pair : viewing_keyframes) {
        if (pair.second > max_count) {
            max_count = pair.second;
            kf_ref = pair.first;
        }
    }

    // Get local keyframes
    std::vector<std::pair<KeyFrame *, int>> sorted_keyframes;
    for (const auto &pair : viewing_keyframes) {
        sorted_keyframes.push_back(pair);
    }
    std::sort(sorted_keyframes.begin(), sorted_keyframes.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    std::vector<KeyFrame *> local_keyframes;
    std::set<MapPoint *> local_points;

    constexpr int kMaxNumOfKeyframesInLocalMap = 20;
    int count = 0;
    for (const auto &pair : sorted_keyframes) {
        if (count >= kMaxNumOfKeyframesInLocalMap)
            break;

        KeyFrame *kf = pair.first;
        local_keyframes.push_back(kf);

        std::vector<MapPoint *> kf_points = kf->get_matched_points();
        for (MapPoint *p : kf_points) {
            local_points.insert(p);
        }
        count++;
    }

    return std::make_tuple(kf_ref, local_keyframes, local_points);
}

// LocalWindowMap implementation
LocalWindowMap::LocalWindowMap(Map *map, int local_window)
    : LocalMapBase(map), local_window(local_window) {}

std::set<KeyFrame *> LocalWindowMap::update_keyframes(KeyFrame *kf_ref) {
    std::lock_guard<std::mutex> lock(_lock);
    keyframes = std::set<KeyFrame *>(map->get_last_keyframes(local_window).begin(),
                                     map->get_last_keyframes(local_window).end());
    return keyframes;
}

std::vector<KeyFrame *> LocalWindowMap::get_best_neighbors(KeyFrame *kf_ref, int N) {
    return map->get_last_keyframes(N);
}

std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
LocalWindowMap::update(KeyFrame *kf_ref) {
    update_keyframes(kf_ref);
    return update_from_keyframes(keyframes);
}

// LocalCovisibilityMap implementation
LocalCovisibilityMap::LocalCovisibilityMap(Map *map) : LocalMapBase(map) {}

std::set<KeyFrame *> LocalCovisibilityMap::update_keyframes(KeyFrame *kf_ref) {
    std::lock_guard<std::mutex> lock(_lock);

    keyframes.clear();
    keyframes.insert(kf_ref);

    std::vector<KeyFrame *> neighbor_kfs = kf_ref->get_covisible_keyframes();
    for (KeyFrame *kf : neighbor_kfs) {
        if (!kf->is_bad()) {
            keyframes.insert(kf);
        }
    }

    return keyframes;
}

std::vector<KeyFrame *> LocalCovisibilityMap::get_best_neighbors(KeyFrame *kf_ref, int N) {
    return kf_ref->get_best_covisible_keyframes(N);
}

std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
LocalCovisibilityMap::update(KeyFrame *kf_ref) {
    update_keyframes(kf_ref);
    return update_from_keyframes(keyframes);
}

} // namespace pyslam