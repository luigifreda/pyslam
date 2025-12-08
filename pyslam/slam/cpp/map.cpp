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
#include "utils/messages.h"

#include <algorithm>
#include <cmath>

#include "optimizer_g2o.h"

namespace pyslam {

// Map implementation
Map::Map() : max_point_id(0), max_frame_id(0), max_keyframe_id(0), viewer_scale(-1.0f) {
    local_map = std::make_unique<LocalCovisibilityMap>(this);
}

void Map::reset() {
    std::lock_guard<MapMutex> lock_map(_lock);
    std::lock_guard<MapMutex> lock_update(_update_lock);

    // Clear all data structures
    frames.clear();
    keyframes.clear();
    points.clear();

    keyframe_origins.clear();
    keyframes_map.clear();

    // Reset local map
    if (local_map) {
        local_map->reset();
    }
}

void Map::reset_session() {
    std::lock_guard<MapMutex> lock_map(_lock);
    std::lock_guard<MapMutex> lock_update(_update_lock);

    // Reset session-specific data
    if (reloaded_session_map_info) {
        // First, collect keyframes to remove
        std::vector<KeyFramePtr> keyframes_to_remove;
        for (auto &kf : keyframes) {
            if (kf->kid >= reloaded_session_map_info->max_keyframe_id) {
                keyframes_to_remove.push_back(kf);
            }
        }
        for (auto &kf : keyframes_to_remove) {
            kf->set_bad();
            keyframes.erase(kf);
            auto it = keyframes_map.find(kf->id);
            if (it != keyframes_map.end()) {
                keyframes_map.erase(it);
            }
            auto it2 = keyframe_origins.find(kf);
            if (it2 != keyframe_origins.end()) {
                keyframe_origins.erase(it2);
            }
        }

        // Similarly for points
        std::vector<MapPointPtr> points_to_remove;
        for (auto &p : points) {
            if (p->id >= reloaded_session_map_info->max_point_id) {
                points_to_remove.push_back(p);
            }
        }
        for (auto &p : points_to_remove) {
            p->set_bad();
            auto it = points.find(p);
            if (it != points.end()) {
                points.erase(it);
            }
        }

        // Similarly for frames
        std::vector<FramePtr> frames_to_remove;
        for (auto &f : frames) {
            if (f->id >= reloaded_session_map_info->max_frame_id) {
                frames_to_remove.push_back(f);
            }
        }
        for (auto &f : frames_to_remove) {
            frames.erase(std::remove(frames.begin(), frames.end(), f), frames.end());
        }

        // Reset the session of the local map
        local_map->reset_session(keyframes_to_remove, points_to_remove);
    } else {
        // If no session info, do a full reset
        reset();
    }
}

void Map::delete_map() {
    std::lock_guard<MapMutex> lock(_lock);
    for (auto &frame : frames) {
        frame->reset_points();
    }
    for (auto &keyframe : keyframes) {
        keyframe->reset_points();
    }
}

// Point operations
std::unordered_set<MapPointPtr> Map::get_points() const {
    std::lock_guard<MapMutex> lock(_lock);
    return points;
}

std::vector<MapPointPtr> Map::get_points_vector() const {
    std::lock_guard<MapMutex> lock(_lock);
    return std::vector<MapPointPtr>(points.begin(), points.end());
}

int Map::num_points() const {
    std::lock_guard<MapMutex> lock(_lock);
    return static_cast<int>(points.size());
}

int Map::add_point(MapPointPtr &point) {
    if (!point) {
        return -1;
    }
    std::lock_guard<MapMutex> lock(_lock);
    const int ret = max_point_id;
    point->id = ret;
    point->map = this;
    max_point_id++;
    points.insert(point);
    return ret;
}

void Map::remove_point(MapPointPtr point) {
    if (!point) {
        return;
    }

    std::lock_guard<MapMutex> lock(_lock);
    points.erase(point);
    // point->map = nullptr;
    point->delete_point();
}

void Map::remove_point_no_lock(MapPointPtr point) {
    if (!point) {
        return;
    }
    points.erase(point);
    // point->map = nullptr;
    point->delete_point();
}
// Frame operations
FramePtr Map::get_frame(int idx) const {
    std::lock_guard<MapMutex> lock(_lock);

    // Support negative indexing like Python
    if (idx < 0) {
        int positive_idx = static_cast<int>(frames.size()) + idx;
        if (positive_idx < 0 || positive_idx >= static_cast<int>(frames.size())) {
            return nullptr;
        }
        auto it = frames.begin();
        std::advance(it, positive_idx);
        return *it;
    }

    // For non-negative indices, use direct deque indexing
    if (idx >= 0 && idx < static_cast<int>(frames.size())) {
        return frames[idx];
    }

    return nullptr;
}

std::vector<FramePtr> Map::get_frames() const {
    std::lock_guard<MapMutex> lock(_lock);
    return std::vector<FramePtr>(frames.begin(), frames.end());
}

int Map::num_frames() const {
    std::lock_guard<MapMutex> lock(_lock);
    return static_cast<int>(frames.size());
}

int Map::add_frame(FramePtr &frame, bool override_id) {
    if (!frame) {
        MSG_WARN("Map::add_frame() - Frame is nullptr");
        return -1;
    }

    std::lock_guard<MapMutex> lock(_lock);
    int ret = frame->id;
    // Add to frames deque (with size limit)
    if (frames.size() >= Parameters::kMaxLenFrameDeque) {
        frames.pop_front();
    }
    frames.push_back(frame);

    // Update max frame ID
    if (override_id) {
        ret = max_frame_id;
        frame->id = ret;
        max_frame_id++;
    } else {
        max_frame_id = std::max(max_frame_id, frame->id + 1);
    }
    return ret;
}

void Map::remove_frame(FramePtr &frame) {
    if (!frame) {
        return;
    }

    std::lock_guard<MapMutex> lock(_lock);

    // Remove from frames deque
    auto it = std::find(frames.begin(), frames.end(), frame);
    if (it != frames.end()) {
        frames.erase(it);
    }
}

// KeyFrame operations
KeyFrameIdSet Map::get_keyframes() const {
    std::lock_guard<MapMutex> lock(_lock);
    return keyframes;
}

std::vector<KeyFramePtr> Map::get_keyframes_vector() const {
    std::lock_guard<MapMutex> lock(_lock);
    return std::vector<KeyFramePtr>(keyframes.begin(), keyframes.end());
}

KeyFramePtr Map::get_last_keyframe() const {
    std::lock_guard<MapMutex> lock(_lock);
    if (keyframes.empty()) {
        return nullptr;
    }
    return *keyframes.rbegin(); // Last element in set
}

std::vector<KeyFramePtr> Map::get_last_keyframes(int local_window_size) const {
    std::lock_guard<MapMutex> lock(_lock);

    std::vector<KeyFramePtr> result;
    auto it = keyframes.rbegin();
    for (int i = 0; i < local_window_size && it != keyframes.rend(); ++i, ++it) {
        result.push_back(*it);
    }
    return result;
}

int Map::num_keyframes() const {
    std::lock_guard<MapMutex> lock(_lock);
    return static_cast<int>(keyframes.size());
}

int Map::num_keyframes_session() const {
    std::lock_guard<MapMutex> lock(_lock);
    if (reloaded_session_map_info) {
        return static_cast<int>(keyframes.size()) - reloaded_session_map_info->num_keyframes;
    }
    return static_cast<int>(keyframes.size());
}

int Map::add_keyframe(KeyFramePtr &keyframe) {
    if (!keyframe) {
        return -1;
    }

    std::lock_guard<MapMutex> lock(_lock);
    assert(keyframe->is_keyframe);
    int ret = max_keyframe_id;
    keyframe->kid = ret;
    keyframe->is_keyframe = true;
    keyframe->map = this;
    keyframes.insert(keyframe);
    keyframes_map[keyframe->id] = keyframe;
    max_keyframe_id++;
    return ret;
}

void Map::remove_keyframe(KeyFramePtr keyframe) {
    if (!keyframe) {
        return;
    }

    std::lock_guard<MapMutex> lock(_lock);
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
double Map::optimize(int local_window_size, bool verbose, int rounds, bool use_robust_kernel,
                     bool do_cull_points, bool *abort_flag) {
    // TODO: implement support for gtsam

    const auto result = OptimizerG2o::bundle_adjustment(
        get_keyframes_vector(), get_points_vector(), local_window_size, false, rounds, 0,
        use_robust_kernel, abort_flag, false, verbose);
    if (do_cull_points) {
        remove_points_with_big_reproj_err(get_points_vector());
    }
    return result.mean_squared_error;
}

double Map::locally_optimize(KeyFramePtr &kf_ref, bool verbose, int rounds, bool *abort_flag) {
    // TODO: implement support for gtsam
    auto [keyframes, points, ref_keyframes] = local_map->update(kf_ref);
    auto keyframes_vector = std::vector<KeyFramePtr>(keyframes.begin(), keyframes.end());
    auto points_vector = std::vector<MapPointPtr>(points.begin(), points.end());
    auto ref_keyframes_vector =
        std::vector<KeyFramePtr>(ref_keyframes.begin(), ref_keyframes.end());
    const auto [mean_squared_error, outlier_ratio] =
        OptimizerG2o::local_bundle_adjustment<MapMutex>(keyframes_vector, points_vector,
                                                        ref_keyframes_vector, false, verbose,
                                                        rounds, abort_flag, nullptr);
    return mean_squared_error;
}

// Session management
bool Map::is_reloaded() const { return reloaded_session_map_info.get() != nullptr; }

void Map::set_reloaded_session_info(const ReloadedSessionMapInfo &info) {
    reloaded_session_map_info = std::make_unique<ReloadedSessionMapInfo>(info);
}

const ReloadedSessionMapInfo *Map::get_reloaded_session_info() const {
    return reloaded_session_map_info.get();
}

// Helper methods

void Map::update_keyframes_map() {
    std::lock_guard<MapMutex> lock(_lock);
    keyframes_map.clear();
    for (const auto &kf : keyframes) {
        keyframes_map[kf->id] = kf;
    }
}

// LocalMapBase implementation
LocalMapBase::LocalMapBase(Map *map) : map(map) {}

void LocalMapBase::reset() {
    std::lock_guard<MapMutex> lock(_lock);
    keyframes.clear();
    points.clear();
    ref_keyframes.clear();
}

void LocalMapBase::reset_session(const std::vector<KeyFramePtr> &keyframes_to_remove,
                                 const std::vector<MapPointPtr> &points_to_remove) {
    std::lock_guard<MapMutex> lock(_lock);

    if (keyframes_to_remove.empty() && points_to_remove.empty()) {
        reset();
    } else {
        for (KeyFramePtr kf : keyframes_to_remove) {
            keyframes.erase(kf);
            ref_keyframes.erase(kf);
        }
        for (MapPointPtr p : points_to_remove) {
            points.erase(p);
        }
    }
}

bool LocalMapBase::is_empty() const {
    std::lock_guard<MapMutex> lock(_lock);
    return keyframes.empty();
}

std::unordered_set<MapPointPtr> LocalMapBase::get_points() const {
    std::lock_guard<MapMutex> lock(_lock);
    return points;
}

int LocalMapBase::num_points() const {
    std::lock_guard<MapMutex> lock(_lock);
    return static_cast<int>(points.size());
}

KeyFrameIdSet LocalMapBase::get_keyframes() const {
    std::lock_guard<MapMutex> lock(_lock);
    return keyframes;
}

int LocalMapBase::num_keyframes() const {
    std::lock_guard<MapMutex> lock(_lock);
    return static_cast<int>(keyframes.size());
}

template <typename Container>
std::tuple<KeyFrameIdSet, std::unordered_set<MapPointPtr>, KeyFrameIdSet>
LocalMapBase::update_from_keyframes(const Container &local_keyframes) {
    KeyFrameIdSet good_keyframes;
    std::unordered_set<MapPointPtr> good_points;
    KeyFrameIdSet ref_keyframes;

    // Filter out bad keyframes
    for (auto &kf : local_keyframes) {
        if (!kf->is_bad()) {
            good_keyframes.insert(kf);
        }
    }

    // Get all good points from local keyframes
    for (auto &kf : good_keyframes) {
        std::vector<MapPointPtr> matched_points = kf->get_matched_good_points();
        for (const auto &p : matched_points) {
            good_points.insert(p);
        }
    }

    // Get reference keyframes
    for (const auto &p : good_points) {
        std::vector<KeyFramePtr> point_keyframes = p->keyframes();
        for (const auto &kf : point_keyframes) {
            if (!kf->is_bad() && good_keyframes.find(kf) == good_keyframes.end()) {
                ref_keyframes.insert(kf);
            }
        }
    }

    {
        std::lock_guard<MapMutex> lock(_lock);
        this->keyframes = good_keyframes;
        this->points = good_points;
        this->ref_keyframes = ref_keyframes;
    }

    return std::make_tuple(good_keyframes, good_points, ref_keyframes);
}

std::tuple<KeyFramePtr, std::vector<KeyFramePtr>, std::vector<MapPointPtr>>
LocalMapBase::get_frame_covisibles(const FramePtr &frame) {
    std::vector<MapPointPtr> frame_points = frame->get_matched_good_points();

    if (frame_points.empty()) {
        return std::make_tuple(nullptr, std::vector<KeyFramePtr>(), std::vector<MapPointPtr>());
    }

    // Count keyframes viewing the points
    std::unordered_map<KeyFramePtr, int> viewing_keyframes;
    for (const auto &p : frame_points) {
        std::vector<KeyFramePtr> point_keyframes = p->keyframes();
        for (const auto &kf : point_keyframes) {
            if (!kf->is_bad()) {
                viewing_keyframes[kf]++;
            }
        }
    }

    if (viewing_keyframes.empty()) {
        return std::make_tuple(nullptr, std::vector<KeyFramePtr>(), std::vector<MapPointPtr>());
    }

    // Get local keyframes
    std::vector<std::pair<KeyFramePtr, int>> sorted_keyframes;
    for (const auto &pair : viewing_keyframes) {
        sorted_keyframes.push_back(pair);
    }
    std::sort(sorted_keyframes.begin(), sorted_keyframes.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Reference keyframe is the most common
    KeyFramePtr kf_ref = sorted_keyframes.front().first;

    std::vector<KeyFramePtr> local_keyframes;
    std::unordered_set<MapPointPtr> local_points;

    int count = 0;
    for (const auto &pair : sorted_keyframes) {
        if (count >= Parameters::kMaxNumOfKeyframesInLocalMap)
            break;

        KeyFramePtr kf = pair.first;
        local_keyframes.push_back(kf);

        std::vector<MapPointPtr> kf_points = kf->get_matched_points();
        for (const auto &p : kf_points) {
            local_points.insert(p);
        }
        count++;
    }

    return std::make_tuple(kf_ref, local_keyframes,
                           std::vector<MapPointPtr>(local_points.begin(), local_points.end()));
}

// LocalWindowMap implementation
LocalWindowMap::LocalWindowMap(Map *map, int local_window_size)
    : LocalMapBase(map), local_window_size(local_window_size) {}

KeyFrameIdSet LocalWindowMap::update_keyframes(const KeyFramePtr &kf_ref) {
    std::lock_guard<MapMutex> lock(_lock);
    keyframes = KeyFrameIdSet(map->get_last_keyframes(local_window_size).begin(),
                              map->get_last_keyframes(local_window_size).end());
    return keyframes;
}

std::vector<KeyFramePtr> LocalWindowMap::get_best_neighbors(const KeyFramePtr &kf_ref, int N) {
    return map->get_last_keyframes(N);
}

std::tuple<KeyFrameIdSet, std::unordered_set<MapPointPtr>, KeyFrameIdSet>
LocalWindowMap::update(const KeyFramePtr &kf_ref) {
    update_keyframes(kf_ref);
    return update_from_keyframes(keyframes);
}

// LocalCovisibilityMap implementation
LocalCovisibilityMap::LocalCovisibilityMap(Map *map) : LocalMapBase(map) {}

KeyFrameIdSet LocalCovisibilityMap::update_keyframes(const KeyFramePtr &kf_ref) {
    std::lock_guard<MapMutex> lock(_lock);

    keyframes.clear();
    keyframes.insert(kf_ref);

    std::vector<KeyFramePtr> neighbor_kfs = kf_ref->get_covisible_keyframes();
    for (const auto &kf : neighbor_kfs) {
        if (!kf->is_bad()) {
            keyframes.insert(kf);
        }
    }
    return keyframes;
}

std::vector<KeyFramePtr> LocalCovisibilityMap::get_best_neighbors(const KeyFramePtr &kf_ref,
                                                                  int N) {
    return kf_ref->get_best_covisible_keyframes(N);
}

std::tuple<KeyFrameIdSet, std::unordered_set<MapPointPtr>, KeyFrameIdSet>
LocalCovisibilityMap::update(const KeyFramePtr &kf_ref) {
    update_keyframes(kf_ref);
    return update_from_keyframes(keyframes);
}

} // namespace pyslam