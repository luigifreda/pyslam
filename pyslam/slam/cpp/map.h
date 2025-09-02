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

#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pyslam {

// Forward declarations
class LocalCovisibilityMap;
class LocalWindowMap;

class ReloadedSessionMapInfo {
  public:
    int num_keyframes;
    int num_points;
    int max_point_id;
    int max_frame_id;
    int max_keyframe_id;

    ReloadedSessionMapInfo(int nkf = 0, int np = 0, int mpid = 0, int mfid = 0, int mkfid = 0)
        : num_keyframes(nkf), num_points(np), max_point_id(mpid), max_frame_id(mfid),
          max_keyframe_id(mkfid) {}
};

class Map {
  private:
    mutable std::mutex _lock;
    mutable std::mutex _update_lock;

  public:
    // Core data structures
    std::deque<Frame *> frames;     // Limited size deque for frames
    std::set<KeyFrame *> keyframes; // OrderedSet equivalent
    std::set<MapPoint *> points;    // set equivalent

    // KeyFrame origins (first keyframes where map is rooted)
    std::set<KeyFrame *> keyframe_origins; // OrderedSet equivalent

    // Fast lookup maps
    std::unordered_map<int, KeyFrame *> keyframes_map; // frame_id -> keyframe

    // ID counters
    int max_point_id;    // 0 is the first point id
    int max_frame_id;    // 0 is the first frame id
    int max_keyframe_id; // 0 is the first keyframe id (kid)

    // Session information
    std::unique_ptr<ReloadedSessionMapInfo> reloaded_session_map_info;

    // Local map
    std::unique_ptr<LocalCovisibilityMap> local_map;

    // Viewer scale
    float viewer_scale;

    // Constructor - matches Python: __init__(self)
    Map();

    // Destructor
    ~Map() = default;

    // Core operations
    void reset();
    void reset_session();
    void delete_map();

    // Lock properties
    std::mutex &lock() { return _lock; }
    std::mutex &update_lock() { return _update_lock; }

    // Point operations
    std::set<MapPoint *> get_points() const;
    int num_points() const;
    void add_point(MapPoint *point);
    void remove_point(MapPoint *point);

    // Frame operations
    Frame *get_frame(int idx) const;
    std::vector<Frame *> get_frames() const;
    int num_frames() const;
    void add_frame(Frame *frame, bool override_id = false);
    void remove_frame(Frame *frame);

    // KeyFrame operations
    std::set<KeyFrame *> get_keyframes() const;
    KeyFrame *get_last_keyframe() const;
    std::vector<KeyFrame *> get_last_keyframes(int local_window = 5) const;
    int num_keyframes() const;
    int num_keyframes_session() const;
    void add_keyframe(KeyFrame *keyframe);
    void remove_keyframe(KeyFrame *keyframe);

    // Visualization
    void draw_feature_trails(cv::Mat &img);

    // Point management
    void add_points(const std::vector<Eigen::Vector3d> &points3d,
                    const std::vector<bool> &mask_pts3d, Frame *f, KeyFrame *kf,
                    const std::vector<int> &idxs, const cv::Mat &img);
    void add_stereo_points(const std::vector<Eigen::Vector3d> &points3d,
                           const std::vector<bool> &mask_pts3d, Frame *f, KeyFrame *kf,
                           const std::vector<int> &idxs, const cv::Mat &img);

    // Point filtering
    void remove_points_with_big_reproj_err(const std::vector<MapPoint *> &points);
    float compute_mean_reproj_error(const std::vector<MapPoint *> &points = {}) const;

    // Optimization
    void optimize(int num_iterations = 10);
    void locally_optimize(KeyFrame *kf_ref, int num_iterations = 5);

    // Serialization
    std::string to_json(const std::string &out_json = "{}") const;
    std::string serialize() const;
    void from_json(const std::string &loaded_json);
    void deserialize(const std::string &s);
    void save(const std::string &filename) const;
    void load(const std::string &filename);

    // Session management
    bool is_reloaded() const;
    void set_reloaded_session_info(const ReloadedSessionMapInfo &info);
    const ReloadedSessionMapInfo *get_reloaded_session_info() const;

  private:
    // Helper methods
    void initialize_local_map();
    void update_keyframes_map();
    void cleanup_isolated_elements();
    void update_map_statistics();
};

// LocalMapBase class - matches Python LocalMapBase exactly
class LocalMapBase {
  protected:
    mutable std::mutex _lock;

  public:
    Map *map;

    // Core data structures
    std::set<KeyFrame *> keyframes;     // OrderedSet equivalent
    std::set<MapPoint *> points;        // set equivalent
    std::set<KeyFrame *> ref_keyframes; // set equivalent

    // Constructor - matches Python: __init__(self, map=None)
    LocalMapBase(Map *map = nullptr);

    // Destructor
    virtual ~LocalMapBase() = default;

    // Core operations
    void reset();
    void reset_session(const std::vector<KeyFrame *> &keyframes_to_remove = {},
                       const std::vector<MapPoint *> &points_to_remove = {});

    // Lock property
    std::mutex &lock() { return _lock; }

    // Status
    bool is_empty() const;

    // Access methods
    std::set<MapPoint *> get_points() const;
    int num_points() const;
    std::set<KeyFrame *> get_keyframes() const;
    int num_keyframes() const;

    // Update methods
    std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
    update_from_keyframes(const std::set<KeyFrame *> &local_keyframes);
    std::tuple<KeyFrame *, std::vector<KeyFrame *>, std::set<MapPoint *>>
    get_frame_covisibles(Frame *frame);
};

// LocalWindowMap class - matches Python LocalWindowMap exactly
class LocalWindowMap : public LocalMapBase {
  public:
    // Constructor - matches Python: __init__(self, map=None,
    // local_window=Parameters.kLocalBAWindow)
    LocalWindowMap(Map *map = nullptr, int local_window = 5);

    // Destructor
    ~LocalWindowMap() = default;

    int local_window; // length of the local window

    // Update methods
    std::set<KeyFrame *> update_keyframes(KeyFrame *kf_ref = nullptr);
    std::vector<KeyFrame *> get_best_neighbors(KeyFrame *kf_ref = nullptr, int N = 20);
    std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
    update(KeyFrame *kf_ref = nullptr);
};

// LocalCovisibilityMap class - matches Python LocalCovisibilityMap exactly
class LocalCovisibilityMap : public LocalMapBase {
  public:
    // Constructor - matches Python: __init__(self, map=None)
    LocalCovisibilityMap(Map *map = nullptr);

    // Destructor
    ~LocalCovisibilityMap() = default;

    // Update methods
    std::set<KeyFrame *> update_keyframes(KeyFrame *kf_ref);
    std::vector<KeyFrame *> get_best_neighbors(KeyFrame *kf_ref, int N = 20);
    std::tuple<std::set<KeyFrame *>, std::set<MapPoint *>, std::set<KeyFrame *>>
    update(KeyFrame *kf_ref);
};

} // namespace pyslam