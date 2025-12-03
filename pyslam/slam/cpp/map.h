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
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "smart_pointers.h"

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

using KeyFrameIdSet = std::set<KeyFramePtr, KeyFrameIdCompare>;

class Map : public std::enable_shared_from_this<Map> {

  private:
    mutable std::mutex _lock;
    mutable std::mutex _update_lock;

  public:
    constexpr static int kLargeBAWindowSize = 20;
    constexpr static std::size_t kMaxLenFrameDeque = 20;

  public:
    // Core data structures
    std::deque<FramePtr> frames;  // Limited size deque for frames
    KeyFrameIdSet keyframes;      // OrderedSet equivalent
    std::set<MapPointPtr> points; // set equivalent

    // KeyFrame origins (first keyframes where map is rooted)
    std::set<KeyFramePtr, KeyFrameIdCompare> keyframe_origins; // OrderedSet equivalent
    // Fast lookup maps
    std::unordered_map<int, KeyFramePtr> keyframes_map; // frame_id -> keyframe

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
    std::set<MapPointPtr> get_points() const;
    std::vector<MapPointPtr> get_points_vector() const;
    int num_points() const;
    int add_point(MapPointPtr &point);
    void remove_point(MapPointPtr point); // no reference passing here!

    // Frame operations
    FramePtr get_frame(int idx) const;
    std::vector<FramePtr> get_frames() const;
    int num_frames() const;
    int add_frame(FramePtr &frame, bool override_id = false);
    void remove_frame(FramePtr &frame);

    // KeyFrame operations
    KeyFrameIdSet get_keyframes() const;
    std::vector<KeyFramePtr> get_keyframes_vector() const;
    KeyFramePtr get_last_keyframe() const;
    std::vector<KeyFramePtr> get_last_keyframes(int local_window_size = 5) const;
    int num_keyframes() const;
    int num_keyframes_session() const;
    int add_keyframe(KeyFramePtr &keyframe);
    void remove_keyframe(KeyFramePtr keyframe); // no reference passing here!

    // Visualization
    cv::Mat draw_feature_trails(cv::Mat &img);

    // Point management
    std::tuple<int, std::vector<bool>, std::vector<MapPointPtr>>
    add_points(const std::vector<Eigen::Vector3d> &points3d,
               const std::optional<std::vector<bool>> &mask_pts3d, KeyFramePtr &kf1,
               KeyFramePtr &kf2, const std::vector<int> &idxs1, const std::vector<int> &idxs2,
               const cv::Mat &img, bool do_check = true, double cos_max_parallax = 0.9998,
               std::optional<double> far_points_threshold = std::nullopt);
    int add_stereo_points(const std::vector<Eigen::Vector3d> &points3d,
                          const std::optional<std::vector<bool>> &mask_pts3d, FramePtr &f,
                          KeyFramePtr &kf, const std::vector<int> &idxs, const cv::Mat &img);

    // Point filtering
    void remove_points_with_big_reproj_err(const std::vector<MapPointPtr> &points);
    float compute_mean_reproj_error(const std::vector<MapPointPtr> &points = {}) const;

    // Optimization
    double optimize(int local_window_size = kLargeBAWindowSize, bool verbose = false,
                    int rounds = 10, bool use_robust_kernel = false, bool do_cull_points = false,
                    bool *abort_flag = nullptr);
    double locally_optimize(KeyFramePtr &kf_ref, bool verbose = false, int rounds = 5,
                            bool *abort_flag = nullptr);

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
    void update_keyframes_map();
};

// LocalMapBase class - matches Python LocalMapBase exactly
class LocalMapBase {
  protected:
    mutable std::mutex _lock;

  public:
    MapPtr map;

    // Core data structures
    KeyFrameIdSet keyframes;      // OrderedSet equivalent
    std::set<MapPointPtr> points; // set equivalent
    KeyFrameIdSet ref_keyframes;  // set equivalent

    // Constructor - matches Python: __init__(self, map=None)
    LocalMapBase(const MapPtr &map = nullptr);

    // Destructor
    virtual ~LocalMapBase() = default;

    // Core operations
    void reset();
    void reset_session(const std::vector<KeyFramePtr> &keyframes_to_remove = {},
                       const std::vector<MapPointPtr> &points_to_remove = {});

    // Lock property
    std::mutex &lock() { return _lock; }

    // Status
    bool is_empty() const;

    // Access methods
    std::set<MapPointPtr> get_points() const;
    int num_points() const;
    KeyFrameIdSet get_keyframes() const;
    int num_keyframes() const;

    // Update methods
    template <typename Container>
    std::tuple<KeyFrameIdSet, std::set<MapPointPtr>, KeyFrameIdSet>
    update_from_keyframes(const Container &local_keyframes);
    std::tuple<KeyFramePtr, std::vector<KeyFramePtr>, std::set<MapPointPtr>>
    get_frame_covisibles(const FramePtr &frame);
};

// LocalWindowMap class - matches Python LocalWindowMap exactly
class LocalWindowMap : public LocalMapBase {
  public:
    // Constructor - matches Python: __init__(self, map=None,
    // local_window_size=Parameters.kLocalBAWindowSize)
    LocalWindowMap(const MapPtr &map = nullptr, int local_window_size = 5);

    // Destructor
    ~LocalWindowMap() = default;

    int local_window_size; // length of the local window

    // Update methods
    KeyFrameIdSet update_keyframes(const KeyFramePtr &kf_ref = nullptr);
    std::vector<KeyFramePtr> get_best_neighbors(const KeyFramePtr &kf_ref = nullptr, int N = 20);
    std::tuple<KeyFrameIdSet, std::set<MapPointPtr>, KeyFrameIdSet>
    update(const KeyFramePtr &kf_ref = nullptr);
};

// LocalCovisibilityMap class - matches Python LocalCovisibilityMap exactly
class LocalCovisibilityMap : public LocalMapBase {
  public:
    // Constructor - matches Python: __init__(self, map=None)
    LocalCovisibilityMap(const MapPtr &map = nullptr);

    // Destructor
    ~LocalCovisibilityMap() = default;

    // Update methods
    KeyFrameIdSet update_keyframes(const KeyFramePtr &kf_ref);
    std::vector<KeyFramePtr> get_best_neighbors(const KeyFramePtr &kf_ref, int N = 20);
    std::tuple<KeyFrameIdSet, std::set<MapPointPtr>, KeyFrameIdSet>
    update(const KeyFramePtr &kf_ref);
};

} // namespace pyslam
