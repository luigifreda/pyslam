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

#include <Eigen/Dense>

#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "smart_pointers.h"

namespace pyslam {

// Forward declarations
class KeyFrame;
class Frame;
class MapPoint;
class Map;

// MapPointBase class - matches Python MapPointBase exactly
class MapPointBase {
  protected:
    // Thread-safe ID management
    static std::atomic<int> _id;
    static std::mutex _id_lock;

    mutable std::mutex _lock_pos;
    mutable std::mutex _lock_features;

  public:
    // Core data members
    int id;
    MapPtr map; // Pointer to Map object

    // Observations
    std::map<KeyFramePtr, int> _observations;
    std::map<FramePtr, int> _frame_views;

    // Status flags
    bool _is_bad = false;
    int _num_observations = 0;
    int num_times_visible = 0;
    int num_times_found = 0;
    int last_frame_id_seen = -1;

    // Replacement tracking
    MapPointPtr replacement;

    // Loop correction
    int corrected_by_kf = -1;
    int corrected_reference = -1;
    KeyFramePtr kf_ref;

    // Constructor - matches Python: __init__(self, id=None)
    MapPointBase(int id = -1);

    // Destructor
    virtual ~MapPointBase() = default;

    // Comparison operators
    bool operator==(const MapPointBase &other) const { return id == other.id; }
    bool operator<(const MapPointBase &other) const { return id < other.id; }
    bool operator<=(const MapPointBase &other) const { return id <= other.id; }

    // Hash function - match Python __hash__
    size_t hash() const { return std::hash<int>{}(id); }

    // String representations
    std::string observations_string() const;
    std::string frame_views_string() const;
    std::string to_string() const;

    // Observation access
    std::vector<std::pair<KeyFramePtr, int>> observations() const;
    std::vector<std::pair<KeyFramePtr, int>> observations_iter() const;
    std::vector<KeyFramePtr> keyframes() const;
    std::vector<KeyFramePtr> keyframes_iter() const;

    // Observation operations
    bool is_in_keyframe(const KeyFramePtr &keyframe) const;
    int get_observation_idx(const KeyFramePtr &keyframe) const;
    bool add_observation_no_lock_(KeyFramePtr &keyframe, int idx);
    bool add_observation(KeyFramePtr keyframe, int idx);         // no reference passing here!
    void remove_observation(KeyFramePtr keyframe, int idx = -1); // no reference passing here!

    // Frame view access
    std::vector<std::pair<FramePtr, int>> frame_views() const;
    std::vector<std::pair<FramePtr, int>> frame_views_iter() const;
    std::vector<FramePtr> frames() const;
    std::vector<FramePtr> frames_iter() const;

    // Frame view operations
    bool is_in_frame(const FramePtr &frame) const;
    bool add_frame_view(FramePtr &frame, int idx);
    void remove_frame_view(FramePtr frame, int idx = -1); // no reference passing here!

    // Status properties
    bool is_bad() const;
    int num_observations() const;
    bool is_good_with_min_obs(int minObs) const;
    std::pair<bool, bool> is_bad_and_is_good_with_min_obs(int minObs) const;
    bool is_bad_or_is_in_keyframe(const KeyFramePtr &keyframe) const;

    // Statistics
    void increase_visible(int num_times = 1);
    void increase_found(int num_times = 1);
    float get_found_ratio() const;
};

// MapPoint class - matches Python MapPoint exactly
class MapPoint : public MapPointBase, public std::enable_shared_from_this<MapPoint> {

  private:
    // Global lock for position updates
    static std::mutex global_lock_;

    // Temporary storage for ID-based data during deserialization
    std::vector<std::pair<int, int>> _observations_id_data;
    std::vector<std::pair<int, int>> _frame_views_id_data;
    int _kf_ref_id = -1;

  public:
    // Core geometric data  (private)
    Eigen::Vector3d _pt; // position in world frame
    Eigen::Vector3d normal;
    float _min_distance, _max_distance;

    // Visual data  (private)
    Eigen::Matrix<unsigned char, 3, 1> color;
    cv::Mat semantic_des;
    cv::Mat des; // best descriptor

    // Reference information
    int first_kid;

    // Update counters
    int num_observations_on_last_update_des;
    int num_observations_on_last_update_normals;
    int num_observations_on_last_update_semantics;

    // GBA support
    Eigen::Vector3d pt_GBA;
    int GBA_kf_id;

  public:
    // Constructor - matches Python: MapPoint(position, color, keyframe=None,
    // idxf=None, id=None)
    MapPoint(const Eigen::Vector3d &position, const Eigen::Matrix<unsigned char, 3, 1> &color,
             const KeyFramePtr &keyframe = nullptr, int idxf = -1, int id = -1);

    explicit MapPoint(int id = -1) : MapPointBase(id) {}

    // Destructor
    ~MapPoint() {
        // Clear references to prevent circular dependencies during shutdown
        _observations.clear();
        _frame_views.clear();
        kf_ref = nullptr;
    }

    // Delete copy constructor
    MapPoint(const MapPoint &) = delete;
    MapPoint &operator=(const MapPoint &) = delete;
    // Delete move constructor
    MapPoint(MapPoint &&) = delete;
    MapPoint &operator=(MapPoint &&) = delete;

    // Position access
    Eigen::Vector3d pt() const;
    Eigen::Vector4d homogeneous() const;
    void update_position(const Eigen::Vector3d &position);

    // Distance properties
    float min_distance() const;
    float max_distance() const;
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, float, float> get_all_pos_info() const;

    // Reference keyframe
    KeyFramePtr get_reference_keyframe() const;

    // Descriptor operations
    std::vector<cv::Mat> descriptors() const;
    float min_des_distance(const cv::Mat &descriptor) const;

    // Point lifecycle
    void delete_point();
    void set_bad();
    MapPointPtr get_replacement() const;
    Eigen::Vector3d get_normal() const;
    void replace_with(MapPointPtr &p);

    // Update operations
    void update_normal_and_depth(const FramePtr &frame = nullptr, int idxf = -1,
                                 bool force = false);
    void update_best_descriptor(bool force = false);
    void update_semantics(void *semantic_fusion_method = nullptr, bool force = false);
    void update_info();

    // Detection level prediction
    int predict_detection_level(float dist) const;

    // JSON serialization
    std::string to_json() const;
    static MapPointPtr from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                  const std::vector<FramePtr> &frames,
                                  const std::vector<KeyFramePtr> &keyframes);

    // Numpy serialization
    pybind11::tuple state_tuple() const;              // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &); // fills this object from the tuple

    // GBA operations
    void set_pt_GBA(const Eigen::Vector3d &pt_GBA);
    void set_GBA_kf_id(int GBA_kf_id);

    // Cleanup method for proper shutdown
    void clear_references() {
        _observations.clear();
        _frame_views.clear();
        kf_ref = nullptr;
    }

  private:
    // Helper methods
    void update_normal_and_depth_from_keyframe(const KeyFramePtr &kf, int idx);
    void compute_distance_range(int octave_level, float distance);
    void normalize_vector(const Eigen::Vector3d &v, Eigen::Vector3d &result) const;
};

// Hash function for MapPoint to use in unordered containers
struct MapPointHash {
    size_t operator()(const MapPoint *mp) const { return mp->hash(); }
    size_t operator()(const MapPointPtr &mp) const { return mp->hash(); }
};

// Equality function for MapPoint pointers
struct MapPointEqual {
    bool operator()(const MapPoint *lhs, const MapPoint *rhs) const {
        return lhs == rhs || (lhs && rhs && *lhs == *rhs);
    }
    bool operator()(const MapPointPtr &lhs, const MapPointPtr &rhs) const {
        return lhs == rhs || (lhs && rhs && *lhs == *rhs);
    }
};

} // namespace pyslam