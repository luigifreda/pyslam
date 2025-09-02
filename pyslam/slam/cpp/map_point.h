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
#include <iterator>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyslam {

// Forward declarations
class KeyFrame;
class Frame;
class MapPoint;

// MapPointBase class - matches Python MapPointBase exactly
class MapPointBase {
protected:
  // Thread-safe ID management - match Python exactly
  static std::atomic<int> _id;
  static std::mutex _id_lock;

  mutable std::mutex _lock_pos;
  mutable std::mutex _lock_features;

public:
  // Core data members - match Python exactly
  int id;
  void *map; // Pointer to Map object (void* to avoid circular dependency)

  // Observations - match Python exactly
  std::unordered_map<KeyFrame *, int> _observations;
  std::unordered_map<Frame *, int> _frame_views;

  // Status flags - match Python exactly
  bool _is_bad;
  int _num_observations;
  int num_times_visible;
  int num_times_found;
  int last_frame_id_seen;

  // Replacement tracking - match Python exactly
  MapPoint *replacement;

  // Loop correction - match Python exactly
  int corrected_by_kf;
  int corrected_reference;
  KeyFrame *kf_ref;

  // Constructor - matches Python: __init__(self, id=None)
  MapPointBase(int id = -1);

  // Destructor
  virtual ~MapPointBase() = default;

  // Comparison operators - match Python exactly
  bool operator==(const MapPointBase &other) const { return id == other.id; }
  bool operator<(const MapPointBase &other) const { return id < other.id; }
  bool operator<=(const MapPointBase &other) const { return id <= other.id; }

  // Hash function - match Python __hash__
  size_t hash() const { return std::hash<int>{}(id); }

  // String representations - match Python exactly
  std::string observations_string() const;
  std::string frame_views_string() const;
  std::string to_string() const;

  // Observation access - match Python exactly
  std::vector<std::pair<KeyFrame *, int>> observations() const;
  std::vector<std::pair<KeyFrame *, int>> observations_iter() const;
  std::vector<KeyFrame *> keyframes() const;
  std::vector<KeyFrame *> keyframes_iter() const;

  // Observation operations - match Python exactly
  bool is_in_keyframe(KeyFrame *keyframe) const;
  int get_observation_idx(KeyFrame *keyframe) const;
  bool add_observation_no_lock_(KeyFrame *keyframe, int idx);
  bool add_observation(KeyFrame *keyframe, int idx);
  std::pair<bool, bool> add_observation_if_not_bad(KeyFrame *keyframe, int idx);
  void remove_observation(KeyFrame *keyframe, int idx = -1);

  // Frame view access - match Python exactly
  std::vector<std::pair<Frame *, int>> frame_views() const;
  std::vector<std::pair<Frame *, int>> frame_views_iter() const;
  std::vector<Frame *> frames() const;
  std::vector<Frame *> frames_iter() const;

  // Frame view operations - match Python exactly
  bool is_in_frame(Frame *frame) const;
  bool add_frame_view(Frame *frame, int idx);
  void remove_frame_view(Frame *frame, int idx = -1);

  // Status properties - match Python exactly
  bool is_bad() const;
  int num_observations() const;
  bool is_good_with_min_obs(int minObs) const;
  std::pair<bool, bool> is_bad_and_is_good_with_min_obs(int minObs) const;
  bool is_bad_or_is_in_keyframe(KeyFrame *keyframe) const;

  // Statistics - match Python exactly
  void increase_visible(int num_times = 1);
  void increase_found(int num_times = 1);
  float get_found_ratio() const;
};

// MapPoint class - matches Python MapPoint exactly
class MapPoint : public MapPointBase {
private:
  // Global lock for position updates - match Python exactly
  static std::mutex global_lock_;

public:
  // Core geometric data - match Python exactly (private)
  Eigen::Vector3d _pt; // position in world frame
  Eigen::Vector3d normal;
  float _min_distance, _max_distance;

  // Visual data - match Python exactly (private)
  Eigen::Matrix<unsigned char, 3, 1> color;
  cv::Mat semantic_des;
  cv::Mat des; // best descriptor

  // Reference information - match Python exactly
  int first_kid;

  // Update counters - match Python exactly
  int num_observations_on_last_update_des;
  int num_observations_on_last_update_normals;
  int num_observations_on_last_update_semantics;

  // GBA support - match Python exactly
  Eigen::Vector3d pt_GBA;
  int GBA_kf_id;

public:
  // Constructor - matches Python: MapPoint(position, color, keyframe=None,
  // idxf=None, id=None)
  MapPoint(const Eigen::Vector3d &position,
           const Eigen::Matrix<unsigned char, 3, 1> &color,
           KeyFrame *keyframe = nullptr, int idxf = -1, int id = -1);

  // Destructor
  ~MapPoint() = default;

  // Position access - match Python exactly
  Eigen::Vector3d pt() const;
  Eigen::Vector4d homogeneous() const;
  void update_position(const Eigen::Vector3d &position);

  // Distance properties - match Python exactly
  float min_distance() const;
  float max_distance() const;
  std::tuple<Eigen::Vector3d, Eigen::Vector3d, float, float>
  get_all_pos_info() const;

  // Reference keyframe - match Python exactly
  KeyFrame *get_reference_keyframe() const;

  // Descriptor operations - match Python exactly
  std::vector<cv::Mat> descriptors() const;
  float min_des_distance(const cv::Mat &descriptor) const;

  // Point lifecycle - match Python exactly
  void delete_point();
  void set_bad();
  MapPoint *get_replacement() const;
  Eigen::Vector3d get_normal() const;
  void replace_with(MapPoint *p);

  // Update operations - match Python exactly
  void update_normal_and_depth(Frame *frame = nullptr, int idxf = -1,
                               bool force = false);
  void update_best_descriptor(bool force = false);
  void update_semantics(void *semantic_fusion_method = nullptr,
                        bool force = false);
  void update_info();

  // Detection level prediction - match Python exactly
  int predict_detection_level(float dist) const;

  // JSON serialization - match Python exactly
  std::string to_json() const;
  static MapPoint from_json(const std::string &json_str);
  void replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                const std::vector<Frame *> &frames,
                                const std::vector<KeyFrame *> &keyframes);

  // GBA operations - match Python exactly
  void set_pt_GBA(const Eigen::Vector3d &pt_GBA);
  void set_GBA_kf_id(int GBA_kf_id);

private:
  // Helper methods
  void update_normal_and_depth_from_keyframe(KeyFrame *kf, int idx);
  void compute_distance_range(int octave_level, float distance);
  void normalize_vector(const Eigen::Vector3d &v,
                        Eigen::Vector3d &result) const;
  float descriptor_distance(const cv::Mat &d1, const cv::Mat &d2) const;
};

// Hash function for MapPoint to use in unordered containers
struct MapPointHash {
  size_t operator()(const MapPoint *mp) const { return mp->hash(); }
};

// Equality function for MapPoint pointers
struct MapPointEqual {
  bool operator()(const MapPoint *lhs, const MapPoint *rhs) const {
    return lhs == rhs || (lhs && rhs && *lhs == *rhs);
  }
};

} // namespace pyslam