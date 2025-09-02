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

#include "map_point.h"
#include "frame.h"
#include "keyframe.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace pyslam {

// Static member definitions
std::atomic<int> MapPointBase::_id{0};
std::mutex MapPointBase::_id_lock;
std::mutex MapPoint::global_lock_;

// MapPointBase constructor - matches Python: __init__(self, id=None)
MapPointBase::MapPointBase(int id)
    : id(id), map(nullptr), _is_bad(false), _num_observations(0),
      num_times_visible(1), num_times_found(1), last_frame_id_seen(-1),
      replacement(nullptr), corrected_by_kf(0), corrected_reference(0),
      kf_ref(nullptr) {
  if (id < 0) {
    std::lock_guard<std::mutex> lock(_id_lock);
    this->id = _id++;
  }
}

// MapPoint constructor - matches Python: MapPoint(position, color,
// keyframe=None, idxf=None, id=None)
MapPoint::MapPoint(const Eigen::Vector3d &position,
                   const Eigen::Matrix<unsigned char, 3, 1> &color,
                   KeyFrame *keyframe, int idxf, int id)
    : MapPointBase(id), _pt(position), color(color), semantic_des(), des(),
      first_kid(-1), num_observations_on_last_update_des(1),
      num_observations_on_last_update_normals(1),
      num_observations_on_last_update_semantics(1),
      pt_GBA(Eigen::Vector3d::Zero()), GBA_kf_id(0), _min_distance(0.0f),
      _max_distance(std::numeric_limits<float>::infinity()),
      normal(Eigen::Vector3d(0, 0, 1)) // Default normal
{
  kf_ref = keyframe;

  if (keyframe != nullptr) {
    first_kid = keyframe->kid;
    update_normal_and_depth_from_keyframe(keyframe, idxf);
  }
}

// MapPointBase methods - match Python exactly
std::vector<std::pair<KeyFrame *, int>> MapPointBase::observations() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return std::vector<std::pair<KeyFrame *, int>>(_observations.begin(),
                                                 _observations.end());
}

std::vector<std::pair<KeyFrame *, int>>
MapPointBase::observations_iter() const {
  return std::vector<std::pair<KeyFrame *, int>>(_observations.begin(),
                                                 _observations.end());
}

std::vector<KeyFrame *> MapPointBase::keyframes() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  std::vector<KeyFrame *> keyframes;
  for (const auto &obs : _observations) {
    keyframes.push_back(obs.first);
  }
  return keyframes;
}

std::vector<KeyFrame *> MapPointBase::keyframes_iter() const {
  std::vector<KeyFrame *> keyframes;
  for (const auto &obs : _observations) {
    keyframes.push_back(obs.first);
  }
  return keyframes;
}

bool MapPointBase::is_in_keyframe(KeyFrame *keyframe) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return _observations.find(keyframe) != _observations.end();
}

int MapPointBase::get_observation_idx(KeyFrame *keyframe) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  auto it = _observations.find(keyframe);
  if (it != _observations.end()) {
    return it->second;
  }
  return -1;
}

bool MapPointBase::add_observation_no_lock_(KeyFrame *keyframe, int idx) {
  if (_observations.find(keyframe) == _observations.end()) {
    keyframe->set_point_match(static_cast<MapPoint *>(this), idx);
    _observations[keyframe] = idx;

    // Update observation count (simplified - would need stereo check)
    _num_observations++;
    return true;
  }
  return false;
}

bool MapPointBase::add_observation(KeyFrame *keyframe, int idx) {
  std::lock_guard<std::mutex> lock(_lock_features);
  return add_observation_no_lock_(keyframe, idx);
}

std::pair<bool, bool>
MapPointBase::add_observation_if_not_bad(KeyFrame *keyframe, int idx) {
  std::lock_guard<std::mutex> lock(_lock_features);
  if (_is_bad) {
    return std::make_pair(false, _is_bad);
  } else {
    return std::make_pair(add_observation_no_lock_(keyframe, idx), _is_bad);
  }
}

void MapPointBase::remove_observation(KeyFrame *keyframe, int idx) {
  std::lock_guard<std::mutex> lock(_lock_features);

  auto it = _observations.find(keyframe);
  if (it != _observations.end()) {
    // Remove point association from keyframe
    if (idx >= 0) {
      keyframe->remove_point_match(idx);
    } else {
      keyframe->remove_point(static_cast<MapPoint *>(this));
    }

    // Remove from observations
    _observations.erase(it);
    _num_observations = std::max(0, _num_observations - 1);

    // Check if point becomes bad
    _is_bad = (_num_observations <= 2);

    // Update reference keyframe if needed
    if (kf_ref == keyframe && !_observations.empty()) {
      kf_ref = _observations.begin()->first;
    }
  }
}

std::vector<std::pair<Frame *, int>> MapPointBase::frame_views() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return std::vector<std::pair<Frame *, int>>(_frame_views.begin(),
                                              _frame_views.end());
}

std::vector<std::pair<Frame *, int>> MapPointBase::frame_views_iter() const {
  return std::vector<std::pair<Frame *, int>>(_frame_views.begin(),
                                              _frame_views.end());
}

std::vector<Frame *> MapPointBase::frames() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  std::vector<Frame *> frames;
  for (const auto &view : _frame_views) {
    frames.push_back(view.first);
  }
  return frames;
}

std::vector<Frame *> MapPointBase::frames_iter() const {
  std::vector<Frame *> frames;
  for (const auto &view : _frame_views) {
    frames.push_back(view.first);
  }
  return frames;
}

bool MapPointBase::is_in_frame(Frame *frame) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return _frame_views.find(frame) != _frame_views.end();
}

bool MapPointBase::add_frame_view(Frame *frame, int idx) {
  std::lock_guard<std::mutex> lock(_lock_features);
  if (_frame_views.find(frame) == _frame_views.end()) {
    frame->set_point_match(static_cast<MapPoint *>(this), idx);
    _frame_views[frame] = idx;
    return true;
  }
  return false;
}

void MapPointBase::remove_frame_view(Frame *frame, int idx) {
  std::lock_guard<std::mutex> lock(_lock_features);

  auto it = _frame_views.find(frame);
  if (it != _frame_views.end()) {
    // Remove point association from frame
    if (idx >= 0) {
      frame->remove_point_match(idx);
    } else {
      frame->remove_point(static_cast<MapPoint *>(this));
    }

    // Remove from frame views
    _frame_views.erase(it);
  }
}

bool MapPointBase::is_bad() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return _is_bad;
}

int MapPointBase::num_observations() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return _num_observations;
}

bool MapPointBase::is_good_with_min_obs(int minObs) const {
  return !_is_bad && _num_observations >= minObs;
}

std::pair<bool, bool>
MapPointBase::is_bad_and_is_good_with_min_obs(int minObs) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  bool good = !_is_bad && _num_observations >= minObs;
  return std::make_pair(_is_bad, good);
}

bool MapPointBase::is_bad_or_is_in_keyframe(KeyFrame *keyframe) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return _is_bad || (_observations.find(keyframe) != _observations.end());
}

void MapPointBase::increase_visible(int num_times) {
  std::lock_guard<std::mutex> lock(_lock_features);
  num_times_visible += num_times;
}

void MapPointBase::increase_found(int num_times) {
  std::lock_guard<std::mutex> lock(_lock_features);
  num_times_found += num_times;
}

float MapPointBase::get_found_ratio() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return static_cast<float>(num_times_found) /
         static_cast<float>(num_times_visible);
}

// MapPoint methods - match Python exactly
Eigen::Vector3d MapPoint::pt() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return _pt;
}

Eigen::Vector4d MapPoint::homogeneous() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return Eigen::Vector4d(_pt.x(), _pt.y(), _pt.z(), 1.0);
}

void MapPoint::update_position(const Eigen::Vector3d &position) {
  std::lock_guard<std::mutex> lock(global_lock_);
  std::lock_guard<std::mutex> lock_pos(_lock_pos);
  _pt = position;
}

float MapPoint::min_distance() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return _min_distance;
}

float MapPoint::max_distance() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return _max_distance;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, float, float>
MapPoint::get_all_pos_info() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return std::make_tuple(_pt, normal, _min_distance, _max_distance);
}

KeyFrame *MapPoint::get_reference_keyframe() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return kf_ref;
}

std::vector<cv::Mat> MapPoint::descriptors() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  std::vector<cv::Mat> descs;
  for (const auto &obs : _observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;
    if (kf != nullptr && idx >= 0) {
      cv::Mat kf_descriptor = kf->des.row(idx);
      if (!kf_descriptor.empty()) {
        descs.push_back(kf_descriptor);
      }
    }
  }
  return descs;
}

float MapPoint::min_des_distance(const cv::Mat &descriptor) const {
  std::lock_guard<std::mutex> lock(_lock_features);
  return descriptor_distance(des, descriptor);
}

void MapPoint::delete_point() {
  std::vector<std::pair<KeyFrame *, int>> observations;

  {
    std::lock_guard<std::mutex> lock_features(_lock_features);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);

    _is_bad = true;
    _num_observations = 0;
    observations = std::vector<std::pair<KeyFrame *, int>>(
        _observations.begin(), _observations.end());
    _observations.clear();
  }

  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;
    kf->remove_point_match(idx);
  }
}

void MapPoint::set_bad() {
  std::vector<std::pair<KeyFrame *, int>> observations;

  {
    std::lock_guard<std::mutex> lock_features(_lock_features);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);

    _is_bad = true;
    _num_observations = 0;
    observations = std::vector<std::pair<KeyFrame *, int>>(
        _observations.begin(), _observations.end());
    _observations.clear();
  }

  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;
    kf->remove_point_match(idx);
  }

  if (map != nullptr) {
    // Cast void* back to Map* and call remove_point
    // This requires proper Map class definition
  }
}

MapPoint *MapPoint::get_replacement() const {
  std::lock_guard<std::mutex> lock_features(_lock_features);
  std::lock_guard<std::mutex> lock_pos(_lock_pos);
  return replacement;
}

Eigen::Vector3d MapPoint::get_normal() const {
  std::lock_guard<std::mutex> lock(_lock_pos);
  return normal;
}

void MapPoint::replace_with(MapPoint *p) {
  if (p == nullptr || p->id == id) {
    return;
  }

  std::vector<std::pair<KeyFrame *, int>> observations;
  int num_times_visible, num_times_found;

  {
    std::lock_guard<std::mutex> lock_features(_lock_features);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);

    observations = std::vector<std::pair<KeyFrame *, int>>(
        _observations.begin(), _observations.end());
    _observations.clear();
    num_times_visible = this->num_times_visible;
    num_times_found = this->num_times_found;
    _is_bad = true;
    _num_observations = 0;
    replacement = p;
  }

  // Replace point observations in keyframes
  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;

    if (p->add_observation(kf, idx)) {
      // Point p was NOT in kf => added new observation in p
      kf->replace_point_match(p, idx);
    } else {
      // Point p is already in kf => just remove this point match from kf
      kf->remove_point_match(idx);
    }
  }

  p->increase_visible(num_times_visible);
  p->increase_found(num_times_found);
  p->update_best_descriptor(true);
}

void MapPoint::update_normal_and_depth(Frame *frame, int idxf, bool force) {
  bool skip = false;
  std::vector<std::pair<KeyFrame *, int>> observations;
  KeyFrame *kf_ref_local;
  int idx_ref;
  Eigen::Vector3d position;

  {
    std::lock_guard<std::mutex> lock_features(_lock_features);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);

    if (_is_bad) {
      return;
    }
    if (_num_observations > num_observations_on_last_update_normals || force) {
      num_observations_on_last_update_normals = _num_observations;
      observations = std::vector<std::pair<KeyFrame *, int>>(
          _observations.begin(), _observations.end());
      kf_ref_local = kf_ref;
      idx_ref = _observations[kf_ref];
      position = _pt;
    } else {
      skip = true;
    }
  }

  if (skip || observations.empty()) {
    return;
  }

  // Compute normal from observations
  std::vector<Eigen::Vector3d> normals;
  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    Eigen::Vector3d Ow = kf->Ow();
    Eigen::Vector3d direction = position - Ow;
    Eigen::Vector3d normalized_direction;
    normalize_vector(direction, normalized_direction);
    normals.push_back(normalized_direction);
  }

  // Compute mean normal
  Eigen::Vector3d mean_normal = Eigen::Vector3d::Zero();
  for (const auto &n : normals) {
    mean_normal += n;
  }
  mean_normal /= normals.size();
  normalize_vector(mean_normal, mean_normal);

  // Compute distance range
  int level = kf_ref_local->octaves[idx_ref];
  float scale_factor = std::pow(2.0f, level);
  float dist = (position - kf_ref_local->Ow()).norm();

  {
    std::lock_guard<std::mutex> lock_pos(_lock_pos);
    _max_distance = dist * scale_factor;
    _min_distance = _max_distance / std::pow(2.0f, 7); // Assuming 8 levels
    normal = mean_normal;
  }
}

void MapPoint::update_best_descriptor(bool force) {
  bool skip = false;
  std::vector<std::pair<KeyFrame *, int>> observations;

  {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (_is_bad) {
      return;
    }
    if (_num_observations > num_observations_on_last_update_des || force) {
      num_observations_on_last_update_des = _num_observations;
      observations = std::vector<std::pair<KeyFrame *, int>>(
          _observations.begin(), _observations.end());
    } else {
      skip = true;
    }
  }

  if (skip || observations.empty()) {
    return;
  }

  // Collect descriptors from observations
  std::vector<cv::Mat> descriptors;
  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;
    if (kf != nullptr && !kf->is_bad()) {
      cv::Mat kf_descriptor = kf->des.row(idx);
      if (!kf_descriptor.empty()) {
        descriptors.push_back(kf_descriptor);
      }
    }
  }

  if (descriptors.size() >= 2) {
    // Find best descriptor (simplified implementation)
    cv::Mat best_descriptor = descriptors[0];
    float min_median_distance = std::numeric_limits<float>::max();

    for (size_t i = 0; i < descriptors.size(); ++i) {
      std::vector<float> distances;
      for (size_t j = 0; j < descriptors.size(); ++j) {
        if (i != j) {
          distances.push_back(
              descriptor_distance(descriptors[i], descriptors[j]));
        }
      }

      if (!distances.empty()) {
        std::sort(distances.begin(), distances.end());
        float median_distance = distances[distances.size() / 2];
        if (median_distance < min_median_distance) {
          min_median_distance = median_distance;
          best_descriptor = descriptors[i];
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(_lock_features);
      des = best_descriptor.clone();
    }
  }
}

void MapPoint::update_semantics(void *semantic_fusion_method, bool force) {
  bool skip = false;
  std::vector<std::pair<KeyFrame *, int>> observations;

  {
    std::lock_guard<std::mutex> lock(_lock_features);
    if (_is_bad) {
      return;
    }
    if (_num_observations > num_observations_on_last_update_semantics ||
        force) {
      num_observations_on_last_update_semantics = _num_observations;
      observations = std::vector<std::pair<KeyFrame *, int>>(
          _observations.begin(), _observations.end());
    } else {
      skip = true;
    }
  }

  if (skip || observations.empty()) {
    return;
  }

  // Collect semantic descriptors from observations
  cv::Mat semantics;
  for (const auto &obs : observations) {
    KeyFrame *kf = obs.first;
    int idx = obs.second;
    if (kf != nullptr && !kf->kps_sem.empty()) {
      const auto &kf_semantic = kf->kps_sem.row(idx);
      semantics.push_back(kf_semantic);
    }
  }

  if (semantics.rows >= 2) {

    // TODO: fuse semantic descriptors
#if 0
         // Fuse semantic descriptors (simplified implementation)
         const auto& fused_semantic = semantics.row(0);
         
         {
             std::lock_guard<std::mutex> lock(_lock_features);
             semantic_des = fused_semantic;
         }
#endif
  }
}

void MapPoint::update_info() {
  update_normal_and_depth();
  update_best_descriptor();
}

int MapPoint::predict_detection_level(float dist) const {
  std::lock_guard<std::mutex> lock(_lock_pos);

  std::cout << "predict_detection_level: " << dist << std::endl;

  // TODO: use FeatureTrackerShared.feature_manager.log_scale_factor

  if (_max_distance <= 0.0f || dist <= 0.0f) {
    return 0;
  }

  float ratio = _max_distance / dist;
  int level = static_cast<int>(std::ceil(
      std::log(ratio) / std::log(2.0f))); // Assuming scale factor of 2

  // Clamp to valid range (assuming 8 levels)
  return std::max(0, std::min(7, level));
}

std::string MapPoint::to_json() const {
  std::ostringstream oss;
  oss << "{";
  oss << "\"id\": " << id << ", ";
  oss << "\"pt\": [" << _pt.x() << ", " << _pt.y() << ", " << _pt.z() << "], ";
  oss << "\"color\": [" << static_cast<int>(color[0]) << ", "
      << static_cast<int>(color[1]) << ", " << static_cast<int>(color[2])
      << "], ";
  oss << "\"num_observations\": " << _num_observations << ", ";
  oss << "\"is_bad\": " << (_is_bad ? "true" : "false");
  oss << "}";
  return oss.str();
}

MapPoint MapPoint::from_json(const std::string &json_str) {
  // This is a simplified implementation
  // The actual implementation would parse JSON properly

  // For now, return a default MapPoint
  Eigen::Vector3d position(0, 0, 0);
  Eigen::Matrix<unsigned char, 3, 1> color;
  color << 255, 255, 255;
  return MapPoint(position, color);
}

void MapPoint::replace_ids_with_objects(
    const std::vector<MapPoint *> &points, const std::vector<Frame *> &frames,
    const std::vector<KeyFrame *> &keyframes) {
  // This is a simplified implementation
  // The actual implementation would replace ID references with object pointers
}

void MapPoint::set_pt_GBA(const Eigen::Vector3d &pt_GBA) {
  std::lock_guard<std::mutex> lock(_lock_pos);
  this->pt_GBA = pt_GBA;
}

void MapPoint::set_GBA_kf_id(int GBA_kf_id) {
  std::lock_guard<std::mutex> lock(_lock_features);
  this->GBA_kf_id = GBA_kf_id;
}

// Helper methods
void MapPoint::update_normal_and_depth_from_keyframe(KeyFrame *kf, int idx) {
  if (kf == nullptr) {
    return;
  }

  // Compute normal from keyframe
  Eigen::Vector3d Ow = kf->Ow();
  Eigen::Vector3d direction = _pt - Ow;
  double norm = direction.norm();

  if (norm > 1e-6) {
    normal = direction / norm;

    // Compute distance range based on octave level
    if (idx >= 0) {
      int octave_level = kf->octaves[idx];
      compute_distance_range(octave_level, static_cast<float>(norm));
    }
  }
}

void MapPoint::compute_distance_range(int octave_level, float distance) {
  // Simplified distance range computation
  // The actual implementation would use proper scale factors

  float scale_factor = std::pow(2.0f, octave_level);
  _max_distance = distance * scale_factor;
  _min_distance = _max_distance / std::pow(2.0f, 7); // Assuming 8 levels
}

void MapPoint::normalize_vector(const Eigen::Vector3d &v,
                                Eigen::Vector3d &result) const {
  double norm = v.norm();
  if (norm > 1e-6) {
    result = v / norm;
  } else {
    result = Eigen::Vector3d(0, 0, 1);
  }
}

float MapPoint::descriptor_distance(const cv::Mat &d1,
                                    const cv::Mat &d2) const {
  // Simplified descriptor distance computation
  // The actual implementation would use proper distance metrics

  if (d1.empty() || d2.empty() || d1.rows != d2.rows || d1.cols != d2.cols) {
    return std::numeric_limits<float>::max();
  }

  cv::Mat diff = d1 - d2;
  return static_cast<float>(cv::norm(diff));
}

// String representation methods
std::string MapPointBase::observations_string() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  std::ostringstream oss;
  oss << "observations: [";
  bool first = true;
  for (const auto &obs : _observations) {
    if (!first)
      oss << ", ";
    oss << "(" << obs.first->kid << ", " << obs.second << ")";
    first = false;
  }
  oss << "]";
  return oss.str();
}

std::string MapPointBase::frame_views_string() const {
  std::lock_guard<std::mutex> lock(_lock_features);
  std::ostringstream oss;
  oss << "views: [";
  bool first = true;
  for (const auto &view : _frame_views) {
    if (!first)
      oss << ", ";
    oss << "(" << view.first->id << ", " << view.second << ")";
    first = false;
  }
  oss << "]";
  return oss.str();
}

std::string MapPointBase::to_string() const {
  std::ostringstream oss;
  oss << "MapPoint " << id << " { ";
  oss << observations_string() << ", ";
  oss << frame_views_string();
  oss << " }";
  return oss.str();
}

} // namespace pyslam