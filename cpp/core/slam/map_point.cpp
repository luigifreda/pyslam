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
#include "map.h"
#include <algorithm>
#include <cmath>
#include <sstream>

#include "feature_shared_resources.h"
#include "semantic_fusion_methods.h"
#include "semantic_mapping_shared_resources.h"
#include "utils/messages.h"

#include "config_parameters.h"

namespace pyslam {

// Static member definitions
std::atomic<int> MapPointBase::_id{0};
std::mutex MapPointBase::_id_lock;
std::mutex MapPoint::global_lock;

// =====================================================
// MapPointBase methods
// =====================================================

// MapPointBase constructor - matches Python: __init__(self, id=None)
MapPointBase::MapPointBase(int id)
    : id(id), _is_bad(false), _num_observations(0), num_times_visible(1), num_times_found(1),
      last_frame_id_seen(-1), replacement(nullptr), corrected_by_kf(0), corrected_reference(0),
      kf_ref(nullptr) {
    if (id < 0) {
        std::lock_guard<std::mutex> lock(_id_lock);
        this->id = _id++;
    }
}

int MapPointBase::next_id() {
    std::lock_guard<std::mutex> lock(_id_lock);
    return _id;
}

void MapPointBase::set_id(int id) {
    std::lock_guard<std::mutex> lock(_id_lock);
    _id = id;
}

std::vector<std::pair<KeyFramePtr, int>> MapPointBase::observations() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(), _observations.end());
}

std::vector<std::pair<KeyFramePtr, int>> MapPointBase::observations_iter() const {
    return std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(), _observations.end());
}

std::vector<KeyFramePtr> MapPointBase::keyframes() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<KeyFramePtr> keyframes;
    for (const auto &obs : _observations) {
        keyframes.push_back(obs.first);
    }
    return keyframes;
}

std::vector<KeyFramePtr> MapPointBase::keyframes_iter() const {
    std::vector<KeyFramePtr> keyframes;
    for (const auto &obs : _observations) {
        keyframes.push_back(obs.first);
    }
    return keyframes;
}

bool MapPointBase::is_in_keyframe(const KeyFramePtr &keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return _observations.find(keyframe) != _observations.end();
}

int MapPointBase::get_observation_idx(const KeyFramePtr &keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    const auto it = _observations.find(keyframe);
    if (it != _observations.end()) {
        return it->second;
    }
    return -1;
}

int MapPointBase::get_frame_view_idx(const FramePtr &frame) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    const auto it = _frame_views.find(frame);
    if (it != _frame_views.end()) {
        return it->second;
    }
    return -1;
}

bool MapPointBase::add_observation_no_lock_(KeyFramePtr &keyframe, int idx) {
    if (idx < 0) {
        MSG_ERROR("MapPointBase: add_observation_no_lock_ - idx is negative");
        return false;
    }
    bool success = false;
    if (_observations.find(keyframe) == _observations.end()) {
        // if the point is not in the keyframe observations, add it
        _observations[keyframe] = idx;
        if (keyframe->kps_ur.size() > 0 && keyframe->kps_ur[idx] >= 0) {
            _num_observations += 2;
        } else {
            _num_observations += 1;
        }
        success = true;
    }
    if (success) {
        // add point association in keyframe
        auto this_as_mp = dynamic_cast<MapPoint *>(this);
        if (this_as_mp == nullptr) {
            throw std::runtime_error("MapPointBase is not a MapPoint");
        }
        keyframe->set_point_match(this_as_mp->shared_from_this(), idx);
    }
    return success;
}

bool MapPointBase::add_observation(KeyFramePtr keyframe, int idx) {
    std::lock_guard<std::mutex> lock(_lock_features);
    return add_observation_no_lock_(keyframe, idx);
}

void MapPointBase::remove_observation(KeyFramePtr keyframe, int idx, bool map_no_lock) {
    if (idx < 0) {
        MSG_ERROR("MapPointBase: remove_observation - idx is negative");
        return;
    }
    bool kf_remove_point_match = false;
    bool kf_remove_point = false;
    bool set_bad = false;

    {
        std::lock_guard<std::mutex> lock(_lock_features);
        auto it = _observations.find(keyframe);
        if (it != _observations.end()) {
            // Remove point association from keyframe
            if (idx >= 0) {
                kf_remove_point_match = true;
            } else {
                kf_remove_point = true;
            }

            // Remove from observations
            _observations.erase(it);

            const bool is_stereo_observation = keyframe->is_stereo_observation(idx);
            if (is_stereo_observation) {
                _num_observations = std::max(0, _num_observations - 2);
            } else {
                _num_observations = std::max(0, _num_observations - 1);
            }

            // Check if point becomes bad
            set_bad = (_num_observations <= 2);

            // Update reference keyframe if needed
            if (kf_ref == keyframe && !_observations.empty()) {
                kf_ref = _observations.begin()->first;
            }
        }
    }

    if (kf_remove_point_match) {
        keyframe->remove_point_match(idx);
    }
    auto this_as_mp = dynamic_cast<MapPoint *>(this);
    if (this_as_mp == nullptr) {
        throw std::runtime_error("MapPointBase is not a MapPoint");
    }
    if (kf_remove_point) {
        keyframe->remove_point(this_as_mp->shared_from_this());
    }
    if (set_bad) {
        this_as_mp->set_bad(map_no_lock);
    }
}

std::vector<std::pair<FramePtr, int>> MapPointBase::frame_views() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return std::vector<std::pair<FramePtr, int>>(_frame_views.begin(), _frame_views.end());
}

std::vector<std::pair<FramePtr, int>> MapPointBase::frame_views_iter() const {
    return std::vector<std::pair<FramePtr, int>>(_frame_views.begin(), _frame_views.end());
}

std::vector<FramePtr> MapPointBase::frames() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<FramePtr> frames;
    for (const auto &view : _frame_views) {
        frames.push_back(view.first);
    }
    return frames;
}

std::vector<FramePtr> MapPointBase::frames_iter() const {
    std::vector<FramePtr> frames;
    for (const auto &view : _frame_views) {
        frames.push_back(view.first);
    }
    return frames;
}

bool MapPointBase::is_in_frame(const FramePtr &frame) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return _frame_views.find(frame) != _frame_views.end();
}

bool MapPointBase::add_frame_view(FramePtr &frame, int idx) {
    bool success = false;
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        if (_frame_views.find(frame) == _frame_views.end()) {
            _frame_views[frame] = idx;
            success = true;
        }
    }
    if (success) {
        auto this_as_mp = dynamic_cast<MapPoint *>(this);
        if (this_as_mp == nullptr) {
            throw std::runtime_error("MapPointBase is not a MapPoint");
        }
        frame->set_point_match(this_as_mp->shared_from_this(), idx);
    }
    return success;
}

void MapPointBase::remove_frame_view(FramePtr frame, int idx) {
    bool frame_remove_point_match = false;
    bool frame_remove_point = false;
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        auto it = _frame_views.find(frame);
        if (it != _frame_views.end()) {
            // Remove point association from frame
            if (idx >= 0) {
                frame_remove_point_match = true;
            } else {
                frame_remove_point = true;
            }

            // Remove from frame views
            _frame_views.erase(it);
        }
    }
    if (frame_remove_point_match) {
        frame->remove_point_match(idx);
    }
    auto this_as_mp = dynamic_cast<MapPoint *>(this);
    if (this_as_mp == nullptr) {
        throw std::runtime_error("MapPointBase is not a MapPoint");
    }
    if (frame_remove_point) {
        frame->remove_point(this_as_mp->shared_from_this());
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
    // no lock here
    return !_is_bad && _num_observations >= minObs;
}

std::pair<bool, bool> MapPointBase::is_bad_and_is_good_with_min_obs(int minObs) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    const bool is_bad = _is_bad.load();
    const bool good = !is_bad && _num_observations >= minObs;
    return std::make_pair(is_bad, good);
}

bool MapPointBase::is_bad_or_is_in_keyframe(const KeyFramePtr &keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return _is_bad ? true : (_observations.find(keyframe) != _observations.end());
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
    return static_cast<float>(num_times_found) / static_cast<float>(num_times_visible);
}

// =====================================================
// MapPoint methods
// =====================================================

MapPoint::MapPoint(const Eigen::Vector3d &position, const Eigen::Matrix<unsigned char, 3, 1> &color)
    : MapPointBase(-1), _pt(position), color(color) {}

MapPoint::MapPoint(const Eigen::Vector3d &position, const Eigen::Matrix<unsigned char, 3, 1> &color,
                   const FramePtr &keyframe, const int idxf, const int id)
    : MapPointBase(id), _pt(position), color(color) {
    if (keyframe) {
        if (keyframe->is_keyframe) {
            std::cout << "MapPoint: MapPoint constructor - casting input frame to KeyFrame"
                      << std::endl;
            KeyFramePtr cast_keyframe = std::dynamic_pointer_cast<KeyFrame>(keyframe);
            if (cast_keyframe) {
                kf_ref = cast_keyframe;
                first_kid = cast_keyframe->kid;
            } else {
                MSG_ERROR("MapPoint: MapPoint constructor - keyframe is not a KeyFrame");
            }
        }
        update_normal_and_depth_from_frame(keyframe, idxf);
        if (idxf >= 0) {
            des = keyframe->des.row(idxf);
            if (!keyframe->kps_sem.empty()) {
                semantic_des = keyframe->kps_sem.row(idxf);
                // Convert semantic descriptor to RGB using the color map (feature-type aware)
                const cv::Vec3b sem_rgb =
                    SemanticMappingSharedResources::semantic_color_map->semantic_to_color(
                        semantic_des, SemanticMappingSharedResources::semantic_feature_type, false);
                semantic_color << static_cast<unsigned char>(sem_rgb[0]),
                    static_cast<unsigned char>(sem_rgb[1]), static_cast<unsigned char>(sem_rgb[2]);
            }
        }
    }
}

MapPoint::MapPoint(const Eigen::Vector3d &position, const Eigen::Matrix<unsigned char, 3, 1> &color,
                   const KeyFramePtr &keyframe, const int idxf, const int id)
    : MapPointBase(id), _pt(position), color(color) {
    kf_ref = keyframe;
    if (keyframe) {
        if (keyframe->is_keyframe) {
            first_kid = keyframe->kid;
        }
        update_normal_and_depth_from_frame(keyframe, idxf);
        if (idxf >= 0) {
            des = keyframe->des.row(idxf);
        }
    }
}

Eigen::Vector3d MapPoint::pt() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return _pt;
}

Eigen::Vector4d MapPoint::homogeneous() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return Eigen::Vector4d(_pt.x(), _pt.y(), _pt.z(), 1.0);
}

void MapPoint::update_position(const Eigen::Vector3d &position) {
    std::lock_guard<std::mutex> lock(global_lock);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);
    _pt = position;
}

float MapPoint::min_distance() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return Parameters::kMinDistanceToleranceFactor * _min_distance;
}

float MapPoint::max_distance() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return Parameters::kMaxDistanceToleranceFactor * _max_distance;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, float, float> MapPoint::get_all_pos_info() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return std::make_tuple(_pt, normal, Parameters::kMinDistanceToleranceFactor * _min_distance,
                           Parameters::kMaxDistanceToleranceFactor * _max_distance);
}

KeyFramePtr MapPoint::get_reference_keyframe() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    return kf_ref;
}

std::vector<cv::Mat> MapPoint::descriptors() const {
    std::lock_guard<std::mutex> lock(_lock_features);
    std::vector<cv::Mat> descs;
    for (const auto &obs : _observations) {
        const auto &kf = obs.first;
        int idx = obs.second;
        if (kf && idx >= 0) {
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
    return pyslam::descriptor_distance(des, descriptor, FeatureSharedResources::norm_type);
}

void MapPoint::delete_point() {
    if (!_is_bad) {
        set_bad();
    }
}

void MapPoint::set_bad(bool map_no_lock) {
    if (_is_bad) {
        return;
    }
    std::vector<std::pair<KeyFramePtr, int>> observations;

    {
        std::lock_guard<std::mutex> lock_features(_lock_features);
        std::lock_guard<std::mutex> lock_pos(_lock_pos);

        _is_bad = true;
        _num_observations = 0;
        observations =
            std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(), _observations.end());
        _observations.clear();
    }

    for (const auto &obs : observations) {
        KeyFramePtr kf = obs.first;
        int idx = obs.second;
        kf->remove_point_match(idx);
    }

    if (map) {
        if (map_no_lock) {
            map->remove_point_no_lock(this->shared_from_this());
        } else {
            map->remove_point(this->shared_from_this());
        }
    }
}

MapPointPtr MapPoint::get_replacement() const {
    std::lock_guard<std::mutex> lock_features(_lock_features);
    std::lock_guard<std::mutex> lock_pos(_lock_pos);
    return replacement;
}

Eigen::Vector3d MapPoint::get_normal() const {
    std::lock_guard<std::mutex> lock(_lock_pos);
    return normal;
}

void MapPoint::replace_with(MapPointPtr &p) {
    if (!p || p->id == id) {
        return;
    }

    std::vector<std::pair<KeyFramePtr, int>> observations;
    int num_times_visible = 0, num_times_found = 0;

    {
        std::lock_guard<std::mutex> lock_features(_lock_features);
        std::lock_guard<std::mutex> lock_pos(_lock_pos);

        observations =
            std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(), _observations.end());
        _observations.clear();
        num_times_visible = this->num_times_visible;
        num_times_found = this->num_times_found;
        _is_bad = true;
        _num_observations = 0;
        replacement = p;
    }

    // Replace point observations in keyframes
    for (const auto &obs : observations) {
        KeyFramePtr kf = obs.first;
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

    if (map) {
        map->remove_point(this->shared_from_this());
    } else {
        MSG_WARN("MapPoint: replace_with() - map is nullptr");
    }
}

void MapPoint::update_normal_and_depth(const bool force) {
    bool skip = false;
    std::vector<std::pair<KeyFramePtr, int>> observations;
    KeyFramePtr kf_ref_local;
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
            observations = std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(),
                                                                    _observations.end());
            kf_ref_local = kf_ref;
            if (!kf_ref_local) {
                MSG_ERROR("MapPoint: update_normal_and_depth - kf_ref_local is nullptr");
                return;
            }
            auto it = _observations.find(kf_ref);
            if (it == _observations.end()) {
                MSG_ERROR("MapPoint: update_normal_and_depth - kf_ref not found in observations");
                return;
            }
            idx_ref = it->second;
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
        const auto kf = obs.first;
        const Eigen::Vector3d Ow = kf->Ow();
        const Eigen::Vector3d direction = position - Ow;
        Eigen::Vector3d normalized_direction;
        normalize_vector(direction, normalized_direction);
        normals.push_back(normalized_direction);
    }

    // Compute mean normal (Assuming the normals are close enough to be averaged)
    Eigen::Vector3d mean_normal = Eigen::Vector3d::Zero();
    for (const auto &n : normals) {
        mean_normal += n;
    }
    mean_normal /= normals.size();
    normalize_vector(mean_normal, mean_normal);

    // Compute distance range
    const int level = kf_ref_local->octaves[idx_ref];
    const float scale_factor = FeatureSharedResources::scale_factors[level];
    const float dist = (position - kf_ref_local->Ow()).norm();

    {
        std::lock_guard<std::mutex> lock_pos(_lock_pos);
        _max_distance = dist * scale_factor;
        _min_distance =
            _max_distance /
            FeatureSharedResources::scale_factors[FeatureSharedResources::num_levels - 1];
        normal = mean_normal;
    }
}

void MapPoint::update_best_descriptor(const bool force) {
    bool skip = false;
    std::vector<std::pair<KeyFramePtr, int>> observations;

    {
        std::lock_guard<std::mutex> lock(_lock_features);
        if (_is_bad) {
            return;
        }
        if (_num_observations > num_observations_on_last_update_des || force) {
            num_observations_on_last_update_des = _num_observations;
            observations = std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(),
                                                                    _observations.end());
        } else {
            skip = true;
        }
    }

    if (skip || observations.empty()) {
        return;
    }

    // Collect descriptors from observations
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(observations.size());
    for (const auto &obs : observations) {
        const auto &kf = obs.first;
        const int idx = obs.second;
        if (kf && !kf->is_bad()) {
            descriptors.push_back(kf->des.row(idx));
        }
    }

    if (descriptors.size() >= 2) {

        const int num_descriptors = descriptors.size();
        // compute the descriptor distances (all pairs)
        std::vector<std::vector<float>> distances_matrix =
            compute_distances_matrix(descriptors, FeatureSharedResources::norm_type);

        // find the descriptor with the least median distance
        int best_descriptor_idx = 0;
        float min_median_distance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < num_descriptors; ++i) {
            std::vector<float> ith_row_distances = distances_matrix[i];
            std::sort(ith_row_distances.begin(), ith_row_distances.end());
            float median_distance;
            const size_t size = ith_row_distances.size();
            if (size % 2 == 0 && size > 0) {
                // Even number: average of two middle values
                median_distance =
                    (ith_row_distances[size / 2 - 1] + ith_row_distances[size / 2]) / 2.0f;
            } else {
                // Odd number: middle value
                median_distance = ith_row_distances[size / 2];
            }
            if (median_distance < min_median_distance) {
                min_median_distance = median_distance;
                best_descriptor_idx = i;
            }
        }

        {
            std::lock_guard<std::mutex> lock(_lock_features);
            des = descriptors[best_descriptor_idx].clone();
        }
    } else if (descriptors.size() == 1) {
        // Single descriptor case - just use it
        {
            std::lock_guard<std::mutex> lock(_lock_features);
            des = descriptors[0].clone();
        }
    }
}

// NOTE: the input function pointer is not used in the C++ code,
// but it is used in the Python code to pass the semantic fusion method.
// Here, we directly use the semantic feature type to determine the semantic fusion method.
void MapPoint::update_semantics(void *semantic_fusion_method, const bool force) {
    bool skip = false;
    std::vector<std::pair<KeyFramePtr, int>> observations;

    {
        std::lock_guard<std::mutex> lock(_lock_features);
        if (_is_bad) {
            return;
        }
        if (_num_observations > num_observations_on_last_update_semantics || force) {
            num_observations_on_last_update_semantics = _num_observations;
            observations = std::vector<std::pair<KeyFramePtr, int>>(_observations.begin(),
                                                                    _observations.end());
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
        const auto &kf = obs.first;
        const int idx = obs.second;
        if (kf && !kf->kps_sem.empty() && !kf->is_bad()) {
            const auto &kf_semantic = kf->kps_sem.row(idx);
            semantics.push_back(kf_semantic);
        }
    }

    if (semantics.rows >= 2 || semantic_color == Vec3b::Zero()) {
        // Fuse semantic descriptors
        const auto &fused_semantic =
            semantic_fusion(semantics, FeatureSharedResources::semantic_feature_type);

        {
            std::lock_guard<std::mutex> lock(_lock_features);
            semantic_des = fused_semantic;
            // Convert semantic descriptor to RGB using the color map (feature-type aware)
            const cv::Vec3b sem_rgb =
                SemanticMappingSharedResources::semantic_color_map->semantic_to_color(
                    semantic_des, SemanticMappingSharedResources::semantic_feature_type, false);
            semantic_color << static_cast<unsigned char>(sem_rgb[0]),
                static_cast<unsigned char>(sem_rgb[1]), static_cast<unsigned char>(sem_rgb[2]);
        }
    }
}

void MapPoint::update_info() {
    update_normal_and_depth();
    update_best_descriptor();
}

int MapPoint::predict_detection_level(const float dist) const {
    std::lock_guard<std::mutex> lock(_lock_pos);

    if (_max_distance <= 0.0f || dist <= 0.0f) {
        return 0;
    }

    float ratio = _max_distance / std::max(dist, 1e-8f);
    int level = static_cast<int>(std::ceil(
        std::log(ratio) / FeatureSharedResources::log_scale_factor)); // Assuming scale factor of 2

    // Clamp to valid range (assuming 8 levels)
    return std::max(0, std::min(FeatureSharedResources::num_levels - 1, level));
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
void MapPoint::update_normal_and_depth_from_frame(const FramePtr &kf, int idx) {
    if (!kf) {
        return;
    }

    // Compute normal from keyframe
    Eigen::Vector3d Ow = kf->Ow();
    Eigen::Vector3d direction = _pt - Ow;
    double norm = direction.norm();

    if (norm > kMin3dVectorNorm) {
        normal = direction / norm;
    } else {
        MSG_WARN("MapPoint: update_normal_and_depth_from_frame - norm is too small");
        normal = Eigen::Vector3d(0, 0, 1);
    }
    // Compute distance range based on octave level
    if (idx >= 0) {
        int octave_level = kf->octaves[idx];
        compute_distance_range(octave_level, static_cast<float>(norm));
    }
}

void MapPoint::compute_distance_range(int octave_level, float distance) {

    const float scale_factor = FeatureSharedResources::scale_factors[octave_level];
    _max_distance = distance * scale_factor;
    _min_distance = _max_distance /
                    FeatureSharedResources::scale_factors[FeatureSharedResources::num_levels - 1];
}

void MapPoint::normalize_vector(const Eigen::Vector3d &v, Eigen::Vector3d &result) const {
    double norm = v.norm();
    if (norm > kMin3dVectorNorm) {
        result = v / norm;
    } else {
        MSG_WARN("MapPoint: normalize_vector - norm is too small");
        result = Eigen::Vector3d(0, 0, 1);
    }
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

std::vector<int> MapPoint::predict_detection_levels(const std::vector<MapPointPtr> &points,
                                                    const std::vector<float> &dists) {
    if (points.empty() || dists.empty()) {
        return std::vector<int>();
    }
    MSG_FORCED_ASSERT(points.size() == dists.size(),
                      "MapPoint: predict_detection_levels - points and dists have different sizes");
    std::vector<int> levels;
    levels.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        levels.push_back(points[i]->predict_detection_level(dists[i]));
    }
    return levels;
}

} // namespace pyslam