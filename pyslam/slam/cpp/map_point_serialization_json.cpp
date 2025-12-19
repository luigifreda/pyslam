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

#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

#include "semantic_serialization.h"
#include "utils/serialization_json.h"

#include <nlohmann/json.hpp>

namespace pyslam {

std::string MapPoint::to_json() const {
    nlohmann::json json_obj;

    // Basic fields
    json_obj["id"] = id;

    // Position
    json_obj["pt"] = {_pt.x(), _pt.y(), _pt.z()};

    // Color
    json_obj["color"] = {static_cast<int>(color[0]), static_cast<int>(color[1]),
                         static_cast<int>(color[2])};

    // Semantic color
    json_obj["semantic_color"] = {static_cast<int>(semantic_color[0]),
                                  static_cast<int>(semantic_color[1]),
                                  static_cast<int>(semantic_color[2])};

    // Observations (convert to ID references)
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        nlohmann::json observations_array = nlohmann::json::array();
        for (const auto &obs : _observations) {
            const auto &kf = obs.first;
            int idx = obs.second;
            if (kf) {
                observations_array.push_back({kf->id, idx});
            }
        }
        json_obj["_observations"] = observations_array;
    }

    // Frame views (convert to ID references)
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        nlohmann::json frame_views_array = nlohmann::json::array();
        for (const auto &view : _frame_views) {
            const auto &frame = view.first;
            int idx = view.second;
            if (frame) {
                frame_views_array.push_back({frame->id, idx});
            }
        }
        json_obj["_frame_views"] = frame_views_array;
    }

    // Status fields
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        json_obj["_is_bad"] = _is_bad.load();
        json_obj["_num_observations"] = _num_observations;
        json_obj["num_times_visible"] = num_times_visible;
        json_obj["num_times_found"] = num_times_found;
        json_obj["last_frame_id_seen"] = last_frame_id_seen;
    }

    // Descriptor - use raw format (npRaw) to match Python's cv_mat_to_json_raw format
    // The flexible deserialization will handle both npRaw and npB64 formats when loading
    if (!des.empty()) {
        json_obj["des"] = cv_mat_to_json_raw(des);
    } else {
        json_obj["des"] = nullptr;
    }

    // Semantic descriptor
    if (!semantic_des.empty()) {
        json_obj["semantic_des"] =
            serialize_semantic_des(semantic_des, FeatureSharedResources::semantic_feature_type);
    } else {
        json_obj["semantic_des"] = nullptr;
    }

    // Distance fields
    {
        std::lock_guard<std::mutex> lock(_lock_pos);
        json_obj["_min_distance"] = _min_distance;
        json_obj["_max_distance"] = _max_distance;
        json_obj["normal"] = {normal.x(), normal.y(), normal.z()};
    }

    // Reference information
    json_obj["first_kid"] = first_kid;
    json_obj["kf_ref"] = kf_ref ? kf_ref->id : -1;

    return json_obj.dump();
}

MapPointPtr MapPoint::from_json(const std::string &json_str) {
    nlohmann::json json_obj = nlohmann::json::parse(json_str);

    // Parse basic fields
    int id = safe_json_get(json_obj, "id", -1);
    Eigen::Vector3d pt = safe_parse_vector3d<double>(json_obj, "pt");

    auto color_array = safe_json_get_array<int>(json_obj, "color");
    Eigen::Matrix<unsigned char, 3, 1> color;
    if (color_array.size() >= 3) {
        color << static_cast<unsigned char>(color_array[0]),
            static_cast<unsigned char>(color_array[1]), static_cast<unsigned char>(color_array[2]);
    } else {
        color << 0, 0, 0; // Default black color
    }

    auto semantic_color_array = safe_json_get_array<int>(json_obj, "semantic_color");
    Eigen::Matrix<unsigned char, 3, 1> semantic_color;
    if (semantic_color_array.size() >= 3) {
        semantic_color << static_cast<unsigned char>(semantic_color_array[0]),
            static_cast<unsigned char>(semantic_color_array[1]),
            static_cast<unsigned char>(semantic_color_array[2]);
    } else {
        semantic_color << 0, 0, 0; // Default black color
    }

    // Create MapPoint with basic constructor
    auto p = std::make_shared<MapPoint>(pt, color, KeyFramePtr(nullptr), -1, id);

    // Parse observations (stored as ID references, will be resolved later)
    if (json_obj.contains("_observations") && !json_obj["_observations"].is_null()) {
        auto observations_array = json_obj["_observations"].get<std::vector<std::vector<int>>>();
        for (const auto &obs : observations_array) {
            if (obs.size() >= 2) {
                int kf_id = obs[0];
                int idx = obs[1];
                // Store as temporary ID-based data for later resolution
                // We'll use a different approach - store in a static map or pass to
                // replace_ids_with_objects
                p->_observations_id_data.push_back({kf_id, idx});
            }
        }
    }

    // Parse frame views (stored as ID references, will be resolved later)
    if (json_obj.contains("_frame_views") && !json_obj["_frame_views"].is_null()) {
        auto frame_views_array = json_obj["_frame_views"].get<std::vector<std::vector<int>>>();
        for (const auto &view : frame_views_array) {
            if (view.size() >= 2) {
                int frame_id = view[0];
                int idx = view[1];
                // Store as temporary ID-based data for later resolution
                p->_frame_views_id_data.push_back({frame_id, idx});
            }
        }
    }

    // Parse status fields
    p->_is_bad = safe_json_get(json_obj, "_is_bad", false);
    p->_num_observations = safe_json_get(json_obj, "_num_observations", 0);
    p->num_times_visible = safe_json_get(json_obj, "num_times_visible", 1);
    p->num_times_found = safe_json_get(json_obj, "num_times_found", 1);
    p->last_frame_id_seen = safe_json_get(json_obj, "last_frame_id_seen", -1);

    // Parse descriptor - use flexible parser to handle both base64 (npB64) and raw (npRaw) formats
    // This matches how frames handle descriptors for consistency
    if (json_obj.contains("des") && !json_obj["des"].is_null()) {
        try {
            p->des = safe_parse_cv_mat_flexible(json_obj, "des");
        } catch (const std::exception &e) {
            // If parsing fails, leave as empty matrix
            p->des = cv::Mat();
        }
    }

    // Parse distance fields
    p->_min_distance = safe_json_get(json_obj, "_min_distance", 0.0f);
    p->_max_distance =
        safe_json_get(json_obj, "_max_distance", std::numeric_limits<float>::infinity());

    // Parse normal
    p->normal = safe_parse_vector3d<double>(json_obj, "normal");
    if (p->normal.norm() < kMin3dVectorNorm) {
        p->normal = Eigen::Vector3d(0, 0, 1); // Default normal
    }

    // Parse reference information
    p->first_kid = safe_json_get(json_obj, "first_kid", -1);
    // Store kf_ref ID for later resolution
    p->_kf_ref_id = safe_json_get(json_obj, "kf_ref", -1);

    // Parse semantic descriptor
    if (json_obj.contains("semantic_des") && !json_obj["semantic_des"].is_null()) {
        auto semantic_result = deserialize_semantic_des(json_obj["semantic_des"]);
        p->semantic_des = semantic_result.first;
        // Note: semantic_type is not stored in MapPoint, but could be used for validation
    }
    p->semantic_color = Vec3b(semantic_color[0], semantic_color[1], semantic_color[2]);

    return p;
}

void MapPoint::replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                        const std::vector<FramePtr> &frames,
                                        const std::vector<KeyFramePtr> &keyframes) {
    // Pre-build dictionaries for efficient lookups
    std::unordered_map<int, KeyFramePtr> keyframes_dict;
    std::unordered_map<int, FramePtr> frames_dict;

    for (auto &kf : keyframes) {
        if (kf) {
            keyframes_dict[kf->id] = kf;
        }
    }

    for (auto &frame : frames) {
        if (frame) {
            frames_dict[frame->id] = frame;
        }
    }

    auto get_keyframe_with_id = [&keyframes_dict](int id) -> KeyFramePtr {
        auto it = keyframes_dict.find(id);
        return (it != keyframes_dict.end()) ? it->second : nullptr;
    };

    auto get_frame_with_id = [&frames_dict](int id) -> FramePtr {
        auto it = frames_dict.find(id);
        return (it != frames_dict.end()) ? it->second : nullptr;
    };

    // Replace _observations from ID-based data
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        _observations.clear();
        for (const auto &[kf_id, idx] : _observations_id_data) {
            auto kf = get_keyframe_with_id(kf_id);
            if (kf) {
                _observations[kf] = idx;
            }
        }
        // Clear the temporary ID data
        _observations_id_data.clear();
    }

    // Replace _frame_views from ID-based data
    {
        std::lock_guard<std::mutex> lock(_lock_features);
        _frame_views.clear();
        for (const auto &[frame_id, idx] : _frame_views_id_data) {
            FramePtr frame = get_frame_with_id(frame_id);
            if (frame) {
                _frame_views[frame] = idx;
            }
        }
        // Clear the temporary ID data
        _frame_views_id_data.clear();
    }

    // Replace kf_ref from ID
    if (_kf_ref_id >= 0) {
        kf_ref = get_keyframe_with_id(_kf_ref_id);
        _kf_ref_id = -1; // Clear the temporary ID
    }
}

} // namespace pyslam