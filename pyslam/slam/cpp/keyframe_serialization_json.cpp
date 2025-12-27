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

#include "keyframe.h"
#include "utils/serialization_json.h"

#include <nlohmann/json.hpp>

#include <Eigen/Dense>

namespace pyslam {

std::string KeyFrameGraph::to_json() const {
    std::lock_guard<std::mutex> lock(_lock_connections);

    nlohmann::json json_obj;

    // Parent ID (or null if null)
    if (parent) {
        json_obj["parent"] = parent->id;
    } else {
        json_obj["parent"] = nullptr;
    }

    // Children IDs
    std::vector<int> children_ids;
    for (const auto &child : children) {
        if (child) {
            children_ids.push_back(child->id);
        }
    }
    json_obj["children"] = children_ids;

    // Loop edges IDs
    std::vector<int> loop_edges_ids;
    for (const auto &edge : loop_edges) {
        if (edge) {
            loop_edges_ids.push_back(edge->id);
        }
    }
    json_obj["loop_edges"] = loop_edges_ids;

    // Boolean flags
    json_obj["init_parent"] = init_parent;
    json_obj["not_to_erase"] = not_to_erase;
    json_obj["is_first_connection"] = is_first_connection;

    // Connected keyframes weights (as array of [id, weight] pairs)
    std::vector<std::pair<int, int>> connected_keyframes_ids_weights;
    for (const auto &[kf, weight] : connected_keyframes_weights) {
        if (kf) {
            connected_keyframes_ids_weights.push_back({kf->id, weight});
        }
    }
    json_obj["connected_keyframes_weights"] = connected_keyframes_ids_weights;

    // Ordered keyframes weights (as array of [id, weight] pairs)
    std::vector<std::pair<int, int>> ordered_keyframes_ids_weights;
    for (const auto &[kf, weight] : ordered_keyframes_weights) {
        if (kf) {
            ordered_keyframes_ids_weights.push_back({kf->id, weight});
        }
    }
    json_obj["ordered_keyframes_weights"] = ordered_keyframes_ids_weights;

    return json_obj.dump();
}

void KeyFrameGraph::init_from_json(const std::string &json_str) {
    std::lock_guard<std::mutex> lock(_lock_connections);

    nlohmann::json json_obj = nlohmann::json::parse(json_str);

    // Clear existing collections
    children.clear();
    loop_edges.clear();
    connected_keyframes_weights.clear();
    ordered_keyframes_weights.clear();
    parent = nullptr;

    // Boolean flags can be set directly
    init_parent = safe_json_get(json_obj, "init_parent", false);
    not_to_erase = safe_json_get(json_obj, "not_to_erase", false);
    is_first_connection = safe_json_get(json_obj, "is_first_connection", true);

    // Store IDs in temporary variables for later conversion to objects
    if (json_obj.contains("parent") && !json_obj["parent"].is_null()) {
        _parent_id_temp = json_obj["parent"].get<int>();
    } else {
        _parent_id_temp = -1;
    }

    // Store children IDs
    _children_ids_temp = safe_json_get_array<int>(json_obj, "children");

    // Store loop edges IDs
    _loop_edges_ids_temp = safe_json_get_array<int>(json_obj, "loop_edges");

    // Store connected keyframes weights IDs
    _connected_keyframes_ids_weights_temp.clear();
    if (json_obj.contains("connected_keyframes_weights") &&
        !json_obj["connected_keyframes_weights"].is_null()) {
        try {
            const auto json_connected_keyframe_ids_weights =
                json_obj["connected_keyframes_weights"];
            _connected_keyframes_ids_weights_temp.reserve(
                json_connected_keyframe_ids_weights.size());
            for (const auto &pair : json_connected_keyframe_ids_weights) {
                if (pair.is_array() && pair.size() >= 2) {
                    int id = pair[0].get<int>();
                    int weight = pair[1].get<int>();
                    _connected_keyframes_ids_weights_temp.push_back({id, weight});
                }
            }
        } catch (const std::exception &e) {
            // If parsing fails, leave empty
        }
    }

    // Store ordered keyframes weights IDs
    _ordered_keyframes_ids_weights_temp.clear();
    if (json_obj.contains("ordered_keyframes_weights") &&
        !json_obj["ordered_keyframes_weights"].is_null()) {
        try {
            const auto json_ordered_keyframe_ids_weights = json_obj["ordered_keyframes_weights"];
            _ordered_keyframes_ids_weights_temp.reserve(json_ordered_keyframe_ids_weights.size());
            for (const auto &pair : json_ordered_keyframe_ids_weights) {
                if (pair.is_array() && pair.size() >= 2) {
                    int id = pair[0].get<int>();
                    int weight = pair[1].get<int>();
                    _ordered_keyframes_ids_weights_temp.push_back({id, weight});
                }
            }
        } catch (const std::exception &e) {
            // If parsing fails, leave empty
        }
    }
}

void KeyFrameGraph::replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                             const std::vector<FramePtr> &frames,
                                             const std::vector<KeyFramePtr> &keyframes) {
    std::lock_guard<std::mutex> lock(_lock_connections);

    // Pre-build a dictionary for efficient lookups
    std::unordered_map<int, KeyFramePtr> keyframes_dict;
    for (const auto &kf : keyframes) {
        if (kf) {
            keyframes_dict[kf->id] = kf; // Use id instead of kid
        }
    }

    // Replace parent
    if (_parent_id_temp != -1) {
        auto it = keyframes_dict.find(_parent_id_temp);
        parent = (it != keyframes_dict.end()) ? it->second : nullptr;
    } else {
        parent = nullptr;
    }

    // Replace children
    children.clear();
    for (int child_id : _children_ids_temp) {
        if (child_id == -1)
            continue;
        auto it = keyframes_dict.find(child_id);
        if (it != keyframes_dict.end() && it->second) {
            children.insert(it->second);
        }
    }

    // Replace loop edges
    loop_edges.clear();
    for (int edge_id : _loop_edges_ids_temp) {
        if (edge_id == -1)
            continue;
        auto it = keyframes_dict.find(edge_id);
        if (it != keyframes_dict.end() && it->second) {
            loop_edges.insert(it->second);
        }
    }

    // Replace connected_keyframes_weights
    connected_keyframes_weights.clear();
    for (const auto &[kf_id, weight] : _connected_keyframes_ids_weights_temp) {
        if (kf_id == -1)
            continue;
        auto it = keyframes_dict.find(kf_id);
        if (it != keyframes_dict.end() && it->second) {
            connected_keyframes_weights[it->second] = weight;
        }
    }

    // Replace ordered_keyframes_weights
    ordered_keyframes_weights.clear();
    for (const auto &[kf_id, weight] : _ordered_keyframes_ids_weights_temp) {
        if (kf_id == -1)
            continue;
        auto it = keyframes_dict.find(kf_id);
        if (it != keyframes_dict.end() && it->second) {
            ordered_keyframes_weights.push_back({it->second, weight});
        }
    }
}

// ===============================
// KeyFrame
// ===============================

std::string KeyFrame::to_json() const {
    nlohmann::json json_obj;

    // First, serialize the Frame data
    std::string frame_json_str = Frame::to_json();
    nlohmann::json frame_json = nlohmann::json::parse(frame_json_str);

    // Add KeyFrame-specific fields
    frame_json["is_keyframe"] = true; // KeyFrames are always keyframes
    frame_json["kid"] = kid;
    frame_json["_is_bad"] = _is_bad;
    frame_json["lba_count"] = lba_count;
    frame_json["to_be_erased"] = to_be_erased;
    // Serialize _pose_Tcp as direct array (aligned with Frame::pose serialization)
    frame_json["_pose_Tcp"] = eigen_matrix_to_json_array(_pose_Tcp.Tcw());
    frame_json["is_Tcw_GBA_valid"] = is_Tcw_GBA_valid;

    // Add loop closing and relocalization fields
    // frame_json["g_des"] = cv_mat_to_json_raw(g_des);
    frame_json["loop_query_id"] = loop_query_id;
    frame_json["num_loop_words"] = num_loop_words;
    frame_json["loop_score"] = loop_score;
    frame_json["reloc_query_id"] = reloc_query_id;
    frame_json["num_reloc_words"] = num_reloc_words;
    frame_json["reloc_score"] = reloc_score;

    // Add GBA fields
    frame_json["GBA_kf_id"] = GBA_kf_id;
    frame_json["Tcw_GBA"] = eigen_matrix_to_json_array(Tcw_GBA);
    frame_json["Tcw_before_GBA"] = eigen_matrix_to_json_array(Tcw_before_GBA);

    // Add KeyFrameGraph data
    std::string graph_json_str = KeyFrameGraph::to_json();
    nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);

    // Merge the JSON objects
    json_obj.update(frame_json);
    json_obj.update(graph_json);

    return json_obj.dump();
}

KeyFramePtr KeyFrame::from_json(const std::string &json_str) {
    nlohmann::json json_obj = nlohmann::json::parse(json_str);

    // Create Frame from JSON
    FramePtr frame = Frame::from_json(json_str);

    // Create KeyFrame from Frame
    auto kf = KeyFrameNewPtr(frame);

    // Set KeyFrame-specific fields
    kf->kid = safe_json_get(json_obj, "kid", -1);
    kf->_is_bad = safe_json_get(json_obj, "_is_bad", false);
    kf->lba_count = safe_json_get(json_obj, "lba_count", 0);
    kf->to_be_erased = safe_json_get(json_obj, "to_be_erased", false);
    kf->is_Tcw_GBA_valid = safe_json_get(json_obj, "is_Tcw_GBA_valid", false);

    // Set loop closing and relocalization fields
    // if (json_obj.contains("g_des") && !json_obj["g_des"].is_null()) {
    //     try {
    //         kf->g_des = json_to_cv_mat_raw(json_obj["g_des"]);
    //     } catch (const std::exception &e) {
    //         // If parsing fails, leave as empty cv::Mat
    //         kf->g_des = cv::Mat();
    //     }
    // } else {
    //     kf->g_des = cv::Mat();
    // }
    kf->loop_query_id = safe_json_get(json_obj, "loop_query_id", -1);
    kf->num_loop_words = safe_json_get(json_obj, "num_loop_words", 0);
    kf->loop_score = safe_json_get(json_obj, "loop_score", 0.0f);
    kf->reloc_query_id = safe_json_get(json_obj, "reloc_query_id", -1);
    kf->num_reloc_words = safe_json_get(json_obj, "num_reloc_words", 0);
    kf->reloc_score = safe_json_get(json_obj, "reloc_score", 0.0f);

    // Set GBA fields
    kf->GBA_kf_id = safe_json_get(json_obj, "GBA_kf_id", 0);
    safe_parse_pose_matrix(json_obj, "Tcw_GBA", kf->Tcw_GBA);
    safe_parse_pose_matrix(json_obj, "Tcw_before_GBA", kf->Tcw_before_GBA);

    // Parse pose relative to parent - deserialize as matrix array (same format as serialization)
    Eigen::Matrix4d Tcp_matrix = Eigen::Matrix4d::Identity();
    if (safe_parse_pose_matrix(json_obj, "_pose_Tcp", Tcp_matrix)) {
        kf->_pose_Tcp.set_from_matrix(Tcp_matrix);
    }

    // Initialize KeyFrameGraph from JSON
    kf->KeyFrameGraph::init_from_json(json_str);

    return kf;
}

void KeyFrame::replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                        const std::vector<FramePtr> &frames,
                                        const std::vector<KeyFramePtr> &keyframes) {
    // Call parent class methods to replace IDs with objects
    Frame::replace_ids_with_objects(points, frames, keyframes);
    KeyFrameGraph::replace_ids_with_objects(points, frames, keyframes);
}

} // namespace pyslam