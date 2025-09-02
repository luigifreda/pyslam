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
#include "utils/serialization.h"
#include "utils/serialization_numpy.h"
namespace pyslam {

std::string KeyFrameGraph::to_json() const {
    std::lock_guard<std::mutex> lock(_lock_connections);

    nlohmann::json json_obj;

    // Parent ID (or -1 if null)
    json_obj["parent"] = (parent != nullptr) ? parent->id : -1;

    // Children IDs
    std::vector<int> children_ids;
    for (const auto &child : children) {
        if (child != nullptr) {
            children_ids.push_back(child->id);
        }
    }
    json_obj["children"] = children_ids;

    // Loop edges IDs
    std::vector<int> loop_edges_ids;
    for (const auto &edge : loop_edges) {
        if (edge != nullptr) {
            loop_edges_ids.push_back(edge->id);
        }
    }
    json_obj["loop_edges"] = loop_edges_ids;

    // Boolean flags
    json_obj["not_to_erase"] = not_to_erase;
    json_obj["is_first_connection"] = is_first_connection;

    // Connected keyframes weights (as array of [id, weight] pairs)
    std::vector<std::pair<int, int>> connected_weights;
    for (const auto &pair : connected_keyframes_weights) {
        if (pair.first != nullptr) {
            connected_weights.push_back({pair.first->id, pair.second});
        }
    }
    json_obj["connected_keyframes_weights"] = connected_weights;

    // Ordered keyframes weights (as array of [id, weight] pairs)
    std::vector<std::pair<int, int>> ordered_weights;
    for (const auto &pair : ordered_keyframes_weights) {
        if (pair.first != nullptr) {
            ordered_weights.push_back({pair.first->id, pair.second});
        }
    }
    json_obj["ordered_keyframes_weights"] = ordered_weights;

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
    not_to_erase = json_obj["not_to_erase"].get<bool>();
    is_first_connection = json_obj["is_first_connection"].get<bool>();

    // Store IDs in temporary variables for later conversion to objects
    _parent_id_temp = json_obj["parent"].get<int>();

    // Store children IDs
    _children_ids_temp.clear();
    for (const auto &child_id : json_obj["children"]) {
        _children_ids_temp.push_back(child_id.get<int>());
    }

    // Store loop edges IDs
    _loop_edges_ids_temp.clear();
    for (const auto &edge_id : json_obj["loop_edges"]) {
        _loop_edges_ids_temp.push_back(edge_id.get<int>());
    }

    // Store connected keyframes weights IDs
    _connected_keyframes_weights_ids_temp.clear();
    for (const auto &pair : json_obj["connected_keyframes_weights"]) {
        int id = pair[0].get<int>();
        int weight = pair[1].get<int>();
        _connected_keyframes_weights_ids_temp.push_back({id, weight});
    }

    // Store ordered keyframes weights IDs
    _ordered_keyframes_weights_ids_temp.clear();
    for (const auto &pair : json_obj["ordered_keyframes_weights"]) {
        int id = pair[0].get<int>();
        int weight = pair[1].get<int>();
        _ordered_keyframes_weights_ids_temp.push_back({id, weight});
    }
}

void KeyFrameGraph::replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                             const std::vector<Frame *> &frames,
                                             const std::vector<KeyFrame *> &keyframes) {
    std::lock_guard<std::mutex> lock(_lock_connections);

    // Pre-build a dictionary for efficient lookups
    std::unordered_map<int, KeyFrame *> keyframes_dict;
    for (const auto &kf : keyframes) {
        if (kf != nullptr) {
            keyframes_dict[kf->kid] = kf;
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
        auto it = keyframes_dict.find(child_id);
        if (it != keyframes_dict.end()) {
            children.insert(it->second);
        }
    }

    // Replace loop edges
    loop_edges.clear();
    for (int edge_id : _loop_edges_ids_temp) {
        auto it = keyframes_dict.find(edge_id);
        if (it != keyframes_dict.end()) {
            loop_edges.insert(it->second);
        }
    }

    // Replace connected_keyframes_weights
    connected_keyframes_weights.clear();
    for (const auto &pair : _connected_keyframes_weights_ids_temp) {
        int id = pair.first;
        int weight = pair.second;
        auto it = keyframes_dict.find(id);
        if (it != keyframes_dict.end()) {
            connected_keyframes_weights[it->second] = weight;
        }
    }

    // Replace ordered_keyframes_weights
    ordered_keyframes_weights.clear();
    for (const auto &pair : _ordered_keyframes_weights_ids_temp) {
        int id = pair.first;
        int weight = pair.second;
        auto it = keyframes_dict.find(id);
        if (it != keyframes_dict.end()) {
            ordered_keyframes_weights[it->second] = weight;
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
    frame_json["_pose_Tcp"] = _pose_Tcp.to_json();

    // Add KeyFrameGraph data
    std::string graph_json_str = KeyFrameGraph::to_json();
    nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);

    // Merge the JSON objects
    json_obj.update(frame_json);
    json_obj.update(graph_json);

    return json_obj.dump();
}

std::unique_ptr<KeyFrame> KeyFrame::from_json(const std::string &json_str) {
    nlohmann::json json_obj = nlohmann::json::parse(json_str);

    // Create Frame from JSON
    Frame frame = Frame::from_json(json_str);

    // Create KeyFrame from Frame
    auto kf = std::make_unique<KeyFrame>(&frame);

    // Set KeyFrame-specific fields
    kf->kid = json_obj["kid"].get<int>();
    kf->_is_bad = json_obj["_is_bad"].get<bool>();
    kf->lba_count = json_obj["lba_count"].get<int>();
    kf->to_be_erased = json_obj["to_be_erased"].get<bool>();

    // Parse pose relative to parent
    if (json_obj.contains("_pose_Tcp") && !json_obj["_pose_Tcp"].is_null()) {
        kf->_pose_Tcp = CameraPose::from_json(json_obj["_pose_Tcp"].get<std::string>());
    }

    // Initialize KeyFrameGraph from JSON
    kf->KeyFrameGraph::init_from_json(json_str);

    return kf;
}

void KeyFrame::replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                        const std::vector<Frame *> &frames,
                                        const std::vector<KeyFrame *> &keyframes) {
    // Call parent class methods to replace IDs with objects
    Frame::replace_ids_with_objects(points, frames, keyframes);
    KeyFrameGraph::replace_ids_with_objects(points, frames, keyframes);
}

//=======================================
//         Numpy serialization
//=======================================

// KeyFrame::state_tuple()
py::tuple KeyFrame::state_tuple() const {
    const int version = 1;

    // 1) Serialize the Frame base as a nested tuple
    py::tuple frame_state = Frame::state_tuple();

    // 2) KeyFrameGraph — parent/children/loop/covisibility as objects
    std::shared_ptr<KeyFrame> parent_sp;
    if (KeyFrameGraph::parent) {
        auto base = static_cast<Frame *>(KeyFrameGraph::parent)->Frame::shared_from_this();
        parent_sp = std::static_pointer_cast<KeyFrame>(base);
    }

    std::vector<std::shared_ptr<KeyFrame>> children_vec;
    children_vec.reserve(KeyFrameGraph::children.size());
    for (auto *ch : KeyFrameGraph::children) {
        if (ch) {
            auto base = static_cast<Frame *>(ch)->Frame::shared_from_this();
            children_vec.emplace_back(std::static_pointer_cast<KeyFrame>(base));
        }
    }

    std::vector<std::shared_ptr<KeyFrame>> loop_vec;
    loop_vec.reserve(KeyFrameGraph::loop_edges.size());
    for (auto *le : KeyFrameGraph::loop_edges) {
        if (le) {
            auto base = static_cast<Frame *>(le)->Frame::shared_from_this();
            loop_vec.emplace_back(std::static_pointer_cast<KeyFrame>(base));
        }
    }

    std::vector<std::pair<std::shared_ptr<KeyFrame>, int>> conn_weights;
    conn_weights.reserve(KeyFrameGraph::connected_keyframes_weights.size());
    for (auto &kv : KeyFrameGraph::connected_keyframes_weights) {
        if (kv.first) {
            auto base = static_cast<Frame *>(kv.first)->Frame::shared_from_this();
            conn_weights.emplace_back(std::static_pointer_cast<KeyFrame>(base), kv.second);
        }
    }

    std::vector<std::pair<std::shared_ptr<KeyFrame>, int>> ordered_conn;
    ordered_conn.reserve(KeyFrameGraph::ordered_keyframes_weights.size());
    for (auto &kv : KeyFrameGraph::ordered_keyframes_weights) {
        if (kv.first) {
            auto base = static_cast<Frame *>(kv.first)->Frame::shared_from_this();
            ordered_conn.emplace_back(std::static_pointer_cast<KeyFrame>(base), kv.second);
        }
    }

    return py::make_tuple(version, frame_state,

                          // ---- KeyFrame core ----
                          kid, _is_bad, to_be_erased, lba_count,

                          // pose relative to parent
                          _pose_Tcp.Tcw(), // or store as matrix; reconstruct on restore

                          // ---- loop & relocalization ----
                          cvmat_to_numpy(g_des), loop_query_id, num_loop_words, loop_score,
                          reloc_query_id, num_reloc_words, reloc_score,

                          // ---- GBA ----
                          GBA_kf_id,
                          Tcw_GBA.Tcw(), // store matrices to avoid needing CameraPose ctor
                          Tcw_before_GBA.Tcw(),

                          // ---- KeyFrameGraph ----
                          KeyFrameGraph::init_parent, parent_sp, children_vec, loop_vec,
                          KeyFrameGraph::not_to_erase, conn_weights, ordered_conn,
                          KeyFrameGraph::is_first_connection
                          // Map* (KeyFrame::map) intentionally omitted (often owned externally)
    );
}

// KeyFrame::restore_from_state()
void KeyFrame::restore_from_state(const py::tuple &t) {
    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1)
        throw std::runtime_error("Unsupported KeyFrame pickle version");

    // 1) Restore Frame base first
    {
        auto frame_state = t[idx++].cast<py::tuple>();
        Frame::restore_from_state(frame_state);
    }

    // 2) KeyFrame core
    kid = t[idx++].cast<int>();
    _is_bad = t[idx++].cast<bool>();
    to_be_erased = t[idx++].cast<bool>();
    lba_count = t[idx++].cast<int>();

    // pose relative to parent (Tcp matrix)
    {
        const Eigen::Matrix4d Tcp = t[idx++].cast<Eigen::Matrix4d>();
        // Rebuild _pose_Tcp via R|t setter (avoid requiring CameraPose ctor signature)
        Eigen::Matrix3d R = Tcp.topLeftCorner<3, 3>();
        Eigen::Vector3d tt = Tcp.topRightCorner<3, 1>();
        _pose_Tcp.set_from_rotation_and_translation(R, tt);
    }

    // loop & relocalization
    g_des = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    loop_query_id = t[idx++].cast<int>();
    num_loop_words = t[idx++].cast<int>();
    loop_score = t[idx++].cast<float>();
    reloc_query_id = t[idx++].cast<int>();
    num_reloc_words = t[idx++].cast<int>();
    reloc_score = t[idx++].cast<float>();

    // GBA
    GBA_kf_id = t[idx++].cast<int>();
    {
        const Eigen::Matrix4d TcwG = t[idx++].cast<Eigen::Matrix4d>();
        Eigen::Matrix3d R = TcwG.topLeftCorner<3, 3>();
        Eigen::Vector3d tt = TcwG.topRightCorner<3, 1>();
        Tcw_GBA.set_from_rotation_and_translation(R, tt);
    }
    {
        const Eigen::Matrix4d TcwB = t[idx++].cast<Eigen::Matrix4d>();
        Eigen::Matrix3d R = TcwB.topLeftCorner<3, 3>();
        Eigen::Vector3d tt = TcwB.topRightCorner<3, 1>();
        Tcw_before_GBA.set_from_rotation_and_translation(R, tt);
    }

    // ---- KeyFrameGraph ----
    KeyFrameGraph::init_parent = t[idx++].cast<bool>();

    {
        auto parent_sp = t[idx++].cast<std::shared_ptr<KeyFrame>>();
        KeyFrameGraph::parent = parent_sp ? parent_sp.get() : nullptr;
    }

    {
        auto children_vec = t[idx++].cast<std::vector<std::shared_ptr<KeyFrame>>>();
        KeyFrameGraph::children.clear();
        for (auto &sp : children_vec)
            if (sp)
                KeyFrameGraph::children.insert(sp.get());
    }

    {
        auto loop_vec = t[idx++].cast<std::vector<std::shared_ptr<KeyFrame>>>();
        KeyFrameGraph::loop_edges.clear();
        for (auto &sp : loop_vec)
            if (sp)
                KeyFrameGraph::loop_edges.insert(sp.get());
    }

    KeyFrameGraph::not_to_erase = t[idx++].cast<bool>();

    {
        auto conn = t[idx++].cast<std::vector<std::pair<std::shared_ptr<KeyFrame>, int>>>();
        KeyFrameGraph::connected_keyframes_weights.clear();
        for (auto &kv : conn)
            if (kv.first)
                KeyFrameGraph::connected_keyframes_weights[kv.first.get()] = kv.second;
    }
    {
        auto ord = t[idx++].cast<std::vector<std::pair<std::shared_ptr<KeyFrame>, int>>>();
        KeyFrameGraph::ordered_keyframes_weights.clear();
        for (auto &kv : ord)
            if (kv.first)
                KeyFrameGraph::ordered_keyframes_weights[kv.first.get()] = kv.second;
    }

    KeyFrameGraph::is_first_connection = t[idx++].cast<bool>();

    // ---- finalize ----
    // map pointer left as-is/null (external owner)
    // locks and atomics are fine; nothing to serialize
}

} // namespace pyslam