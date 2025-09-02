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
#include "camera_pose.h"
#include "frame.h"
#include "map.h"
#include "map_point.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <sstream>

namespace pyslam {

// Static member definitions
std::atomic<int> KeyFrame::next_kid_{0};
std::mutex KeyFrame::kid_mutex_;

// KeyFrameGraph implementation
KeyFrameGraph::KeyFrameGraph() : parent(nullptr), not_to_erase(false), is_first_connection(true) {}

std::string KeyFrameGraph::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"parent\": " << (parent != nullptr ? parent->id : -1) << ", ";
    oss << "\"children\": " << children.size() << ", ";
    oss << "\"loop_edges\": " << loop_edges.size() << ", ";
    oss << "\"not_to_erase\": " << (not_to_erase ? "true" : "false") << ", ";
    oss << "\"connected_keyframes_weights\": " << connected_keyframes_weights.size() << ", ";
    oss << "\"ordered_keyframes_weights\": " << ordered_keyframes_weights.size() << ", ";
    oss << "\"is_first_connection\": " << (is_first_connection ? "true" : "false") << "}";
    return oss.str();
}

void KeyFrameGraph::init_from_json(const std::string &json_str) {
    std::istringstream iss(json_str);
    nlohmann::json json;
    iss >> json;

    // These should be handled differently since we only have sizes, not actual
    // data parent = json["parent"];  // This is an ID, not a pointer children =
    // json["children"];  // This is a size, not a set loop_edges =
    // json["loop_edges"];  // This is a size, not a set
    // connected_keyframes_weights = json["connected_keyframes_weights"];  // This
    // is a size, not a map ordered_keyframes_weights =
    // json["ordered_keyframes_weights"];  // This is a size, not a map

    // Only the boolean and simple values can be directly assigned:
    not_to_erase = json["not_to_erase"];
    is_first_connection = json["is_first_connection"];

    // The collections would need to be reconstructed from actual data,
    // not just their sizes. This function appears to be incomplete.
}

void KeyFrameGraph::add_child(KeyFrame *child) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    if (std::find(children.begin(), children.end(), child) == children.end()) {
        children.insert(child);
    }
}

void KeyFrameGraph::erase_child(KeyFrame *child) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) {
        children.erase(it);
    }
}

void KeyFrameGraph::set_parent(KeyFrame *parent) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    parent = parent;
}

std::set<KeyFrame *> KeyFrameGraph::get_children() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return children;
}

KeyFrame *KeyFrameGraph::get_parent() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return parent;
}

void KeyFrameGraph::add_loop_edge(KeyFrame *kf) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    if (std::find(loop_edges.begin(), loop_edges.end(), kf) == loop_edges.end()) {
        loop_edges.insert(kf);
    }
}

std::set<KeyFrame *> KeyFrameGraph::get_loop_edges() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return loop_edges;
}

void KeyFrameGraph::add_connection(KeyFrame *kf, int weight) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    connected_keyframes_weights[kf] = weight;
}

void KeyFrameGraph::update_best_covisibles() {
    std::lock_guard<std::mutex> lock(_lock_connections);

    // Convert map to vector for sorting
    std::vector<std::pair<KeyFrame *, int>> covisibles_vec;
    covisibles_vec.reserve(connected_keyframes_weights.size());

    for (const auto &pair : connected_keyframes_weights) {
        covisibles_vec.push_back(pair);
    }

    // Sort the vector
    std::sort(covisibles_vec.begin(), covisibles_vec.end(),
              [](const std::pair<KeyFrame *, int> &a, const std::pair<KeyFrame *, int> &b) {
                  return a.second > b.second;
              });

    // Update ordered_keyframes_weights
    ordered_keyframes_weights.clear();
    for (const auto &pair : covisibles_vec) {
        ordered_keyframes_weights.insert(pair);
    }
}

// KeyFrameGraph missing methods
void KeyFrameGraph::reset_covisibility() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    connected_keyframes_weights.clear();
    ordered_keyframes_weights.clear();
}

void KeyFrameGraph::erase_connection(KeyFrame *keyframe) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    auto it = connected_keyframes_weights.find(keyframe);
    if (it != connected_keyframes_weights.end()) {
        connected_keyframes_weights.erase(it);
        update_best_covisibles();
    }
}

std::vector<KeyFrame *> KeyFrameGraph::get_connected_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFrame *> connected;
    for (const auto &pair : connected_keyframes_weights) {
        connected.push_back(pair.first);
    }
    return connected;
}

std::vector<KeyFrame *> KeyFrameGraph::get_covisible_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFrame *> covisible;
    for (const auto &pair : ordered_keyframes_weights) {
        covisible.push_back(pair.first);
    }
    return covisible;
}

std::vector<KeyFrame *> KeyFrameGraph::get_best_covisible_keyframes(int N) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFrame *> best_covisible;
    int count = 0;
    for (const auto &pair : ordered_keyframes_weights) {
        if (count >= N)
            break;
        best_covisible.push_back(pair.first);
        count++;
    }
    return best_covisible;
}

std::vector<KeyFrame *> KeyFrameGraph::get_covisible_by_weight(int weight) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFrame *> covisible;
    for (const auto &pair : ordered_keyframes_weights) {
        if (pair.second > weight) {
            covisible.push_back(pair.first);
        }
    }
    return covisible;
}

int KeyFrameGraph::get_weight(KeyFrame *keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    auto it = connected_keyframes_weights.find(keyframe);
    return (it != connected_keyframes_weights.end()) ? it->second : 0;
}

// KeyFrame implementation
KeyFrame::KeyFrame(Frame *frame, const cv::Mat &img, const cv::Mat &img_right, const cv::Mat &depth,
                   int kid)
    : Frame(*frame) // Copy frame data
      ,
      kid(kid), _is_bad(false), to_be_erased(false), lba_count(0), loop_query_id(-1),
      num_loop_words(0), loop_score(0.0f), reloc_query_id(-1), num_reloc_words(0),
      reloc_score(0.0f), GBA_kf_id(-1), map(nullptr) {

    // Set as keyframe
    is_keyframe = true;

    // Initialize pose relative to parent - use default constructor, not
    // make_unique
    _pose_Tcp = CameraPose();

    // Initialize GBA poses - use default constructor, not make_unique
    Tcw_GBA = CameraPose();
    Tcw_before_GBA = CameraPose();

    // Copy images if provided
    if (!img.empty()) {
        this->img = img.clone();
    }
    if (!img_right.empty()) {
        this->img_right = img_right.clone();
    }
    if (!depth.empty()) {
        this->depth_img = depth.clone();
    }
}

// KeyFrame::KeyFrame(const KeyFrame& other)
//     : Frame(other)
//     , kid(other.kid)
//     , _is_bad(other._is_bad)
//     , to_be_erased(other.to_be_erased)
//     , lba_count(other.lba_count)
//     , loop_query_id(other.loop_query_id)
//     , num_loop_words(other.num_loop_words)
//     , loop_score(other.loop_score)
//     , reloc_query_id(other.reloc_query_id)
//     , num_reloc_words(other.num_reloc_words)
//     , reloc_score(other.reloc_score)
//     , GBA_kf_id(other.GBA_kf_id)
//     , map(other.map) {

//     is_keyframe = other.is_keyframe;
//     _pose_Tcp = other._pose_Tcp;

//     if(!other.img.empty()) {
//         this->img = other.img.clone();
//     }
//     if(!other.img_right.empty()) {
//         this->img_right = other.img_right.clone();
//     }
//     if(!other.depth_img.empty()) {
//         this->depth_img = other.depth_img.clone();
//     }
// }

// KeyFrame& KeyFrame::operator=(const KeyFrame& other) {
//     Frame::operator=(other);
//     kid = other.kid;
//     _is_bad = other._is_bad;
//     to_be_erased = other.to_be_erased;
//     lba_count = other.lba_count;
//     loop_query_id = other.loop_query_id;
//     num_loop_words = other.num_loop_words;
//     loop_score = other.loop_score;
//     reloc_query_id = other.reloc_query_id;
//     num_reloc_words = other.num_reloc_words;
//     reloc_score = other.reloc_score;
//     GBA_kf_id = other.GBA_kf_id;
//     map = other.map;

//     is_keyframe = other.is_keyframe;
//     _pose_Tcp = other._pose_Tcp;
// }

void KeyFrame::init_observations() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    for (size_t idx = 0; idx < points.size(); ++idx) {
        MapPoint *p = points[idx];
        if (p != nullptr && !p->is_bad()) {
            p->add_observation(this, static_cast<int>(idx));
            p->update_info();
        }
    }
}

void KeyFrame::update_connections() {
    // TODO: Implement
}

std::string KeyFrame::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"kid\": " << kid << ", ";
    oss << "\"id\": " << this->id << ", ";
    oss << "\"timestamp\": " << timestamp << ", ";
    oss << "\"is_bad\": " << (is_bad() ? "true" : "false") << ", ";
    oss << "\"lba_count\": " << lba_count;
    oss << "}";
    return oss.str();
}

KeyFrame KeyFrame::from_json(const std::string &json_str) {
    // TODO: Implement proper JSON parsing
    // For now, return a default KeyFrame
    Frame empty_frame(nullptr);
    return KeyFrame(&empty_frame);
}

bool KeyFrame::is_bad() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return _is_bad;
}

CameraPose KeyFrame::Tcp() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return _pose_Tcp;
}

void KeyFrame::set_not_erase() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    not_to_erase = true;
}

void KeyFrame::set_erase() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    if (loop_edges.empty()) {
        not_to_erase = false;
    }
    if (to_be_erased) {
        set_bad();
    }
}

void KeyFrame::set_bad() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    if (kid == 0) {
        return; // Don't mark the first keyframe as bad
    }

    if (not_to_erase) {
        to_be_erased = true;
        return;
    }

    // Remove covisibility connections
    for (auto &pair : connected_keyframes_weights) {
        KeyFrame *kf_connected = pair.first;
        kf_connected->erase_connection(this);
    }

    // Remove feature observations
    for (size_t idx = 0; idx < points.size(); ++idx) {
        MapPoint *p = points[idx];
        if (p != nullptr) {
            p->remove_observation(this, static_cast<int>(idx));
        }
    }

    reset_covisibility();

    // Update spanning tree - reassign children
    if (parent != nullptr) {
        std::set<KeyFrame *> parent_candidates = {parent};

        // Reassign children based on covisibility weights
        std::vector<KeyFrame *> remaining_children(children.begin(), children.end());
        children.clear();

        for (KeyFrame *child : remaining_children) {
            if (child->is_bad())
                continue;

            auto covisible = child->get_covisible_keyframes();
            KeyFrame *best_parent = nullptr;
            int max_weight = -1;

            for (KeyFrame *candidate : parent_candidates) {
                if (std::find(covisible.begin(), covisible.end(), candidate) != covisible.end()) {
                    int w = child->get_weight(candidate);
                    if (w > max_weight) {
                        best_parent = candidate;
                        max_weight = w;
                    }
                }
            }

            if (best_parent) {
                child->set_parent(best_parent);
                parent_candidates.insert(child);
            } else {
                child->set_parent(parent);
            }
        }

        parent->erase_child(this);
        _pose_Tcp.update(this->Tcw() * parent->Twc());
    }

    _is_bad = true;

    if (map != nullptr) {
        map->remove_keyframe(this);
    }
}

} // namespace pyslam
