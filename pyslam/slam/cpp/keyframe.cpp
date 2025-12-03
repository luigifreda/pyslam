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
#include "utils/messages.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>
#include <sstream>

namespace pyslam {

// Threshold constant (equivalent to Parameters.kMinNumOfCovisiblePointsForCreatingConnection =
// 15)
constexpr int kMinNumOfCovisiblePointsForCreatingConnection = 15;

// Static member definitions
std::atomic<int> KeyFrame::next_kid_{0};
std::mutex KeyFrame::kid_mutex_;

// KeyFrameGraph implementation
KeyFrameGraph::KeyFrameGraph() : parent(nullptr), not_to_erase(false), is_first_connection(true) {}

void KeyFrameGraph::add_child(KeyFramePtr child) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    add_child_no_lock(child);
}

void KeyFrameGraph::add_child_no_lock(KeyFramePtr child) {
    if (std::find(children.begin(), children.end(), child) == children.end()) {
        children.insert(child);
    }
}

void KeyFrameGraph::erase_child(KeyFramePtr child) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    erase_child_no_lock(child);
}

void KeyFrameGraph::erase_child_no_lock(KeyFramePtr child) {
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) {
        children.erase(it);
    }
}

void KeyFrameGraph::set_parent_no_lock(KeyFramePtr &parent) {
    // Safely cast this to KeyFrame* for comparison
    auto this_as_kf = dynamic_cast<KeyFrame *>(this);
    if (!this_as_kf) {
        MSG_ERROR("KeyFrameGraph::set_parent called on standalone KeyFrameGraph object");
        throw std::runtime_error(
            "KeyFrameGraph::set_parent called on standalone KeyFrameGraph object");
    }
    if (parent->id == this_as_kf->id) {
        return;
    }
    // Set the parent
    this->parent = parent;
    // Add this keyframe as a child of the parent
    parent->add_child(WrapPtr(this_as_kf));
}

void KeyFrameGraph::set_parent(KeyFramePtr &parent) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    set_parent_no_lock(parent);
}

std::set<KeyFramePtr> KeyFrameGraph::get_children() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return children;
}

KeyFramePtr KeyFrameGraph::get_parent() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return parent;
}

bool KeyFrameGraph::has_child(const KeyFramePtr &keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return std::find(children.begin(), children.end(), keyframe) != children.end();
}

void KeyFrameGraph::add_loop_edge(KeyFramePtr &kf) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    not_to_erase = true;
    // Add the loop edge only if it is not already in the set
    if (std::find(loop_edges.begin(), loop_edges.end(), kf) == loop_edges.end()) {
        loop_edges.insert(kf);
    }
}

std::set<KeyFramePtr> KeyFrameGraph::get_loop_edges() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return loop_edges;
}

void KeyFrameGraph::add_connection_no_lock(KeyFramePtr kf, int weight) {
    connected_keyframes_weights[kf] = weight;
    update_best_covisibles_no_lock_();
}

void KeyFrameGraph::add_connection(KeyFramePtr kf, int weight) {

    std::lock_guard<std::mutex> lock(_lock_connections);
    add_connection_no_lock(kf, weight);
}

void KeyFrameGraph::update_best_covisibles_no_lock_() {

    // Convert map to vector for sorting
    std::vector<std::pair<KeyFramePtr, int>> covisibles_vec;
    covisibles_vec.reserve(connected_keyframes_weights.size());

    for (const auto &pair : connected_keyframes_weights) {
        covisibles_vec.push_back(pair);
    }

    // Sort the vector
    std::sort(covisibles_vec.begin(), covisibles_vec.end(),
              [](const std::pair<KeyFramePtr, int> &a, const std::pair<KeyFramePtr, int> &b) {
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

void KeyFrameGraph::erase_connection_no_lock(KeyFramePtr keyframe) {
    auto it = connected_keyframes_weights.find(keyframe);
    if (it != connected_keyframes_weights.end()) {
        connected_keyframes_weights.erase(it);
        update_best_covisibles_no_lock_();
    }
}

void KeyFrameGraph::erase_connection(KeyFramePtr keyframe) {
    std::lock_guard<std::mutex> lock(_lock_connections);
    erase_connection_no_lock(keyframe);
}

std::vector<KeyFramePtr> KeyFrameGraph::get_connected_keyframes_no_lock() const {
    std::vector<KeyFramePtr> connected;
    for (const auto &pair : connected_keyframes_weights) {
        connected.push_back(pair.first);
    }
    return connected;
}

std::vector<KeyFramePtr> KeyFrameGraph::get_connected_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return get_connected_keyframes_no_lock();
}

std::vector<KeyFramePtr> KeyFrameGraph::get_covisible_keyframes_no_lock() const {
    std::vector<KeyFramePtr> covisible;
    for (const auto &pair : ordered_keyframes_weights) {
        covisible.push_back(pair.first);
    }
    return covisible;
}

std::vector<KeyFramePtr> KeyFrameGraph::get_covisible_keyframes() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return get_covisible_keyframes_no_lock();
}

std::vector<KeyFramePtr> KeyFrameGraph::get_best_covisible_keyframes(int N) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFramePtr> best_covisible;
    int count = 0;
    for (const auto &pair : ordered_keyframes_weights) {
        if (count >= N)
            break;
        best_covisible.push_back(pair.first);
        count++;
    }
    return best_covisible;
}

std::vector<KeyFramePtr> KeyFrameGraph::get_covisible_by_weight(int weight) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::vector<KeyFramePtr> covisible;
    for (const auto &pair : ordered_keyframes_weights) {
        if (pair.second > weight) {
            covisible.push_back(pair.first);
        }
    }
    return covisible;
}

int KeyFrameGraph::get_weight_no_lock(const KeyFramePtr &keyframe) const {
    auto it = connected_keyframes_weights.find(keyframe);
    return (it != connected_keyframes_weights.end()) ? it->second : 0;
}

int KeyFrameGraph::get_weight(const KeyFramePtr &keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return get_weight_no_lock(keyframe);
}

// KeyFrame implementation
KeyFrame::KeyFrame(const FramePtr &frame, const cv::Mat &img, const cv::Mat &img_right,
                   const cv::Mat &depth, int kid)
    : Frame(frame->camera), kid(kid), _is_bad(false), to_be_erased(false), lba_count(0),
      loop_query_id(-1), num_loop_words(0), loop_score(0.0f), reloc_query_id(-1),
      num_reloc_words(0), reloc_score(0.0f), GBA_kf_id(-1), map(nullptr) {

    Frame::copy_from(*frame);

    // Set as keyframe
    is_keyframe = true;

    // Initialize pose relative to parent - use default constructor, not
    // make_unique
    _pose_Tcp = CameraPose();

    // Initialize GBA poses - use default constructor, not make_unique
    Tcw_GBA = Eigen::Matrix4d::Identity();
    Tcw_before_GBA = Eigen::Matrix4d::Identity();

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

void KeyFrame::init_observations() {
    std::lock_guard<std::mutex> lock(_lock_features);
    for (size_t idx = 0; idx < points.size(); ++idx) {
        const auto &p = points[idx];
        if (p && !p->is_bad()) {
            p->add_observation(WrapPtr(this), static_cast<int>(idx));
            p->update_info();
        }
    }
}

void KeyFrame::update_connections() {
    // Get all matched good points from this keyframe
    const auto points = get_matched_good_points();
    int num_points = static_cast<int>(points.size());

    if (num_points == 0) {
        // Equivalent to Printer.orange("KeyFrame: update_connections - frame without points")
        MSG_WARN("KeyFrame: update_connections - frame without points");
        return;
    }

    // Build a counter for viewing keyframes (equivalent to Counter in Python)
    std::unordered_map<KeyFramePtr, int> viewing_keyframes;
    for (const auto &p : points) {
        const auto point_keyframes = p->keyframes();
        for (const auto &kf : point_keyframes) {
            if (kf->kid != this->kid) {
                viewing_keyframes[kf]++;
            }
        }
    }

    if (viewing_keyframes.empty()) {
        return;
    }

    // Order the keyframes: sort by weight in descending order
    // Convert to vector for sorting (equivalent to most_common() in Python Counter)
    std::vector<std::pair<KeyFramePtr, int>> covisible_keyframes;
    covisible_keyframes.reserve(viewing_keyframes.size());

    for (const auto &pair : viewing_keyframes) {
        covisible_keyframes.push_back(pair);
    }

    // Sort by weight in descending order
    std::sort(covisible_keyframes.begin(), covisible_keyframes.end(),
              [](const std::pair<KeyFramePtr, int> &a, const std::pair<KeyFramePtr, int> &b) {
                  return a.second > b.second;
              });

    // Get keyframe that shares most points
    KeyFramePtr kf_max = covisible_keyframes[0].first;
    int w_max = covisible_keyframes[0].second;

    std::lock_guard<std::mutex> lock(_lock_connections);

    if (w_max >= kMinNumOfCovisiblePointsForCreatingConnection) {
        // Set connected_keyframes_weights to viewing_keyframes
        connected_keyframes_weights = viewing_keyframes;
        ordered_keyframes_weights.clear();

        for (const auto &pair : covisible_keyframes) {
            KeyFramePtr kf = pair.first;
            int w = pair.second;

            if (w >= kMinNumOfCovisiblePointsForCreatingConnection) {
                kf->add_connection_no_lock(WrapPtr(this), w);
                ordered_keyframes_weights[kf] = w;
            } else {
                break; // Since sorted, no more will meet threshold
            }
        }
    } else {
        // Only add the one with maximum counter
        connected_keyframes_weights.clear();
        connected_keyframes_weights[kf_max] = w_max;

        ordered_keyframes_weights.clear();
        ordered_keyframes_weights[kf_max] = w_max;

        kf_max->add_connection_no_lock(WrapPtr(this), w_max);
    }

    // Update spanning tree
    // We need to avoid setting the parent to None or self or a bad keyframe
    if (is_first_connection && kid != 0 && kf_max && (kf_max->id != this->id) &&
        !kf_max->is_bad()) {
        set_parent_no_lock(kf_max);
        is_first_connection = false;
    }
}

bool KeyFrame::is_bad() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return _is_bad;
}

Eigen::Matrix4d KeyFrame::Tcp() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return _pose_Tcp.get_matrix();
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
    if (kid <= 0) {
        return; // Don't mark the first keyframe as bad
    }

    if (not_to_erase) {
        to_be_erased = true;
        return;
    }

    // --- 1. Remove covisibility connections ---
    auto connected_keyframes = get_connected_keyframes_no_lock();
    for (auto &kf_connected : connected_keyframes) {
        kf_connected->erase_connection_no_lock(WrapPtr(this));
    }

    // --- 2. Remove feature observations ---
    for (size_t idx = 0; idx < points.size(); ++idx) {
        const auto &p = points[idx];
        if (p) {
            p->remove_observation(WrapPtr(this), static_cast<int>(idx));
        }
    }

    reset_covisibility();

    // --- 3. Update spanning tree ---
    // Each children must be connected to a new parent

    MSG_FORCED_ASSERT(parent, "KeyFrame: set_bad - parent is nullptr");

    std::set<KeyFramePtr> parent_candidates = {parent};

    // Prevent infinite loop due to malformed graph
    int max_iters = children.size() * 100;
    int iters = 0;

    // Reassign children based on covisibility weights
    std::vector<KeyFramePtr> remaining_children(children.begin(), children.end());
    children.clear();

    while (remaining_children.size() > 0 && iters < max_iters) {
        iters++;
        KeyFramePtr best_child;
        KeyFramePtr best_parent;
        int max_weight = -1;

        for (auto &child : remaining_children) {
            if (child->is_bad())
                continue;

            auto covisibles = child->get_covisible_keyframes_no_lock();
            // Intersect with parent candidates
            for (auto &candidate : parent_candidates) {
                // If the candidate is in the covisible keyframes of the child
                if (std::find(covisibles.begin(), covisibles.end(), candidate) !=
                    covisibles.end()) {
                    int w = child->get_weight_no_lock(candidate);
                    if (w > max_weight) {
                        best_child = child;
                        best_parent = candidate;
                        max_weight = w;
                    }
                }
            }

            if (best_parent && best_child) {
                best_child->set_parent(best_parent);
                parent_candidates.insert(child);
                remaining_children.erase(
                    std::find(remaining_children.begin(), remaining_children.end(), best_child));
            } else {
                break;
            }

            if (iters >= max_iters) {
                MSG_WARN("KeyFrame: set_bad - max iterations reached");
            }
        }

        // --- 4. Reassign unconnected children to original parent ---
        for (auto &child : remaining_children) {
            child->set_parent_no_lock(parent);
        }

        // --- 5. Cleanup ---
        parent->erase_child_no_lock(WrapPtr(this));
        _pose_Tcp.update(this->Tcw() * parent->Twc());
        _is_bad = true;

        if (map) {
            map->remove_keyframe(WrapPtr(this));
        }
    }
}

} // namespace pyslam
