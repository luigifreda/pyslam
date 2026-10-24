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
#include "config_parameters.h"
#include "frame.h"
#include "map.h"
#include "map_point.h"
#include "utils/messages.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <mutex>

namespace pyslam {

// Static member definitions
std::atomic<int> KeyFrame::next_kid_{0};
std::mutex KeyFrame::kid_mutex_;

// ======================================================
// KeyFrameGraph implementation
// ======================================================

KeyFrameGraph::KeyFrameGraph() : parent(nullptr) {}

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

void KeyFrameGraph::set_parent_no_lock(KeyFramePtr parent) {

    if (parent.get() == this) {
        return;
    }
    // Set the parent
    this->parent = parent;
    auto this_as_kf = std::dynamic_pointer_cast<KeyFrame>(this->shared_from_this());
    if (this_as_kf == nullptr) {
        throw std::runtime_error("KeyFrameGraph is not a KeyFrame");
    }
    auto self = this_as_kf->KeyFrameGraph::downcasted_shared_from_this<KeyFrame>();
    if (!self) {
        MSG_ERROR("KeyFrameGraph could not be downcasted to KeyFrame");
        return;
    }
    // Add this keyframe as a child of the parent
    parent->add_child(self);
}

void KeyFrameGraph::set_parent(KeyFramePtr parent) {
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

bool KeyFrameGraph::has_child(const KeyFramePtr keyframe) const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    return std::find(children.begin(), children.end(), keyframe) != children.end();
}

void KeyFrameGraph::add_loop_edge(KeyFramePtr kf) {
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
              [](const std::pair<KeyFrameGraphPtr, int> &a,
                 const std::pair<KeyFrameGraphPtr, int> &b) { return a.second > b.second; });

    // Update ordered_keyframes_weights
    ordered_keyframes_weights = std::move(covisibles_vec);
}

// KeyFrameGraph missing methods
void KeyFrameGraph::reset_covisibility() {
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
    covisible.reserve(ordered_keyframes_weights.size());
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
    best_covisible.reserve(std::min(N, static_cast<int>(ordered_keyframes_weights.size())));
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

std::unordered_map<int, int> KeyFrameGraph::get_connected_keyframes_weights() const {
    std::lock_guard<std::mutex> lock(_lock_connections);
    std::unordered_map<int, int> weights_map;
    for (const auto &[kf, weight] : connected_keyframes_weights) {
        if (kf) {
            weights_map[kf->id] = weight;
        }
    }
    return weights_map;
}

// ======================================================
// KeyFrame implementation
// ======================================================

// KeyFrame implementation
KeyFrame::KeyFrame(const FramePtr &frame, const cv::Mat &img, const cv::Mat &img_right,
                   const cv::Mat &depth, int kid)
    : Frame(frame->camera, cv::Mat(), cv::Mat(), cv::Mat(), CameraPose(), frame ? frame->id : -1,
            frame ? frame->timestamp : 0.0, frame ? frame->img_id : -1),
      kid(kid), _is_bad(false), to_be_erased(false), lba_count(0), loop_query_id(-1),
      num_loop_words(0), loop_score(0.0f), reloc_query_id(-1), num_reloc_words(0),
      reloc_score(0.0f), GBA_kf_id(0), map(nullptr) {

    if (frame) {
        Frame::copy_from(*frame);
    }
    // We don't preserve KeyFrame outliers and reinitialize all the flags to false
    outliers = std::vector<bool>(frame->outliers.size(), false);

    // Set as keyframe
    is_keyframe = true;

    // Initialize pose relative to parent - use default constructor, not
    // make_unique
    _pose_Tcp = CameraPose();

    // Initialize GBA poses - use default constructor, not make_unique
    is_Tcw_GBA_valid = false;
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
    auto self = KeyFrameGraph::downcasted_shared_from_this<KeyFrame>();
    if (!self) {
        MSG_ERROR("KeyFrameGraph could not be downcasted to KeyFrame");
        return;
    }
    for (size_t idx = 0; idx < points.size(); ++idx) {
        const auto &p = points[idx];
        if (p && !p->is_bad()) {
            if (p->add_observation(self, static_cast<int>(idx))) {
                p->update_info();
            }
        }
    }
}

void KeyFrame::update_connections() {
    // for all map points of this keyframe check in which other keyframes they are seen
    // build a counter for these other keyframes

    // Get all matched good points from this keyframe
    const auto points = get_matched_good_points();
    int num_points = static_cast<int>(points.size());

    if (num_points == 0) {
        // Equivalent to Printer.orange("KeyFrame: update_connections - frame without points")
        MSG_WARN("KeyFrame: update_connections - frame without points");
        return;
    }

    // Build a counter for viewing keyframes with stable (first-seen) order
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

    // Sort the vector by weight in descending order
    std::sort(covisible_keyframes.begin(), covisible_keyframes.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; }); // descending order

    auto kf_max = covisible_keyframes[0].first;
    auto w_max = covisible_keyframes[0].second;

    auto self = KeyFrameGraph::downcasted_shared_from_this<KeyFrame>();
    if (!self) {
        MSG_ERROR("KeyFrameGraph could not be downcasted to KeyFrame");
        return;
    }

    std::lock_guard<std::mutex> lock(_lock_connections);

    connected_keyframes_weights = std::move(viewing_keyframes);

    if (w_max >= Parameters::kMinNumOfCovisiblePointsForCreatingConnection) {
        ordered_keyframes_weights.clear();
        ordered_keyframes_weights.reserve(covisible_keyframes.size());

        // Here we keep the weight-decreasing-order of the covisible_keyframes
        for (const auto &[kf, w] : covisible_keyframes) {
            if (w >= Parameters::kMinNumOfCovisiblePointsForCreatingConnection) {
                kf->add_connection_no_lock(self, w);
                ordered_keyframes_weights.push_back({kf, w});
            } else {
                break; // Since sorted, no more will meet threshold
            }
        }
    } else {
        ordered_keyframes_weights.clear();
        ordered_keyframes_weights.push_back({kf_max, w_max});

        kf_max->add_connection_no_lock(self, w_max);
    }

    // Update spanning tree
    // We need to avoid setting the parent to None or self or a bad keyframe
    if (is_first_connection && (kid != 0) && (kf_max) && ((kf_max->id != this->id)) &&
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
    {
        std::lock_guard<std::mutex> lock(_lock_connections);
        if (loop_edges.empty()) {
            not_to_erase = false;
        }
    }
    if (to_be_erased) {
        set_bad();
    }
}

void KeyFrame::set_bad() {
    std::lock_guard<std::mutex> lock(_lock_connections);
    if (kid <= 0) {
        return;
    }

    if (not_to_erase) {
        to_be_erased = true;
        return;
    }
    auto self = KeyFrameGraph::downcasted_shared_from_this<KeyFrame>();
    if (!self) {
        MSG_ERROR("KeyFrameGraph could not be downcasted to KeyFrame");
        return;
    }

    // 1) Remove covisibility connections
    auto connected_keyframes = get_connected_keyframes_no_lock();
    for (auto &kf_connected : connected_keyframes) {
        kf_connected->erase_connection_no_lock(self);
    }

    // 2) Remove feature observations
    for (size_t idx = 0; idx < points.size(); ++idx) {
        const auto &p = points[idx];
        if (p) {
            p->remove_observation(self, static_cast<int>(idx));
        }
    }

    reset_covisibility();

    // 3) Update spanning tree
    MSG_FORCED_ASSERT(parent, "KeyFrame: set_bad - parent is nullptr");

    std::set<KeyFramePtr> parent_candidates = {std::static_pointer_cast<KeyFrame>(parent)};

    std::vector<KeyFramePtr> remaining_children(children.begin(), children.end());
    children.clear();

    // Reassign children via covisibility to any parent candidate; one child per iteration
    int iters = 0;
    const int max_iters = static_cast<int>(remaining_children.size()) * 100;
    while (!remaining_children.empty() && iters < max_iters) {
        iters++;

        KeyFramePtr best_child;
        KeyFramePtr best_parent;
        int max_weight = -1;

        // Find best reassignment across all remaining children
        for (auto &child : remaining_children) {
            if (child->is_bad())
                continue;

            auto covisibles = child->get_covisible_keyframes_no_lock();
            for (auto &candidate : parent_candidates) {
                if (std::find(covisibles.begin(), covisibles.end(), candidate) !=
                    covisibles.end()) {
                    int w = child->get_weight_no_lock(candidate);
                    if (w > max_weight) {
                        max_weight = w;
                        best_child = child;
                        best_parent = candidate;
                    }
                }
            }
        }

        // If no covisible parent candidate found, stop trying to reassign via covisibility
        if (!best_child || !best_parent)
            break;

        // Commit the best reassignment found in this iteration
        best_child->set_parent(best_parent);
        parent_candidates.insert(best_child);

        // Safely remove the chosen child (no range-iteration here)
        auto it = std::find(remaining_children.begin(), remaining_children.end(), best_child);
        if (it != remaining_children.end())
            remaining_children.erase(it);

        if (iters >= max_iters) {
            MSG_WARN("KeyFrame: set_bad - max iterations reached");
            break;
        }
    }

    // 4) Reassign any still-unconnected children to the original parent
    for (auto &child : remaining_children) {
        child->set_parent_no_lock(parent);
    }

    // 5) Cleanup
    auto parent_kf = std::static_pointer_cast<KeyFrame>(parent);
    parent->erase_child_no_lock(self);
    _pose_Tcp.update(this->Tcw() * parent_kf->Twc());
    _is_bad = true;

    if (map) {
        map->remove_keyframe(self);
    }
}

} // namespace pyslam
