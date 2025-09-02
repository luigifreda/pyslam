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

#include "camera_pose.h"
#include "frame.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pyslam {

// Forward declarations
class Map;
class MapPoint;
class KeyFrame;

// KeyFrameGraph class - matches Python KeyFrameGraph exactly
class KeyFrameGraph {
  protected:
    mutable std::mutex _lock_connections;

  public:
    // Data members
    bool init_parent; // is parent initialized?
    KeyFrame *parent;
    std::set<KeyFrame *> children;
    std::set<KeyFrame *> loop_edges;
    bool not_to_erase; // if there is a loop edge then you cannot erase this
                       // keyframe
    std::unordered_map<KeyFrame *, int> connected_keyframes_weights; // Counter equivalent
    std::map<KeyFrame *, int> ordered_keyframes_weights;             // OrderedDict equivalent
    bool is_first_connection;

    // Constructor - matches Python __init__ exactly
    KeyFrameGraph();

    // Destructor
    virtual ~KeyFrameGraph() = default;

    // Copy constructor and assignment
    // KeyFrameGraph(const KeyFrameGraph& other);
    // KeyFrameGraph& operator=(const KeyFrameGraph& other);

    // Serialization
    std::string to_json() const;
    void init_from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                  const std::vector<Frame *> &frames,
                                  const std::vector<KeyFrame *> &keyframes);

    // Spanning tree methods
    void add_child(KeyFrame *keyframe);
    void erase_child(KeyFrame *keyframe);
    void set_parent(KeyFrame *keyframe);
    std::set<KeyFrame *> get_children() const;
    KeyFrame *get_parent() const;
    bool has_child(KeyFrame *keyframe) const;

    // Loop edges methods
    void add_loop_edge(KeyFrame *keyframe);
    std::set<KeyFrame *> get_loop_edges() const;

    // Covisibility methods
    void reset_covisibility();
    void add_connection(KeyFrame *keyframe, int weight);
    void erase_connection(KeyFrame *keyframe);
    void update_best_covisibles();
    std::vector<KeyFrame *> get_connected_keyframes() const;
    std::vector<KeyFrame *> get_covisible_keyframes() const;
    std::vector<KeyFrame *> get_best_covisible_keyframes(int N) const;
    std::vector<KeyFrame *> get_covisible_by_weight(int weight) const;
    int get_weight(KeyFrame *keyframe) const;
};

// KeyFrame class - inherits from Frame and KeyFrameGraph, matches Python
// KeyFrame exactly
class KeyFrame : public Frame, public KeyFrameGraph {
  private:
    // Thread-safe ID management
    static std::atomic<int> next_kid_;
    static std::mutex kid_mutex_;

  public:
    // Data members
    int kid; // keyframe id (keyframe counter-id, different from frame.id)
    bool _is_bad;
    bool to_be_erased;
    int lba_count; // how many time this keyframe has adjusted by LBA

    // Pose relative to parent
    CameraPose _pose_Tcp; // pose relative to parent: self.Tcw @ self.parent.Twc

    // Loop closing and relocalization
    cv::Mat g_des; // global descriptor for loop closing
    int loop_query_id;
    int num_loop_words;
    float loop_score;
    int reloc_query_id;
    int num_reloc_words;
    float reloc_score;

    // Global Bundle Adjustment
    int GBA_kf_id;
    CameraPose Tcw_GBA;
    CameraPose Tcw_before_GBA;

    // Map reference
    Map *map;

    // Constructor - matches Python __init__ exactly
    KeyFrame(Frame *frame, const cv::Mat &img = cv::Mat(), const cv::Mat &img_right = cv::Mat(),
             const cv::Mat &depth = cv::Mat(), int kid = -1);

    // Destructor
    ~KeyFrame() = default;

    // Copy constructor and assignment
    // KeyFrame(const KeyFrame& other);
    // KeyFrame& operator=(const KeyFrame& other);

    // Serialization
    std::string to_json() const;
    static KeyFrame from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPoint *> &points,
                                  const std::vector<Frame *> &frames,
                                  const std::vector<KeyFrame *> &keyframes);

    void init_observations();

    // Connection management methods
    void update_connections();

    // Properties
    CameraPose Tcp() const; // pose relative to parent: self.Tcw @ self.parent.Twc
    bool is_bad() const;

    void set_not_erase();
    void set_erase();
    void set_bad();
};

} // namespace pyslam
