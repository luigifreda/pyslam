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
#include "utils/inheritable_shared_from_this.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#ifdef USE_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

namespace pyslam {

// Forward declarations
class Map;
class MapPoint;
class KeyFrame;

// KeyFrameGraph class - matches Python KeyFrameGraph exactly
class KeyFrameGraph : public inheritable_enable_shared_from_this<KeyFrameGraph> {
  protected:
    mutable std::mutex _lock_connections;

  public:
    // Data members
    bool init_parent = false; // is parent initialized?
    KeyFramePtr parent;
    std::set<KeyFramePtr> children;
    std::set<KeyFramePtr> loop_edges;
    bool not_to_erase = false; // if there is a loop edge then you cannot erase this
                               // keyframe
    std::unordered_map<KeyFramePtr, int> connected_keyframes_weights;   // Counter equivalent
    std::vector<std::pair<KeyFramePtr, int>> ordered_keyframes_weights; // OrderedDict equivalent
    bool is_first_connection = true;

  public:
    // Constructor
    KeyFrameGraph();

    // Destructor
    virtual ~KeyFrameGraph() = default;

    // Copy constructor and assignment - deleted due to mutexes
    KeyFrameGraph(const KeyFrameGraph &other) = delete;
    KeyFrameGraph &operator=(const KeyFrameGraph &other) = delete;
    KeyFrameGraph(KeyFrameGraph &&other) noexcept = delete;
    KeyFrameGraph &operator=(KeyFrameGraph &&other) noexcept = delete;

    // Serialization
    std::string to_json() const;
    void init_from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                  const std::vector<FramePtr> &frames,
                                  const std::vector<KeyFramePtr> &keyframes);

    // Spanning tree methods
    void add_child(KeyFramePtr keyframe);         // no reference passing here!
    void add_child_no_lock(KeyFramePtr keyframe); // no reference passing here!

    void erase_child(KeyFramePtr keyframe);         // no reference passing here!
    void erase_child_no_lock(KeyFramePtr keyframe); // no reference passing here!

    void set_parent(KeyFramePtr keyframe);
    void set_parent_no_lock(KeyFramePtr keyframe);

    std::set<KeyFramePtr> get_children() const;
    KeyFramePtr get_parent() const;
    bool has_child(const KeyFramePtr keyframe) const;

    // Loop edges methods
    void add_loop_edge(KeyFramePtr keyframe);
    std::set<KeyFramePtr> get_loop_edges() const;

    // Covisibility methods
    void reset_covisibility();

    void add_connection(KeyFramePtr keyframe, int weight); // no reference passing here!
    void add_connection_no_lock(KeyFramePtr keyframe,
                                int weight); // no reference passing here!

    void erase_connection(KeyFramePtr keyframe);         // no reference passing here!
    void erase_connection_no_lock(KeyFramePtr keyframe); // no reference passing here!

    void update_best_covisibles_no_lock_();

    std::vector<KeyFramePtr> get_connected_keyframes() const;
    std::vector<KeyFramePtr> get_connected_keyframes_no_lock() const;

    std::vector<KeyFramePtr> get_covisible_keyframes() const;
    std::vector<KeyFramePtr> get_covisible_keyframes_no_lock() const;

    std::vector<KeyFramePtr> get_best_covisible_keyframes(int N) const;
    std::vector<KeyFramePtr> get_covisible_by_weight(int weight) const;

    int get_weight(const KeyFramePtr &keyframe) const;
    int get_weight_no_lock(const KeyFramePtr &keyframe) const;

    // Get all connected keyframes with their weights (thread-safe)
    std::unordered_map<int, int> get_connected_keyframes_weights() const;

  protected:
    // Temporary storage for deserialization (IDs before object replacement)
    int _parent_id_temp;
    std::vector<int> _children_ids_temp;
    std::vector<int> _loop_edges_ids_temp;
    std::vector<std::pair<int, int>> _connected_keyframes_ids_weights_temp;
    std::vector<std::pair<int, int>> _ordered_keyframes_ids_weights_temp;
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
    int kid = -1; // keyframe id (keyframe counter-id, different from frame.id)
    bool _is_bad = false;
    bool to_be_erased = false;
    int lba_count = 0; // how many time this keyframe has adjusted by LBA

    // Pose relative to parent
    CameraPose _pose_Tcp; // pose relative to parent: self.Tcw @ self.parent.Twc

// Loop closing and relocalization
#ifdef USE_PYTHON
    pybind11::object g_des =
        pybind11::none(); // global descriptor for loop closing (accepts any type from Python)
#else
    // global descriptor for loop closing (accepts any type)
    cv::Mat g_des;
    // GlobalDescriptor g_des;  // WIP

#endif
    int loop_query_id = -1; // query id for loop closing
    int num_loop_words = 0;
    float loop_score = 0.0f;
    int reloc_query_id = -1; // query id for relocalization
    int num_reloc_words = 0;
    float reloc_score = 0.0f;

    // Global Bundle Adjustment
    int GBA_kf_id = 0;
    bool is_Tcw_GBA_valid = false;
    Eigen::Matrix4d Tcw_GBA = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Tcw_before_GBA = Eigen::Matrix4d::Identity();

    // Map reference
    Map *map = nullptr; // Pointer to Map object

    // Constructor
    KeyFrame(const FramePtr &frame, const cv::Mat &img = cv::Mat(),
             const cv::Mat &img_right = cv::Mat(), const cv::Mat &depth = cv::Mat(), int kid = -1);

    explicit KeyFrame(int id) : Frame(id) {}

    // Destructor
    ~KeyFrame() {
        // Clear references to prevent circular dependencies during shutdown
        clear_references();
        _is_bad = true;
    }

    // Deleted due to mutexes
    KeyFrame(const KeyFrame &other) = delete;
    KeyFrame &operator=(const KeyFrame &other) = delete;
    KeyFrame(KeyFrame &&other) noexcept = delete;
    KeyFrame &operator=(KeyFrame &&other) noexcept = delete;

    // Serialization
    std::string to_json() const;
    static KeyFramePtr from_json(const std::string &json_str);
    void replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                  const std::vector<FramePtr> &frames,
                                  const std::vector<KeyFramePtr> &keyframes);

    void init_observations();

    // Cleanup method for proper shutdown
    void clear_references() {
        children.clear();
        loop_edges.clear();
        connected_keyframes_weights.clear();
        ordered_keyframes_weights.clear();
        parent = nullptr;
        map = nullptr;
    }

    // Connection management methods
    void update_connections();

    // Properties
    Eigen::Matrix4d Tcp() const; // pose relative to parent: self.Tcw @ self.parent.Twc
    bool is_bad() const;

    void set_not_erase();
    void set_erase();
    void set_bad();

#ifdef USE_PYTHON
    // Numpy serialization
    pybind11::tuple state_tuple(bool need_lock = true) const; // builds the versioned tuple
    void restore_from_state(const pybind11::tuple &,
                            bool need_lock = true); // fills this object from the tuple
#endif
};

struct KeyFrameIdCompare {
    bool operator()(const KeyFrame *a, const KeyFrame *b) const noexcept { return a->id < b->id; }
    bool operator()(const KeyFramePtr &a, const KeyFramePtr &b) const noexcept {
        return a->id < b->id;
    }
};

} // namespace pyslam
