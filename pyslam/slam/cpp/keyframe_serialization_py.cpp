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

#include "utils/optional_lock.h"
#include "utils/serialization_numpy.h"

namespace pyslam {

//=======================================
//         Numpy serialization
//=======================================

py::tuple KeyFrame::state_tuple(bool need_lock) const {
    const int version = 1;

    CONDITIONAL_LOCK(_lock_connections, need_lock);
    CONDITIONAL_LOCK(_lock_pose, need_lock);
    CONDITIONAL_LOCK(_lock_features, need_lock);

    // 1) Serialize the Frame base as a nested tuple
    py::tuple frame_state = Frame::state_tuple(false /* no lock needed for Frame base */);

    // 2) KeyFrameGraph — parent/children/loop/covisibility as objects
    int parent_id = KeyFrameGraph::parent ? KeyFrameGraph::parent->id : -1;

    std::vector<int> children_id_data;
    children_id_data.reserve(KeyFrameGraph::children.size());
    for (auto &ch : KeyFrameGraph::children) {
        if (ch) {
            children_id_data.emplace_back(ch->id);
        }
    }

    std::vector<int> loop_id_data;
    loop_id_data.reserve(KeyFrameGraph::loop_edges.size());
    for (auto &le : KeyFrameGraph::loop_edges) {
        if (le) {
            loop_id_data.emplace_back(le->id);
        }
    }

    std::vector<std::pair<int, int>> connected_keyframes_ids_weights;
    connected_keyframes_ids_weights.reserve(KeyFrameGraph::connected_keyframes_weights.size());
    for (auto &[kf, weight] : KeyFrameGraph::connected_keyframes_weights) {
        if (kf) {
            connected_keyframes_ids_weights.emplace_back(kf->id, weight);
        }
    }

    std::vector<std::pair<int, int>> ordered_keyframes_ids_weights;
    ordered_keyframes_ids_weights.reserve(KeyFrameGraph::ordered_keyframes_weights.size());
    for (auto &[kf, weight] : KeyFrameGraph::ordered_keyframes_weights) {
        if (kf) {
            ordered_keyframes_ids_weights.emplace_back(kf->id, weight);
        }
    }

    return py::make_tuple(version, frame_state,

                          // ---- KeyFrame core ----
                          kid, _is_bad, to_be_erased, lba_count,

                          // pose relative to parent
                          _pose_Tcp.Tcw(), // or store as matrix; reconstruct on restore

                          // ---- loop & relocalization ----
                          // cvmat_to_numpy(g_des),
                          loop_query_id, num_loop_words, loop_score, reloc_query_id,
                          num_reloc_words, reloc_score,

                          // ---- GBA ----
                          GBA_kf_id, is_Tcw_GBA_valid,
                          Tcw_GBA, // store matrices to avoid needing CameraPose ctor
                          Tcw_before_GBA,

                          // ---- KeyFrameGraph ----
                          KeyFrameGraph::init_parent, parent_id, children_id_data, loop_id_data,
                          KeyFrameGraph::not_to_erase, connected_keyframes_ids_weights,
                          ordered_keyframes_ids_weights, KeyFrameGraph::is_first_connection
                          // Map* (KeyFrame::map) intentionally omitted (often owned externally)
    );
}

void KeyFrame::restore_from_state(const py::tuple &t, bool need_lock) {
    CONDITIONAL_LOCK(_lock_connections, need_lock);
    CONDITIONAL_LOCK(_lock_pose, need_lock);
    CONDITIONAL_LOCK(_lock_features, need_lock);

    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1)
        throw std::runtime_error("Unsupported KeyFrame pickle version");

    // 1) Restore Frame base first
    {
        auto frame_state = t[idx++].cast<py::tuple>();
        Frame::restore_from_state(frame_state, false /* no lock needed for Frame base */);
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
    // g_des = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    loop_query_id = t[idx++].cast<int>();
    num_loop_words = t[idx++].cast<int>();
    loop_score = t[idx++].cast<float>();
    reloc_query_id = t[idx++].cast<int>();
    num_reloc_words = t[idx++].cast<int>();
    reloc_score = t[idx++].cast<float>();

    // GBA
    GBA_kf_id = t[idx++].cast<int>();
    is_Tcw_GBA_valid = t[idx++].cast<bool>();
    {
        Tcw_GBA = t[idx++].cast<Eigen::Matrix4d>();
    }
    {
        Tcw_before_GBA = t[idx++].cast<Eigen::Matrix4d>();
    }

    // ---- KeyFrameGraph ----
    KeyFrameGraph::init_parent = t[idx++].cast<bool>();

    _parent_id_temp = t[idx++].cast<int>();

    _children_ids_temp = t[idx++].cast<std::vector<int>>();

    _loop_edges_ids_temp = t[idx++].cast<std::vector<int>>();

    KeyFrameGraph::not_to_erase = t[idx++].cast<bool>();

    _connected_keyframes_ids_weights_temp = t[idx++].cast<std::vector<std::pair<int, int>>>();

    _ordered_keyframes_ids_weights_temp = t[idx++].cast<std::vector<std::pair<int, int>>>();

    KeyFrameGraph::is_first_connection = t[idx++].cast<bool>();

    // ---- finalize ----
    // map pointer left as-is/null (external owner)

    // Note: call replace_ids_with_objects(...) after this to resolve IDs to objects.
}

} // namespace pyslam