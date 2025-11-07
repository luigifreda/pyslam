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

#include "semantic_serialization.h"
#include "utils/optional_lock.h"
#include "utils/serialization_numpy.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pyslam {

//=======================================
//         Numpy serialization
//=======================================

py::tuple MapPoint::state_tuple(bool need_lock) const {
    // Acquire feature lock before position lock to stay consistent with MapPoint methods.
    CONDITIONAL_LOCK(_lock_features, need_lock);
    CONDITIONAL_LOCK(_lock_pos, need_lock);

    int version = 1;

    // Observations: KeyFrame objects + idx
    std::vector<std::pair<int, int>> obs_id_data;
    obs_id_data.reserve(_observations.size());
    for (const auto &[kf, idx] : _observations) {
        if (kf) {
            obs_id_data.emplace_back(kf->id, idx);
        }
    }

    // Frame views: Frame objects + idx
    std::vector<std::pair<int, int>> fviews_id_data;
    fviews_id_data.reserve(_frame_views.size());
    for (const auto &[frame, idx] : _frame_views) {
        if (frame) {
            fviews_id_data.emplace_back(frame->id, idx);
        }
    }

    int kf_ref_id = kf_ref ? kf_ref->id : -1;

    // Handle empty descriptor matrices
    auto des_type = des.empty() ? CV_32F : des.type();
    auto des_obj = des.empty() ? py::array() : cvmat_to_numpy(des);
    auto semantic_des_type = semantic_des.empty() ? CV_32F : semantic_des.type();
    auto semantic_des_obj = semantic_des.empty() ? py::array() : cvmat_to_numpy(semantic_des);

    // NOTE: this does not make sense in a new spawned process
    // size_t map_ptr = reinterpret_cast<size_t>(map);

    // Note: order matters, keep in sync with restore_from_state
    return py::make_tuple(
        version, id, static_cast<bool>(_is_bad), _num_observations, num_times_visible,
        num_times_found, last_frame_id_seen,
        _pt,    // Eigen::Vector3d
        normal, // Eigen::Vector3d
        _min_distance, _max_distance, py::array(3, color.data()), des_type, des_obj,
        semantic_des_type, semantic_des_obj, py::array(3, semantic_color.data()), first_kid,
        obs_id_data, fviews_id_data, kf_ref_id, num_observations_on_last_update_des,
        num_observations_on_last_update_normals, num_observations_on_last_update_semantics, pt_GBA,
        is_pt_GBA_valid, GBA_kf_id, corrected_by_kf, corrected_reference);
}

void MapPoint::restore_from_state(const py::tuple &t, bool need_lock) {
    // Keep lock order aligned with state_tuple() to prevent deadlocks.
    CONDITIONAL_LOCK(_lock_features, need_lock);
    CONDITIONAL_LOCK(_lock_pos, need_lock);

    int idx = 0;
    int version = t[idx++].cast<int>();
    if (version != 1)
        throw std::runtime_error("Unsupported MapPoint pickle version");

    id = t[idx++].cast<int>();
    _is_bad = t[idx++].cast<bool>();

    _num_observations = t[idx++].cast<int>();
    num_times_visible = t[idx++].cast<int>();
    num_times_found = t[idx++].cast<int>();
    last_frame_id_seen = t[idx++].cast<int>();

    _pt = t[idx++].cast<Eigen::Vector3d>();

    normal = t[idx++].cast<Eigen::Vector3d>();
    _min_distance = t[idx++].cast<float>();
    _max_distance = t[idx++].cast<float>();

    py::array color_np = t[idx++].cast<py::array>();
    const int des_type = t[idx++].cast<int>();
    des = numpy_to_cvmat(t[idx++].cast<py::array>(), des_type);
    const int semantic_des_type = t[idx++].cast<int>();
    semantic_des = numpy_to_cvmat(t[idx++].cast<py::array>(), semantic_des_type);
    py::array semantic_color_np = t[idx++].cast<py::array>();
    first_kid = t[idx++].cast<int>();

    std::vector<std::pair<int, int>> obs_id_data =
        t[idx++].cast<std::vector<std::pair<int, int>>>();
    std::vector<std::pair<int, int>> fviews_id_data =
        t[idx++].cast<std::vector<std::pair<int, int>>>();

    int kf_ref_id = t[idx++].cast<int>();

    num_observations_on_last_update_des = t[idx++].cast<int>();
    num_observations_on_last_update_normals = t[idx++].cast<int>();
    num_observations_on_last_update_semantics = t[idx++].cast<int>();

    pt_GBA = t[idx++].cast<Eigen::Vector3d>();

    is_pt_GBA_valid = t[idx++].cast<bool>();
    GBA_kf_id = t[idx++].cast<int>();
    corrected_by_kf = t[idx++].cast<int>();
    corrected_reference = t[idx++].cast<int>();

    // NOTE: this does not make sense in a new spawned process
    // map = reinterpret_cast<Map *>(t[idx++].cast<size_t>());

    // Color
    auto color_data = color_np.unchecked<uint8_t>();
    for (int i = 0; i < 3; i++)
        color[i] = color_data(i);

    // Semantic color
    auto semantic_color_data = semantic_color_np.unchecked<uint8_t>();
    for (int i = 0; i < 3; i++)
        semantic_color[i] = semantic_color_data(i);

    // Reattach observations and frame views (store ID pairs; resolved later)
    _observations_id_data.clear();
    for (const auto &[kf_id, idx_obs] : obs_id_data) {
        _observations_id_data.emplace_back(kf_id, idx_obs);
    }

    _frame_views_id_data.clear();
    for (const auto &[frame_id, idx_view] : fviews_id_data) {
        _frame_views_id_data.emplace_back(frame_id, idx_view);
    }

    _kf_ref_id = kf_ref_id;

    // Note: call replace_ids_with_objects(...) after this to resolve IDs to objects.
}

} // namespace pyslam
