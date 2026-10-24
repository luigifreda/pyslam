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

#include "map.h"

#include "utils/serialization_numpy.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pyslam {

//=======================================
//         Numpy serialization
//=======================================

py::tuple MapStateData::state_tuple() const {
    int version = 1;

    // Convert vectors to numpy arrays
    auto poses_array = vector_to_numpy(poses);
    auto pose_timestamps_array = vector_to_numpy(pose_timestamps);
    auto fov_centers_array = vector_to_numpy(fov_centers);
    auto fov_centers_colors_array = vector_to_numpy(fov_centers_colors);
    auto points_array = vector_to_numpy(points);
    auto colors_array = vector_to_numpy(colors);
    auto semantic_colors_array = vector_to_numpy(semantic_colors);
    auto covisibility_graph_array = vector_to_numpy(covisibility_graph);
    auto spanning_tree_array = vector_to_numpy(spanning_tree);
    auto loops_array = vector_to_numpy(loops);

    return py::make_tuple(version, poses_array, pose_timestamps_array, fov_centers_array,
                          fov_centers_colors_array, points_array, colors_array,
                          semantic_colors_array, covisibility_graph_array, spanning_tree_array,
                          loops_array);
}

void MapStateData::restore_from_state(const py::tuple &t) {
    int idx = 0;
    int version = t[idx++].cast<int>();
    if (version != 1)
        throw std::runtime_error("Unsupported MapStateData pickle version");

    poses = numpy_to_vector<Mat4d>(t[idx++].cast<py::array>());
    pose_timestamps = numpy_to_vector<double>(t[idx++].cast<py::array>());
    fov_centers = numpy_to_vector<Vec3d>(t[idx++].cast<py::array>());
    fov_centers_colors = numpy_to_vector<Vec3d>(t[idx++].cast<py::array>());
    points = numpy_to_vector<Vec3d>(t[idx++].cast<py::array>());
    colors = numpy_to_vector<Vec3f>(t[idx++].cast<py::array>());
    semantic_colors = numpy_to_vector<Vec3f>(t[idx++].cast<py::array>());
    covisibility_graph = numpy_to_vector<Vec6d>(t[idx++].cast<py::array>());
    spanning_tree = numpy_to_vector<Vec6d>(t[idx++].cast<py::array>());
    loops = numpy_to_vector<Vec6d>(t[idx++].cast<py::array>());
}

} // namespace pyslam