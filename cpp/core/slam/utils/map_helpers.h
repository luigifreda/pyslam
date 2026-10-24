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

#include "frame.h"
#include "map_point.h"
#include <vector>

namespace pyslam {

// Helper function to safely validate MapPoint pointers
inline bool is_valid_mappoint(const pyslam::MapPointPtr &mp) {
    // return mp && !mp->is_bad() && mp->id >= 0;
    return mp && mp->id >= 0;
}

// Helper function to filter valid MapPoints from a container
template <typename Container>
std::vector<pyslam::MapPointPtr> filter_valid_mappoints(const Container &container) {
    std::vector<pyslam::MapPointPtr> valid_points;
    valid_points.reserve(container.size());

    for (const auto &point : container) {
        if (is_valid_mappoint(point)) {
            valid_points.push_back(point);
        }
    }

    return valid_points;
}

} // namespace pyslam