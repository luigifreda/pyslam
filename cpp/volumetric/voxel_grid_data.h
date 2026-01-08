/**
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

#include <array>
#include <vector>

#include "bounding_boxes.h"

namespace volumetric {

// Output structure for voxel grid data extraction
// Contains points, colors, and optionally semantic data
struct VoxelGridData {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<float, 3>> colors;

    // Semantic data (only populated if IncludeSemantics is true)
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<int> object_ids;

    std::vector<OrientedBoundingBox3D> oriented_bounding_boxes;
    std::vector<BoundingBox3D> bounding_boxes;
};

} // namespace volumetric