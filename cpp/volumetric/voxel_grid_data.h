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

#include "bounding_boxes.h"
#include <array>
#include <memory>
#include <vector>

namespace volumetric {

// A collection of data for a voxel grid

// Output structure for voxel grid data extraction
// Contains points, colors, and optionally semantic data
struct VoxelGridData {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<float, 3>> colors;

    // Semantic data (only populated if IncludeSemantics is true)
    std::vector<int> class_ids;
    std::vector<int> object_ids;

    std::vector<float> confidences; // confidences for each voxel
};

// Output structure for object data extraction
// Contains points, colors, and object data
struct ObjectData {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<float, 3>> colors;

    int object_id;
    int class_id;

    float confidence_min; // minimum confidence across all voxels in this object (range: [0, 1])
    float confidence_max; // maximum confidence across all voxels in this object (range: [0, 1])

    OrientedBoundingBox3D oriented_bounding_box;

    using Ptr = std::shared_ptr<ObjectData>;
};

// Output structure for class data extraction
// Contains points, colors, and class data
struct ClassData {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<float, 3>> colors;

    int class_id;

    float confidence_min; // minimum confidence across all objects in this class (range: [0, 1])
    float confidence_max; // maximum confidence across all objects in this class (range: [0, 1])

    using Ptr = std::shared_ptr<ClassData>;
};

} // namespace volumetric