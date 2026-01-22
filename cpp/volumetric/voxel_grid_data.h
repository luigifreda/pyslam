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

//=================================================================================================
// Voxel Grid Data
//=================================================================================================

// Output structure for voxel grid data extraction
// Contains points, colors, and optionally semantic data
template <typename Tpos, typename Tcolor> struct VoxelGridDataT {

    std::vector<std::array<Tpos, 3>> points;
    std::vector<std::array<Tcolor, 3>> colors;

    // Semantic data (only populated if IncludeSemantics is true)
    std::vector<int> class_ids;
    std::vector<int> object_ids;

    std::vector<float> confidences; // confidences for each voxel

    using Ptr = std::shared_ptr<VoxelGridDataT>;
    using PosScalar = Tpos;
    using ColorScalar = Tcolor;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using VoxelGridData = VoxelGridDataT<double, float>;
using VoxelGridDataD = VoxelGridDataT<double, float>;
using VoxelGridDataF = VoxelGridDataT<float, float>;

//=================================================================================================
// Object Data
//=================================================================================================

// Output structure for object data extraction
// Contains points, colors, and object data
template <typename Tpos, typename Tcolor> struct ObjectDataT {
    std::vector<std::array<Tpos, 3>> points;
    std::vector<std::array<Tcolor, 3>> colors;

    int object_id;
    int class_id;

    float confidence_min; // minimum confidence across all voxels in this object (range: [0, 1])
    float confidence_max; // maximum confidence across all voxels in this object (range: [0, 1])

    OrientedBoundingBox3D oriented_bounding_box;

    using Ptr = std::shared_ptr<ObjectDataT>;
    using PosScalar = Tpos;
    using ColorScalar = Tcolor;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using ObjectData = ObjectDataT<double, float>;
using ObjectDataD = ObjectDataT<double, float>;
using ObjectDataF = ObjectDataT<float, float>;

template <typename Tpos, typename Tcolor> struct ObjectDataGroupT {
    std::vector<std::shared_ptr<ObjectDataT<Tpos, Tcolor>>> object_vector;
    std::vector<int> class_ids;  // redundant but convenient for drawing
    std::vector<int> object_ids; // redundant but convenient for drawing

    using Ptr = std::shared_ptr<ObjectDataGroupT>;
    using PosScalar = Tpos;
    using ColorScalar = Tcolor;
    using ObjectDataType = ObjectDataT<Tpos, Tcolor>;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using ObjectDataGroup = ObjectDataGroupT<double, float>;
using ObjectDataGroupD = ObjectDataGroupT<double, float>;
using ObjectDataGroupF = ObjectDataGroupT<float, float>;

//=================================================================================================
// Class Data
//=================================================================================================

// Output structure for class data extraction
// Contains points, colors, and class data
template <typename Tpos, typename Tcolor> struct ClassDataT {
    std::vector<std::array<Tpos, 3>> points;
    std::vector<std::array<Tcolor, 3>> colors;

    int class_id;

    float confidence_min; // minimum confidence across all objects in this class (range: [0, 1])
    float confidence_max; // maximum confidence across all objects in this class (range: [0, 1])

    using Ptr = std::shared_ptr<ClassDataT>;
    using PosScalar = Tpos;
    using ColorScalar = Tcolor;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using ClassData = ClassDataT<double, float>;
using ClassDataD = ClassDataT<double, float>;
using ClassDataF = ClassDataT<float, float>;

template <typename Tpos, typename Tcolor> struct ClassDataGroupT {
    std::vector<std::shared_ptr<ClassDataT<Tpos, Tcolor>>> class_vector;
    std::vector<int> class_ids; // redundant but convenient for drawing

    using Ptr = std::shared_ptr<ClassDataGroupT>;
    using PosScalar = Tpos;
    using ColorScalar = Tcolor;
    using ClassDataType = ClassDataT<Tpos, Tcolor>;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using ClassDataGroup = ClassDataGroupT<double, float>;
using ClassDataGroupD = ClassDataGroupT<double, float>;
using ClassDataGroupF = ClassDataGroupT<float, float>;

} // namespace volumetric