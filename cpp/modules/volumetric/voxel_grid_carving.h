
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
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "camera_frustrum.h"
#include "image_utils.h"
#include "voxel_data.h"
#include "voxel_hashing.h"

#include <opencv2/opencv.hpp>

namespace volumetric {

// Carve the voxel grid using the camera frustrum and camera depths.
// Reset all the voxels that are inside the camera frustrum and have a depth less than their
// camera depth - depth threshold.
// Inputs:
// - camera_frustrum: the camera frustrum
// - depth_image: the depth image
// - depth_threshold: the depth threshold for carving (default: 1e-3)
template <typename VoxelGridT, typename VoxelDataT>
void carve(VoxelGridT &voxel_grid, const CameraFrustrum &camera_frustrum,
           const cv::Mat &depth_image, const float depth_threshold) {

    if (depth_image.empty()) {
        std::cout << "volumetric::carve: Depth image is empty" << std::endl;
        return;
    }
    if (!check_image_size(depth_image, camera_frustrum.get_height(), camera_frustrum.get_width(),
                          "Depth")) {
        return;
    }
    cv::Mat depth_image_float = convert_image_type_if_needed(depth_image, CV_32F, "Depth");

    const auto callback = [&](VoxelDataT &v, const VoxelKey &key,
                              const typename VoxelDataT::Pos3 &pos_w,
                              const ImagePoint &image_point) {
        const float point_depth = image_point.depth;
        const float image_depth = depth_image_float.at<float>(image_point.v, image_point.u);

        // Skip invalid depth values (0, NaN, or inf)
        if (image_depth <= 0.0f || !std::isfinite(image_depth)) {
            return;
        }

        // Carve voxels that are inconsistent with the depth image:
        // Voxels in front of the depth image (point_depth < image_depth - threshold)
        // These are likely occluded or inconsistent
        if (point_depth < image_depth - depth_threshold) {
            v.reset();
        }
    };
    voxel_grid.iterate_voxels_in_camera_frustrum(camera_frustrum, callback);
};

} // namespace volumetric