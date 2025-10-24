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

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pyslam {

/**
 * Generates a color map for generic semantic segmentation
 * @param num_classes Number of classes
 * @return Vector of cv::Vec3b with RGB values in [0, 255]
 */
std::vector<cv::Vec3b> get_generic_color_map(int num_classes);

/**
 * Generates n visually distinct RGB colors using HSV color space
 * @param n Number of colors to generate
 * @param s Saturation (0-1)
 * @param v Brightness/Value (0-1)
 * @return Vector of cv::Vec3b with RGB values in [0, 255]
 */
std::vector<cv::Vec3b> generate_hsv_color_map(int n, double s = 0.65, double v = 0.95);

/**
 * Load the mapping that associates pascal VOC classes with label colors
 * @return Vector of cv::Vec3b with dimensions (21, 3)
 */
std::vector<cv::Vec3b> get_voc_color_map();

/**
 * Get VOC dataset labels
 * @return Vector of strings with VOC class names
 */
std::vector<std::string> get_voc_labels();

/**
 * Load the mapping that associates cityscapes classes with label colors
 * @return Vector of cv::Vec3b with dimensions (19, 3)
 */
std::vector<cv::Vec3b> get_cityscapes_color_map();

/**
 * Get Cityscapes dataset labels
 * @return Vector of strings with Cityscapes class names
 */
std::vector<std::string> get_cityscapes_labels();

/**
 * Load the mapping that associates NYU40 classes with label colors
 * @return Vector of cv::Vec3b with dimensions (41, 3)
 */
std::vector<cv::Vec3b> get_nyu40_color_map();

/**
 * Get NYU40 dataset labels
 * @return Vector of strings with NYU40 class names
 */
std::vector<std::string> get_nyu40_labels();

/**
 * Returns a mapping from ADE20K class IDs to ScanNet20 class IDs
 * @return Vector of integers where each index corresponds to an ADE20K class ID
 */
std::vector<int> get_ade20k_to_scannet40_map();

/**
 * Returns the ADE20K color map
 * @param bgr If true, returns the color map in BGR format instead of RGB
 * @return Vector of cv::Vec3b representing the ADE20K color map
 */
std::vector<cv::Vec3b> get_ade20k_color_map(bool bgr = false);

/**
 * Get ADE20K dataset labels
 * @return Vector of strings with ADE20K class names
 */
std::vector<std::string> get_ade20k_labels();

} // namespace pyslam