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

#include "semantic_labels.h"
#include "semantic_types.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pyslam {

/**
 * Transforms a similarity map to a visual RGB image using a colormap.
 *
 * @param sim_map Similarity image of shape (H, W)
 * @param colormap OpenCV colormap (e.g., cv::COLORMAP_JET)
 * @param sim_scale Scale factor for similarity values
 * @param bgr If true, returns BGR image; otherwise RGB
 * @return RGB/BGR image (H, W, 3) visualizing similarity
 */
cv::Mat similarity_heatmap_image(const cv::Mat &sim_map, int colormap = cv::COLORMAP_JET,
                                 double sim_scale = 1.0, bool bgr = false);

/**
 * Generates a similarity color for a single point.
 *
 * @param sim_point Similarity of point (0.0-1.0)
 * @param colormap OpenCV colormap (e.g., cv::COLORMAP_JET)
 * @param sim_scale Scale factor for similarity values
 * @param bgr If true, returns BGR color; otherwise RGB
 * @return RGB/BGR color for the similarity point
 */
cv::Vec3b similarity_heatmap_point(double sim_point, int colormap = cv::COLORMAP_JET,
                                   double sim_scale = 1.0, bool bgr = false);

/**
 * Converts a class label image to an RGB image.
 *
 * @param label_img 2D array of class labels
 * @param semantics_color_map List or array of class RGB colors
 * @param bgr If true, color map is in BGR format
 * @param ignore_labels Labels to ignore (use original RGB image)
 * @param rgb_image Original RGB image for ignored labels
 * @return RGB image as a cv::Mat
 */
cv::Mat labels_to_image(const cv::Mat &label_img, const std::vector<cv::Vec3b> &semantics_color_map,
                        bool bgr = false, const std::vector<int> &ignore_labels = {},
                        const cv::Mat &rgb_image = cv::Mat());

/**
 * Converts an RGB label image to a class label image.
 *
 * @param rgb_labels Input RGB image as a cv::Mat
 * @param label_map List or array of class RGB colors
 * @return 2D array of class labels
 */
cv::Mat rgb_to_class(const cv::Mat &rgb_labels, const std::vector<cv::Vec3b> &label_map);

/**
 * Converts a single label to its corresponding color.
 *
 * @param label Class label
 * @param semantics_color_map Color map for labels
 * @param bgr If true, returns BGR color; otherwise RGB
 * @return RGB/BGR color for the label
 */
cv::Vec3b single_label_to_color(int label, const std::vector<cv::Vec3b> &semantics_color_map,
                                bool bgr = false);

/**
 * Factory function to get color map based on semantic dataset type.
 *
 * @param semantic_dataset_type Type of semantic dataset
 * @param num_classes Number of classes (required for CUSTOM_SET)
 * @return Color map for the dataset
 */
std::vector<cv::Vec3b> labels_color_map_factory(SemanticDatasetType semantic_dataset_type,
                                                int num_classes = 0);

/**
 * Factory function to get label names based on semantic dataset type.
 *
 * @param semantic_dataset_type Type of semantic dataset
 * @return Label names for the dataset
 */
std::vector<std::string> labels_name_factory(SemanticDatasetType semantic_dataset_type);

/**
 * Factory function to get information weights based on semantic dataset type.
 *
 * @param semantic_dataset_type Type of semantic dataset
 * @param num_classes Number of classes (required for CUSTOM_SET)
 * @return Information weights for the dataset
 */
std::vector<double> information_weights_factory(SemanticDatasetType semantic_dataset_type,
                                                int num_classes = 0);

// Information weights implementations for each dataset
std::vector<double> get_voc_information_weights();
std::vector<double> get_cityscapes_information_weights();
std::vector<double> get_ade20k_information_weights();
std::vector<double> get_nyu40_information_weights();
std::vector<double> get_trivial_information(int num_classes);

} // namespace pyslam