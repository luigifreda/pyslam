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
#include "semantic_utils.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace pyslam {

class SemanticColorMap {

  public:
    /**
     * Constructor for SemanticColorMap
     * @param semantic_dataset_type Type of semantic dataset
     * @param num_classes Number of classes (required for CUSTOM_SET)
     */
    SemanticColorMap(const SemanticDatasetType &semantic_dataset_type, int num_classes = 0);

    /**
     * Destructor
     */
    ~SemanticColorMap();

    /**
     * Initialize the color map with a specific dataset type
     * @param semantic_dataset_type Type of semantic dataset
     * @param num_classes Number of classes (required for CUSTOM_SET)
     */
    void init(SemanticDatasetType semantic_dataset_type, int num_classes = 0);

    /**
     * Get the color map
     * @return Vector of cv::Vec3b colors
     */
    const std::vector<cv::Vec3b> &get_color_map() const;

    /**
     * Get the label names
     * @return Vector of string labels
     */
    const std::vector<std::string> &get_labels() const;

    /**
     * Get the information weights
     * @return Vector of double weights
     */
    const std::vector<double> &get_information_weights() const;

    /**
     * Convert semantic labels to RGB image
     * @param semantics Input semantic labels (CV_8UC1 or CV_32SC1)
     * @param bgr If true, output BGR format; otherwise RGB
     * @return RGB/BGR image
     */
    cv::Mat to_rgb(const cv::Mat &semantics, bool bgr = false) const;

    /**
     * Convert semantic labels to RGB image with ignored labels
     * @param semantics Input semantic labels
     * @param bgr If true, output BGR format; otherwise RGB
     * @param ignore_labels Labels to ignore (use original RGB image)
     * @param rgb_image Original RGB image for ignored labels
     * @return RGB/BGR image
     */
    cv::Mat to_rgb(const cv::Mat &semantics, bool bgr, const std::vector<int> &ignore_labels,
                   const cv::Mat &rgb_image) const;

    /**
     * Convert similarity map to heatmap visualization
     * @param semantics Input similarity map (CV_32FC1)
     * @param bgr If true, output BGR format; otherwise RGB
     * @param colormap OpenCV colormap (default: cv::COLORMAP_JET)
     * @param sim_scale Scale factor for similarity values
     * @return RGB/BGR heatmap image
     */
    cv::Mat to_heatmap(const cv::Mat &semantics, bool bgr = false,
                       const int colormap = cv::COLORMAP_JET, double sim_scale = 1.0) const;

    /**
     * Convert single label to color
     * @param label Class label
     * @param bgr If true, return BGR color; otherwise RGB
     * @return RGB/BGR color
     */
    cv::Vec3b single_label_to_color(int label, bool bgr = false) const;

    /**
     * Convert single similarity value to color
     * @param sim_value Similarity value (0.0-1.0)
     * @param bgr If true, return BGR color; otherwise RGB
     * @param colormap OpenCV colormap
     * @param sim_scale Scale factor for similarity values
     * @return RGB/BGR color
     */
    cv::Vec3b single_similarity_to_color(double sim_value, bool bgr = false,
                                         int colormap = cv::COLORMAP_JET,
                                         double sim_scale = 1.0) const;

    /**
     * Convert a single semantic descriptor to color.
     * Accepts either a 1x1 label matrix (CV_32S) or a 1xC probability/feature row (CV_32F).
     * For FEATURE_SIMILARITY datasets, colors by the maximum similarity value using a heatmap.
     */
    cv::Vec3b semantic_to_color(const cv::Mat &semantic_des,
                                const SemanticFeatureType &feature_type, bool bgr = false) const;

    /**
     * Get the dataset type
     * @return SemanticDatasetType
     */
    SemanticDatasetType get_dataset_type() const;

    /**
     * Get the number of classes
     * @return Number of classes
     */
    int get_num_classes() const;

    /**
     * Check if the color map supports similarity visualization
     * @return True if similarity visualization is supported
     */
    bool supports_similarity() const;

  protected:
    SemanticDatasetType semantic_dataset_type_;
    std::vector<cv::Vec3b> color_map_;
    std::vector<std::string> labels_;
    std::vector<double> information_weights_;
    int num_classes_;

    /**
     * Load color map based on dataset type
     */
    void load_color_map();

    /**
     * Load labels based on dataset type
     */
    void load_labels();

    /**
     * Load information weights based on dataset type
     */
    void load_information_weights();
};

} // namespace pyslam

/*
Usage example:

// Create color map for Cityscapes dataset
SemanticColorMap color_map(SemanticDatasetType::CITYSCAPES);

// Convert semantic labels to RGB image
cv::Mat semantic_labels = ...; // Your semantic segmentation result
cv::Mat rgb_image = color_map.to_rgb(semantic_labels, false); // RGB format

// Convert single label to color
cv::Vec3b road_color = color_map.single_label_to_color(0, false); // Road class

// For similarity visualization
SemanticColorMap similarity_map(SemanticDatasetType::FEATURE_SIMILARITY);
cv::Mat similarity_result = ...; // Your similarity map
cv::Mat heatmap = similarity_map.to_heatmap(similarity_result, false, cv::COLORMAP_JET, 3.0);


*/