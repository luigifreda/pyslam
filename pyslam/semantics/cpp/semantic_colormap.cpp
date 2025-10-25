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

#include "semantic_colormap.h"
#include <stdexcept>

namespace pyslam {

SemanticColorMap::SemanticColorMap(const SemanticDatasetType &semantic_dataset_type,
                                   int num_classes)
    : semantic_dataset_type_(semantic_dataset_type), num_classes_(num_classes) {
    init(semantic_dataset_type, num_classes);
}

SemanticColorMap::~SemanticColorMap() = default;

void SemanticColorMap::init(SemanticDatasetType semantic_dataset_type, int num_classes) {
    semantic_dataset_type_ = semantic_dataset_type;
    num_classes_ = num_classes;

    load_color_map();
    load_labels();
    load_information_weights();
}

void SemanticColorMap::load_color_map() {
    switch (semantic_dataset_type_) {
    case SemanticDatasetType::VOC:
        color_map_ = get_voc_color_map();
        break;
    case SemanticDatasetType::CITYSCAPES:
        color_map_ = get_cityscapes_color_map();
        break;
    case SemanticDatasetType::ADE20K:
        color_map_ = get_ade20k_color_map(false); // RGB format
        break;
    case SemanticDatasetType::NYU40:
        color_map_ = get_nyu40_color_map();
        break;
    case SemanticDatasetType::CUSTOM_SET:
        if (num_classes_ <= 0) {
            throw std::invalid_argument("num_classes must be > 0 for CUSTOM_SET");
        }
        color_map_ = get_generic_color_map(num_classes_);
        break;
    case SemanticDatasetType::FEATURE_SIMILARITY:
        // For similarity, we don't need a predefined color map
        // Colors will be generated dynamically using colormaps
        color_map_.clear();
        break;
    default:
        throw std::invalid_argument("Unknown semantic dataset type");
    }
}

void SemanticColorMap::load_labels() {
    switch (semantic_dataset_type_) {
    case SemanticDatasetType::VOC:
        labels_ = get_voc_labels();
        break;
    case SemanticDatasetType::CITYSCAPES:
        labels_ = get_cityscapes_labels();
        break;
    case SemanticDatasetType::ADE20K:
        labels_ = get_ade20k_labels();
        break;
    case SemanticDatasetType::NYU40:
        labels_ = get_nyu40_labels();
        break;
    case SemanticDatasetType::CUSTOM_SET:
        // For custom set, labels are not predefined
        labels_.clear();
        break;
    case SemanticDatasetType::FEATURE_SIMILARITY:
        // For similarity, labels are not predefined
        labels_.clear();
        break;
    default:
        throw std::invalid_argument("Unknown semantic dataset type");
    }
}

void SemanticColorMap::load_information_weights() {
    switch (semantic_dataset_type_) {
    case SemanticDatasetType::VOC:
        information_weights_ = get_voc_information_weights();
        break;
    case SemanticDatasetType::CITYSCAPES:
        information_weights_ = get_cityscapes_information_weights();
        break;
    case SemanticDatasetType::ADE20K:
        information_weights_ = get_ade20k_information_weights();
        break;
    case SemanticDatasetType::NYU40:
        information_weights_ = get_nyu40_information_weights();
        break;
    case SemanticDatasetType::CUSTOM_SET:
        information_weights_ = get_trivial_information(num_classes_);
        break;
    case SemanticDatasetType::FEATURE_SIMILARITY:
        // For similarity, use trivial weights
        information_weights_ = {1.0};
        break;
    default:
        throw std::invalid_argument("Unknown semantic dataset type");
    }
}

const std::vector<cv::Vec3b> &SemanticColorMap::get_color_map() const { return color_map_; }

const std::vector<std::string> &SemanticColorMap::get_labels() const { return labels_; }

const std::vector<double> &SemanticColorMap::get_information_weights() const {
    return information_weights_;
}

cv::Mat SemanticColorMap::to_rgb(const cv::Mat &semantics, bool bgr) const {
    return labels_to_image(semantics, color_map_, bgr);
}

cv::Mat SemanticColorMap::to_rgb(const cv::Mat &semantics, bool bgr,
                                 const std::vector<int> &ignore_labels,
                                 const cv::Mat &rgb_image) const {
    return labels_to_image(semantics, color_map_, bgr, ignore_labels, rgb_image);
}

cv::Mat SemanticColorMap::to_heatmap(const cv::Mat &semantics, bool bgr, const int colormap,
                                     double sim_scale) const {
    if (semantic_dataset_type_ != SemanticDatasetType::FEATURE_SIMILARITY) {
        throw std::runtime_error(
            "Heatmap visualization only supported for FEATURE_SIMILARITY dataset type");
    }
    return similarity_heatmap_image(semantics, colormap, sim_scale, bgr);
}

cv::Vec3b SemanticColorMap::single_label_to_color(int label, bool bgr) const {
    if (semantic_dataset_type_ == SemanticDatasetType::FEATURE_SIMILARITY) {
        throw std::runtime_error(
            "single_label_to_color not supported for FEATURE_SIMILARITY dataset type");
    }
    return ::pyslam::single_label_to_color(label, color_map_, bgr);
}

cv::Vec3b SemanticColorMap::single_similarity_to_color(double sim_value, bool bgr, int colormap,
                                                       double sim_scale) const {
    if (semantic_dataset_type_ != SemanticDatasetType::FEATURE_SIMILARITY) {
        throw std::runtime_error(
            "single_similarity_to_color only supported for FEATURE_SIMILARITY dataset type");
    }
    return similarity_heatmap_point(sim_value, colormap, sim_scale, bgr);
}

SemanticDatasetType SemanticColorMap::get_dataset_type() const { return semantic_dataset_type_; }

int SemanticColorMap::get_num_classes() const { return num_classes_; }

bool SemanticColorMap::supports_similarity() const {
    return semantic_dataset_type_ == SemanticDatasetType::FEATURE_SIMILARITY;
}

} // namespace pyslam