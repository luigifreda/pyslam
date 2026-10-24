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

#include <stdexcept>
#include <string>

namespace pyslam {

/**
 * NOTES:
 * In order to add a new SEMANTIC representation:
 * - add a new enum value in SemanticFeatureType
 * - configure it in your semantic_segmentation*
 * - add its usage in the semantic_mapping* class that you want to use
 */

/**
 * Enumeration of semantic feature types for representing different kinds of semantic information
 */
enum class SemanticFeatureType : int {
    NONE = -1,
    LABEL = 0,              // [1] One value with the categorical label of the class
    PROBABILITY_VECTOR = 1, // [N] A vector of distribution parameters (categorical or Dirichlet)
                            // over N categorical classes
    FEATURE_VECTOR = 2      // [D] A feature vector from an encoder (e.g., CLIP or DiNO)
                            // with D dimensions
};

/**
 * Get the CV_DEPTH for a semantic feature type
 * This is used to normalize kps_sem type to avoid mixed-type issues downstream
 * Policy: LABEL -> CV_32S;
 * PROBABILITY_VECTOR/FEATURE_VECTOR -> CV_32F
 * @param type The semantic feature type
 * @return The CV_DEPTH for the semantic feature type
 */
int get_cv_depth_for_semantic_feature_type(const SemanticFeatureType &type);

/**
 * Enumeration of semantic entity types for associating semantics with different geometric
 * entities
 */
enum class SemanticEntityType : int {
    NONE = -1,
    POINT = 0, // The semantics are associated to each point
    OBJECT = 1 // The semantics are associated to each object
};

enum class SemanticDatasetType : int {
    NONE = -1,
    CITYSCAPES = 0,
    ADE20K = 1,
    VOC = 2,
    NYU40 = 3,
    FEATURE_SIMILARITY = 4,
    CUSTOM_SET = 5
};

enum class SemanticSegmentationType : int {
    NONE = -1,
    DEEPLABV3 = 0, // Semantics from torchvision DeepLab's v3
    SEGFORMER = 1, // Semantics from transformer's Segformer
    CLIP = 2,      // Semantics from CLIP's segmentation head
};

// ------------------------------------------------------------
//  serialization
// ------------------------------------------------------------

/**
 * Convert SemanticFeatureType enum to string representation
 * @param type The semantic feature type
 * @return String representation of the enum value
 */
inline std::string semantic_feature_type_to_string(SemanticFeatureType type) {
    switch (type) {
    case SemanticFeatureType::LABEL:
        return "LABEL";
    case SemanticFeatureType::PROBABILITY_VECTOR:
        return "PROBABILITY_VECTOR";
    case SemanticFeatureType::FEATURE_VECTOR:
        return "FEATURE_VECTOR";
    default:
        return "UNKNOWN";
    }
}

/**
 * Convert string to SemanticFeatureType enum
 * @param str String representation of the enum value
 * @return SemanticFeatureType enum value
 * @throws std::invalid_argument if string is not a valid enum value
 */
inline SemanticFeatureType string_to_semantic_feature_type(const std::string &str) {
    if (str == "LABEL") {
        return SemanticFeatureType::LABEL;
    } else if (str == "PROBABILITY_VECTOR") {
        return SemanticFeatureType::PROBABILITY_VECTOR;
    } else if (str == "FEATURE_VECTOR") {
        return SemanticFeatureType::FEATURE_VECTOR;
    } else {
        throw std::invalid_argument("Invalid SemanticFeatureType: " + str);
    }
}

/**
 * Convert SemanticEntityType enum to string representation
 * @param type The semantic entity type
 * @return String representation of the enum value
 */
inline std::string semantic_entity_type_to_string(SemanticEntityType type) {
    switch (type) {
    case SemanticEntityType::POINT:
        return "POINT";
    case SemanticEntityType::OBJECT:
        return "OBJECT";
    default:
        return "UNKNOWN";
    }
}

/**
 * Convert string to SemanticEntityType enum
 * @param str String representation of the enum value
 * @return SemanticEntityType enum value
 * @throws std::invalid_argument if string is not a valid enum value
 */
inline SemanticEntityType string_to_semantic_entity_type(const std::string &str) {
    if (str == "POINT") {
        return SemanticEntityType::POINT;
    } else if (str == "OBJECT") {
        return SemanticEntityType::OBJECT;
    } else {
        throw std::invalid_argument("Invalid SemanticEntityType: " + str);
    }
}

/**
 * Convert SemanticDatasetType enum to string representation
 * @param type The semantic dataset type
 * @return String representation of the enum value
 */
inline std::string semantic_dataset_type_to_string(SemanticDatasetType type) {
    switch (type) {
    case SemanticDatasetType::CITYSCAPES:
        return "CITYSCAPES";
    case SemanticDatasetType::ADE20K:
        return "ADE20K";
    case SemanticDatasetType::VOC:
        return "VOC";
    case SemanticDatasetType::NYU40:
        return "NYU40";
    case SemanticDatasetType::FEATURE_SIMILARITY:
        return "FEATURE_SIMILARITY";
    case SemanticDatasetType::CUSTOM_SET:
        return "CUSTOM_SET";
    default:
        return "UNKNOWN";
    }
}

/**
 * Convert string to SemanticDatasetType enum
 * @param str String representation of the enum value
 * @return SemanticDatasetType enum value
 * @throws std::invalid_argument if string is not a valid enum value
 */
inline SemanticDatasetType string_to_semantic_dataset_type(const std::string &str) {
    if (str == "CITYSCAPES") {
        return SemanticDatasetType::CITYSCAPES;
    } else if (str == "ADE20K") {
        return SemanticDatasetType::ADE20K;
    } else if (str == "VOC") {
        return SemanticDatasetType::VOC;
    } else if (str == "NYU40") {
        return SemanticDatasetType::NYU40;
    } else if (str == "FEATURE_SIMILARITY") {
        return SemanticDatasetType::FEATURE_SIMILARITY;
    } else if (str == "CUSTOM_SET") {
        return SemanticDatasetType::CUSTOM_SET;
    } else {
        throw std::invalid_argument("Invalid SemanticDatasetType: " + str);
    }
}

/**
 * Convert SemanticSegmentationType enum to string representation
 * @param type The semantic segmentation type
 * @return String representation of the enum value
 */
inline std::string semantic_segmentation_type_to_string(SemanticSegmentationType type) {
    switch (type) {
    case SemanticSegmentationType::DEEPLABV3:
        return "DEEPLABV3";
    case SemanticSegmentationType::SEGFORMER:
        return "SEGFORMER";
    case SemanticSegmentationType::CLIP:
        return "CLIP";
    default:
        return "UNKNOWN";
    }
}

/**
 * Convert string to SemanticSegmentationType enum
 * @param str String representation of the enum value
 * @return SemanticSegmentationType enum value
 * @throws std::invalid_argument if string is not a valid enum value
 */
inline SemanticSegmentationType string_to_semantic_segmentation_type(const std::string &str) {
    if (str == "DEEPLABV3") {
        return SemanticSegmentationType::DEEPLABV3;
    } else if (str == "SEGFORMER") {
        return SemanticSegmentationType::SEGFORMER;
    } else if (str == "CLIP") {
        return SemanticSegmentationType::CLIP;
    } else {
        throw std::invalid_argument("Invalid SemanticSegmentationType: " + str);
    }
}

} // namespace pyslam