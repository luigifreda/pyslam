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
    FEATURE_VECTOR =
        2 // [D] A feature vector from an encoder (e.g., CLIP or DiNO) with D dimensions
};

/**
 * Enumeration of semantic entity types for associating semantics with different geometric entities
 */
enum class SemanticEntityType : int {
    POINT = 0, // The semantics are associated to each point
    OBJECT = 1 // The semantics are associated to each object
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

} // namespace pyslam