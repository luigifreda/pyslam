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

#include "eigen_aliases.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>

#include "semantic_types.h"

namespace pyslam {

// /**
//  * Semantic fusion methods for combining multiple semantic measurements
//  */
// enum class SemanticFusionMethod {
//     NONE = -1,
//     COUNT_LABELS = 0,    // The label with the highest count
//     BAYESIAN_FUSION = 1, // Integrate measurements iteratively in a bayesian manner
//     AVERAGE_FUSION = 2   // Average all measurements
// };

/**
 * Count labels fusion: returns the label with the highest count
 * @param labels vector of integer labels
 * @return the most frequent label
 */
int count_labels_fusion(const std::vector<int> &labels);
int count_labels_fusion(VecNdRef labels);
cv::Mat count_labels_fusion(
    const cv::Mat &labels); // here we return a matrix with the most frequent label (1x1)

/**
 * Bayesian fusion: integrates probability measurements iteratively in a bayesian manner
 * @param probs vector of probability vectors (each inner vector should sum to 1.0)
 * @return fused probability vector
 */
VecNd bayesian_fusion(const std::vector<VecNd> &probs);
VecNd bayesian_fusion(MatNxMdRef probs);
cv::Mat bayesian_fusion(const cv::Mat &probs);

/**
 * Average fusion: averages all feature measurements
 * @param features vector of feature vectors
 * @return averaged feature vector
 */
VecNd average_fusion(const std::vector<VecNd> &features);
VecNd average_fusion(MatNxMdRef features);
cv::Mat average_fusion(const cv::Mat &features);

/**
 * Semantic fusion method: returns the first semantic measurement
 * @param semantics vector of semantic measurements
 * @return the first semantic measurement
 */
inline cv::Mat semantic_fusion(const cv::Mat &semantics, const SemanticFeatureType &feature_type) {
    switch (feature_type) {
    case SemanticFeatureType::LABEL:
        return count_labels_fusion(semantics);
    case SemanticFeatureType::PROBABILITY_VECTOR:
        return bayesian_fusion(semantics);
    case SemanticFeatureType::FEATURE_VECTOR:
        return average_fusion(semantics);
    default:
        std::cerr << "Semantic feature type not supported: " << (int)feature_type << std::endl;
        return cv::Mat();
    }
}

} // namespace pyslam