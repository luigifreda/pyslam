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

#include "semantic_types.h"
#include <nlohmann/json.hpp>

#include <opencv2/opencv.hpp>

namespace pyslam {

// Helper function to serialize semantic descriptors to JSON
nlohmann::json serialize_semantic_des(const cv::Mat &semantic_des,
                                      SemanticFeatureType semantic_type);

// Helper function to deserialize semantic descriptors from JSON
std::pair<cv::Mat, SemanticFeatureType> deserialize_semantic_des(const nlohmann::json &json_data);

} // namespace pyslam