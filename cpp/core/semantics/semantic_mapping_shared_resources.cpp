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

#include "semantic_mapping_shared_resources.h"

namespace pyslam {

SemanticFeatureType SemanticMappingSharedResources::semantic_feature_type =
    SemanticFeatureType::NONE;
SemanticDatasetType SemanticMappingSharedResources::semantic_dataset_type =
    SemanticDatasetType::NONE;
SemanticEntityType SemanticMappingSharedResources::semantic_entity_type = SemanticEntityType::NONE;
SemanticSegmentationType SemanticMappingSharedResources::semantic_segmentation_type =
    SemanticSegmentationType::NONE;

std::shared_ptr<SemanticColorMap> SemanticMappingSharedResources::semantic_color_map;

void SemanticMappingSharedResources::init_color_map(
    const SemanticDatasetType &semantic_dataset_type, int num_classes) {
    semantic_color_map = std::make_shared<SemanticColorMap>(semantic_dataset_type, num_classes);
}

void SemanticMappingSharedResources::reset_color_map() { semantic_color_map.reset(); }

} // namespace pyslam