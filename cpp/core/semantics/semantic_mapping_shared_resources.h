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

#include "semantic_colormap.h"
#include "semantic_fusion_methods.h"
#include "semantic_types.h"
#include <memory>
namespace pyslam {

class SemanticMappingSharedResources {
  public:
    static SemanticFeatureType semantic_feature_type;
    static SemanticDatasetType semantic_dataset_type;
    static SemanticEntityType semantic_entity_type;
    static SemanticSegmentationType semantic_segmentation_type;

    static std::shared_ptr<SemanticColorMap> semantic_color_map;

  public:
    static void init_color_map(const SemanticDatasetType &semantic_dataset_type,
                               int num_classes = 0);
    static void reset_color_map();
};

} // namespace pyslam