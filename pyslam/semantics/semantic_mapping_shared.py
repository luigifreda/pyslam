"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
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
"""

import numpy as np
import os
import sys


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Base class for semantic estimators via inference
class SemanticMappingShared:

    semantic_mapping = None
    semantic_feature_type = None
    semantic_dataset_type = None
    semantic_fusion_method = None
    sem_des_to_rgb = None
    sem_img_to_rgb = None
    get_semantic_weight = None

    @staticmethod
    def set_semantic_mapping(semantic_mapping, force=False):

        if not force and SemanticMappingShared.semantic_mapping is not None:
            raise Exception("SemanticMappingShared: Semantic Estimator is already set!")

        SemanticMappingShared.semantic_mapping = semantic_mapping
        SemanticMappingShared.semantic_feature_type = semantic_mapping.semantic_feature_type
        SemanticMappingShared.semantic_dataset_type = semantic_mapping.semantic_dataset_type
        SemanticMappingShared.semantic_fusion_method = semantic_mapping.semantic_fusion_method
        SemanticMappingShared.sem_des_to_rgb = semantic_mapping.sem_des_to_rgb
        SemanticMappingShared.sem_img_to_rgb = semantic_mapping.sem_img_to_rgb
        SemanticMappingShared.get_semantic_weight = semantic_mapping.get_semantic_weight
