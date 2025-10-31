"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
import traceback

from pyslam.utilities.system import Printer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."

from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from pyslam.semantics.semantic_mapping_base import SemanticMappingType, SemanticMappingBase
    from pyslam.semantics.semantic_types import SemanticFeatureType, SemanticDatasetType


# Base class for semantic estimators via inference
class SemanticMappingShared:

    semantic_mapping: Union["SemanticMappingType", None] = None
    semantic_feature_type: Union["SemanticFeatureType", None] = None
    semantic_dataset_type: Union["SemanticDatasetType", None] = None
    semantic_fusion_method: Union[Callable[[np.ndarray], np.ndarray], None] = None
    sem_des_to_rgb: Union[Callable[[np.ndarray, bool], np.ndarray], None] = None
    sem_img_to_rgb: Union[Callable[[np.ndarray, bool], np.ndarray], None] = None
    get_semantic_weight: Union[Callable[[np.ndarray], float], None] = None

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
        # Initialize the C++ module with the semantic mapping info
        SemanticMappingShared.init_cpp_module(semantic_mapping)

    @staticmethod
    def init_cpp_module(semantic_mapping: "SemanticMappingBase"):
        try:
            from pyslam.slam.cpp import cpp_module
            from pyslam.slam.cpp import CPP_AVAILABLE
            from pyslam.slam import USE_CPP

            if not CPP_AVAILABLE:
                return

            cpp_module.FeatureSharedResources.semantic_feature_type = (
                semantic_mapping.semantic_feature_type
            )

            # Set the semantic mapping info in the C++ module
            cpp_module.SemanticMappingSharedResources.semantic_feature_type = (
                semantic_mapping.semantic_feature_type
            )
            cpp_module.SemanticMappingSharedResources.semantic_dataset_type = (
                semantic_mapping.semantic_dataset_type
            )
            cpp_module.SemanticMappingSharedResources.semantic_fusion_method = (
                semantic_mapping.semantic_fusion_method
            )
            cpp_module.SemanticMappingSharedResources.semantic_entity_type = (
                semantic_mapping.semantic_entity_type
            )
            cpp_module.SemanticMappingSharedResources.semantic_segmentation_type = (
                semantic_mapping.semantic_segmentation_type
            )

            num_classes = semantic_mapping.semantic_segmentation.num_classes()
            cpp_module.SemanticMappingSharedResources.init_color_map(
                semantic_mapping.semantic_dataset_type,
                num_classes,
            )
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot set cpp_module: {e}")
            traceback.print_exc()
