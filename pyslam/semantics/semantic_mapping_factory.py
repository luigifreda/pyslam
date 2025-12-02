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

from pyslam.config_parameters import Parameters
from .semantic_mapping_dense import SemanticMappingDense
from .semantic_mapping_dense_process import SemanticMappingDenseProcess
from .semantic_mapping_base import SemanticMappingType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame


def semantic_mapping_factory(slam: "Slam", headless=False, image_size=(512, 512), **kwargs):

    semantic_mapping_type = kwargs.get("semantic_mapping_type")
    if semantic_mapping_type is None:
        raise ValueError("semantic_mapping_type is not specified in semantic_mapping_config")

    if semantic_mapping_type == SemanticMappingType.DENSE:
        semantic_mapping_dense_class = SemanticMappingDense
        if Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess:
            semantic_mapping_dense_class = SemanticMappingDenseProcess
        return semantic_mapping_dense_class(
            slam=slam,
            headless=headless,
            image_size=image_size,
            semantic_segmentation_type=kwargs.get("semantic_segmentation_type"),
            semantic_dataset_type=kwargs.get("semantic_dataset_type"),
            semantic_feature_type=kwargs.get("semantic_feature_type"),
        )
    else:
        raise ValueError(f"Invalid semantic mapping type: {semantic_mapping_type}")
