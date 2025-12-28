"""
* This file is part of PYSLAM
*
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

import cv2
import numpy as np
import torch

from .semantic_types import SemanticFeatureType
from .semantic_types import SemanticDatasetType
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_color_map_base import SemanticColorMapBase
from .semantic_color_map_detectron2 import SemanticColorMapDetectron2
from .semantic_color_map_standard import SemanticColorMapStandard
from .semantic_color_utils import labels_color_map_factory, labels_to_image


from pyslam.utilities.logging import Printer

kVerbose = True
kTimerVerbose = False

kSemanticMappingSleepTime = 5e-3  # [s]


# ============================================================================
# Factory Function
# ============================================================================


def semantic_color_map_factory(
    semantic_dataset_type: SemanticDatasetType,
    semantic_feature_type: SemanticFeatureType,
    num_classes=None,
    text_embs=None,
    device="cpu",
    sim_scale=1.0,
    semantic_segmentation_type=None,
    metadata=None,
):
    """
    Factory function to create the appropriate semantic color map instance.

    Args:
        semantic_dataset_type: Type of semantic dataset
        semantic_feature_type: Type of semantic feature (LABEL, PROBABILITY_VECTOR, FEATURE_VECTOR)
        num_classes: Number of classes (optional)
        text_embs: Text embeddings for feature-based models (optional)
        device: Device to use ("cpu" or "gpu")
        sim_scale: Scale factor for similarity visualization
        semantic_segmentation_type: Type of semantic segmentation model (optional)
        metadata: Optional detectron2 MetadataCatalog object. If provided, creates Detectron2ColorMap.

    Returns:
        SemanticColorMapBase: Appropriate semantic color map instance
    """
    # If metadata is provided, try to create detectron2-based color map
    if metadata is not None:
        try:
            return SemanticColorMapDetectron2(
                semantic_dataset_type=semantic_dataset_type,
                semantic_feature_type=semantic_feature_type,
                metadata=metadata,
                num_classes=num_classes,
                text_embs=text_embs,
                device=device,
                sim_scale=sim_scale,
                semantic_segmentation_type=semantic_segmentation_type,
            )
        except Exception as e:
            # Fallback to standard color map on error
            Printer.yellow(
                f"semantic_color_map_factory: Failed to create Detectron2ColorMap: {e}, "
                f"falling back to StandardColorMap"
            )
            # Fall through to standard implementation

    # Use standard factory-based color map
    return SemanticColorMapStandard(
        semantic_dataset_type=semantic_dataset_type,
        semantic_feature_type=semantic_feature_type,
        num_classes=num_classes,
        text_embs=text_embs,
        device=device,
        sim_scale=sim_scale,
        semantic_segmentation_type=semantic_segmentation_type,
    )
