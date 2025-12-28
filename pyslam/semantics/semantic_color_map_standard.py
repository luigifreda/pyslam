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

from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_color_map_base import SemanticColorMapBase
from .semantic_color_utils import labels_color_map_factory, need_large_color_map

from pyslam.utilities.logging import Printer


class SemanticColorMapStandard(SemanticColorMapBase):
    """
    Standard semantic color map implementation using factory-based color map creation.

    This class creates color maps using the labels_color_map_factory function,
    suitable for standard semantic segmentation models.
    """

    def __init__(
        self,
        semantic_dataset_type: SemanticDatasetType,
        semantic_feature_type: SemanticFeatureType,
        num_classes=None,
        text_embs=None,
        device="cpu",
        sim_scale=1.0,
        semantic_segmentation_type=None,
    ):
        """
        Initialize standard semantic color map.

        Args:
            semantic_dataset_type: Type of semantic dataset
            semantic_feature_type: Type of semantic feature (LABEL, PROBABILITY_VECTOR, FEATURE_VECTOR)
            num_classes: Number of classes (optional)
            text_embs: Text embeddings for feature-based models (optional)
            device: Device to use ("cpu" or "gpu")
            sim_scale: Scale factor for similarity visualization
            semantic_segmentation_type: Type of semantic segmentation model (optional)
        """
        # Initialize base class
        super().__init__(
            semantic_dataset_type=semantic_dataset_type,
            semantic_feature_type=semantic_feature_type,
            num_classes=num_classes,
            text_embs=text_embs,
            device=device,
            sim_scale=sim_scale,
            semantic_segmentation_type=semantic_segmentation_type,
        )

        # Use factory-based color map creation
        # Check if we need a large color map for open-vocabulary models (Detic/EOV-Seg)
        # These models output category IDs that can be much larger than standard dataset class counts
        needs_large_color_map, model_name = need_large_color_map(semantic_segmentation_type)

        # Determine the appropriate color map size
        # For open-vocabulary models, use a large color map (default 3000) regardless of num_classes
        # because category IDs can exceed num_classes (e.g., Detic outputs IDs like 295, 792, 1076)
        if needs_large_color_map:
            # Use a large color map for open-vocabulary models
            color_map_size = max(num_classes if num_classes is not None else 3000, 3000)
            if Printer:
                Printer.yellow(
                    f"SemanticColorMapStandard: Using large color map for {model_name} "
                    f"({color_map_size} classes) to handle open-vocabulary category IDs"
                )
            self.color_map = labels_color_map_factory(
                semantic_dataset_type,
                semantic_segmentation_type=semantic_segmentation_type,
                needs_large_color_map=True,
                num_classes=color_map_size,
            )
        else:
            # For standard models, use dataset-specific color map
            base_color_map = labels_color_map_factory(
                semantic_dataset_type,
                semantic_segmentation_type=semantic_segmentation_type,
            )

            # If the model outputs more classes than the dataset color map provides,
            # expand the color map to avoid index errors
            if num_classes is not None and num_classes > len(base_color_map):
                Printer.yellow(
                    f"SemanticColorMapStandard: expanding color map to {num_classes} classes "
                    f"(dataset map size: {len(base_color_map)})"
                )
                self.color_map = labels_color_map_factory(
                    semantic_dataset_type,
                    semantic_segmentation_type=semantic_segmentation_type,
                    needs_large_color_map=True,
                    num_classes=num_classes,
                )
            else:
                # Either we don't know num_classes or the dataset map is sufficient
                self.color_map = (
                    base_color_map if num_classes is None else base_color_map[:num_classes]
                )
