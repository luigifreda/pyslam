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
from .semantic_feature_processing import SemanticFeatureProcessing
from .semantic_types import SemanticDatasetType
from .semantic_utils import (
    labels_color_map_factory,
    single_label_to_color,
    similarity_heatmap_point,
    similarity_heatmap_image,
    labels_to_image,
)

kVerbose = True
kTimerVerbose = False

kSemanticMappingSleepTime = 5e-3  # [s]


# This class is used to manage the color map for semantic mapping. It is used by SemanticSegmentationProcess.
class SemanticMappingColorMap:
    def __init__(
        self,
        semantic_dataset_type: SemanticDatasetType,
        semantic_feature_type: SemanticFeatureType,
        num_classes=None,
        text_embs=None,
        device="cpu",
        sim_scale=1.0,
    ):
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_feature_type = semantic_feature_type
        self.num_classes = num_classes
        self.text_embs = text_embs
        self.device = device
        self.sim_scale = sim_scale
        if num_classes is None:
            self.color_map = labels_color_map_factory(semantic_dataset_type)
        else:
            self.color_map = labels_color_map_factory(
                semantic_dataset_type, num_classes=num_classes
            )
        if semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.color_map = None
        # check consistency of text_embs and device
        if device == "cpu":
            if isinstance(text_embs, torch.Tensor):
                text_embs = text_embs.cpu().detach().numpy()
        elif device == "gpu":
            if isinstance(text_embs, np.ndarray):
                text_embs = torch.from_numpy(text_embs).to(device)
        else:
            raise ValueError("Invalid device: {device}")

    def to_rgb(self, semantics, bgr=False):
        # Check if input is an array (image) or scalar (single value)
        # Arrays have ndim >= 1 and size > 1 (or are 1D+ arrays)
        is_array = isinstance(semantics, np.ndarray) and (
            semantics.ndim >= 2 or (semantics.ndim == 1 and semantics.size > 1)
        )

        if is_array:
            # For arrays (images), use sem_img_to_rgb which handles images properly
            return self.sem_img_to_rgb(semantics, bgr=bgr)
        else:
            # For scalars (single values), use the original single_label_to_color logic
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                return similarity_heatmap_point(semantics, colormap=cv2.COLORMAP_JET, bgr=bgr)
            else:
                return single_label_to_color(semantics, self.color_map, bgr=bgr)

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return single_label_to_color(semantic_des, self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return single_label_to_color(np.argmax(semantic_des, axis=-1), self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                # transform features to similarities
                sims = SemanticFeatureProcessing.features_to_sims(
                    semantic_des, self.text_embs, self.device
                )
                return similarity_heatmap_point(
                    sims,
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                label = SemanticFeatureProcessing.features_to_labels(
                    semantic_des, self.text_embs, self.device
                )
                return single_label_to_color(label, self.color_map, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantic_img, self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(np.argmax(semantic_img, axis=-1), self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Transform semantic to tensor
            # TODO(dvdmc): check if doing these operations (and functions below) in CPU is more efficient (it probably is)
            semantics = torch.from_numpy(semantic_img).to(self.device)
            # Convert text_embs to tensor if needed
            if isinstance(self.text_embs, np.ndarray):
                text_embs = torch.from_numpy(self.text_embs).to(self.device)
            else:
                text_embs = self.text_embs.to(self.device)
            # Compute similarity
            sims = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                # For FEATURE_SIMILARITY, take maximum similarity across all text embeddings
                sim_map = sims.max(dim=-1)[0]  # (H, W)
                return similarity_heatmap_image(
                    sim_map.cpu().detach().numpy(),
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                pred = sims.argmax(dim=-1)
                return labels_to_image(pred.cpu().detach().numpy(), self.color_map, bgr=bgr)
