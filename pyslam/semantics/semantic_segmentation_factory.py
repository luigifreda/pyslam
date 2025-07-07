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

import pyslam.config as config
import torch
import platform

from .semantic_types import SemanticFeatureType
from .semantic_utils import SemanticDatasetType

from pyslam.utilities.utils_sys import Printer, import_from
from pyslam.utilities.utils_serialization import SerializableEnum, register_class

try:
    from .semantic_segmentation_base import SemanticSegmentationBase
    from .semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3
    from .semantic_segmentation_segformer import SemanticSegmentationSegformer
    from .semantic_segmentation_clip import SemanticSegmentationCLIP
except ModuleNotFoundError:
    SemanticSegmentationBase = import_from('pyslam.semantics.semantic_segmentation_base', 'SemanticSegmentationBase')
    SemanticSegmentationDeepLabV3 = import_from('pyslam.semantics.semantic_segmentation_deep_lab_v3', 'SemanticSegmentationDeepLabV3')
    SemanticSegmentationSegformer = import_from('pyslam.semantics.semantic_segmentation_segformer', 'SemanticSegmentationSegformer')
    SemanticSegmentationCLIP = import_from('pyslam.semantics.semantic_segmentation_clip', 'SemanticSegmentationCLIP')


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'


@register_class
class SemanticSegmentationType(SerializableEnum):
    DEEPLABV3     = 0     # Semantics from torchvision DeepLab's v3
    SEGFORMER     = 1     # Semantics from transformer's Segformer
    CLIP          = 2     # Semantics from CLIP's segmentation head
    @staticmethod
    def from_string(name: str):
        try:
            return SemanticSegmentationType[name]
        except KeyError:
            raise ValueError(f"Invalid SemanticSegmentationType: {name}")
        
#TODO(dvdmc): Not all properties of the semantic segmentation modules are exposed (e.g. encoder_name, sim_text_query).
def semantic_segmentation_factory(semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                               semantic_feature_type=SemanticFeatureType.LABEL,
                               semantic_dataset_type=SemanticDatasetType.CITYSCAPES, image_size=(512, 512), device=None):
    Printer.green(f"Initializing semantic segmentation: {semantic_segmentation_type}, {semantic_feature_type}, {semantic_dataset_type}, {image_size}")
    if semantic_segmentation_type == SemanticSegmentationType.DEEPLABV3:
        return SemanticSegmentationDeepLabV3(device=device, semantic_dataset_type=semantic_dataset_type, image_size=image_size, semantic_feature_type=semantic_feature_type)
    elif semantic_segmentation_type == SemanticSegmentationType.SEGFORMER:
        return SemanticSegmentationSegformer(device=device, semantic_dataset_type=semantic_dataset_type, image_size=image_size, semantic_feature_type=semantic_feature_type)
    elif semantic_segmentation_type == SemanticSegmentationType.CLIP:
        return SemanticSegmentationCLIP(device=device, semantic_dataset_type=semantic_dataset_type, image_size=image_size, semantic_feature_type=semantic_feature_type)
    else:
        raise ValueError(f'Invalid semantic segmentation type: {semantic_segmentation_type}')