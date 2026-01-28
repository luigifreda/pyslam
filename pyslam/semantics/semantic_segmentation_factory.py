"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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

from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_segmentation_types import SemanticSegmentationType

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from

try:
    from .semantic_segmentation_base import SemanticSegmentationBase
    from .semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3
    from .semantic_segmentation_segformer import SemanticSegmentationSegformer
    from .semantic_segmentation_clip import SemanticSegmentationCLIP
    from .semantic_segmentation_eov_seg import SemanticSegmentationEovSeg
    from .semantic_segmentation_detic import SemanticSegmentationDetic
    from .semantic_segmentation_odise import SemanticSegmentationOdise
    from .semantic_segmentation_rf_detr import SemanticSegmentationRfDetr
    from .semantic_segmentation_yolo import SemanticSegmentationYolo
except ModuleNotFoundError:
    SemanticSegmentationBase = import_from(
        "pyslam.semantics.semantic_segmentation_base", "SemanticSegmentationBase"
    )
    SemanticSegmentationDeepLabV3 = import_from(
        "pyslam.semantics.semantic_segmentation_deep_lab_v3", "SemanticSegmentationDeepLabV3"
    )
    SemanticSegmentationSegformer = import_from(
        "pyslam.semantics.semantic_segmentation_segformer", "SemanticSegmentationSegformer"
    )
    SemanticSegmentationCLIP = import_from(
        "pyslam.semantics.semantic_segmentation_clip", "SemanticSegmentationCLIP"
    )
    SemanticSegmentationEovSeg = import_from(
        "pyslam.semantics.semantic_segmentation_eov_seg", "SemanticSegmentationEovSeg"
    )
    SemanticSegmentationDetic = import_from(
        "pyslam.semantics.semantic_segmentation_detic", "SemanticSegmentationDetic"
    )
    SemanticSegmentationOdise = import_from(
        "pyslam.semantics.semantic_segmentation_odise", "SemanticSegmentationOdise"
    )
    SemanticSegmentationRfDetr = import_from(
        "pyslam.semantics.semantic_segmentation_rf_detr", "SemanticSegmentationRfDetr"
    )
    SemanticSegmentationYolo = import_from(
        "pyslam.semantics.semantic_segmentation_yolo", "SemanticSegmentationYolo"
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.semantics.semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3
    from pyslam.semantics.semantic_segmentation_segformer import SemanticSegmentationSegformer
    from pyslam.semantics.semantic_segmentation_clip import SemanticSegmentationCLIP
    from pyslam.semantics.semantic_segmentation_eov_seg import SemanticSegmentationEovSeg
    from pyslam.semantics.semantic_segmentation_detic import SemanticSegmentationDetic
    from pyslam.semantics.semantic_segmentation_odise import SemanticSegmentationOdise
    from pyslam.semantics.semantic_segmentation_rf_detr import SemanticSegmentationRfDetr
    from pyslam.semantics.semantic_segmentation_yolo import SemanticSegmentationYolo


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


def semantic_segmentation_factory(
    semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
    semantic_feature_type=SemanticFeatureType.LABEL,
    semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
    image_size=(512, 512),
    device=None,
    encoder_name=None,
    **kwargs,
):
    Printer.green(
        f"Initializing semantic segmentation: {semantic_segmentation_type}, {semantic_feature_type}, {semantic_dataset_type}, {image_size}"
    )
    if semantic_segmentation_type == SemanticSegmentationType.DEEPLABV3:
        # encoder_name defaults to "resnet50" for DeepLabV3
        encoder_name = encoder_name if encoder_name else "resnet50"
        return SemanticSegmentationDeepLabV3(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            encoder_name=encoder_name,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.SEGFORMER:
        # encoder_name defaults to "b0" for Segformer
        encoder_name = encoder_name if encoder_name else "b0"
        return SemanticSegmentationSegformer(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            encoder_name=encoder_name,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.CLIP:
        # encoder_name defaults to "ViT-L/14@336px" for CLIP
        encoder_name = encoder_name if encoder_name else "ViT-L/14@336px"
        return SemanticSegmentationCLIP(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            encoder_name=encoder_name,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.EOV_SEG:
        # For EOV-Seg, config_file and model_weights can be passed via kwargs
        return SemanticSegmentationEovSeg(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.DETIC:
        # For Detic, config_file, model_weights, vocabulary, etc. can be passed via kwargs
        return SemanticSegmentationDetic(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.ODISE:
        # For ODISE, config_file, model_weights, vocab, label_list, etc. can be passed via kwargs
        return SemanticSegmentationOdise(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.RFDETR:
        # For RF-DETR, model_variant, weights_path, weights_name, etc. can be passed via kwargs
        return SemanticSegmentationRfDetr(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            **kwargs,
        )
    elif semantic_segmentation_type == SemanticSegmentationType.YOLO:
        # For YOLO, model_name, weights_path, model_url, etc. can be passed via kwargs
        return SemanticSegmentationYolo(
            device=device,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            semantic_feature_type=semantic_feature_type,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid semantic segmentation type: {semantic_segmentation_type}")
