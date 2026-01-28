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
import multiprocessing as mp
from enum import Enum

from . import semantic_types
from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Logging, LoggerQueue
from pyslam.utilities.file_management import create_folder

from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_color_utils import instance_ids_to_rgb

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


kVerbose = True

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame


# Base class for semantic estimators via inference
class SemanticSegmentationBase:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
    logging_manager, logger = None, None

    def __init__(self, model, transform, device, semantic_feature_type):
        self.model = model
        self.transform = transform
        self.device = device
        self.semantic_feature_type = semantic_feature_type
        self.semantics = None  # predicted semantic labels
        self.semantic_instances = None  # predicted instance IDs
        self.init_print()

    def infer(self, image) -> SemanticSegmentationOutput:
        """
        Run semantic segmentation inference on an image.
        Args:
            image: numpy array of shape (H, W, 3) in BGR format (OpenCV format)
        Returns:
            SemanticSegmentationOutput: object containing semantics and optionally instances
        """
        raise NotImplementedError

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        """
        Convert semantic predictions to a visualization RGB/BGR image.

        This method is intended for visualization-only outputs and may include
        overlays/annotations depending on the backend implementation.
        """
        return NotImplementedError

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        """
        Convert semantic predictions to an RGB/BGR image suitable for processing.

        This should be a pure color-map conversion (no overlays/annotations).
        """
        return NotImplementedError

    def instances_to_rgb(self, instances, bgr=False):
        """
        Convert instance IDs to RGB visualization using hash-based color table.

        This method is designed for instance IDs which can be arbitrary integers
        (including negative values like -1 for unlabeled). Unlike semantic class IDs,
        instance IDs don't have a fixed range and need a hash-based color mapping.

        Args:
            instances: numpy array of shape (H, W) with instance IDs, or None
            bgr: If True, return BGR format; otherwise RGB

        Returns:
            RGB/BGR image array of shape (H, W, 3), or None if instances is None
        """
        return instance_ids_to_rgb(instances, bgr=bgr)

    def num_classes(self):
        return NotImplementedError

    def features_to_sims(self, semantics):
        return NotImplementedError

    def features_to_labels(self, semantics):
        return NotImplementedError

    def init_print(self):
        if kVerbose:
            if Parameters.kSemanticMappingDebugAndPrintToFile:
                # redirect the prints of semantic segmentation to the file logs/semantic_segmentation.log (by default)
                # you can watch the output in separate shell by running:
                # $ tail -f logs/semantic_segmentation.log

                logging_file = Parameters.kLogsFolder + "/semantic_segmentation.log"
                create_folder(logging_file)
                if SemanticSegmentationBase.logging_manager is None:
                    # Note: Each process has its own memory space, so singleton pattern works per-process
                    SemanticSegmentationBase.logging_manager = LoggerQueue.get_instance(
                        logging_file
                    )
                    SemanticSegmentationBase.logger = (
                        SemanticSegmentationBase.logging_manager.get_logger(
                            "semantic_segmentation_logger"
                        )
                    )

                def print_file(*args, **kwargs):
                    try:
                        if SemanticSegmentationBase.logger is not None:
                            message = " ".join(
                                str(arg) for arg in args
                            )  # Convert all arguments to strings and join with spaces
                            return SemanticSegmentationBase.logger.info(message, **kwargs)
                    except:
                        print("Error printing: ", args, kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return print(message, **kwargs)

            SemanticSegmentationBase.print = staticmethod(print_file)
