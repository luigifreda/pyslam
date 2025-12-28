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
        self.semantics = None
        self.init_print()

    # Run semantic segmentation inference on an image.
    # Args:
    #     image: numpy array of shape (H, W, 3) in BGR format (OpenCV format)
    # Returns:
    #     SemanticSegmentationOutput: object containing semantics and optionally instances
    def infer(self, image) -> SemanticSegmentationOutput:
        raise NotImplementedError

    def to_rgb(self, semantics, bgr=False):
        return NotImplementedError

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
