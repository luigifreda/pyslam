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
from enum import Enum

from . import semantic_types
from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Logging

from pyslam.slam import KeyFrame

from .semantic_segmentation_output import SemanticSegmentationOutput

kVerbose = True

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame


class PerceptionTaskType(Enum):
    NONE = 0
    SEMANTIC_SEGMENTATION = 1
    INSTANCE_SEGMENTATION = 2
    SEMANTIC_OBJECT_DETECTION = 3
    INSTANCE_OBJECT_DETECTION = 4
    LOAD = 5
    SAVE = 6


# keyframe (picklable) data that are needed for loop detection
class PerceptionKeyframeData:
    def __init__(self, keyframe: KeyFrame = None, img=None):
        # keyframe data
        self.id = keyframe.id if keyframe is not None else -1
        self.img = img if img is not None else (keyframe.img if keyframe is not None else None)

        # NOTE: The kid is not actually used for the processing in this whole file
        if isinstance(keyframe, KeyFrame):
            self.kid = keyframe.kid if keyframe is not None else -1
        else:
            self.kid = -1


class PerceptionTask:
    def __init__(
        self,
        keyframe: KeyFrame,
        img,
        task_type=PerceptionTaskType.NONE,
        load_save_path=None,
    ):
        self.task_type = task_type
        self.keyframe_data = PerceptionKeyframeData(keyframe, img)
        self.load_save_path = load_save_path

    def __str__(self) -> str:
        return f"SemanticSegmentationTask: img id = {self.keyframe_data.id}, kid = {self.keyframe_data.kid}, task_type = {self.task_type.name}"


class PerceptionOutput:
    def __init__(
        self,
        task_type,
        inference_output: SemanticSegmentationOutput | None = None,
        inference_color_image=None,
        frame_id=None,
        img=None,
    ):
        self.task_type = task_type
        self.frame_id = frame_id
        self.img = img  # for debugging
        self.inference_output = inference_output
        self.inference_color_image = inference_color_image

    def __str__(self) -> str:
        if self.inference_output is not None:
            if (
                hasattr(self.inference_output, "semantics")
                and self.inference_output.semantics is not None
            ):
                inference_output_str = f"inference_output = {self.inference_output.semantics.shape}"
            elif (
                hasattr(self.inference_output, "instances")
                and self.inference_output.instances is not None
            ):
                inference_output_str = f"inference_output = {self.inference_output.instances.shape}"
            else:
                inference_output_str = f"inference_output type: {type(self.inference_output)}"
        else:
            inference_output_str = "inference_output = None"
        return f"PerceptionOutput: task_type = {self.task_type.name}, {inference_output_str}, frame_id = {self.frame_id}"
