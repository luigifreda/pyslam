"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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

import config
import torch
import platform

from utils_sys import import_from

from utils_serialization import SerializableEnum, register_class

from semantic_estimator_deep_lab_v3 import SemanticEstimatorDeepLabV3
from semantic_estimator_segformer import SemanticEstimatorSegformer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

@register_class
class SemanticEstimatorType(SerializableEnum):
    DEEPLABV3     = 0     # Semantics from torchvision DeepLab's v3
    SEGFORMER     = 1     # Semantics from transformer's Segformer
    @staticmethod
    def from_string(name: str):
        try:
            return SemanticEstimatorType[name]
        except KeyError:
            raise ValueError(f"Invalid SemanticEstimatorType: {name}")
        
def semantic_estimator_factory(semantic_estimator_type=SemanticEstimatorType.DEEPLABV3,
                               device=None):
    if semantic_estimator_type == SemanticEstimatorType.DEEPLABV3:
        return SemanticEstimatorDeepLabV3(device=device)
    elif semantic_estimator_type == SemanticEstimatorType.SEGFORMER:
        return SemanticEstimatorSegformer(device=device)
    else:
        raise ValueError(f'Invalid depth estimator type: {semantic_estimator_type}')