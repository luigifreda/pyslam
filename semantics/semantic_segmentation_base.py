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

import semantic_types

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

# Base class for semantic estimators via inference
class SemanticSegmentationBase:
    def __init__(self, model, transform, device, semantic_feature_type):
        self.model = model
        self.transform = transform
        self.device = device
        self.semantic_feature_type = semantic_feature_type

        self.semantics = None

    def infer(self, image):
        raise NotImplementedError
    
    # TODO(dvdmc): test if this works directly for transforming a single label
    def to_rgb(self, semantics, bgr=False):
        return NotImplementedError