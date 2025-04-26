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

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

# Base class for semantic estimators via inference
class SemanticEstimator:
    def __init__(self, model, transform, device, semantics_rgb_map):
        self.model = model
        self.transform = transform
        self.device = device
        self.semantics_rgb_map = semantics_rgb_map

        self.semantics = None

    # Return the predicted semantic map
    def infer(self, image):
        raise NotImplementedError