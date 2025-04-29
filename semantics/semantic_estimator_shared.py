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

import semantic_feature_types

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

# Base class for semantic estimators via inference
class SemanticEstimatorShared:
    
    semantic_estimator           = None
    semantic_feature_type        = None
    semantics_rgb_map            = None
    fuse_semantic_descriptors    = None

    @staticmethod
    def set_semantic_estimator(self, semantic_estimator, force=False):

        if not force and SemanticEstimatorShared.semantic_estimator is not None:
            raise Exception("SemanticEstimatorShared: Semantic Estimator is already set!")
        
        SemanticEstimatorShared.semantic_estimator          = semantic_estimator
        SemanticEstimatorShared.semantic_feature_type       = semantic_estimator.semantic_feature_type
        SemanticEstimatorShared.semantics_rgb_map           = semantic_estimator.semantics_rgb_map
        SemanticEstimatorShared.semantic_fusion_method      = semantic_estimator.semantic_fusion_method