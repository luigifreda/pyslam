"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
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


class SemanticSegmentationOutput:
    """
    Container for semantic segmentation model inference results.

    Attributes:
        semantics: numpy array of shape (H, W) for LABEL, (H, W, num_classes) for PROBABILITY_VECTOR,
                   or (H, W, D) for FEATURE_VECTOR
        instances: numpy array of shape (H, W) for instance IDs (optional, None if not available)
    """

    def __init__(self, semantics=None, instances=None):
        self.semantics = semantics  # numpy array: (H, W) for LABEL, (H, W, num_classes) for PROBABILITY_VECTOR, or (H, W, D) for FEATURE_VECTOR
        self.instances = instances  # numpy array of shape (H, W) for INSTANCES (optional)
