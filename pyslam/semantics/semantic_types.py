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

from pyslam.utilities.serialization import SerializableEnum, register_class

"""
NOTES:
In order to add a new SEMANTIC representation:
- add a new enum in SemanticFeatureType
- configure it in your semantic_segmentation*
- adds its usage in the semantic_mapping* class that you want to use
"""


@register_class
class SemanticFeatureType(SerializableEnum):
    NONE = -1
    LABEL = 0  # [1] One value with the categorical label of the class
    PROBABILITY_VECTOR = 1  # [N] A vector of distribution parameters (categorical or Dirichlet) over N categorical classes
    FEATURE_VECTOR = (
        2  # [D] A feature vector from an encoder (e.g., CLIP or DiNO) with D dimensions
    )


@register_class
class SemanticEntityType(SerializableEnum):
    POINT = 0  # The semantics are associated to each point
    OBJECT = 1  # The semantics are associated to each object


@register_class
class SemanticDatasetType(SerializableEnum):
    CITYSCAPES = 0  # Cityscapes dataset (#classes: 19)
    ADE20K = 1  # ADE20K dataset (#classes: 150)
    VOC = 2  # VOC dataset (#classes: 21)
    NYU40 = 3  # NYU40 dataset (#classes: 41)
    FEATURE_SIMILARITY = 4  # Feature similarity dataset
    CUSTOM_SET = 5  # Custom set dataset (#classes: custom)


# See also
# pyslam/semantics/semantic_segmentation_types.py
