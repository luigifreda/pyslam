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


# Panoptic/instance: DETIC, ODISE, EOV_SEG
# Semantic only: DEEPLABV3, SEGFORMER, CLIP
@register_class
class SemanticSegmentationType(SerializableEnum):
    DEEPLABV3 = 0  # Semantics from torchvision DeepLab's v3 [Semantic only:]
    SEGFORMER = 1  # Semantics from transformer's Segformer [Semantic only]
    CLIP = 2  # Semantics from CLIP's segmentation head [Semantic only]
    EOV_SEG = (
        3  # Semantics from EOV-Seg (Efficient Open Vocabulary Segmentation) [Panoptic/instance]
    )
    DETIC = 4  # Semantics from Detic (Detecting Twenty-thousand Classes) [Panoptic/instance]
    ODISE = 5  # Semantics from ODISE (Open-vocabulary DIffusion-based panoptic SEgmentation) [Panoptic/instance]

    @staticmethod
    def from_string(name: str):
        try:
            return SemanticSegmentationType[name]
        except KeyError:
            raise ValueError(f"Invalid SemanticSegmentationType: {name}")
