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

from .semantic_types import SemanticDatasetType


def semantic_information_weights_factory(
    semantic_dataset_type=SemanticDatasetType.CITYSCAPES, **kwargs
):
    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        if "num_classes" not in kwargs:
            raise ValueError("num_classes must be provided if semantic_dataset_type is CUSTOM_SET")
        return get_trivial_information_weights(kwargs["num_classes"])
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))


def get_voc_information_weights():
    return [
        1.0,  # aeroplane
        1.0,  # bicycle
        1.0,  # bird
        1.0,  # boat
        1.0,  # bottle
        1.0,  # bus
        1.0,  # car
        1.0,  # cat
        1.0,  # chair
        1.0,  # cow
        1.0,  # diningtable
        1.0,  # dog
        1.0,  # horse
        1.0,  # motorbike
        1.0,  # person
        1.0,  # pottedplant
        1.0,  # sheep
        1.0,  # sofa
        1.0,  # train
        1.0,  # tvmonitor
    ]


def get_cityscapes_information_weights():
    return [
        1.0,  # road
        1.0,  # sidewalk
        1.0,  # building
        1.0,  # wall
        1.0,  # fence
        1.0,  # pole
        1.0,  # traffic light
        1.0,  # traffic sign
        0.001,  # vegetation
        1.0,  # terrain
        1.0,  # sky
        1.0,  # person
        1.0,  # rider
        1.0,  # car
        1.0,  # truck
        1.0,  # bus
        1.0,  # train
        1.0,  # motorcycle
        1.0,  # bicycle
    ]


def get_ade20k_information_weights():
    return get_trivial_information_weights(num_classes=150)


def get_nyu40_information_weights():
    return get_trivial_information_weights(num_classes=41)


def get_trivial_information_weights(num_classes):
    return np.ones(num_classes)
