"""
* This file is part of PYSLAM
*
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
import cv2

from semantic_labels import get_ade20k_color_map, get_cityscapes_color_map, get_nyu40_color_map, get_voc_color_map, get_generic_color_map
from semantic_labels import get_ade20k_labels, get_cityscapes_labels, get_nyu40_labels, get_voc_labels
from utils_serialization import SerializableEnum, register_class


def similarity_heatmap_image(sim_map, colormap=cv2.COLORMAP_JET, sim_scale=1.0, bgr=False):
    """
    Transforms a similarity map to a visual RGB image using a colormap.

    Args:
        sim_map (np.ndarray): Similarity image of shape (H, W)
        colormap (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)

    Returns:
        np.ndarray: RGB image (H, W, 3) visualizing similarity
    """
    # Normalize to 0–255 for colormap (skip min-max if values are already in [0,1])
    sim_map = np.clip(sim_map*sim_scale, 0.0, 1.0)
    sim_map = ((1-sim_map) * 255).astype(np.uint8)

    # Apply colormap and convert to RGB
    sim_color = cv2.applyColorMap(sim_map, colormap)
    if bgr:
        sim_color = cv2.cvtColor(sim_color, cv2.COLOR_BGR2RGB)
    return sim_color


def similarity_heatmap_point(sim_point, colormap=cv2.COLORMAP_JET, sim_scale=1.0, bgr=False):
    """
    Generates a similarity color for a single point.

    Args:
        sim_point (np.ndarray): Similarity of point (0.0-1.0)
        colormap (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)

    Returns:
        np.ndarray: RGB image (H, W, 3) visualizing similarity
    """
    sim_point = np.clip(sim_point*sim_scale, 0.0, 1.0)
    sim_point = ((1-sim_point) * 255).astype(np.uint8)
    sim_color = cv2.applyColorMap(sim_point, colormap)
    if bgr:
        sim_color = cv2.cvtColor(sim_color, cv2.COLOR_BGR2RGB)
    return sim_color[0][0]


# create a scaled image of uint8 from a image of semantics
def labels_to_image(
    label_img, semantics_color_map, bgr=False, ignore_labels=[], rgb_image=None
):
    """
    Converts a class label image to an RGB image.
    Args:
        label_img: 2D array of class labels.
        label_map: List or array of class RGB colors.
    Returns:
        rgb_output: RGB image as a NumPy array.
    """
    semantics_color_map = np.array(semantics_color_map, dtype=np.uint8)
    if bgr:
        semantics_color_map = semantics_color_map[:, ::-1]

    rgb_output = semantics_color_map[label_img]

    if len(ignore_labels) > 0:
        if rgb_image is None:
            raise ValueError("rgb_image must be provided if ignore_labels is not empty")
        else:
            mask = np.isin(label_img, ignore_labels)
            rgb_output[mask] = rgb_image[mask]
    return rgb_output


def rgb_to_class(rgb_labels, label_map):
    """
    Converts an RGB label image to a class label image.
    Args:
        rgb_labels: Input RGB image as a NumPy array.
        label_map: List or array of class RGB colors.
    Returns:
        class_image: 2D array of class labels.
    """
    rgb_np = np.array(rgb_labels, dtype=np.uint8)[:, :, :3]
    label_map = np.array(label_map, dtype=np.uint8)

    reshaped = rgb_np.reshape(-1, 3)
    class_image = np.zeros(reshaped.shape[0], dtype=np.uint8)

    # Create a LUT for color matching
    for class_label, class_color in enumerate(label_map):
        matches = np.all(reshaped == class_color, axis=1)
        class_image[matches] = class_label

    return class_image.reshape(rgb_np.shape[:2])


def single_label_to_color(label, semantics_color_map, bgr=False):
    label = int(label)  # ensure label is a Python int
    color = semantics_color_map[label]
    if bgr:
        color = color[::-1]
    return color


@register_class
class SemanticDatasetType(SerializableEnum):
    CITYSCAPES = 0
    ADE20K = 1
    VOC = 2
    NYU40 = 3
    FEATURE_SIMILARITY = 4
    CUSTOM_SET = 5


# We map from SLAM datasets to semantic datasets
def labels_color_map_factory(semantic_dataset_type=SemanticDatasetType.CITYSCAPES, **kwargs):
    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_color_map()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_color_map()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_color_map()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_color_map()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        if 'num_classes' not in kwargs:
            raise ValueError("num_classes must be provided if semantic_dataset_type is CUSTOM_SET")
        return get_generic_color_map(kwargs['num_classes'])
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))

def labels_name_factory(semantic_dataset_type=SemanticDatasetType.CITYSCAPES):
    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_labels()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_labels()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_labels()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_labels()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        raise ValueError("CUSTOM_SET does not have predefined labels")
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))

def information_weights_factory(semantic_dataset_type=SemanticDatasetType.CITYSCAPES, **kwargs):
    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_information_weights()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        if 'num_classes' not in kwargs:
            raise ValueError("num_classes must be provided if semantic_dataset_type is CUSTOM_SET")
        return get_trivial_information(kwargs['num_classes'])
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
    return get_trivial_information(num_classes=150)

def get_nyu40_information_weights():
    return get_trivial_information(num_classes=41)

def get_trivial_information(num_classes):
    return np.ones(num_classes)