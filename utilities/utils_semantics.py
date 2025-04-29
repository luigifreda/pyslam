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

from semantic_feature_types import SemanticFeatureTypes

# TODO(@dvdmc): add PCA for open vocab semantics
  
# create a scaled image of uint8 from a image of semantics
def labels_to_image(label_img, semantics_rgb_map, bgr=False, ignore_labels=[], rgb_image=None):
    """
    Converts a class label image to an RGB image.
    Args:
        label_img: 2D array of class labels.
        label_map: List or array of class RGB colors.
    Returns:
        rgb_output: RGB image as a NumPy array.
    """
    semantics_rgb_map = np.array(semantics_rgb_map, dtype=np.uint8)
    if bgr:
        semantics_rgb_map = semantics_rgb_map[:, ::-1]

    rgb_output = semantics_rgb_map[label_img]

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

def single_label_to_color(label, semantics_rgb_map, bgr=False):
    color = semantics_rgb_map[label]
    if bgr:
        color = color[::-1]
    return color

def labels_map_factory(dataset_name="cityscapes"):
    if dataset_name == "voc":
        return get_voc_labels()
    elif dataset_name == "cityscapes":
        return get_cityscapes_labels()
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
    
def get_voc_labels():
    """Load the mapping that associates pascal VOC classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    color_map = np.array(
        [
            [0, 0, 0],  # 0=background
            [0, 64, 0],  # 1=aeroplane # TEMPORAL CHANGE
            [0, 128, 0],  # 2=bicycle
            [128, 128, 0],  # 3=bird
            [0, 0, 128],  # 4=boat
            [128, 0, 128],  # 5=bottle
            [0, 128, 128],  # 6=bus
            [128, 128, 128],  # 7=car
            [64, 0, 0],  # 8=cat
            [192, 0, 0],  # 9=chair
            [64, 128, 0],  # 10=cow
            [192, 128, 0],  # 11=diningtable
            [64, 0, 128],  # 12=dog
            [192, 0, 128],  # 13=horse
            [64, 128, 128],  # 14=motorbike
            [192, 128, 128],  # 15=person
            [0, 64, 0],  # 16=potted plant
            [128, 64, 0],  # 17=sheep
            [0, 192, 0],  # 18=sofa
            [128, 192, 0],  # 19=train
            [0, 64, 128],  # 20=tv/monitor
        ]
    )
    return color_map

def get_cityscapes_labels():
    """Load the mapping that associates cityscapes classes with label colors
    Returns:
        np.ndarray with dimensions (19, 3)
    """
    color_map = np.array(
        [
        [128, 64, 128],    # 0=road
        [244, 35, 232],    # 1=sidewalk
        [70, 70, 70],      # 2=building
        [102, 102, 156],   # 3=wall
        [190, 153, 153],   # 4=fence
        [153, 153, 153],   # 5=pole
        [250, 170, 30],    # 6=traffic light
        [220, 220, 0],     # 7=traffic sign
        [107, 142, 35],    # 8=vegetation
        [152, 251, 152],   # 9=terrain
        [70, 130, 180],    # 10=sky
        [220, 20, 60],     # 11=person
        [255, 0, 0],       # 12=rider
        [0, 0, 142],       # 13=car
        [0, 0, 70],        # 14=truck
        [0, 60, 100],      # 15=bus
        [0, 80, 100],      # 16=train
        [0, 0, 230],       # 17=motorcycle
        [119, 11, 32],     # 18=bicycle
    ]
    )
    return color_map

def information_weights_factory(dataset_name="cityscapes"):
    if dataset_name == "voc":
        return get_voc_information_weights()
    elif dataset_name == "cityscapes":
        return get_cityscapes_information_weights()
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
    
def get_voc_information_weights():
    return [1.0, # aeroplane
            1.0, # bicycle
            1.0, # bird
            1.0, # boat
            1.0, # bottle
            1.0, # bus
            1.0, # car
            1.0, # cat
            1.0, # chair
            1.0, # cow
            1.0, # diningtable
            1.0, # dog
            1.0, # horse
            1.0, # motorbike
            1.0, # person
            1.0, # pottedplant
            1.0, # sheep
            1.0, # sofa
            1.0, # train
            1.0, # tvmonitor
            ]

def get_cityscapes_information_weights():
    return [1.0, # road
            1.0, # sidewalk
            1.0, # building
            1.0, # wall
            1.0, # fence
            1.0, # pole
            1.0, # traffic light
            1.0, # traffic sign
            0.001, # vegetation
            1.0, # terrain
            1.0, # sky
            0.04, # person
            0.002, # rider
            0.002, # car
            0.002, # truck
            0.002, # bus
            0.002, # train
            0.002, # motorcycle
            0.002, # bicycle
            ]