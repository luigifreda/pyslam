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
import cv2

from .semantic_types import SemanticDatasetType


from .semantic_labels import (
    get_ade20k_color_map,
    get_cityscapes_color_map,
    get_nyu40_color_map,
    get_voc_color_map,
    get_generic_color_map,
    get_open_vocab_color_map,
)
from .semantic_labels import (
    get_ade20k_labels,
    get_cityscapes_labels,
    get_nyu40_labels,
    get_voc_labels,
)
from .semantic_segmentation_types import SemanticSegmentationType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.logging import Printer


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
    sim_map = np.clip(sim_map * sim_scale, 0.0, 1.0)
    sim_map = ((1 - sim_map) * 255).astype(np.uint8)

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
    sim_point = np.clip(sim_point * sim_scale, 0.0, 1.0)
    sim_point = ((1 - sim_point) * 255).astype(np.uint8)
    sim_color = cv2.applyColorMap(sim_point, colormap)
    if bgr:
        sim_color = cv2.cvtColor(sim_color, cv2.COLOR_BGR2RGB)
    return sim_color[0][0]


# create a scaled image of uint8 from a image of semantics
def labels_to_image(label_img, semantic_color_map, bgr=False, ignore_labels=None, rgb_image=None):
    """
    Converts a class label image to an RGB image.
    Args:
        label_img: 2D array of class labels.
        label_map: List or array of class RGB colors.
    Returns:
        rgb_output: RGB image as a NumPy array.
    """
    semantic_color_map = np.array(semantic_color_map, dtype=np.uint8)
    if bgr:
        semantic_color_map = semantic_color_map[:, ::-1]

    # Clamp label values to valid range [0, num_classes-1] to prevent index errors
    # This handles cases where values might be corrupted during dtype conversion or multiprocessing
    label_img = np.asarray(label_img)
    num_classes = len(semantic_color_map)
    if not np.issubdtype(label_img.dtype, np.integer):
        label_img = label_img.astype(np.int64, copy=False)
    if label_img.size > 0 and (label_img.max() >= num_classes or label_img.min() < 0):
        Printer.red(
            f"labels_to_image: label_img has values out of range: {label_img.min()} - {label_img.max()}"
        )
        label_img = np.clip(label_img, 0, num_classes - 1)

    rgb_output = semantic_color_map[label_img]

    if ignore_labels is not None and len(ignore_labels) > 0:
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


def single_label_to_color(label, semantic_color_map, bgr=False):
    label = int(label)  # ensure label is a Python int
    color = semantic_color_map[label]
    if bgr:
        color = color[::-1]
    return color


def instance_ids_to_rgb(instance_ids, bgr=False, unlabeled_color=(0, 0, 0)):
    """
    Convert instance IDs to RGB colors using hash-based color table.

    This function is designed for instance IDs which can be arbitrary integers
    (including negative values like -1 for unlabeled). Unlike semantic class IDs,
    instance IDs don't have a fixed range and need a hash-based color mapping.

    Uses C++ implementation from color_utils module.

    Args:
        instance_ids: 1D or 2D numpy array of instance IDs (can include -1 for unlabeled)
        bgr: If True, return BGR format; otherwise RGB
        unlabeled_color: RGB tuple for unlabeled instances (default: black)

    Returns:
        RGB/BGR image array of shape (H, W, 3) for 2D input or (N, 3) for 1D input,
        with dtype uint8 and values in [0, 255]
    """
    # Handle None or empty arrays
    if instance_ids is None:
        return None

    instance_ids = np.asarray(instance_ids)
    if instance_ids.size == 0:
        # Return empty array with correct shape
        if len(instance_ids.shape) == 1:
            return np.zeros((0, 3), dtype=np.uint8)
        else:
            return np.zeros((*instance_ids.shape, 3), dtype=np.uint8)

    # Try to use C++ implementation from color_utils module
    try:
        import color_utils

        # Create an instance of IdsColorTable
        color_table = color_utils.IdsColorTable()
        # Convert unlabeled_color to cv::Vec3b format (BGR) as numpy array
        # unlabeled_color is in RGB format, convert to BGR and then to numpy array
        unlabeled_bgr = np.array(
            tuple(reversed(unlabeled_color)), dtype=np.uint8
        )  # RGB to BGR, then to numpy array
        result = color_table.ids_to_rgb(instance_ids, bgr=bgr, unlabeled_color=unlabeled_bgr)
        return result
    except (ImportError, AttributeError):
        # Fallback: raise error (no Python fallback anymore)
        raise RuntimeError("color_utils C++ module not available. Please rebuild the C++ modules.")


def need_large_color_map(semantic_segmentation_type: SemanticSegmentationType):
    """
    Check if a semantic segmentation type needs a large color map (open-vocabulary models).

    Open-vocabulary models (like EOV-Seg and Detic) output category IDs that can be much larger
    than standard dataset class counts, so they need large color maps (e.g., 3000 classes).

    Args:
        semantic_segmentation_type: SemanticSegmentationType enum value

    Returns:
        tuple: (needs_large_color_map: bool, model_name: str or None)
            - needs_large_color_map: True if the type needs a large color map
            - model_name: Human-readable model name if it needs large color map, None otherwise
    """
    if semantic_segmentation_type is None:
        return False, None

    # Direct comparison with enum values (most reliable and type-safe)

    if semantic_segmentation_type == SemanticSegmentationType.EOV_SEG:
        return True, "EOV-Seg"
    elif semantic_segmentation_type == SemanticSegmentationType.DETIC:
        return True, "Detic"

    return False, None


# We map from SLAM datasets to semantic datasets
def labels_color_map_factory(
    semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
    semantic_segmentation_type=None,
    needs_large_color_map=None,
    **kwargs,
):
    """
    Factory function to get the appropriate color map for semantic segmentation.

    Args:
        semantic_dataset_type: The semantic dataset type (CITYSCAPES, ADE20K, etc.)
        semantic_segmentation_type: Optional semantic segmentation type. If EOV_SEG or DETIC is specified,
                                    uses a large color map suitable for open-vocabulary models.
                                    Ignored if needs_large_color_map is explicitly provided.
        needs_large_color_map: Optional explicit flag to request a large color map (for open-vocabulary models).
                              If None, automatically determined from semantic_segmentation_type.
                              If True, uses a large color map (3000 classes by default).
        **kwargs: Additional arguments (e.g., num_classes for CUSTOM_SET or for large color maps)

    Returns:
        np.ndarray: Color map array of shape (num_classes, 3)
    """
    # Determine if we need a large color map
    # If explicitly provided, use it; otherwise check from semantic_segmentation_type
    model_name = None
    if needs_large_color_map is None and semantic_segmentation_type is not None:
        needs_large_color_map, model_name = need_large_color_map(semantic_segmentation_type)
    elif needs_large_color_map is True and semantic_segmentation_type is not None:
        # If explicitly set to True, still try to get model name for better debug output
        _, model_name = need_large_color_map(semantic_segmentation_type)

    # If EOV-Seg or Detic is being used, use the large color map regardless of dataset type
    # This is because open-vocabulary models output category IDs that can be much larger than standard
    # dataset class counts (e.g., 1432 vs 150 for ADE20K, or 1203 for LVIS)
    if needs_large_color_map:
        # Debug: show what we received
        try:
            from pyslam.utilities.logging import Printer

            if semantic_segmentation_type is not None:
                seg_type_str = (
                    semantic_segmentation_type.name
                    if hasattr(semantic_segmentation_type, "name")
                    else str(semantic_segmentation_type)
                )
                Printer.yellow(
                    f"labels_color_map_factory: semantic_segmentation_type={seg_type_str}, "
                    f"needs_large_color_map={needs_large_color_map}"
                )
            else:
                Printer.yellow(
                    f"labels_color_map_factory: needs_large_color_map={needs_large_color_map} (explicitly set)"
                )
        except:
            pass

        num_classes = kwargs.get("num_classes", 3000)
        color_map = get_open_vocab_color_map(num_classes=num_classes)
        # Debug output
        try:
            from pyslam.utilities.logging import Printer

            display_name = model_name if model_name else "Open-vocabulary model"
            Printer.green(
                f"labels_color_map_factory: Using {display_name} color map with {len(color_map)} classes (requested {num_classes})"
            )
        except:
            display_name = model_name if model_name else "Open-vocabulary model"
            print(f"DEBUG: Using {display_name} color map with {len(color_map)} classes")
        return color_map

    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_color_map()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_color_map()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_color_map()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_color_map()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        if "num_classes" not in kwargs:
            raise ValueError("num_classes must be provided if semantic_dataset_type is CUSTOM_SET")
        return get_generic_color_map(kwargs["num_classes"])
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))
