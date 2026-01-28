#!/usr/bin/env -S python3 -O
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
import os
import sys


from pyslam.config import Config

config = Config()

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.utilities.system import getchar
from pyslam.utilities.logging import Printer
from pyslam.utilities.img_management import (
    float_to_color,
    convert_float_to_colored_uint8_image,
    LoopCandidateImgs,
    ImgWriter,
)

import math
import cv2
import numpy as np

from pyslam.io.dataset_factory import dataset_factory
from pyslam.slam.frame import Frame, FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.semantics.semantic_segmentation_factory import semantic_segmentation_factory
from pyslam.semantics.semantic_segmentation_types import SemanticSegmentationType
from pyslam.semantics.semantic_color_utils import labels_color_map_factory
from pyslam.semantics.semantic_types import SemanticFeatureType, SemanticDatasetType

from pyslam.config_parameters import Parameters


Parameters.kLoopClosingDebugAndPrintToFile = False
Parameters.kLoopClosingDebugWithSimmetryMatrix = True
Parameters.kLoopClosingDebugWithLoopDetectionImages = True

check_disconnected_instances = True
check_output = True


# basic output checks
def check_output_shapes(semantics, instances, image_shape, feature_type):
    h, w = image_shape[:2]
    assert semantics is not None, "semantics is None"
    if feature_type == SemanticFeatureType.LABEL:
        assert semantics.shape == (h, w), f"label semantics shape {semantics.shape} != {(h, w)}"
    else:
        assert semantics.shape[:2] == (
            h,
            w,
        ), f"prob semantics shape {semantics.shape} has wrong H,W"


# check if the semantics are of the correct dtype
def check_semantics_dtype(semantics, feature_type):
    if feature_type == SemanticFeatureType.LABEL:
        assert semantics.dtype in (
            np.int32,
            np.int64,
            np.uint16,
            np.uint8,
        ), f"label semantics dtype looks wrong: {semantics.dtype}"
    else:
        assert np.issubdtype(
            semantics.dtype, np.floating
        ), f"prob semantics dtype looks wrong: {semantics.dtype}"


# check if the label ids are in the correct range
def check_label_range(semantics, num_classes, feature_type):
    if feature_type == SemanticFeatureType.LABEL:
        if semantics.size > 0:
            min_id = int(semantics.min())
            max_id = int(semantics.max())
            assert min_id >= 0, f"label id < 0: {min_id}"
            assert max_id < num_classes, f"label id >= num_classes: {max_id} >= {num_classes}"


# check if the probability vectors are simplex
def check_probability_simplex(semantics, feature_type, atol=1e-3):
    if feature_type != SemanticFeatureType.PROBABILITY_VECTOR:
        return
    sums = semantics.sum(axis=-1)
    max_err = float(np.max(np.abs(sums - 1.0)))
    assert max_err < atol, f"prob sums not ~1.0 (max err {max_err})"


# check if the instances are consistent
def check_instances_consistency(instances):
    if instances is None:
        return
    assert instances.ndim == 2, f"instances should be 2D, got {instances.ndim}D"
    assert instances.dtype == np.int32, f"instances dtype should be int32, got {instances.dtype}"
    if instances.size > 0:
        min_id = int(instances.min())
        assert min_id >= 0, f"instance id < 0: {min_id}"


# check if the instances are connected
def check_instances_connected(instances, min_pixels=50):
    if instances is None:
        return
    instance_ids = np.unique(instances)
    for instance_id in instance_ids:
        if instance_id <= 0:
            continue
        mask = instances == instance_id
        if np.count_nonzero(mask) < min_pixels:
            continue
        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        assert num_labels <= 2, f"instance id {instance_id} has disconnected components"


# check if the instances are disconnected
def warn_on_disconnected_instances(instances, min_pixels=50, max_components_report=6):
    if instances is None:
        return
    instance_ids = np.unique(instances)
    for instance_id in instance_ids:
        if instance_id <= 0:
            continue
        mask = instances == instance_id
        if np.count_nonzero(mask) < min_pixels:
            continue
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        if num_labels > 2:  # background + multiple components
            Printer.red(f"instance id {instance_id} has {num_labels - 1} disconnected components")
            component_ids = list(range(1, num_labels))
            if max_components_report is not None:
                component_ids = component_ids[:max_components_report]
            component_masks = [(labels == comp_id) for comp_id in component_ids]
            # Loop over each connected component mask for the current instance id.
            for i in range(len(component_masks)):
                # For component i, compute a distance transform on the inverse of the mask.
                # That produces, for every pixel, the Euclidean distance to the nearest pixel
                # inside component i. Pixels inside the component have distance 0.
                inv_mask_i = 1 - component_masks[i].astype(
                    np.uint8
                )  # inverse of the mask: 1 for background, 0 for component i
                dist_map = cv2.distanceTransform(inv_mask_i, cv2.DIST_L2, 3)
                # compute the minimum boundary-to-boundary distance between components
                for j in range(i + 1, len(component_masks)):
                    # select pixels belonging to the other component
                    other_pixels = component_masks[j]
                    if not np.any(other_pixels):
                        continue
                    # min distance from any pixel in component j to component i
                    min_dist = float(dist_map[other_pixels].min())
                    Printer.red(
                        # report component pair and their separation in pixels
                        f"  components {component_ids[i]}-{component_ids[j]} min distance: {min_dist:.1f}px"
                    )


if __name__ == "__main__":

    dataset = dataset_factory(config)

    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 2000
    print("tracker_config: ", tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared.
    FeatureTrackerShared.set_feature_tracker(feature_tracker)

    # Select your semantic segmentation configuration (see the file semantics/semantic_segmentation_factory.py)

    semantic_segmentation_type = SemanticSegmentationType.DETIC
    semantic_feature_type = SemanticFeatureType.LABEL
    # semantic_dataset_type = SemanticDatasetType.ADE20K
    semantic_dataset_type = SemanticDatasetType.CITYSCAPES
    image_size = (512, 512)
    device = None  # autodetect
    semantic_segmentation = semantic_segmentation_factory(
        semantic_segmentation_type=semantic_segmentation_type,
        semantic_feature_type=semantic_feature_type,
        semantic_dataset_type=semantic_dataset_type,
        image_size=image_size,
        device=device,
    )
    Printer.green(f"semantic_segmentation_type: {semantic_segmentation_type.name}")
    Printer.green(f"semantic_feature_type: {semantic_feature_type.name}")
    Printer.green(f"semantic_dataset_type: {semantic_dataset_type.name}")
    Printer.green(f"num classes: {semantic_segmentation.num_classes()}")

    semantic_color_map = None
    if semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY:
        semantic_color_map = labels_color_map_factory(semantic_dataset_type)

    img_writer = ImgWriter(font_scale=0.7)

    # Create windows before the loop to avoid delay on first display
    cv2.namedWindow("img")
    cv2.namedWindow("semantic prediction viz")
    cv2.namedWindow("semantic class map")
    cv2.namedWindow("semantic instance map")

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    key = None
    while True:

        timestamp, img = None, None

        if dataset.is_ok:
            timestamp = dataset.getTimestamp()  # get current timestamp
            img = dataset.getImageColor(img_id)

        if img is not None:
            print("----------------------------------------")
            print(f"processing img {img_id}, img.shape: {img.shape}")

            inference_result = semantic_segmentation.infer(img)
            curr_semantic_prediction = inference_result.semantics
            curr_semantic_instances = inference_result.instances

            if check_output:
                check_output_shapes(
                    curr_semantic_prediction,
                    curr_semantic_instances,
                    img.shape,
                    semantic_feature_type,
                )
                check_semantics_dtype(curr_semantic_prediction, semantic_feature_type)
                check_label_range(
                    curr_semantic_prediction,
                    semantic_segmentation.num_classes(),
                    semantic_feature_type,
                )
                check_probability_simplex(curr_semantic_prediction, semantic_feature_type)
                check_instances_consistency(curr_semantic_instances)
                # Enable strict connectedness check if you want hard failures.
                # check_instances_connected(curr_semantic_instances, min_pixels=50)

            # Get the visualization RGB image with possible overlays/annotations
            semantic_color_img_viz = semantic_segmentation.sem_img_to_viz_rgb(
                curr_semantic_prediction, bgr=True
            )
            semantic_color_img = semantic_segmentation.sem_img_to_rgb(
                curr_semantic_prediction, bgr=True
            )
            if curr_semantic_instances is not None:
                # Use instances_to_rgb method from semantic segmentation class
                # This uses hash-based color mapping for instance IDs (which can be arbitrary integers)
                semantic_color_instances_img = semantic_segmentation.instances_to_rgb(
                    curr_semantic_instances, bgr=True
                )
                if check_disconnected_instances:
                    warn_on_disconnected_instances(curr_semantic_instances)
            else:
                semantic_color_instances_img = None

            img_writer.write(img, f"id: {img_id}", (30, 30))

            cv2.imshow("img", img)

            cv2.imshow("semantic prediction viz", semantic_color_img_viz)

            cv2.imshow("semantic class map", semantic_color_img)

            if semantic_color_instances_img is not None:
                cv2.imshow("semantic instance map", semantic_color_instances_img)

            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(100)

        if key == ord("q") or key == 27:
            break

        img_id += 1
