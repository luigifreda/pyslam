"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
import time
import cv2
import numpy as np
import torch

from collections import defaultdict

from threading import RLock, Thread, Condition
from queue import Queue

from pyslam.config_parameters import Parameters

from .semantic_segmentation_factory import semantic_segmentation_factory
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_fusion_methods import SemanticFusionMethods
from .semantic_mapping_base import SemanticMappingType, SemanticMappingBase
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_mapping_configs import SemanticMappingConfig
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_color_utils import (
    labels_color_map_factory,
    single_label_to_color,
    similarity_heatmap_point,
)
from .semantic_information_weights_factory import (
    semantic_information_weights_factory,
)

from pyslam.utilities.timer import TimerFps
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.logging import Printer, Logging
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue

from pyslam.viz.qimage_thread import QimageViewer
import traceback
import platform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame

kVerbose = True
kTimerVerbose = False

kSemanticMappingSleepTime = 5e-3  # [s]


class SemanticMappingDenseBase(SemanticMappingBase):
    """
    Base class for dense semantic mapping. It is used by both SemanticMappingDense and SemanticMappingDenseProcess.
    """

    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(
        self,
        slam: "Slam",
        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        semantic_feature_type=SemanticFeatureType.LABEL,
        image_size=(512, 512),
        headless=False,
    ):
        super().__init__(
            slam,
            SemanticMappingType.DENSE,
            semantic_segmentation_type,
            semantic_dataset_type,
            semantic_feature_type,
        )

        self.semantic_feature_type = semantic_feature_type
        self.semantic_segmentation_type = semantic_segmentation_type
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_fusion_method = SemanticFusionMethods.get_semantic_fusion_method(
            semantic_feature_type
        )

        self.semantic_mapping_config = SemanticMappingConfig(
            semantic_segmentation_type=semantic_segmentation_type,
            semantic_feature_type=self.semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
        )

        Printer.green(f"semantic_segmentation_type: {semantic_segmentation_type.name}")

        if semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY:
            # Pass semantic_segmentation_type to use EOV-Seg color map when needed
            self.semantic_color_map = labels_color_map_factory(
                semantic_dataset_type, semantic_segmentation_type=semantic_segmentation_type
            )
            self.semantic_sigma2_factor = semantic_information_weights_factory(
                semantic_dataset_type
            )
        else:
            self.semantic_color_map = None
            self.semantic_sigma2_factor = [1.0]

        self.timer_verbose = kTimerVerbose
        self.timer_inference = TimerFps("Inference", is_verbose=self.timer_verbose)
        self.timer_update_keyframe = TimerFps("Update KeyFrame", is_verbose=self.timer_verbose)
        self.timer_update_mappoints = TimerFps("Update MapPoints", is_verbose=self.timer_verbose)

        self.headless = headless
        self.draw_semantic_mapping_init = False
        # Cache whether casting predictions from int64 to int32 is value-safe
        self._is_sem_pred_cast_to_int32_safe = None

        self.curr_semantic_inference: SemanticSegmentationOutput | None = None

    def _ensure_cpp_compatible_array(self, array, target_dtype, cast_safety_cache=None):
        """
        Ensure a numpy array is compatible with C++ pybind11 bindings.
        The C++ type caster requires exact dtype (e.g., np.int32, np.uint16), not platform-specific int types.

        Args:
            array: numpy array or None
            target_dtype: target numpy dtype (e.g., np.int32, np.uint16)
            cast_safety_cache: optional cache flag for int64->int32 safety check (only used for int32)

        Returns:
            tuple: (converted_array, updated_cast_safety_cache)
            - If array is None, returns (None, cast_safety_cache)
            - Otherwise returns (C-contiguous array with target_dtype, updated_cache)
        """
        if array is None or not isinstance(array, np.ndarray):
            return array, cast_safety_cache

        if not Parameters.USE_CPP_CORE:
            return array, cast_safety_cache

        # For int32 target dtype, handle int64 safely using the base class method
        # Check if dtype is integer type (int64, int32, intc, etc.)
        if target_dtype == np.int32 and np.issubdtype(array.dtype, np.integer):
            # For int64, use the safe conversion method that checks value ranges
            if array.dtype == np.int64:
                array, cast_safety_cache = SemanticMappingBase.ensure_int32_prediction(
                    array, cast_safety_cache
                )

        # Ensure the dtype is exactly target_dtype that pybind11 will recognize
        # Create a fresh array with explicit dtype to ensure pybind11 compatibility
        # Use np.array() with copy=True and explicit dtype to create a new array
        # The C++ code now has a fallback to check dtype name, but we still ensure
        # the dtype is target_dtype for maximum compatibility
        if array.dtype != target_dtype or not array.flags["C_CONTIGUOUS"]:
            array = np.array(array, dtype=target_dtype, copy=True)

        return array, cast_safety_cache

    def update_kf_cur_semantics(self):
        inference_output = self.curr_semantic_inference
        curr_semantic_prediction = (
            inference_output.semantics if inference_output is not None else None
        )
        curr_semantic_instances = (
            inference_output.instances if inference_output is not None else None
        )

        # Ensure semantics use int32 for C++ compatibility
        # The C++ type caster requires exact np.int32 dtype, not platform-specific int types
        if Parameters.USE_CPP_CORE:
            curr_semantic_prediction, self._is_sem_pred_cast_to_int32_safe = (
                self._ensure_cpp_compatible_array(
                    curr_semantic_prediction, np.int32, self._is_sem_pred_cast_to_int32_safe
                )
            )
            # Ensure instances use int32 for C++ compatibility (supports large panoptic segment IDs)
            curr_semantic_instances, _ = self._ensure_cpp_compatible_array(
                curr_semantic_instances, np.int32, None  # No safety cache needed for int32
            )

        if inference_output is not None:
            inference_output.semantics = curr_semantic_prediction
            inference_output.instances = curr_semantic_instances

        Printer.green(f"#semantic inference, timing: {self.timer_inference.last_elapsed}")
        # TODO(dvdmc): the prints don't work for some reason. They block the Thread ?
        # SemanticMappingBase.print(f'#semantic inference, timing: {self.timer_pts_culling.last_elapsed}')

        # update keypoints of current keyframe
        self.timer_update_keyframe.start()
        self.kf_cur.set_semantics(curr_semantic_prediction)
        if curr_semantic_instances is not None:
            self.kf_cur.set_semantic_instances(curr_semantic_instances)
        else:
            # Avoid passing None into C++ bindings; clear only when supported.
            if hasattr(self.kf_cur, "semantic_instances_img"):
                try:
                    self.kf_cur.semantic_instances_img = None
                except Exception:
                    pass
        self.timer_update_keyframe.refresh()
        Printer.green(f"#set KF semantics, timing: {self.timer_update_keyframe.last_elapsed}")
        # SemanticMappingBase.print(f'#keypoints: {self.kf_cur.num_keypoints()}, timing: {self.timer_update_keyframe.last_elapsed}')

        # update map points of current keyframe
        self.timer_update_mappoints.start()
        self.kf_cur.update_points_semantics(self.semantic_fusion_method)
        self.timer_update_mappoints.refresh()
        Printer.green(f"#set MPs semantics, timing: {self.timer_update_mappoints.last_elapsed}")
        # SemanticMappingBase.print(f'#map points: {self.kf_cur.num_points()}, timing: {self.timer_update_mappoints.last_elapsed}')

        self.draw_semantic_prediction()

    def draw_semantic_prediction(self):
        if self.headless:
            return
        inference_output = self.curr_semantic_inference
        curr_semantic_prediction = (
            inference_output.semantics if inference_output is not None else None
        )
        draw = False
        use_cv2_for_drawing = (
            platform.system() != "Darwin"
        )  # under mac we can't use cv2 imshow here

        if curr_semantic_prediction is not None:
            if not self.draw_semantic_mapping_init:
                if use_cv2_for_drawing:
                    cv2.namedWindow("Semantic prediction")  # to get a resizable window
                self.draw_semantic_mapping_init = True
            draw = True
            semantic_color_img = self.semantic_segmentation.sem_img_to_viz_rgb(
                curr_semantic_prediction, bgr=True
            )
            if use_cv2_for_drawing:
                cv2.imshow("Semantic prediction", semantic_color_img)
            else:
                QimageViewer.get_instance().draw(semantic_color_img, "Semantic prediction")

        if draw:
            if use_cv2_for_drawing:
                cv2.waitKey(1)

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return single_label_to_color(semantic_des, self.semantic_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return single_label_to_color(
                np.argmax(semantic_des, axis=-1), self.semantic_color_map, bgr=bgr
            )
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                # transform features to similarities
                sims = self.semantic_segmentation.features_to_sims(semantic_des)
                return similarity_heatmap_point(
                    sims,
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.semantic_segmentation.sim_scale,
                    bgr=bgr,
                )
            else:
                label = self.semantic_segmentation.features_to_labels(semantic_des)
                return single_label_to_color(label, self.semantic_color_map, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_segmentation.sem_img_to_rgb(semantic_img, bgr=bgr)

    def get_semantic_weight(self, semantic_des):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return self.semantic_sigma2_factor[semantic_des]
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return self.semantic_sigma2_factor[np.argmax(semantic_des, axis=-1)]
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            return self.semantic_sigma2_factor[
                self.semantic_segmentation.features_to_labels(semantic_des)
            ]

    def set_query_word(self, query_word):
        if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.semantic_segmentation.set_query_word(query_word)
        else:
            raise NotImplementedError


class SemanticMappingDense(SemanticMappingDenseBase):
    """
    Dense semantic mapping. It is used to run the semantic mapping on a main thread.
    """

    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(
        self,
        slam: "Slam",
        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        semantic_feature_type=SemanticFeatureType.LABEL,
        image_size=(512, 512),
        headless=False,
    ):
        super().__init__(
            slam,
            semantic_segmentation_type,
            semantic_dataset_type,
            semantic_feature_type,
            image_size,
            headless,
        )

        # Initialize the semantic segmentation to be used in the thread
        self.semantic_segmentation = semantic_segmentation_factory(
            semantic_segmentation_type=semantic_segmentation_type,
            semantic_feature_type=self.semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            encoder_name=(
                self.semantic_mapping_config.encoder_name
                if self.semantic_mapping_config.encoder_name
                else None
            ),
        )

    def semantic_mapping_impl(self):
        # do dense semantic segmentation inference
        self.timer_inference.start()
        inference_output = self.semantic_segmentation.infer(self.img_cur)
        self.curr_semantic_inference = inference_output
        self.timer_inference.refresh()

        self.update_kf_cur_semantics()
