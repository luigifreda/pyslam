"""
* This file is part of PYSLAM
*
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

from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_mapping_base import SemanticMappingBase
from .semantic_mapping_dense import SemanticMappingDenseBase
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_segmentation_process import SemanticSegmentationProcess
from .perception_tasks import PerceptionTaskType, PerceptionTask, PerceptionOutput

from pyslam.slam import KeyFrame

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
kPrintTrackebackDetails = True

kSemanticMappingOnSeparateThread = Parameters.kSemanticMappingOnSeparateThread
kSemanticMappingDebugAndPrintToFile = Parameters.kSemanticMappingDebugAndPrintToFile

kSemanticMappingSleepTime = 5e-3  # [s]


def override(method):
    """Decorator to indicate a method overrides a base class method."""
    return method


# This class is used to run semantic mapping by moving the image segmentation to a separate process.
# The main thread (from SemanticMappingBase) is responsible for managing the semantic mapping,
# while the separate process (spawned here) is responsible for the image segmentation.
class SemanticMappingDenseProcess(SemanticMappingDenseBase):
    """
    Dense semantic mapping. It is used to run the semantic mapping on a separate process.
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

        # Initialize the semantic segmentation process
        self.semantic_segmentation_process = SemanticSegmentationProcess(
            slam=slam,
            semantic_mapping_config=self.semantic_mapping_config,
        )
        self.semantic_segmentation = self.semantic_segmentation_process

        # Update the base class semantic_color_map to use the one from the semantic segmentation process
        # This ensures consistency between the sparse semantic map colors and the semantic color image
        # when using a separate process for semantic segmentation
        # Note: The base class expects a numpy array, while semantic_segmentation_process.semantic_color_map
        # is a SemanticColorMap object, so we extract the color_map array from it
        if self.semantic_segmentation_process.semantic_color_map.color_map is not None:
            self.semantic_color_map = (
                self.semantic_segmentation_process.semantic_color_map.color_map
            )

    def is_ready(self):
        return self.is_running and self.semantic_segmentation_process.is_ready()

    @override
    def request_reset(self):
        SemanticMappingBase.print("SemanticMapping: Requesting reset...")
        if self.reset_requested:
            SemanticMappingBase.print("SemanticMapping: reset already requested...")
            return
        with self.reset_mutex:
            self.reset_requested = True
        while True:
            with self.semantic_segmentation_process.queue_condition:
                self.semantic_segmentation_process.queue_condition.notify_all()  # to unblock self.pop_keyframe()
            with self.reset_mutex:
                if not self.reset_requested:
                    break
            time.sleep(0.1)
            SemanticMappingBase.print("SemanticMapping: waiting for reset...")
        SemanticMappingBase.print("SemanticMapping: ...Reset done.")

    @override
    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                SemanticMappingBase.print("SemanticMapping: reset_if_requested() starting...")
                self.semantic_segmentation_process.request_reset()
                empty_queue(self.queue)
                self.reset_requested = False
                SemanticMappingBase.print("SemanticMapping: reset_if_requested() ...done")

    @override
    def quit(self):
        SemanticMappingBase.print("SemanticMapping: quitting...")
        if self.is_running and self.work_thread is not None:
            self.is_running = False
            self.work_thread.join(timeout=Parameters.kMultithreadingThreadJoinDefaultTimeout)
        self.semantic_segmentation_process.quit()

        # Clean up C++ semantic mapping resources
        from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared

        SemanticMappingShared.cleanup_cpp_module()

        if QimageViewer.is_running():
            QimageViewer.get_instance().quit()
        SemanticMappingBase.print("SemanticMapping: done")

    # Step in the main loop of the semantic mapping thread (see SemanticMappingBase.run())
    # Depending on the implementation the step might just add semantics to new frames, keyframes or it might
    # segment objects and track 3D segments
    @override
    def step(self):
        if self.map.num_keyframes() > 0:
            if not self.stop_requested:
                work_done = False

                # Process all available pushed keyframes from the main thread and add them to the segmentation queue
                # Use non-blocking get when queue is not empty for better performance
                if not self.queue.empty():
                    ret = self.pop_keyframe(timeout=0.0)  # non-blocking
                    if ret is not None:
                        self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret
                        if self.kf_cur is not None:
                            self.last_processed_kf_img_id = self.kf_cur.img_id
                            self.add_keyframe_task(self.kf_cur, self.img_cur)
                            work_done = True

                # Process all available perception outputs
                # Process multiple outputs per iteration for better throughput
                if not self.semantic_segmentation_process.q_out.empty():
                    perception_output: PerceptionOutput = (
                        self.semantic_segmentation_process.pop_output(timeout=0.0)  # non-blocking
                    )
                    if perception_output is not None:
                        work_done = True
                        self.set_idle(False)
                        try:
                            self.do_semantic_mapping(
                                perception_output
                            )  # => calls self.semantic_mapping_impl()
                        except Exception as e:
                            SemanticMappingBase.print(
                                f"SemanticMapping: encountered exception: {e}"
                            )
                            SemanticMappingBase.print(traceback.format_exc())
                        self.set_idle(True)

                # Only sleep if no work was done - reduces latency when work is available
                if not work_done:
                    time.sleep(kSemanticMappingSleepTime)

            elif self.stop_if_requested():
                self.set_idle(True)
                while self.is_stopped():
                    SemanticMappingBase.print(
                        f"SemanticMapping: stopped, idle: {self._is_idle} ..."
                    )
                    time.sleep(kSemanticMappingSleepTime)
        else:
            msg = "SemanticMapping: waiting for keyframes..."
            # Printer.red(msg)
            # SemanticMappingBase.print(msg)
            time.sleep(kSemanticMappingSleepTime)
        self.reset_if_requested()

    def add_keyframe_task(self, keyframe: KeyFrame, img: np.ndarray):
        try:
            task = PerceptionTask(
                keyframe=keyframe,
                img=img,
                task_type=PerceptionTaskType.SEMANTIC_SEGMENTATION,
            )
            self.semantic_segmentation_process.add_task(task)
        except Exception as e:
            SemanticMappingBase.print(f"SemanticMapping: add_keyframe_task: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                SemanticMappingBase.print(f"\t traceback details: {traceback_details}")
            return False
        return True

    @override
    def do_semantic_mapping(self, perception_output: PerceptionOutput):
        SemanticMappingBase.print("semantic mapping: starting...")

        Printer.cyan("@semantic mapping")
        time_start = time.time()

        # CRITICAL: Match the prediction to the correct keyframe using frame_id
        # The prediction might be for a different keyframe than self.kf_cur due to async processing
        if perception_output.frame_id is None:
            Printer.red("semantic mapping: perception_output has no frame_id")
            return

        # Find the correct keyframe matching the prediction's frame_id
        kf_target = None
        keyframes = None  # Initialize for use in error message
        if self.kf_cur is not None and self.kf_cur.id == perception_output.frame_id:
            # Current keyframe matches - use it
            kf_target = self.kf_cur
        else:
            # Need to find the correct keyframe from the map
            keyframes = self.map.get_keyframes()
            for kf in keyframes:
                if kf.id == perception_output.frame_id:
                    kf_target = kf
                    break

        if kf_target is None:
            # This can happen when:
            # 1. The keyframe was removed from the map (e.g., during loop closure or optimization)
            # 2. The semantic segmentation result arrived after the keyframe was removed
            # 3. This is expected behavior in async processing - just skip this result
            if keyframes is None:
                keyframes = self.map.get_keyframes()
            keyframes_list = list(keyframes)
            if len(keyframes_list) > 0:
                current_kf_ids = [kf.id for kf in keyframes_list[-5:]]  # Last 5 keyframes
                max_kf_id = max(kf.id for kf in keyframes_list)
                min_kf_id = min(kf.id for kf in keyframes_list)
                SemanticMappingBase.print(
                    f"semantic mapping: skipping frame_id {perception_output.frame_id} "
                    f"(keyframe was removed from map). "
                    f"Current keyframe IDs: {min_kf_id}-{max_kf_id}, recent: {current_kf_ids}"
                )
            else:
                SemanticMappingBase.print(
                    f"semantic mapping: skipping frame_id {perception_output.frame_id} "
                    f"(no keyframes in map)"
                )
            return

        # Temporarily set kf_cur to the correct keyframe for this prediction
        kf_cur_original = self.kf_cur
        self.kf_cur = kf_target

        if kSemanticMappingOnSeparateThread:
            SemanticMappingBase.print("..................................")
            if kf_cur_original is not None and kf_cur_original.id != kf_target.id:
                SemanticMappingBase.print(
                    f"WARNING: frame_id mismatch - using KF {kf_target.id} (was {kf_cur_original.id}) for frame_id {perception_output.frame_id}"
                )
            SemanticMappingBase.print(
                f"processing KF: {self.kf_cur.id} (frame_id: {perception_output.frame_id}), queue size: {self.queue_size()}"
            )

        self.semantic_mapping_impl(perception_output)

        # Restore original kf_cur if it was more recent (higher ID)
        # Otherwise keep kf_target as it was the one we just processed
        if kf_cur_original is not None and kf_cur_original.id > kf_target.id:
            self.kf_cur = kf_cur_original

        elapsed_time = time.time() - time_start
        self.time_semantic_mapping = elapsed_time
        SemanticMappingBase.print(f"semantic mapping elapsed time: {elapsed_time}")

    def _ensure_cpp_compatible_array_after_multiprocessing(
        self, array, target_dtype, cast_safety_cache=None
    ):
        """
        Ensure a numpy array is compatible with C++ pybind11 bindings after multiprocessing
        pickling/unpickling. After multiprocessing, numpy arrays may have platform-specific
        dtypes (e.g., np.intc instead of np.int32) that pybind11 doesn't recognize.
        Only convert dtype if necessary for C++ compatibility, preserving the actual values.

        Args:
            array: numpy array or None
            target_dtype: target numpy dtype (e.g., np.int32, np.uint16)
            cast_safety_cache: optional cache flag for int64->int32 safety check (only used for int32)

        Returns:
            tuple: (converted_array, updated_cast_safety_cache)
            - If array is None or not a numpy array, returns (array, cast_safety_cache)
            - Otherwise returns (C-contiguous array with compatible dtype, updated_cache)
        """
        if array is None or not isinstance(array, np.ndarray):
            return array, cast_safety_cache

        if not Parameters.USE_CPP_CORE:
            return array, cast_safety_cache

        if not np.issubdtype(array.dtype, np.integer):
            return array, cast_safety_cache

        # For int32 target dtype, handle int64 safely using the base class method
        if target_dtype == np.int32 and array.dtype == np.int64:
            array, cast_safety_cache = SemanticMappingBase.ensure_int32_prediction(
                array, cast_safety_cache
            )

        # Only convert dtype if necessary for C++ compatibility
        # Preserve original dtype if it's already compatible (uint8, uint16, int8, int16, int32)
        # Only convert platform-specific types (like intc) that pybind11 doesn't recognize
        dtype_name = array.dtype.name
        is_contiguous = array.flags["C_CONTIGUOUS"]

        # Determine compatible dtype names based on target_dtype
        if target_dtype == np.int32:
            compatible_dtype_names = ["int8", "int16", "int32"]
        elif target_dtype == np.uint16:
            compatible_dtype_names = ["uint8", "uint16"]
        else:
            # For other target dtypes, only allow exact match
            compatible_dtype_names = [target_dtype.name] if hasattr(target_dtype, "name") else []

        # Check if dtype needs conversion: only convert if it's a platform-specific type
        # (like 'intc') or if it's not C-contiguous. Preserve compatible dtypes.
        needs_conversion = dtype_name not in compatible_dtype_names or not is_contiguous

        if needs_conversion:
            # Use astype() to preserve values while changing dtype to target_dtype
            # This ensures pybind11 compatibility while preserving semantic label values
            array = np.ascontiguousarray(array.astype(target_dtype, copy=False))

        return array, cast_safety_cache

    @override
    def semantic_mapping_impl(self, perception_output: PerceptionOutput):
        # process the dense semantic segmentation inference output
        self.timer_inference.start()
        # inference_output is now a SemanticSegmentationOutput object containing both semantics and instances
        self.curr_semantic_inference = perception_output.inference_output
        self.curr_semantic_prediction_color_image = perception_output.inference_color_image

        curr_semantic_prediction = (
            self.curr_semantic_inference.semantics
            if self.curr_semantic_inference is not None
            else None
        )
        curr_semantic_instances = (
            self.curr_semantic_inference.instances
            if self.curr_semantic_inference is not None
            else None
        )

        # CRITICAL: After multiprocessing pickling/unpickling, numpy arrays may have platform-specific
        # dtypes (e.g., np.intc instead of np.int32) that pybind11 doesn't recognize.
        # Only convert dtype if necessary for C++ compatibility, preserving the actual values.
        curr_semantic_prediction, self._is_sem_pred_cast_to_int32_safe = (
            self._ensure_cpp_compatible_array_after_multiprocessing(
                curr_semantic_prediction, np.int32, self._is_sem_pred_cast_to_int32_safe
            )
        )
        curr_semantic_instances, _ = self._ensure_cpp_compatible_array_after_multiprocessing(
            curr_semantic_instances, np.int32, None  # No safety cache needed for int32
        )

        if self.curr_semantic_inference is not None:
            self.curr_semantic_inference.semantics = curr_semantic_prediction
            self.curr_semantic_inference.instances = curr_semantic_instances

        self.timer_inference.refresh()

        self.update_kf_cur_semantics()

    @override
    def draw_semantic_prediction(self):
        """
        Override to use the pre-computed inference_color_image from the separate process
        instead of regenerating it. This ensures the displayed image matches what was
        computed in the separate process.
        """
        if self.headless:
            return
        draw = False
        use_cv2_for_drawing = (
            platform.system() != "Darwin"
        )  # under mac we can't use cv2 imshow here

        curr_semantic_prediction = (
            self.curr_semantic_inference.semantics
            if self.curr_semantic_inference is not None
            else None
        )
        if curr_semantic_prediction is not None:
            if not self.draw_semantic_mapping_init:
                if use_cv2_for_drawing:
                    cv2.namedWindow("Semantic prediction")  # to get a resizable window
                self.draw_semantic_mapping_init = True
            draw = True

            # Use the pre-computed color image from the separate process if available
            # This ensures consistency with what was computed in the separate process
            if (
                hasattr(self, "curr_semantic_prediction_color_image")
                and self.curr_semantic_prediction_color_image is not None
            ):
                semantic_color_img = self.curr_semantic_prediction_color_image
            else:
                # Fallback to generating it (shouldn't happen in normal operation)
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

    @override
    def sem_des_to_rgb(self, semantic_des, bgr=False):
        """
        Override to use the color map from SemanticSegmentationProcess instead of the base class.
        This ensures consistency between the sparse semantic map colors and the semantic color image.
        """
        return self.semantic_segmentation_process.semantic_color_map.sem_des_to_rgb(
            semantic_des, bgr=bgr
        )

    @override
    def sem_img_to_rgb(self, semantic_img, bgr=False):
        """
        Override to use the color map from SemanticSegmentationProcess instead of the base class.
        This ensures consistency between the sparse semantic map colors and the semantic color image.
        """
        return self.semantic_segmentation_process.semantic_color_map.sem_img_to_rgb(
            semantic_img, bgr=bgr
        )
