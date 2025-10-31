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

from .semantic_segmentation_factory import SemanticSegmentationType
from .semantic_mapping_base import (
    SemanticMappingType,
    SemanticMappingBase,
)
from .semantic_mapping_dense import SemanticMappingDenseBase
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_segmentation_process import SemanticSegmentationProcess
from .perception_tasks import PerceptionTaskType, PerceptionTask, PerceptionOutput

from pyslam.slam import KeyFrame

from pyslam.utilities.timer import TimerFps
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.system import Printer, Logging
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


# This class is used to manage semantic mapping on a separate process. It does the same job as SemanticMappingDense,
# but on a separate process.
class SemanticMappingDenseProcess(SemanticMappingDenseBase):
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
            self.work_thread.join(timeout=5)
        self.semantic_segmentation_process.quit()
        if QimageViewer.is_running():
            QimageViewer.get_instance().quit()
        SemanticMappingBase.print("SemanticMapping: done")

    # Depending on the implementation the step might just add semantics to new frames, keyframes or it might
    # segment objects and track 3D segments
    @override
    def step(self):
        if self.map.num_keyframes() > 0:
            if not self.stop_requested:
                work_done = False

                # Process all available keyframes and add them to the segmentation queue
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

        if self.kf_cur is None:
            Printer.red("semantic mapping: no keyframe to process")
            return

        if kSemanticMappingOnSeparateThread:
            SemanticMappingBase.print("..................................")
            SemanticMappingBase.print(
                "processing KF: ", self.kf_cur.id, ", queue size: ", self.queue_size()
            )

        self.semantic_mapping_impl(perception_output)

        elapsed_time = time.time() - time_start
        self.time_semantic_mapping = elapsed_time
        SemanticMappingBase.print(f"semantic mapping elapsed time: {elapsed_time}")

    @override
    def semantic_mapping_impl(self, perception_output: PerceptionOutput):
        # process the dense semantic segmentation inference output
        self.timer_inference.start()
        self.curr_semantic_prediction = perception_output.inference_output
        self.curr_semantic_prediction_color_image = perception_output.inference_color_image
        self.timer_inference.refresh()

        self.update_kf_cur_semantics()
