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
import time
import math

# import multiprocessing as mp
import torch.multiprocessing as mp

import numpy as np
import cv2
from enum import Enum

from pyslam.utilities.system import Printer, set_rlimit
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue

from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_types import FeatureInfo

from pyslam.utilities.timer import TimerFps

from pyslam.slam import KeyFrame, Frame
from pyslam.slam.feature_tracker_shared import SlamFeatureManagerInfo

from .semantic_mapping_configs import SemanticMappingConfig
from .semantic_segmentation_factory import SemanticSegmentationType, semantic_segmentation_factory
from .semantic_types import SemanticFeatureType, SemanticEntityType
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_clip import SemanticSegmentationCLIP
from .perception_tasks import PerceptionTaskType, PerceptionTask, PerceptionOutput
from .semantic_mapping_color_map import SemanticMappingColorMap


import traceback
import platform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame

kVerbose = True
kTimerVerbose = False
kPrintTrackebackDetails = True

kSemanticMappingSleepTime = 5e-3  # [s]

kSemanticMappingOnSeparateThread = Parameters.kSemanticMappingOnSeparateThread
kSemanticMappingDebugAndPrintToFile = Parameters.kSemanticMappingDebugAndPrintToFile

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# Entry point for semantic segmentation on a parallel process. An instance of SemanticSegmentationProcess is used by SemanticMappingDenseProcess.
# For efficiency, we use multiprocessing to run semantic segmentation tasks in a parallel process. That means on a different CPU core thanks to multiprocessing.
# This wouldn't be possible with python multithreading that runs threads on the same CPU core (due to the GIL).
# A SemanticSegmentationProcess instance is owned by SemanticMappingDenseProcess.
# The latter does the full job of managing
#   (1) semantic segmentation on a separate process
#   (2) semantic mapping on a separate thread
class SemanticSegmentationProcess:
    def __init__(
        self,
        slam: "Slam",
        semantic_mapping_config: SemanticMappingConfig,
        device="cpu",
        headless=False,
    ):
        set_rlimit()

        if mp.get_start_method() != "spawn":
            mp.set_start_method(
                "spawn", force=True
            )  # NOTE: This may generate some pickling problems with multiprocessing
            #       in combination with torch and we need to check it in other places.
            #       This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

        # self.slam_info = SlamFeatureManagerInfo(slam=slam)
        self.semantic_mapping_config = semantic_mapping_config
        self.headless = headless

        # NOTE: We must initialze in the launched process in order to avoid pickling problems.
        # self.semantic_segmentation = semantic_segmentation_factory(**semantic_mapping_config)

        self.time_semantic_segmentation = mp.Value("d", 0.0)

        self.reset_mutex = mp.Lock()
        self.reset_requested = mp.Value("i", -1)

        self.load_request_completed = mp.Value("i", 0)
        self.load_request_condition = mp.Condition()
        self.save_request_completed = mp.Value("i", 0)
        self.save_request_condition = mp.Condition()

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.q_in = self.mp_manager.Queue()
        self.q_out = self.mp_manager.Queue()
        self.q_shared_data = self.mp_manager.Queue()

        self.q_in_condition = mp.Condition()
        self.q_out_condition = mp.Condition()
        self.q_shared_data_condition = mp.Condition()

        self.is_running = mp.Value("i", 0)
        self.is_looping = mp.Value("i", 0)

        # to pass data between the main process and the launched process
        # for instance, the number of classes and the text embeddings
        self.semantic_segmentation_shared_data = self.mp_manager.Dict()
        self._num_classes = None
        self._text_embs = None

        self.start()

        print(
            "SemanticMappingDenseProcess: waiting for semantic segmentation process to be ready..."
        )
        with self.q_shared_data_condition:
            while self.q_shared_data.empty():
                self.q_shared_data_condition.wait()
            shared_data = self.q_shared_data.get()
            self._num_classes = shared_data["num_classes"]
            self._text_embs = shared_data["text_embs"]

        self.semantics_color_map = SemanticMappingColorMap(
            semantic_mapping_config.semantic_dataset_type,
            semantic_mapping_config.semantic_feature_type,
            num_classes=self._num_classes,
            text_embs=self._text_embs,
            device=device,
        )

    def num_classes(self):
        return self._num_classes

    def text_embs(self):
        return self._text_embs

    def to_rgb(self, semantics, bgr=False):
        return self.semantics_color_map.to_rgb(semantics, bgr=bgr)

    def start(self):
        self.process = mp.Process(
            target=self.run,
            args=(
                self.semantic_mapping_config,
                self.headless,
                self.q_in,
                self.q_in_condition,
                self.q_out,
                self.q_out_condition,
                self.q_shared_data,
                self.q_shared_data_condition,
                self.is_running,
                self.is_looping,
                self.reset_mutex,
                self.reset_requested,
                self.load_request_completed,
                self.load_request_condition,
                self.save_request_completed,
                self.save_request_condition,
                self.time_semantic_segmentation,
                self.semantic_segmentation_shared_data,
            ),
        )

        # self.process.daemon = True
        self.process.start()

        if MultiprocessingManager.is_start_method_spawn():
            time.sleep(3)  # give a bit of time for the process to start and initialize

    def is_ready(self):
        return self.is_running.value == 1 and self.is_looping.value == 1

    def save(self, path):
        task_type = PerceptionTaskType.SAVE
        task = PerceptionTask(None, None, task_type, load_save_path=path)
        self.save_request_completed.value = 0
        self.add_task(task)
        with self.save_request_condition:
            while self.save_request_completed.value == 0:
                self.save_request_condition.wait()

    def load(self, path):
        task_type = PerceptionTaskType.LOAD
        task = PerceptionTask(None, None, task_type, load_save_path=path)
        self.load_request_completed.value = 0
        self.add_task(task)
        with self.load_request_condition:
            while self.load_request_completed.value == 0:
                self.load_request_condition.wait()

    def request_reset(self):
        SemanticSegmentationBase.print("SemanticSegmentationProcess: Requesting reset...")
        with self.reset_mutex:
            self.reset_requested.value = 1
        while True:
            with self.reset_mutex:
                with self.q_in_condition:
                    self.q_in_condition.notify_all()  # to unblock q_in_condition.wait() in run() method
                if self.reset_requested.value == 0:
                    break
            time.sleep(0.1)
        SemanticSegmentationBase.print("SemanticSegmentationProcess: ...Reset done.")

    def reset_if_requested(
        self,
        reset_mutex,
        reset_requested,
        semantic_segmentation,
        q_in,
        q_in_condition,
        q_out,
        q_out_condition,
    ):
        # acting within the launched process with the passed mp.Value() (received in input)
        with reset_mutex:
            if reset_requested.value == 1:
                SemanticSegmentationBase.print(
                    "SemanticSegmentationProcess: reset_if_requested()..."
                )
                with q_in_condition:
                    empty_queue(q_in)
                    q_in_condition.notify_all()
                with q_out_condition:
                    empty_queue(q_out)
                    q_out_condition.notify_all()
                # Now reset the loop detector in the launched parallel process
                try:
                    semantic_segmentation.reset()
                except Exception as e:
                    SemanticSegmentationBase.print(
                        f"SemanticSegmentationProcess: reset_if_requested: Exception: {e}"
                    )
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        SemanticSegmentationBase.print(f"\t traceback details: {traceback_details}")
                reset_requested.value = 0

    def quit(self):
        if self.is_running.value == 1:
            SemanticSegmentationBase.print("SemanticSegmentationProcess: quitting...")
            self.is_running.value = 0
            with self.q_in_condition:
                self.q_in.put(None)  # put a None in the queue to signal we have to exit
                self.q_in_condition.notify_all()
            with self.q_out_condition:
                self.q_out_condition.notify_all()
            if self.process.is_alive():
                self.process.join(timeout=5)
            if self.process.is_alive():
                Printer.orange(
                    "Warning: Loop detection process did not terminate in time, forced kill."
                )
                self.process.terminate()
            SemanticSegmentationBase.print("SemanticSegmentationProcess: done")

    def init(
        self,
        semantic_mapping_config: SemanticMappingConfig,
        semantic_segmentation_shared_data: dict,
        q_shared_data,
        q_shared_data_condition,
    ):
        self.semantic_segmentation = semantic_segmentation_factory(
            **semantic_mapping_config.to_dict()
        )
        semantic_segmentation_shared_data["num_classes"] = self.semantic_segmentation.num_classes()
        if (
            hasattr(self.semantic_segmentation, "text_embs")
            and self.semantic_segmentation.text_embs is not None
        ):
            # NOTE: This is a workaround to pass the text embeddings to the launched process.
            semantic_segmentation_shared_data["text_embs"] = (
                self.semantic_segmentation.text_embs.cpu().detach().numpy()
            )
        else:
            semantic_segmentation_shared_data["text_embs"] = None

        with q_shared_data_condition:
            q_shared_data.put(
                {
                    "num_classes": semantic_segmentation_shared_data["num_classes"],
                    "text_embs": semantic_segmentation_shared_data["text_embs"],
                }
            )
            q_shared_data_condition.notify_all()

    # main loop of the semantic segmentation process
    def run(
        self,
        semantic_mapping_config: SemanticMappingConfig,
        headless,
        q_in,
        q_in_condition,
        q_out,
        q_out_condition,
        q_shared_data,
        q_shared_data_condition,
        is_running,
        is_looping,
        reset_mutex,
        reset_requested,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_semantic_segmentation,
        semantic_segmentation_shared_data,
    ):
        is_running.value = 1
        SemanticSegmentationBase.print("SemanticSegmentationProcess: starting...")
        self.init(
            semantic_mapping_config,
            semantic_segmentation_shared_data,
            q_shared_data,
            q_shared_data_condition,
        )
        SemanticSegmentationBase.print(
            f"SemanticSegmentationProcess: initialized with num_classes: {self.num_classes()}"
        )
        # main loop
        is_looping.value = 1
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1 and reset_requested.value != 1:
                    SemanticSegmentationBase.print(
                        "SemanticSegmentationProcess: waiting for new task..."
                    )
                    q_in_condition.wait()
            if not q_in.empty():
                self.semantic_segmenting(
                    self.semantic_segmentation,
                    q_in,
                    q_out,
                    q_out_condition,
                    is_running,
                    load_request_completed,
                    load_request_condition,
                    save_request_completed,
                    save_request_condition,
                    time_semantic_segmentation,
                    headless,
                )
            else:
                SemanticSegmentationBase.print("SemanticSegmentationProcess: q_in is empty...")
                time.sleep(0.01)
            self.reset_if_requested(
                reset_mutex,
                reset_requested,
                self.semantic_segmentation,
                q_in,
                q_in_condition,
                q_out,
                q_out_condition,
            )

        empty_queue(q_in)  # empty the queue before exiting
        SemanticSegmentationBase.print("SemanticSegmentationProcess: loop exit...")

    def semantic_segmenting(
        self,
        semantic_segmentation: SemanticSegmentationBase,
        q_in,
        q_out,
        q_out_condition,
        is_running,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_semantic_segmentation,
        headless,
    ):
        # print('SemanticSegmentationProcess: semantic_segmenting')
        timer = TimerFps("SemanticSegmentationProcess", is_verbose=kTimerVerbose)
        timer.start()
        try:
            if is_running.value == 1:

                # check q_in size and dump a warn message if it is too big
                q_in_size = q_in.qsize()
                if q_in_size >= 10:
                    warn_msg = f"\n!SemanticSegmentationProcess: WARNING: q_in size: {q_in_size} is too big!!!\n"
                    SemanticSegmentationBase.print(warn_msg)
                    Printer.red(warn_msg)

                self.last_input_task = (
                    q_in.get()
                )  # blocking call to get a new input task for semantic segmentation
                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                else:
                    last_output = None
                    try:
                        if self.last_input_task.task_type == PerceptionTaskType.LOAD:
                            path = self.last_input_task.load_save_path
                            SemanticSegmentationBase.print(
                                f"SemanticSegmentationProcess: loading the semantic segmentation state from {path}..."
                            )
                            SemanticSegmentationBase.print(
                                f"SemanticSegmentationProcess: ... done loading the semantic segmentation state from {path}"
                            )
                            last_output = PerceptionOutput(self.last_input_task.task_type)
                        elif self.last_input_task.task_type == PerceptionTaskType.SAVE:
                            path = self.last_input_task.load_save_path
                            SemanticSegmentationBase.print(
                                f"SemanticSegmentationProcess: saving the semantic segmentation state into {path}..."
                            )
                            SemanticSegmentationBase.print(
                                f"SemanticSegmentationProcess: ... done saving the semantic segmentation state into {path}"
                            )
                            last_output = PerceptionOutput(self.last_input_task.task_type)
                        else:
                            # run the other semantic segmentation tasks.
                            # first: check and compute if needed the local descriptors by using the independent local feature manager (if present).
                            inference_output = semantic_segmentation.infer(
                                self.last_input_task.keyframe_data.img
                            )
                            if not headless:
                                inference_color_image = semantic_segmentation.to_rgb(
                                    inference_output, bgr=True
                                )
                            else:
                                inference_color_image = None
                            last_output = PerceptionOutput(
                                self.last_input_task.task_type,
                                inference_output=inference_output,
                                inference_color_image=inference_color_image,
                                frame_id=self.last_input_task.keyframe_data.id,
                                img=self.last_input_task.keyframe_data.img,
                            )
                            if last_output is None:
                                SemanticSegmentationBase.print(
                                    f"SemanticSegmentationProcess: semantic segmentation task failed with None output"
                                )

                    except Exception as e:
                        SemanticSegmentationBase.print(
                            f"SemanticSegmentationProcess: EXCEPTION: {e} !!!"
                        )
                        if kPrintTrackebackDetails:
                            traceback_details = traceback.format_exc()
                            SemanticSegmentationBase.print(
                                f"\t traceback details: {traceback_details}"
                            )

                    if is_running.value == 1 and last_output is not None:
                        # push the computed task output in its output queue
                        if last_output.task_type == PerceptionTaskType.LOAD:
                            with load_request_condition:
                                load_request_completed.value = 1
                                load_request_condition.notify_all()
                        elif last_output.task_type == PerceptionTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()
                        else:
                            # manage other semantic segmentation task output
                            with q_out_condition:
                                # push the computed output in the output queue
                                q_out.put(last_output)
                                q_out_condition.notify_all()
                                SemanticSegmentationBase.print(
                                    f"SemanticSegmentationProcess: pushed new output to queue_out size: {q_out.qsize()}"
                                )

        except Exception as e:
            SemanticSegmentationBase.print(f"SemanticSegmentationProcess: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                SemanticSegmentationBase.print(f"\t traceback details: {traceback_details}")

        timer.refresh()
        time_semantic_segmentation.value = timer.last_elapsed
        SemanticSegmentationBase.print(
            f"SemanticSegmentationProcess: q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, loop-detection-process elapsed time: {time_semantic_segmentation.value}"
        )

    def add_task(self, task: PerceptionTask):
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(task)
                self.q_in_condition.notify_all()

    def pop_output(
        self, q_out=None, q_out_condition=None, timeout=Parameters.kLoopDetectingTimeoutPopKeyframe
    ):
        # Normally, we use self.q_out and self.q_out_condition.
        if q_out is None:
            q_out = self.q_out
        if q_out_condition is None:
            q_out_condition = self.q_out_condition

        if self.is_running.value == 0:
            return None
        with q_out_condition:
            while q_out.empty() and self.is_running.value == 1:
                ok = q_out_condition.wait(timeout=timeout)
                if not ok:
                    SemanticSegmentationBase.print(
                        "SemanticSegmentationProcess: pop_output: timeout"
                    )
                    break  # Timeout occurred
        if q_out.empty():
            return None
        try:
            return q_out.get(timeout=timeout)
        except Exception as e:
            SemanticSegmentationBase.print(
                f"SemanticSegmentationProcess: pop_output: encountered exception: {e}"
            )
            return None
