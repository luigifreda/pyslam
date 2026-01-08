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

from pyslam.utilities.logging import Printer, LoggerQueue
from pyslam.utilities.system import set_rlimit
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue

from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_types import FeatureInfo

from pyslam.utilities.timer import TimerFps

from pyslam.slam import KeyFrame, Frame
from pyslam.slam.feature_tracker_shared import SlamFeatureManagerInfo
from pyslam.loop_closing.loop_detector_configs import (
    LoopDetectorConfigs,
    loop_detector_factory,
    loop_detector_config_check,
    GlobalDescriptorType,
)
from pyslam.loop_closing.loop_detector_base import (
    LoopDetectorTask,
    LoopDetectorTaskType,
    LoopDetectorBase,
    LoopDetectorOutput,
)

import traceback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame


kVerbose = True
kPrintTrackebackDetails = True

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# Entry point for loop detection that generates candidates for loop closure. An instance of LoopDetectingProcess is used by LoopClosing.
# For efficiency, we use multiprocessing to run detection tasks in a parallel process. That means on a different CPU core thanks to multiprocessing.
# This wouldn't be possible with python multithreading that runs threads on the same CPU core (due to the GIL).
# A LoopDetectingProcess instance is owned by LoopClosing. The latter does the full job of managing (1) detection, (2) consistency verification, (3) geometry verification and (4) correction.
class LoopDetectingProcess:
    def __init__(self, slam: "Slam", loop_detector_config=LoopDetectorConfigs.DBOW3):
        set_rlimit()

        global_descriptor_type = loop_detector_config["global_descriptor_type"]
        # NOTE: the following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).
        if (
            global_descriptor_type == GlobalDescriptorType.COSPLACE
            or global_descriptor_type == GlobalDescriptorType.ALEXNET
            or global_descriptor_type == GlobalDescriptorType.NETVLAD
            or global_descriptor_type == GlobalDescriptorType.VLAD
            or global_descriptor_type == GlobalDescriptorType.EIGENPLACES
            or global_descriptor_type == GlobalDescriptorType.MEGALOC
        ):
            if mp.get_start_method() != "spawn":
                mp.set_start_method(
                    "spawn", force=True
                )  # NOTE: This may generate some pickling problems with multiprocessing
                #       in combination with torch and we need to check it in other places.
                #       This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

        self.loop_detector_config = loop_detector_config
        self.slam_info = SlamFeatureManagerInfo(slam=slam)

        # NOTE: We must initialze in the launched process in order to avoid pickling problems.
        # self.loop_detector = loop_detector_factory(**loop_detector_config, slam_info=self.slam_info)
        # if slam is not None:
        #     loop_detector_config_check(self.loop_detector, slam.feature_tracker.feature_manager.descriptor_type)

        self.time_loop_detection = mp.Value("d", 0.0)

        self.last_input_task = None

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
        self.q_out_reloc = self.mp_manager.Queue()

        self.q_in_condition = mp.Condition()
        self.q_out_condition = mp.Condition()
        self.q_out_reloc_condition = mp.Condition()

        self.is_running = mp.Value("i", 0)
        self.is_looping = mp.Value("i", 0)

        self.start()

    def start(self):
        self.process = mp.Process(
            target=self.run,
            args=(
                self.loop_detector_config,
                self.slam_info,
                self.q_in,
                self.q_in_condition,
                self.q_out,
                self.q_out_condition,
                self.q_out_reloc,
                self.q_out_reloc_condition,
                self.is_running,
                self.is_looping,
                self.reset_mutex,
                self.reset_requested,
                self.load_request_completed,
                self.load_request_condition,
                self.save_request_completed,
                self.save_request_condition,
                self.time_loop_detection,
            ),
        )

        # self.process.daemon = True
        self.process.start()

        if MultiprocessingManager.is_start_method_spawn():
            time.sleep(3)  # give a bit of time for the process to start and initialize

    def is_ready(self):
        return self.is_running.value == 1 and self.is_looping.value == 1

    def save(self, path):
        task_type = LoopDetectorTaskType.SAVE
        task = LoopDetectorTask(None, None, task_type, load_save_path=path)
        self.save_request_completed.value = 0
        self.add_task(task)
        with self.save_request_condition:
            while self.save_request_completed.value == 0:
                self.save_request_condition.wait()

    def load(self, path):
        task_type = LoopDetectorTaskType.LOAD
        task = LoopDetectorTask(None, None, task_type, load_save_path=path)
        self.load_request_completed.value = 0
        self.add_task(task)
        with self.load_request_condition:
            while self.load_request_completed.value == 0:
                self.load_request_condition.wait()

    def request_reset(self):
        LoopDetectorBase.print("LoopDetectingProcess: Requesting reset...")
        with self.reset_mutex:
            self.reset_requested.value = 1
        while True:
            with self.reset_mutex:
                with self.q_in_condition:
                    self.q_in_condition.notify_all()  # to unblock q_in_condition.wait() in run() method
                if self.reset_requested.value == 0:
                    break
            time.sleep(0.1)
        LoopDetectorBase.print("LoopDetectingProcess: ...Reset done.")

    def reset_if_requested(
        self,
        reset_mutex,
        reset_requested,
        loop_detector,
        q_in,
        q_in_condition,
        q_out,
        q_out_condition,
        q_out_reloc,
        q_out_reloc_condition,
    ):
        # acting within the launched process with the passed mp.Value() (received in input)
        with reset_mutex:
            if reset_requested.value == 1:
                LoopDetectorBase.print("LoopDetectingProcess: reset_if_requested()...")
                with q_in_condition:
                    empty_queue(q_in)
                    q_in_condition.notify_all()
                with q_out_condition:
                    empty_queue(q_out)
                    q_out_condition.notify_all()
                with q_out_reloc_condition:
                    empty_queue(q_out_reloc)
                    q_out_reloc_condition.notify_all()
                # Now reset the loop detector in the launched parallel process
                try:
                    loop_detector.reset()
                except Exception as e:
                    LoopDetectorBase.print(
                        f"LoopDetectingProcess: reset_if_requested: Exception: {e}"
                    )
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        LoopDetectorBase.print(f"\t traceback details: {traceback_details}")
                reset_requested.value = 0

    def quit(self):
        if self.is_running.value == 1:
            LoopDetectorBase.print("LoopDetectingProcess: quitting...")
            self.is_running.value = 0
            self.is_looping.value = 0
            with self.q_in_condition:
                self.q_in.put(None)  # put a None in the queue to signal we have to exit
                self.q_in_condition.notify_all()
            with self.q_out_condition:
                self.q_out_condition.notify_all()
            with self.q_out_reloc_condition:
                self.q_out_reloc_condition.notify_all()
            if self.process.is_alive():
                # check if "spawn" method is used, so we increse the default timeout
                timeout = (
                    Parameters.kMultiprocessingProcessJoinDefaultTimeout
                    if mp.get_start_method() != "spawn"
                    else 2 * Parameters.kMultiprocessingProcessJoinDefaultTimeout
                )
                self.process.join(timeout=timeout)
            if self.process.is_alive():
                Printer.orange(
                    "Warning: Loop detection process did not terminate in time, forced kill."
                )
                self.process.terminate()

            # Shutdown the manager AFTER the process has exited
            if hasattr(self, "mp_manager") and self.mp_manager is not None:
                try:
                    self.mp_manager.shutdown()
                except Exception as e:
                    LoopDetectorBase.print(f"Warning: Error shutting down manager: {e}")

            LoopDetectorBase.print("LoopDetectingProcess: done")

    def init(self, loop_detector_config, slam_info: SlamFeatureManagerInfo):
        self.loop_detector = loop_detector_factory(**loop_detector_config, slam_info=slam_info)
        if slam_info.feature_descriptor_type is not None:
            loop_detector_config_check(self.loop_detector, slam_info.feature_descriptor_type)
        self.loop_detector.init()

    # main loop of the loop detection process
    def run(
        self,
        loop_detector_config,
        slam_info,
        q_in,
        q_in_condition,
        q_out,
        q_out_condition,
        q_out_reloc,
        q_out_reloc_condition,
        is_running,
        is_looping,
        reset_mutex,
        reset_requested,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_loop_detection,
    ):
        is_running.value = 1
        LoopDetectorBase.print("LoopDetectingProcess: starting...")
        self.init(loop_detector_config, slam_info)
        # main loop
        is_looping.value = 1
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1 and reset_requested.value != 1:
                    LoopDetectorBase.print("LoopDetectingProcess: waiting for new task...")
                    q_in_condition.wait()
            if not q_in.empty():
                self.loop_detecting(
                    self.loop_detector,
                    q_in,
                    q_out,
                    q_out_condition,
                    q_out_reloc,
                    q_out_reloc_condition,
                    is_running,
                    load_request_completed,
                    load_request_condition,
                    save_request_completed,
                    save_request_condition,
                    time_loop_detection,
                )
            else:
                LoopDetectorBase.print("LoopDetectingProcess: q_in is empty...")
                time.sleep(0.01)
            self.reset_if_requested(
                reset_mutex,
                reset_requested,
                self.loop_detector,
                q_in,
                q_in_condition,
                q_out,
                q_out_condition,
                q_out_reloc,
                q_out_reloc_condition,
            )

        empty_queue(q_in)  # empty the queue before exiting
        is_looping.value = 0

        # Clean up LoggerQueue instances in this spawned process before exiting
        LoggerQueue.stop_all_instances()

        LoopDetectorBase.print("LoopDetectingProcess: loop exit...")

    def loop_detecting(
        self,
        loop_detector: LoopDetectorBase,
        q_in,
        q_out,
        q_out_condition,
        q_out_reloc,
        q_out_reloc_condition,
        is_running,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_loop_detection,
    ):
        # print('LoopDetectingProcess: loop_detecting')
        timer = TimerFps("LoopDetectingProcess", is_verbose=kTimerVerbose)
        timer.start()
        try:
            if is_running.value == 1:

                # check q_in size and dump a warn message if it is too big
                q_in_size = q_in.qsize()
                if q_in_size >= 10:
                    warn_msg = (
                        f"\n!LoopDetectingProcess: WARNING: q_in size: {q_in_size} is too big!!!\n"
                    )
                    LoopDetectorBase.print(warn_msg)
                    Printer.red(warn_msg)

                self.last_input_task = (
                    q_in.get()
                )  # blocking call to get a new input task for loop detection
                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                else:
                    last_output = None
                    try:
                        if self.last_input_task.task_type == LoopDetectorTaskType.LOAD:
                            path = self.last_input_task.load_save_path
                            LoopDetectorBase.print(
                                f"LoopDetectingProcess: loading the loop detection state from {path}..."
                            )
                            self.loop_detector.load_db_maps(path)
                            self.loop_detector.load(path)
                            LoopDetectorBase.print(
                                f"LoopDetectingProcess: ... done loading the loop detection state from {path}"
                            )
                            last_output = LoopDetectorOutput(self.last_input_task.task_type)
                        elif self.last_input_task.task_type == LoopDetectorTaskType.SAVE:
                            path = self.last_input_task.load_save_path
                            LoopDetectorBase.print(
                                f"LoopDetectingProcess: saving the loop detection state into {path}..."
                            )
                            self.loop_detector.save_db_maps(path)
                            self.loop_detector.save(path)
                            LoopDetectorBase.print(
                                f"LoopDetectingProcess: ... done saving the loop detection state into {path}"
                            )
                            last_output = LoopDetectorOutput(self.last_input_task.task_type)
                        else:
                            # run the other loop detection tasks.
                            # first: check and compute if needed the local descriptors by using the independent local feature manager (if present).
                            loop_detector.compute_local_des_if_needed(self.last_input_task)
                            # next: run the requested loop detection task
                            last_output = loop_detector.run_task(self.last_input_task)
                            if last_output is None:
                                LoopDetectorBase.print(
                                    f"LoopDetectingProcess: loop detection task failed with None output"
                                )

                    except Exception as e:
                        LoopDetectorBase.print(f"LoopDetectingProcess: EXCEPTION: {e} !!!")
                        if kPrintTrackebackDetails:
                            traceback_details = traceback.format_exc()
                            LoopDetectorBase.print(f"\t traceback details: {traceback_details}")

                    if is_running.value == 1 and last_output is not None:
                        # push the computed task output in its output queue
                        if last_output.task_type == LoopDetectorTaskType.LOAD:
                            with load_request_condition:
                                load_request_completed.value = 1
                                load_request_condition.notify_all()
                        elif last_output.task_type == LoopDetectorTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()
                        elif last_output.task_type == LoopDetectorTaskType.RELOCALIZATION:
                            with q_out_reloc_condition:
                                # push the computed output in the dedicated reloc output queue in order not interfer with the main output queue
                                q_out_reloc.put(last_output)
                                q_out_reloc_condition.notify_all()
                                LoopDetectorBase.print(
                                    f"LoopDetectingProcess: pushed new output to queue_out_reloc size: {q_out_reloc.qsize()}"
                                )
                        else:
                            # manage other loop detection task output
                            with q_out_condition:
                                # push the computed output in the output queue
                                q_out.put(last_output)
                                q_out_condition.notify_all()
                                LoopDetectorBase.print(
                                    f"LoopDetectingProcess: pushed new output to queue_out size: {q_out.qsize()}"
                                )

        except Exception as e:
            LoopDetectorBase.print(f"LoopDetectingProcess: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopDetectorBase.print(f"\t traceback details: {traceback_details}")

        timer.refresh()
        time_loop_detection.value = timer.last_elapsed
        LoopDetectorBase.print(
            f"LoopDetectingProcess: q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, q_out_reloc size: {q_out_reloc.qsize()}, loop-detection-process elapsed time: {time_loop_detection.value}"
        )

    def add_task(self, task: LoopDetectorTask):
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(task)
                self.q_in_condition.notify_all()

    def pop_output(
        self, q_out=None, q_out_condition=None, timeout=Parameters.kLoopDetectingTimeoutPopKeyframe
    ):
        # Normally, we use self.q_out and self.q_out_condition.
        # However, we may need to use self.q_out_reloc and self.q_out_reloc_condition.
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
                    LoopDetectorBase.print("LoopDetectingProcess: pop_output: timeout")
                    break  # Timeout occurred
        if q_out.empty():
            return None
        try:
            return q_out.get(timeout=timeout)
        except Exception as e:
            LoopDetectorBase.print(f"LoopDetectingProcess: pop_output: encountered exception: {e}")
            return None

    def relocalize(self, task: LoopDetectorTask):
        assert task.task_type == LoopDetectorTaskType.RELOCALIZATION
        # push the relocalization task
        self.add_task(task)
        # immediately wait for and get the relocalization output
        detection_output = self.pop_output(
            q_out=self.q_out_reloc, q_out_condition=self.q_out_reloc_condition
        )
        if detection_output is not None:
            assert detection_output.task_type == LoopDetectorTaskType.RELOCALIZATION
        return detection_output
