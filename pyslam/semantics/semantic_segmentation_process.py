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
import sys
import signal

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

from .semantic_mapping_configs import SemanticMappingConfig
from .semantic_segmentation_factory import semantic_segmentation_factory
from .semantic_segmentation_base import SemanticSegmentationBase
from .perception_tasks import PerceptionTaskType, PerceptionTask, PerceptionOutput
from .semantic_color_map_factory import semantic_color_map_factory


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


# Entry point for semantic segmentation on a parallel process. An instance of SemanticSegmentationProcess may be used
# by SemanticMappingDenseProcess, as an alternative to the semantic mapping on a separate thread.
# For efficiency, we use multiprocessing to run semantic segmentation tasks in a parallel process. That means on a
# different CPU core thanks to multiprocessing. This wouldn't be possible with python multithreading that runs
# threads on the same CPU core (due to the GIL).
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
        self._color_map = None
        self._color_map_params = None

        self.start()

        print(
            "SemanticMappingDenseProcess: waiting for semantic segmentation process to be ready..."
        )
        timeout = float("inf")
        start_time = time.time()
        with self.q_shared_data_condition:
            while self.q_shared_data.empty():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    # Check if process is still alive
                    if hasattr(self, "process") and not self.process.is_alive():
                        exit_code = self.process.exitcode
                        raise RuntimeError(
                            f"Semantic segmentation process died during initialization "
                            f"(exit code: {exit_code}). Check logs for errors."
                        )
                    raise TimeoutError(
                        f"Timeout waiting for semantic segmentation process to initialize "
                        f"(waited {elapsed:.1f}s). The process may be stuck during model initialization. "
                        f"Process is {'alive' if (hasattr(self, 'process') and self.process.is_alive()) else 'not alive'}."
                    )
                # Use timeout to periodically check and print status
                self.q_shared_data_condition.wait(timeout=5.0)
                if self.q_shared_data.empty():
                    elapsed = time.time() - start_time
                    process_status = (
                        "alive"
                        if (hasattr(self, "process") and self.process.is_alive())
                        else "not alive"
                    )
                    print(
                        f"Still waiting for semantic segmentation initialization... "
                        f"({elapsed:.1f}s elapsed, process is {process_status})"
                    )
            shared_data = self.q_shared_data.get()
            self._num_classes = shared_data["num_classes"]
            self._text_embs = shared_data["text_embs"]
            # Keep the semantic color map produced in the worker so we can reuse the exact palette
            self._color_map = shared_data.get("color_map")
            self._color_map_params = shared_data.get("color_map_params")

        # Recreate the color map using the parameters (and palette) coming from the worker
        color_map_params = self._color_map_params or {}
        semantic_color_map_kwargs = {
            "semantic_dataset_type": color_map_params.get(
                "semantic_dataset_type", semantic_mapping_config.semantic_dataset_type
            ),
            "semantic_feature_type": color_map_params.get(
                "semantic_feature_type", semantic_mapping_config.semantic_feature_type
            ),
            "num_classes": color_map_params.get("num_classes", self._num_classes),
            "text_embs": color_map_params.get("text_embs", self._text_embs),
            "device": color_map_params.get("device", device),
            "sim_scale": color_map_params.get("sim_scale", 1.0),
            "semantic_segmentation_type": color_map_params.get(
                "semantic_segmentation_type", semantic_mapping_config.semantic_segmentation_type
            ),
        }

        self.semantic_color_map = semantic_color_map_factory(**semantic_color_map_kwargs)

        # If the worker provided an explicit palette, enforce it to keep colors consistent
        if self._color_map is not None:
            color_map_array = np.ascontiguousarray(np.array(self._color_map))
            self.semantic_color_map.color_map = color_map_array
            self._num_classes = len(color_map_array)

    def num_classes(self):
        return self._num_classes

    def text_embs(self):
        return self._text_embs

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        return self.semantic_color_map.to_rgb(semantics, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_color_map.sem_img_to_rgb(semantic_img, bgr=bgr)

    def __getstate__(self):
        """
        Custom pickling: exclude non-picklable multiprocessing primitives.
        This allows SemanticSegmentationProcess to be pickled for passing to spawned processes.
        Only the essential data (semantic_color_map, config, etc.) is preserved.
        """
        state = self.__dict__.copy()
        # Exclude multiprocessing synchronized objects that can't be pickled with spawn
        if "time_semantic_segmentation" in state:
            del state["time_semantic_segmentation"]
        if "reset_mutex" in state:
            del state["reset_mutex"]
        if "reset_requested" in state:
            del state["reset_requested"]
        if "load_request_completed" in state:
            del state["load_request_completed"]
        if "load_request_condition" in state:
            del state["load_request_condition"]
        if "save_request_completed" in state:
            del state["save_request_completed"]
        if "save_request_condition" in state:
            del state["save_request_condition"]
        if "mp_manager" in state:
            del state["mp_manager"]
        if "q_in" in state:
            del state["q_in"]
        if "q_out" in state:
            del state["q_out"]
        if "q_shared_data" in state:
            del state["q_shared_data"]
        if "q_in_condition" in state:
            del state["q_in_condition"]
        if "q_out_condition" in state:
            del state["q_out_condition"]
        if "q_shared_data_condition" in state:
            del state["q_shared_data_condition"]
        if "is_running" in state:
            del state["is_running"]
        if "is_looping" in state:
            del state["is_looping"]
        if "semantic_segmentation_shared_data" in state:
            del state["semantic_segmentation_shared_data"]
        if "process" in state:
            del state["process"]
        # Keep: semantic_color_map, _num_classes, _text_embs, semantic_mapping_config, headless
        return state

    def __setstate__(self, state):
        """
        Custom unpickling: restore state and set excluded multiprocessing primitives to None.
        The excluded attributes are not needed for the callable methods (to_rgb, etc.) to work.
        """
        self.__dict__.update(state)
        # Set excluded multiprocessing primitives to None (they're not needed for callable methods)
        # Note: The process itself won't be running in the spawned process, but that's okay
        # because we only need semantic_color_map for the callable methods

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

        # Note: We don't set daemon=True because we need the process to finish its work
        # Setting daemon=True would cause the process to be killed immediately on exit
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
        SemanticSegmentationBase.print("SemanticSegmentationProcess: quit() called...")

        # Check if process is already stopped
        if not hasattr(self, "process") or not self.process.is_alive():
            SemanticSegmentationBase.print("SemanticSegmentationProcess: process already stopped.")
            return

        if self.is_running.value == 1:
            SemanticSegmentationBase.print("SemanticSegmentationProcess: quitting...")
            self.is_running.value = 0

            # Signal the process to exit
            try:
                with self.q_in_condition:
                    # Put None first to signal immediate exit, then empty remaining items
                    # This ensures the process exits quickly even if queue has many items
                    self.q_in.put(None)  # put a None in the queue to signal we have to exit
                    self.q_in_condition.notify_all()
            except Exception as e:
                SemanticSegmentationBase.print(f"Warning: Error signaling q_in: {e}")

            try:
                with self.q_out_condition:
                    # Notify waiting threads
                    self.q_out_condition.notify_all()
            except Exception as e:
                SemanticSegmentationBase.print(f"Warning: Error signaling q_out: {e}")

            # Wait for the process to finish with longer timeout to allow graceful shutdown
            # Increase timeout to handle large queues (e.g., 23 items * ~1.3s per item = ~30s)
            if self.process.is_alive():
                SemanticSegmentationBase.print(
                    "SemanticSegmentationProcess: waiting for process to finish gracefully..."
                )
                total_timeout = (
                    Parameters.kMultiprocessingProcessJoinDefaultTimeout
                    if mp.get_start_method() != "spawn"
                    else 2 * Parameters.kMultiprocessingProcessJoinDefaultTimeout
                )
                check_interval = 3  # Print status every 3 seconds
                elapsed = 0
                while self.process.is_alive() and elapsed < total_timeout:
                    self.process.join(timeout=check_interval)
                    elapsed += check_interval
                    if self.process.is_alive():
                        remaining = total_timeout - elapsed
                        q_in_size = 0
                        q_out_size = 0
                        try:
                            q_in_size = self.q_in.qsize()
                            q_out_size = self.q_out.qsize()
                        except Exception:
                            # Queue might be closed or unavailable, ignore
                            pass
                        SemanticSegmentationBase.print(
                            f"SemanticSegmentationProcess: still waiting... "
                            f"(elapsed: {elapsed}s, remaining: {remaining}s, "
                            f"q_in: {q_in_size}, q_out: {q_out_size})"
                        )
                # Check if process exited during the wait
                if not self.process.is_alive():
                    SemanticSegmentationBase.print(
                        f"SemanticSegmentationProcess: process exited gracefully after {elapsed}s"
                    )

            # If still alive, force termination
            if self.process.is_alive():
                Printer.orange(
                    "Warning: Semantic segmentation process did not terminate in time, forcing shutdown."
                )
                SemanticSegmentationBase.print("SemanticSegmentationProcess: sending SIGTERM...")
                self.process.terminate()
                # Wait with status updates
                terminate_timeout = (
                    Parameters.kMultiprocessingProcessJoinDefaultTimeout
                    if mp.get_start_method() != "spawn"
                    else 2 * Parameters.kMultiprocessingProcessJoinDefaultTimeout
                )
                check_interval = 1  # Check every second during termination
                elapsed = 0
                while self.process.is_alive() and elapsed < terminate_timeout:
                    self.process.join(timeout=check_interval)
                    elapsed += check_interval
                    if self.process.is_alive():
                        remaining = terminate_timeout - elapsed
                        SemanticSegmentationBase.print(
                            f"SemanticSegmentationProcess: waiting for termination... "
                            f"(elapsed: {elapsed}s, remaining: {remaining}s)"
                        )

                # Force kill if still alive
                if self.process.is_alive():
                    Printer.red("Process still alive after terminate, killing...")
                    SemanticSegmentationBase.print(
                        "SemanticSegmentationProcess: sending SIGKILL..."
                    )
                    self.process.kill()
                    self.process.join(timeout=2)
                    if self.process.is_alive():
                        SemanticSegmentationBase.print(
                            "SemanticSegmentationProcess: process killed."
                        )

            # CRITICAL: Shutdown the manager AFTER the process has exited
            # This prevents the manager from closing queues while the process is still using them
            # Do this regardless of whether process exited cleanly or not
            if hasattr(self, "mp_manager") and self.mp_manager.manager is not None:
                try:
                    SemanticSegmentationBase.print("Shutting down multiprocessing manager...")
                    self.mp_manager.manager.shutdown()
                    SemanticSegmentationBase.print("Manager shut down successfully.")
                except Exception as e:
                    SemanticSegmentationBase.print(f"Warning: Error shutting down manager: {e}")

            # Try to empty queues after manager shutdown (but don't block if they're already closed)
            if not self.process.is_alive():
                # Process exited cleanly - try to empty queues
                try:
                    with self.q_in_condition:
                        empty_queue(self.q_in)
                except Exception as e:
                    SemanticSegmentationBase.print(f"Warning: Error emptying q_in: {e}")

                try:
                    with self.q_out_condition:
                        empty_queue(self.q_out)
                except Exception as e:
                    SemanticSegmentationBase.print(f"Warning: Error emptying q_out: {e}")

                try:
                    with self.q_shared_data_condition:
                        empty_queue(self.q_shared_data)
                except Exception as e:
                    SemanticSegmentationBase.print(f"Warning: Error emptying q_shared_data: {e}")

            # Final check: ensure process is fully dead
            if hasattr(self, "process") and self.process.is_alive():
                Printer.red("CRITICAL: Process still alive after cleanup, forcing kill...")
                try:
                    self.process.kill()
                    self.process.join(timeout=1)
                except Exception as e:
                    SemanticSegmentationBase.print(f"Error killing process: {e}")

            SemanticSegmentationBase.print("SemanticSegmentationProcess: done")

            # Flush any pending output
            sys.stdout.flush()
            sys.stderr.flush()

            # Small delay to allow Python's atexit handlers to run properly
            # This helps prevent hanging when multiprocessing queues are finalized
            time.sleep(0.1)

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

        # Share the exact color map (and its parameters) created by the semantic segmentation model
        color_map_obj = None
        if hasattr(self.semantic_segmentation, "semantic_color_map_obj"):
            color_map_obj = self.semantic_segmentation.semantic_color_map_obj
        elif hasattr(self.semantic_segmentation, "semantic_color_map") and not isinstance(
            getattr(self.semantic_segmentation, "semantic_color_map", None), np.ndarray
        ):
            potential_obj = self.semantic_segmentation.semantic_color_map
            if hasattr(potential_obj, "color_map"):
                color_map_obj = potential_obj

        shared_color_map = None
        color_map_params = None
        if color_map_obj is not None and getattr(color_map_obj, "color_map", None) is not None:
            shared_color_map = np.ascontiguousarray(color_map_obj.color_map)
            color_map_params = {
                "semantic_dataset_type": color_map_obj.semantic_dataset_type,
                "semantic_feature_type": color_map_obj.semantic_feature_type,
                "num_classes": getattr(color_map_obj, "num_classes", None),
                "semantic_segmentation_type": getattr(
                    color_map_obj, "semantic_segmentation_type", None
                ),
                "device": getattr(color_map_obj, "device", "cpu"),
                "sim_scale": getattr(color_map_obj, "sim_scale", 1.0),
            }
            if hasattr(color_map_obj, "text_embs") and color_map_obj.text_embs is not None:
                text_embs = color_map_obj.text_embs
                if hasattr(text_embs, "cpu"):
                    text_embs = text_embs.cpu()
                if hasattr(text_embs, "detach"):
                    text_embs = text_embs.detach()
                color_map_params["text_embs"] = (
                    text_embs.numpy() if hasattr(text_embs, "numpy") else text_embs
                )
            # Keep num_classes in sync with the exported palette size when available
            if color_map_params["num_classes"] is None:
                color_map_params["num_classes"] = len(shared_color_map)
        elif hasattr(self.semantic_segmentation, "semantic_color_map") and isinstance(
            getattr(self.semantic_segmentation, "semantic_color_map", None), np.ndarray
        ):
            shared_color_map = np.ascontiguousarray(self.semantic_segmentation.semantic_color_map)
            color_map_params = {
                "semantic_dataset_type": semantic_mapping_config.semantic_dataset_type,
                "semantic_feature_type": semantic_mapping_config.semantic_feature_type,
                "num_classes": len(shared_color_map),
                "semantic_segmentation_type": semantic_mapping_config.semantic_segmentation_type,
                "device": "cpu",
                "sim_scale": 1.0,
            }

        SemanticSegmentationBase.print(
            f"SemanticSegmentationProcess.init: About to put data into q_shared_data "
            f"(num_classes={semantic_segmentation_shared_data['num_classes']}, "
            f"color_map is {'not ' if shared_color_map is None else ''}None)"
        )
        with q_shared_data_condition:
            q_shared_data.put(
                {
                    "num_classes": semantic_segmentation_shared_data["num_classes"],
                    "text_embs": semantic_segmentation_shared_data["text_embs"],
                    "color_map": shared_color_map,
                    "color_map_params": color_map_params,
                }
            )
            SemanticSegmentationBase.print(
                "SemanticSegmentationProcess.init: Data put into q_shared_data, notifying..."
            )
            q_shared_data_condition.notify_all()
            SemanticSegmentationBase.print("SemanticSegmentationProcess.init: Notification sent")

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
        # Use semantic_segmentation.num_classes() instead of self.num_classes()
        # because self._num_classes is only set in the main process
        num_classes_val = (
            self.semantic_segmentation.num_classes()
            if hasattr(self.semantic_segmentation, "num_classes")
            else None
        )
        SemanticSegmentationBase.print(
            f"SemanticSegmentationProcess: initialized with num_classes: {num_classes_val}"
        )

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            SemanticSegmentationBase.print(
                f"SemanticSegmentationProcess: received signal {signum}, setting is_running to 0"
            )
            is_running.value = 0
            # Notify condition to wake up from wait
            with q_in_condition:
                q_in_condition.notify_all()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # main loop
        is_looping.value = 1
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1 and reset_requested.value != 1:
                    SemanticSegmentationBase.print(
                        "SemanticSegmentationProcess: waiting for new task..."
                    )
                    # Use timeout to periodically check is_running flag
                    q_in_condition.wait(timeout=1.0)
            # Check if we should exit before processing
            if is_running.value == 0:
                SemanticSegmentationBase.print(
                    "SemanticSegmentationProcess: is_running is 0, exiting..."
                )
                break

            if not q_in.empty():
                # Get the task and check if it's None (shutdown signal)
                try:
                    task = q_in.get()
                    if task is None:
                        # Received shutdown signal, exit immediately
                        is_running.value = 0
                        SemanticSegmentationBase.print(
                            "SemanticSegmentationProcess: received shutdown signal, exiting..."
                        )
                        break
                    else:
                        # Process the task directly (pass it to avoid getting it again from queue)
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
                            task=task,  # Pass the task directly
                        )
                except Exception as e:
                    SemanticSegmentationBase.print(
                        f"SemanticSegmentationProcess: error getting task: {e}"
                    )
                    break
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

        # Clean up LoggerQueue instances in this spawned process before exiting
        LoggerQueue.stop_all_instances()

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
        task=None,  # Optional: if provided, use this task instead of getting from queue
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

                # Get task from parameter or from queue
                if task is not None:
                    self.last_input_task = task
                else:
                    self.last_input_task = (
                        q_in.get()
                    )  # blocking call to get a new input task for semantic segmentation

                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                    SemanticSegmentationBase.print(
                        "SemanticSegmentationProcess: received None task, exiting immediately..."
                    )
                    return  # Exit immediately without processing more tasks
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
                                inference_color_image = semantic_segmentation.sem_img_to_viz_rgb(
                                    inference_output.semantics, bgr=True
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
