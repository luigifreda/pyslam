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
import multiprocessing as mp

from collections import defaultdict

from threading import RLock, Thread, Condition
from queue import Queue

from pyslam.config_parameters import Parameters

from .semantic_types import SemanticFeatureType, SemanticEntityType

from pyslam.utilities.timer import TimerFps
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.logging import Printer, Logging, LoggerQueue
from pyslam.utilities.file_management import create_folder
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

kSemanticMappingOnSeparateThread = Parameters.kSemanticMappingOnSeparateThread
kSemanticMappingDebugAndPrintToFile = Parameters.kSemanticMappingDebugAndPrintToFile


@register_class
class SemanticMappingType(SerializableEnum):
    DENSE = 0  # Pixel-wise segmentation to points maps


class SemanticMappingBase:
    """
    Base class for semantic mapping. A thread is to run the semantic mapping.
    """

    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
    logging_manager, logger = None, None

    def __init__(
        self,
        slam: "Slam",
        semantic_mapping_type,
        semantic_segmentation_type,
        semantic_dataset_type,
        semantic_feature_type,
    ):
        self.slam = slam
        self.semantic_mapping_type = semantic_mapping_type
        self.semantic_entity_type = None
        if semantic_mapping_type == SemanticMappingType.DENSE:
            self.semantic_entity_type = SemanticEntityType.POINT

        self.semantic_segmentation_type = semantic_segmentation_type
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_feature_type = semantic_feature_type

        self.queue = Queue()
        self.queue_condition = Condition()

        self.work_thread: Thread | None = None  # main thread managing the semantic mapping
        self.is_running: bool = False

        self._is_idle = True
        self.idle_condition = Condition()

        self.stop_requested = False
        self.do_not_stop = False
        self.stopped = False
        self.stop_mutex = RLock()

        self.reset_requested = False
        self.reset_mutex = RLock()

        self.last_processed_kf_img_id = None

        self.init_print()

    def init_print(self):
        if kVerbose:
            if kSemanticMappingOnSeparateThread:
                if kSemanticMappingDebugAndPrintToFile:
                    # Default log to file: logs/semantic_mapping.log
                    logging_file = os.path.join(Parameters.kLogsFolder, "semantic_mapping.log")
                    create_folder(logging_file)
                    if SemanticMappingBase.logging_manager is None:
                        # Note: Each process has its own memory space, so singleton pattern works per-process
                        SemanticMappingBase.logging_manager = LoggerQueue.get_instance(logging_file)
                        SemanticMappingBase.logger = SemanticMappingBase.logging_manager.get_logger(
                            "semantic_mapping_logger"
                        )

                def print_file(*args, **kwargs):
                    try:
                        if SemanticMappingBase.logger is not None:
                            message = " ".join(str(arg) for arg in args)
                            return SemanticMappingBase.logger.info(message, **kwargs)
                    except:
                        print("Error printing: ", args, kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(str(arg) for arg in args)
                    return print(message, **kwargs)

            SemanticMappingBase.print = staticmethod(print_file)

    @property
    def map(self):
        return self.slam.map

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    def request_reset(self):
        SemanticMappingBase.print("SemanticMapping: Requesting reset...")
        if self.reset_requested:
            SemanticMappingBase.print("SemanticMapping: reset already requested...")
            return
        with self.reset_mutex:
            self.reset_requested = True
        while True:
            with self.queue_condition:
                self.queue_condition.notify_all()  # to unblock self.pop_keyframe()
            with self.reset_mutex:
                if not self.reset_requested:
                    break
            time.sleep(0.1)
            SemanticMappingBase.print("SemanticMapping: waiting for reset...")
        SemanticMappingBase.print("SemanticMapping: ...Reset done.")

    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                SemanticMappingBase.print("SemanticMapping: reset_if_requested() starting...")
                empty_queue(self.queue)
                self.reset_requested = False
                SemanticMappingBase.print("SemanticMapping: reset_if_requested() ...done")

    def start(self):
        SemanticMappingBase.print(f"SemanticMapping: starting...")
        self.work_thread = Thread(target=self.run)
        self.work_thread.start()

    def quit(self):
        SemanticMappingBase.print("SemanticMapping: quitting...")
        if self.is_running and self.work_thread is not None:
            self.is_running = False
            self.work_thread.join(timeout=Parameters.kMultithreadingThreadJoinDefaultTimeout)
        SemanticMappingBase.print("SemanticMapping: done")

    # push the new keyframe and its image into the queue
    def push_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        with self.queue_condition:
            self.queue.put((keyframe, img, img_right, depth))
            self.queue_condition.notify_all()

    # blocking call
    def pop_keyframe(self, timeout=Parameters.kSemanticMappingTimeoutPopKeyframe):
        with self.queue_condition:
            if self.queue.empty():
                while self.queue.empty() and not self.stop_requested and not self.reset_requested:
                    ok = self.queue_condition.wait(timeout=timeout)
                    if not ok:
                        break  # Timeout occurred
                    # SemanticMappingBase.print('SemanticMapping: waiting for keyframe...')
        if self.queue.empty() or self.stop_requested:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except Exception as e:
            SemanticMappingBase.print(f"SemanticMapping: pop_keyframe: encountered exception: {e}")
            return None

    def queue_size(self):
        return self.queue.qsize()

    def is_idle(self):
        with self.idle_condition:
            return self._is_idle

    def is_ready(self):
        return self.is_running

    def set_idle(self, flag):
        with self.idle_condition:
            self._is_idle = flag
            self.idle_condition.notify_all()

    def wait_idle(self, print=print, timeout=None):
        if self.is_running == False:
            return
        with self.idle_condition:
            while not self._is_idle and self.is_running:
                SemanticMappingBase.print("SemanticMapping: waiting for idle...")
                ok = self.idle_condition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(
                        f"SemanticMapping: timeout {timeout}s reached, quit waiting for idle"
                    )
                    return

    def request_stop(self):
        with self.stop_mutex:
            Printer.yellow("requesting a stop for semantic mapping")
            self.stop_requested = True
        with self.queue_condition:
            self.queue_condition.notify_all()  # to unblock self.pop_keyframe()

    def is_stop_requested(self):
        with self.stop_mutex:
            return self.stop_requested

    def stop_if_requested(self):
        with self.stop_mutex:
            if self.stop_requested and not self.do_not_stop:
                self.stopped = True
                SemanticMappingBase.print("SemanticMapping: stopped...")
                return True
            return False

    def is_stopped(self):
        with self.stop_mutex:
            return self.stopped

    def set_do_not_stop(self, value):
        with self.stop_mutex:
            if value and self.stopped:
                return False
            self.do_not_stop = value
            return True

    def release(self):
        if not self.is_running:
            return
        with self.stop_mutex:
            self.stopped = False
            self.stop_requested = False
            # emtpy the queue
            while not self.queue.empty():
                self.queue.get()
            self.set_idle(True)
            SemanticMappingBase.print(f"SemanticMapping: released...")

    # Main loop of the semantic mapping thread
    def run(self):
        self.is_running = True
        while self.is_running:
            self.step()
        empty_queue(self.queue)  # empty the queue before exiting
        SemanticMappingBase.print("SemanticMapping: loop exit...")

    # Depending on the implementation the step might just add semantics to new frames, keyframes or it might
    # segment objects and track 3D segments
    def step(self):
        if self.map.num_keyframes() > 0:
            if not self.stop_requested:

                ret = self.pop_keyframe()  # blocking call
                if ret is not None:
                    (self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur) = ret
                    if self.kf_cur is not None:
                        self.last_processed_kf_img_id = self.kf_cur.img_id

                        self.set_idle(False)
                        try:
                            self.do_semantic_mapping()
                        except Exception as e:
                            SemanticMappingBase.print(
                                f"SemanticMapping: encountered exception: {e}"
                            )
                            SemanticMappingBase.print(traceback.format_exc())
                        self.set_idle(True)

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

    def semantic_mapping_impl(self):
        """
        Semantic mapping implementations assume as possible inputs new KF, RGB images and depth image
        They are provided in self.kf_cur, self.img_cur, self.img_cur_right, and self.depth_cur
        """
        raise NotImplementedError

    def do_semantic_mapping(self):
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

        self.semantic_mapping_impl()

        elapsed_time = time.time() - time_start
        self.time_semantic_mapping = elapsed_time
        SemanticMappingBase.print(f"semantic mapping elapsed time: {elapsed_time}")

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        return NotImplementedError

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return NotImplementedError

    def get_semantic_weight(self, semantic_des):
        return NotImplementedError

    @staticmethod
    def ensure_int32_prediction(prediction: np.ndarray, is_cast_to_int32_safe: bool):
        """Cast int64 predictions to int32 when value-safe, using a cached decision.
        - First time we see int64: compute min/max once and cache whether it's safe.
        - Subsequent frames reuse the cached decision to avoid repeated scans.
        Used only for C++ core compatibility.
        """
        if is_cast_to_int32_safe:
            return prediction.astype(np.int32, copy=False), is_cast_to_int32_safe
        if is_cast_to_int32_safe is None:
            info_i32 = np.iinfo(np.int32)
            vmin = prediction.min(initial=0)
            vmax = prediction.max(initial=0)
            is_cast_to_int32_safe = vmin >= info_i32.min and vmax <= info_i32.max
            if is_cast_to_int32_safe:
                return prediction.astype(np.int32, copy=False), is_cast_to_int32_safe
            else:
                Printer.yellow(
                    f"semantic prediction values out of int32 range [{vmin}, {vmax}], keeping dtype {prediction.dtype}"
                )
        return prediction, is_cast_to_int32_safe

    def __getstate__(self):
        """
        Custom pickling: exclude non-picklable attributes.
        This allows bound methods like sem_img_to_rgb to be pickled
        even though self.slam contains non-picklable objects (like ORBextractor).

        Note: semantic_segmentation and semantic_segmentation_process contain
        multiprocessing synchronized objects (mp.Value, mp.Lock, etc.) that can't be pickled
        with spawn method. The fallback in SemanticMappingShared.import_state() will recreate
        the callables.
        """
        state = self.__dict__.copy()
        # Exclude non-picklable attributes
        # self.slam contains ORBextractor and other C++ objects that can't be pickled
        if "slam" in state:
            del state["slam"]
        # Exclude threading primitives that can't be pickled
        if "queue" in state:
            del state["queue"]
        if "queue_condition" in state:
            del state["queue_condition"]
        if "work_thread" in state:
            del state["work_thread"]
        if "stop_mutex" in state:
            del state["stop_mutex"]
        if "reset_mutex" in state:
            del state["reset_mutex"]
        if "idle_condition" in state:
            del state["idle_condition"]
        # Note: semantic_segmentation might not be picklable (PyTorch models, CUDA tensors)
        # If it's not picklable, the bound method will fail, but the fallback
        # in SemanticMappingShared.import_state() will recreate the callables
        if "semantic_segmentation" in state:
            del state["semantic_segmentation"]
        # semantic_segmentation_process is now picklable (see SemanticSegmentationProcess.__getstate__)
        # It excludes multiprocessing primitives but keeps semantic_color_map which is what we need
        # So we don't exclude it here - let it be pickled
        return state

    def __setstate__(self, state):
        """
        Custom unpickling: restore state and set excluded attributes to None.
        The excluded attributes (like self.slam) are not needed for the
        callable methods (sem_img_to_rgb, etc.) to work.
        """
        self.__dict__.update(state)
        # Set excluded attributes to None (they're not needed for callable methods)
        if not hasattr(self, "slam"):
            self.slam = None
        # Note: Threading primitives are not restored - they're not needed
        # for the callable methods to work in spawned processes
        # Note: If semantic_segmentation was excluded and is None, the callable
        # methods won't work, but SemanticMappingShared.import_state() will recreate them
