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

import multiprocessing as mp
import threading

import numpy as np
import cv2
from enum import Enum
import traceback
import g2o

from pyslam.config_parameters import Parameters

from pyslam.slam import Map, optimizer_g2o, optimizer_gtsam

from pyslam.utilities.logging import Printer, Logging
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue, Value
from pyslam.utilities.timer import TimerFps

import logging


# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .slam import Slam
    from .map import Map
    from . import optimizer_gtsam
    from . import optimizer_g2o


kVerbose = True
kTimerVerbose = False
kPrintTrackebackDetails = True


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class GlobalBundleAdjustment:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(self, slam: "Slam", use_multiprocessing=True):
        self.init_print()
        GlobalBundleAdjustment.print(
            f"GlobalBundleAdjustment: starting with use_multiprocessing: {use_multiprocessing}"
        )
        # self.slam = slam
        self.map: Map = slam.map
        self.local_mapping = slam.local_mapping

        self.use_multiprocessing = use_multiprocessing

        # parameters for GBA
        self.loop_kf_id = -1
        self.rounds = 10
        self.use_robust_kernel = Parameters.kGBAUseRobustKernel

        self.process = None

        if use_multiprocessing:
            self.opt_abort_flag = None
            self.mp_opt_abort_flag = mp.Value("i", False)
            self.time_GBA = mp.Value("d", -1)
            self.mean_squared_error = mp.Value("d", -1)
            self._is_running = mp.Value("i", 0)  # True if the child process is running
            self._is_correcting = mp.Value("i", 0)  # True if the GBA is correcting
            # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
            self.mp_manager = MultiprocessingManager()
            self.q_message = self.mp_manager.Queue()
            self.result_dict_queue = (
                self.mp_manager.Queue()
            )  # we share the dictionary result as an entry of this dedicated multiprocessing queue
        else:
            self.opt_abort_flag = g2o.Flag(False)
            self.mp_opt_abort_flag = None
            self.time_GBA = Value("d", -1)
            self.mean_squared_error = Value("d", -1)
            self._is_running = Value("i", 0)
            self._is_correcting = Value("i", 0)
            self.q_message = []
            self.result_dict_queue = []

    def init_print(self):
        if kVerbose:
            if Parameters.kGBADebugAndPrintToFile:
                # redirect the prints of GBA to the file logs/gba.log
                # you can watch the output in separate shell by running:
                # $ tail -f logs/gba.log

                logging_file = Parameters.kLogsFolder + "/gba.log"
                GlobalBundleAdjustment.local_logger = Logging.setup_file_logger(
                    "gba_logger", logging_file, formatter=Logging.simple_log_formatter
                )

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return GlobalBundleAdjustment.local_logger.info(message, **kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return print(message, **kwargs)

            GlobalBundleAdjustment.print = staticmethod(print_file)

    def start(self, loop_kf_id):
        if self.is_running():
            Printer.red(
                "GlobalBundleAdjustment: GBA is already running! You can only have one GBA running at a time."
            )
            return

        if self.use_multiprocessing:
            empty_queue(
                self.q_message
            )  # clear the output queue, prevent messages from previous runs from interfering with the current run
            empty_queue(self.result_dict_queue)
        else:
            self.q_message.clear()
            self.result_dict_queue.clear()

        self._is_running.value = 0  # reset it to zero, then it is set to 1 in run()
        self._is_correcting.value = 0
        self.time_GBA.value = -1
        self.mean_squared_error.value = -1
        self.loop_kf_id = loop_kf_id

        if self.use_multiprocessing:
            self.mp_opt_abort_flag.value = False  # reset it to False
        else:
            self.opt_abort_flag.value = False

        keyframes = self.map.get_keyframes()
        points = self.map.get_points()

        args = (
            keyframes,
            points,
            self.loop_kf_id,
            self.rounds,
            self.use_robust_kernel,
            self.q_message,
            self.result_dict_queue,
            self._is_running,
            self._is_correcting,
            self.time_GBA,
            self.mean_squared_error,
            self.opt_abort_flag,
            self.mp_opt_abort_flag,
        )

        if self.use_multiprocessing:
            # launch child process
            GlobalBundleAdjustment.print("GlobalBundleAdjustment: starting child process...")
            self.process = mp.Process(target=self.run, args=args)
            self.process.daemon = True
        else:
            # launch thread
            GlobalBundleAdjustment.print("GlobalBundleAdjustment: starting thread...")
            self.process = threading.Thread(target=self.run, args=args)

        self.process.start()
        GlobalBundleAdjustment.print("GlobalBundleAdjustment: process.start() called")

    def is_running(self):
        return self._is_running.value == 1

    def has_finished(self):
        if self._is_running.value == 1:
            return False
        queue_size = self.q_message.qsize() if self.use_multiprocessing else len(self.q_message)
        has_new_messages = queue_size > 0
        if queue_size > 1:
            Printer.red(f"GlobalBundleAdjustment: WARNING: queue_size is {queue_size}!")
        if has_new_messages:
            output = self.q_message.get() if self.use_multiprocessing else self.q_message.pop(0)
            return output == "Finished"
        return False

    def is_correcting(self):
        return self._is_correcting.value == 1

    def abort(self):
        GlobalBundleAdjustment.print("GlobalBundleAdjustment: interrupting GBA...")
        if self.use_multiprocessing:
            self.mp_opt_abort_flag.value = True
        else:
            self.opt_abort_flag.value = True

    def quit(self):
        if self.is_running():
            GlobalBundleAdjustment.print("GlobalBundleAdjustment: quitting...")
            self.abort()
            if self.process.is_alive():
                timeout = (
                    Parameters.kMultiprocessingProcessJoinDefaultTimeout
                    if mp.get_start_method() != "spawn"
                    else 2 * Parameters.kMultiprocessingProcessJoinDefaultTimeout
                )
                self.process.join(timeout=timeout)
            if self.process.is_alive():
                message = "GlobalBundleAdjustment: WARNING: GBA process did not terminate in time, forced kill."
                GlobalBundleAdjustment.print(message)
                Printer.orange(message)
                self.process.terminate()
            if self.use_multiprocessing:
                empty_queue(self.q_message)
                # Shutdown the manager AFTER the process has exited
                if hasattr(self, "mp_manager") and self.mp_manager is not None:
                    try:
                        self.mp_manager.shutdown()
                    except Exception as e:
                        GlobalBundleAdjustment.print(f"Warning: Error shutting down manager: {e}")
            else:
                self.q_message.clear()
            self._is_running.value = 0
            self._is_correcting.value = 0
            GlobalBundleAdjustment.print("GlobalBundleAdjustment: done")

    # def check_GBA_has_finished_and_correct_if_needed(self):
    #     if (
    #         not self.is_running()
    #         and (self.q_message.qsize() if self.use_multiprocessing else len(self.q_message)) > 0
    #     ):
    #         output = self.q_message.get() if self.use_multiprocessing else self.q_message.pop(0)
    #         try:
    #             return self.correct_after_GBA()
    #         except Exception as e:
    #             GlobalBundleAdjustment.print(
    #                 f"GlobalBundleAdjustment: check_GBA_has_finished_and_correct_if_needed: encountered exception: {e}"
    #             )
    #             if kPrintTrackebackDetails:
    #                 traceback_details = traceback.format_exc()
    #                 GlobalBundleAdjustment.print(f"\t traceback details: {traceback_details}")
    #     return False

    def check_GBA_has_finished_and_correct_if_needed(self):
        if self.is_running():
            return False
        received_message = None
        queue_size = None
        try:
            # check if there is a new message in the queue
            queue_size = self.q_message.qsize() if self.use_multiprocessing else len(self.q_message)
            if queue_size > 0:
                received_message = (
                    self.q_message.get() if self.use_multiprocessing else self.q_message.pop(0)
                )
            else:
                # no new message, so GBA has not finished
                return False
            if queue_size > 1:
                Printer.red(f"GlobalBundleAdjustment: WARNING: queue_size is {queue_size}!")
            if received_message != "Finished":
                Printer.red(
                    f"GlobalBundleAdjustment: WARNING: received message is {received_message}!"
                )
        except Exception as e:
            GlobalBundleAdjustment.print(
                f"GlobalBundleAdjustment: check_GBA_has_finished_and_correct_if_needed: encountered exception: {e}"
            )
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f"\t traceback details: {traceback_details}")
            return False

        try:
            return self.correct_after_GBA()
        except Exception as e:
            GlobalBundleAdjustment.print(
                f"GlobalBundleAdjustment: check_GBA_has_finished_and_correct_if_needed: encountered exception: {e}"
            )
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f"\t traceback details: {traceback_details}")
        return False

    def correct_after_GBA(self):
        GlobalBundleAdjustment.print(f"GlobalBundleAdjustment: correct after GBA...")

        self._is_correcting.value = 1

        # Send a stop signal to Local Mapping
        # Avoid new keyframes are inserted while correcting the loop
        self.local_mapping.request_stop()
        # wait till local mapping is idle
        self.local_mapping.wait_idle(timeout=1.0, print=print)
        while self.local_mapping.queue_size() > 0:
            time.sleep(0.1)
            Printer.yellow(
                f"GlobalBundleAdjustment: waiting for local mapping to be idle and queue to be empty..."
            )

        GlobalBundleAdjustment.print("GlobalBundleAdjustment: starting correction ...")
        # get the updates from GBA results and put them in their temporary fields in the map
        loop_kf_id = self.loop_kf_id
        is_result_dict_queue_empty = (
            self.result_dict_queue.empty()
            if self.use_multiprocessing
            else len(self.result_dict_queue) == 0
        )
        if is_result_dict_queue_empty:
            Printer.red("GlobalBundleAdjustment: result_dict_queue is empty!")
            raise Exception("GlobalBundleAdjustment: result_dict_queue is empty!")
        result_dict = (
            self.result_dict_queue.get()
            if self.use_multiprocessing
            else self.result_dict_queue.pop(0)
        )
        keyframe_updates = result_dict["keyframe_updates"]
        point_updates = result_dict["point_updates"]

        keyframes = self.map.get_keyframes()
        points = self.map.get_points()

        num_kf_updates = 0
        num_pt_updates = 0

        num_kf_without_updates = 0
        num_pt_without_updates = 0

        # put frames back
        for kf in keyframes:
            try:
                T = keyframe_updates[kf.id]
                kf.Tcw_GBA = T
                kf.is_Tcw_GBA_valid = True
                kf.GBA_kf_id = loop_kf_id
                num_kf_updates += 1
            except:
                # print(f'GlobalBundleAdjustment: keyframe {kf.id} not in keyframe_updates')
                num_kf_without_updates += 1
                kf.is_Tcw_GBA_valid = False
                kf.GBA_kf_id = -1

        # put points back
        for p in points:
            try:
                p.pt_GBA = point_updates[p.id]
                p.is_pt_GBA_valid = True
                p.GBA_kf_id = loop_kf_id
                num_pt_updates += 1
            except:
                # print(f'GlobalBundleAdjustment: point {p.id} not in point_updates')
                num_pt_without_updates += 1
                p.is_pt_GBA_valid = False
                p.GBA_kf_id = -1

        GlobalBundleAdjustment.print(
            f"GlobalBundleAdjustment: got {num_kf_updates} keyframe updates and {num_pt_updates} point updates after GBA."
        )
        GlobalBundleAdjustment.print(
            f"GlobalBundleAdjustment: {num_kf_without_updates} keyframes without updates and {num_pt_without_updates} points without updates."
        )

        # Update all MapPoints and KeyFrames.
        # Local Mapping was active during BA, that means that there might be new keyframes
        # not included in the Global BA and they are not consistent with the updated map.
        # We need to propagate the correction through the spanning tree.
        try:
            # Get Map Mutex
            with self.map.update_lock:

                print(f"GlobalBundleAdjustment: correcting keyframes...")

                # Correct keyframes starting at map first keyframe
                keyframes_to_check = list(self.map.keyframe_origins)
                while keyframes_to_check:
                    keyframe = keyframes_to_check.pop(0)
                    child_keyframes = keyframe.get_children()
                    Twc = keyframe.Twc()

                    # if keyframe.Tcw_GBA is None:
                    if not keyframe.is_Tcw_GBA_valid:
                        GlobalBundleAdjustment.print(
                            f"GlobalBundleAdjustment: WARNING: keyframe {keyframe.id} (is_bad: {keyframe.is_bad()}) with invalid Tcw_GBA!"
                        )

                    # propagate the correction to children
                    for child in child_keyframes:
                        if child.GBA_kf_id != self.loop_kf_id:
                            # Only propagate if child was NOT optimized in current GBA
                            # if keyframe.Tcw_GBA is not None:
                            if keyframe.is_Tcw_GBA_valid:
                                T_child_c = child.Tcw() @ Twc
                                child.Tcw_GBA = T_child_c @ keyframe.Tcw_GBA
                                child.is_Tcw_GBA_valid = True
                                child.GBA_kf_id = self.loop_kf_id
                        keyframes_to_check.append(child)

                    keyframe.Tcw_before_GBA = keyframe.Tcw()
                    # if keyframe.Tcw_GBA is not None:
                    if keyframe.is_Tcw_GBA_valid:
                        keyframe.update_pose(keyframe.Tcw_GBA)

                print(f"GlobalBundleAdjustment: correcting map points...")

                # Correct MapPoints
                for map_point in self.map.get_points():
                    if map_point.is_bad():
                        continue

                    if map_point.GBA_kf_id == self.loop_kf_id and map_point.is_pt_GBA_valid:
                        # If optimized by Global BA, just update
                        map_point.update_position(map_point.pt_GBA)
                    else:
                        # Update according to the correction of its reference keyframe
                        ref_keyframe = map_point.get_reference_keyframe()
                        if ref_keyframe is None:
                            GlobalBundleAdjustment.print(
                                f"GlobalBundleAdjustment: WARNING: MapPoint {map_point.id} has no reference keyframe!"
                            )
                            continue

                        if ref_keyframe.GBA_kf_id != self.loop_kf_id:
                            continue

                        # Map to non-corrected camera
                        Rcw = ref_keyframe.Tcw_before_GBA[0:3, 0:3]
                        tcw = ref_keyframe.Tcw_before_GBA[0:3, 3]
                        Xc = Rcw @ map_point.pt() + tcw

                        # Backproject using corrected camera
                        Twc = ref_keyframe.Twc()
                        Rwc = Twc[0:3, 0:3]
                        twc = Twc[0:3, 3]

                        map_point.update_position(Rwc @ Xc + twc)
                    map_point.update_normal_and_depth()

                self._is_correcting.value = 0

                self.local_mapping.release()

                GlobalBundleAdjustment.print(f"GlobalBundleAdjustment: map updated!")
                return True

        except Exception as e:
            GlobalBundleAdjustment.print(f"GlobalBundleAdjustment: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f"\t traceback details: {traceback_details}")

        self._is_correcting.value = 0
        self.local_mapping.release()

        return False

    def run(
        self,
        keyframes,
        points,
        loop_kf_id,
        rounds,
        use_robust_kernel,
        q_message,
        result_dict_queue,
        is_running,
        is_correcting,
        time_GBA,
        mean_squared_error,
        opt_abort_flag,
        mp_opt_abort_flag,
    ):
        GlobalBundleAdjustment.print(
            f"GlobalBundleAdjustment: starting global bundle adjustment with loop_kf_id {loop_kf_id}..."
        )
        is_running.value = 1
        # time.sleep(0.1)

        timer = TimerFps("GlobalBundleAdjustment", is_verbose=kTimerVerbose)
        timer.start()

        task_completed = False

        result_dict = {}

        # Ensure containers are lists for pybind
        if not isinstance(keyframes, list):
            keyframes = list(keyframes)
        if not isinstance(points, list):
            points = list(points)

        try:
            if Parameters.kOptimizationBundleAdjustUseGtsam:
                global_bundle_adjustment_fun = optimizer_gtsam.global_bundle_adjustment
            else:
                global_bundle_adjustment_fun = optimizer_g2o.global_bundle_adjustment
            mean_squared_error.value, result_dict = global_bundle_adjustment_fun(
                keyframes=keyframes,
                points=points,
                rounds=rounds,
                loop_kf_id=loop_kf_id,
                use_robust_kernel=use_robust_kernel,
                abort_flag=opt_abort_flag,
                mp_abort_flag=mp_opt_abort_flag,
                result_dict=result_dict,
                verbose=False,
                print=print,
            )
            task_completed = True
        except Exception as e:
            GlobalBundleAdjustment.print(f"GlobalBundleAdjustment: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f"\t traceback details: {traceback_details}")
            mean_squared_error.value = -1
            time_GBA.value = -1

        if self.use_multiprocessing:
            result_dict_queue.put(result_dict)
        else:
            result_dict_queue.append(result_dict)

        if task_completed:
            timer.refresh()
            time_GBA.value = timer.last_elapsed
            # push a simple task-completed signal
            if self.use_multiprocessing:
                q_message.put("Finished")
            else:
                q_message.append("Finished")

        is_running.value = 0
        GlobalBundleAdjustment.print(
            f"GlobalBundleAdjustment: task completed {task_completed}, mean_squared_error: {mean_squared_error.value}, elapsed time: {time_GBA.value}"
        )
