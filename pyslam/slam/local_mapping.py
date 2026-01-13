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

from __future__ import print_function  # This must be the first statement before other statements

import os
import time
import numpy as np
import g2o
from enum import Enum

from collections import defaultdict

from threading import RLock, Thread, Condition
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from pyslam.config_parameters import Parameters

from .frame import compute_frame_matches

from pyslam.slam import EpipolarMatcher  # , LocalMappingCore

kUseCppLocalMappingCore = True  # True: Use the C++ local mapping core when available, False: Force the usage of the Python local mapping core
if kUseCppLocalMappingCore:
    # use the C++ local mapping core
    from pyslam.slam import LocalMappingCore
else:
    # just for debugging this force the usage of the python local mapping core
    from .local_mapping_core import LocalMappingCore


from pyslam.io.dataset_types import SensorType
from pyslam.utilities.timer import TimerFps

from pyslam.utilities.logging import Printer, Logging
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.geom_triangulation import triangulate_normalized_points
from pyslam.utilities.data_management import empty_queue

import multiprocessing as mp
import traceback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .slam import Slam  #
    from .keyframe import KeyFrame
    from .geometry_matchers import EpipolarMatcher
    from .local_mapping_core import LocalMappingCore


kVerbose = True
kTimerVerbose = False

kLocalMappingDebugAndPrintToFile = Parameters.kLocalMappingDebugAndPrintToFile

kUseLargeWindowBA = Parameters.kUseLargeWindowBA

kNumMinObsForKeyFrameTrackedPoints = 3

kLocalMappingSleepTime = 5e-3  # [s]

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


def kf_search_frame_for_triangulation(
    kf1, kf2, kf2_idx, idxs1, idxs2, max_descriptor_distance, is_monocular, result_queue
):
    idxs1_out, idxs2_out, num_found_matches = EpipolarMatcher.search_frame_for_triangulation(
        kf1, kf2, idxs1, idxs2, max_descriptor_distance, is_monocular
    )
    LocalMapping.print(
        f"\t kf_search_frame_for_triangulation: found for ({kf1.id}, {kf2.id}), #potential matches: {num_found_matches}"
    )
    result_queue.put((idxs1_out, idxs2_out, num_found_matches, kf2_idx))


class LocalMapping:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(self, slam: "Slam"):
        self.slam = slam
        self.local_mapping_core = LocalMappingCore(slam.map, slam.sensor_type)

        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings
        self.timer_triangulation = TimerFps("Triangulation", is_verbose=self.timer_verbose)
        self.timer_pts_culling = TimerFps("Culling points", is_verbose=self.timer_verbose)
        self.timer_kf_culling = TimerFps("Culling kfs", is_verbose=self.timer_verbose)
        self.timer_pts_fusion = TimerFps("Fusing points", is_verbose=self.timer_verbose)
        self.timer_large_BA = TimerFps("Large window BA", is_verbose=self.timer_verbose)
        self.time_local_opt = TimerFps("Local optimization", is_verbose=self.timer_verbose)
        self.time_large_opt = TimerFps("Large window optimization", is_verbose=self.timer_verbose)

        self.queue = Queue()
        self.queue_condition = Condition()
        self.work_thread = None  # Thread(target=self.run)
        self.is_running = False

        self._is_idle = True
        self.idle_condition = Condition()

        self.stop_requested = False
        self.do_not_stop = False
        self.stopped = False
        self.stop_mutex = RLock()

        self.reset_requested = False
        self.reset_mutex = RLock()

        self.log_file = None
        self.thread_large_BA = None

        self.depth_cur = None
        self.img_cur_right = None
        self.img_cur = None

        self.mean_ba_chi2_error = None
        self.time_local_mapping = None

        self.far_points_threshold = None  # read and set by Slam
        self.use_fov_centers_based_kf_generation = False  # read and set by Slam
        self.max_fov_centers_distance = -1  # read and set by Slam

        self.last_processed_kf_img_id = None

        self.last_num_triangulated_points = None
        self.total_num_triangulated_points = 0
        self.last_num_fused_points = None
        self.total_num_fused_points = 0
        self.last_num_culled_points = None
        self.total_num_culled_points = 0

        self.last_num_culled_keyframes = None
        self.total_num_culled_keyframes = 0

        self.init_print()

    def init_print(self):
        if kVerbose:
            if Parameters.kLocalMappingOnSeparateThread:
                if kLocalMappingDebugAndPrintToFile:
                    # Default log to file: logs/local_mapping.log
                    logging_file = os.path.join(Parameters.kLogsFolder, "local_mapping.log")
                    LocalMapping.local_logger = Logging.setup_file_logger(
                        "local_mapping_logger", logging_file, formatter=Logging.simple_log_formatter
                    )

                    def file_print(*args, **kwargs):
                        message = " ".join(str(arg) for arg in args)
                        LocalMapping.local_logger.info(message, **kwargs)

                else:

                    def file_print(*args, **kwargs):
                        message = " ".join(str(arg) for arg in args)
                        return print(message, **kwargs)

                LocalMapping.print = staticmethod(file_print)
            else:
                LocalMapping.print = staticmethod(print)
            if hasattr(LocalMappingCore, "print"):
                LocalMappingCore.print = LocalMapping.print

    @property
    def map(self):
        return self.slam.map

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def kf_cur(self):
        return self.local_mapping_core.kf_cur

    @kf_cur.setter
    def kf_cur(self, value):
        self.local_mapping_core.kf_cur = value

    @property
    def kid_last_BA(self):
        return self.local_mapping_core.kid_last_BA

    @kid_last_BA.setter
    def kid_last_BA(self, value):
        self.local_mapping_core.kid_last_BA = value

    @property
    def descriptor_distance_sigma(self):
        return self.slam.tracking.descriptor_distance_sigma

    def request_reset(self):
        LocalMapping.print("LocalMapping: Requesting reset...")
        if self.reset_requested:
            LocalMapping.print("LocalMapping: reset already requested...")
            return
        with self.reset_mutex:
            self.reset_requested = True
        if Parameters.kLocalMappingOnSeparateThread:
            while True:
                with self.queue_condition:
                    self.queue_condition.notify_all()  # to unblock self.pop_keyframe()
                with self.reset_mutex:
                    if not self.reset_requested:
                        break
                time.sleep(0.1)
                LocalMapping.print("LocalMapping: waiting for reset...")
        LocalMapping.print("LocalMapping: ...Reset done.")

    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                LocalMapping.print("LocalMapping: reset_if_requested() starting...")
                empty_queue(self.queue)
                self.reset_requested = False
                self.total_num_triangulated_points = 0
                self.total_num_fused_points = 0
                self.total_num_culled_points = 0
                self.total_num_culled_keyframes = 0
                self.last_num_triangulated_points = None
                self.local_mapping_core.reset()
                LocalMapping.print("LocalMapping: reset_if_requested() ...done")

    def start(self):
        LocalMapping.print(f"LocalMapping: starting...")
        self.work_thread = Thread(target=self.run)
        self.work_thread.start()

    def quit(self):
        LocalMapping.print("LocalMapping: quitting...")
        if self.is_running and self.work_thread is not None:
            self.is_running = False
            self.local_mapping_core.set_opt_abort_flag(True)
            self.work_thread.join(timeout=5)
        LocalMapping.print("LocalMapping: done")

    def set_opt_abort_flag(self, value):
        self.local_mapping_core.set_opt_abort_flag(value)

    # push the new keyframe and its image into the queue
    def push_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        with self.queue_condition:
            self.queue.put((keyframe, img, img_right, depth))
            self.queue_condition.notify_all()
            self.set_opt_abort_flag(True)

    # blocking call
    def pop_keyframe(self, timeout=Parameters.kLocalMappingTimeoutPopKeyframe):
        with self.queue_condition:
            if self.queue.empty():
                while self.queue.empty() and not self.stop_requested and not self.reset_requested:
                    ok = self.queue_condition.wait(timeout=timeout)
                    if not ok:
                        break  # Timeout occurred
                    # LocalMapping.print('LocalMapping: waiting for keyframe...')
        if self.queue.empty() or self.stop_requested:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except Exception as e:
            LocalMapping.print(f"LocalMapping: pop_keyframe: encountered exception: {e}")
            return None

    def queue_size(self):
        return self.queue.qsize()

    def is_idle(self):
        with self.idle_condition:
            return self._is_idle

    def set_idle(self, flag):
        with self.idle_condition:
            self._is_idle = flag
            self.idle_condition.notify_all()

    def wait_idle(self, print=print, timeout=None):
        if self.is_running == False:
            return
        with self.idle_condition:
            while not self._is_idle and self.is_running:
                LocalMapping.print("LocalMapping: waiting for idle...")
                ok = self.idle_condition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(
                        f"LocalMapping: timeout {timeout}s reached, quit waiting for idle"
                    )
                    return

    def interrupt_optimization(self):
        Printer.yellow("interrupting local mapping optimization")
        self.set_opt_abort_flag(True)

    def request_stop(self):
        with self.stop_mutex:
            Printer.yellow("requesting a stop for local mapping optimization")
            self.stop_requested = True
        with self.queue_condition:
            self.set_opt_abort_flag(True)
            self.queue_condition.notify_all()  # to unblock self.pop_keyframe()

    def is_stop_requested(self):
        with self.stop_mutex:
            return self.stop_requested

    def stop_if_requested(self):
        with self.stop_mutex:
            if self.stop_requested and not self.do_not_stop:
                self.stopped = True
                LocalMapping.print("LocalMapping: stopped...")
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

    # release the local mapping thread
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
            LocalMapping.print(f"LocalMapping: released...")

    def run(self):
        self.is_running = True
        while self.is_running:
            self.step()
        empty_queue(self.queue)  # empty the queue before exiting
        LocalMapping.print("LocalMapping: loop exit...")

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
                            self.do_local_mapping()
                        except Exception as e:
                            LocalMapping.print(f"LocalMapping: encountered exception: {e}")
                            LocalMapping.print(traceback.format_exc())
                        self.set_idle(True)
                else:
                    # pop_keyframe returned None (queue empty or stopped)
                    # Ensure we're marked as idle when waiting for keyframes
                    self.set_idle(True)

            elif self.stop_if_requested():
                self.set_idle(True)
                while self.is_stopped():
                    LocalMapping.print(f"LocalMapping: stopped, idle: {self._is_idle} ...")
                    time.sleep(kLocalMappingSleepTime)
        else:
            msg = "LocalMapping: waiting for keyframes..."
            # Printer.red(msg)
            # LocalMapping.print(msg)
            time.sleep(kLocalMappingSleepTime)
        self.reset_if_requested()

    def do_local_mapping(self):
        LocalMapping.print("local mapping: starting...")

        Printer.cyan("@local mapping")
        time_start = time.time()

        if self.kf_cur is None:
            Printer.red("local mapping: no keyframe to process")
            return

        if Parameters.kLocalMappingOnSeparateThread:
            LocalMapping.print("..................................")
            LocalMapping.print(
                "processing KF: ", self.kf_cur.id, ", queue size: ", self.queue_size()
            )

        is_single_thread = not Parameters.kLocalMappingOnSeparateThread

        # LocalMapping.print('descriptor_distance_sigma: ', self.descriptor_distance_sigma)

        self.process_new_keyframe()

        # do map points culling
        self.timer_pts_culling.start()
        num_culled_points = self.cull_map_points()
        self.last_num_culled_points = num_culled_points
        self.total_num_culled_points += num_culled_points
        self.timer_pts_culling.refresh()
        LocalMapping.print(
            f"#culled points: {num_culled_points}, timing: {self.timer_pts_culling.last_elapsed}"
        )

        # create new points by triangulation
        self.timer_triangulation.start()
        total_new_pts = self.create_new_map_points()
        # total_new_pts = self.create_new_map_points_parallel()  # use parallel triangulation
        self.last_num_triangulated_points = total_new_pts
        self.total_num_triangulated_points += total_new_pts
        self.timer_triangulation.refresh()
        LocalMapping.print(
            f"#new map points: {total_new_pts}, timing: {self.timer_triangulation.last_elapsed}"
        )

        if self.queue.empty() or is_single_thread:
            # fuse map points of close keyframes
            self.timer_pts_fusion.start()
            total_fused_pts = self.fuse_map_points()
            self.last_num_fused_points = total_fused_pts
            self.total_num_fused_points += total_fused_pts
            self.timer_pts_fusion.refresh()
            LocalMapping.print(
                f"#fused map points: {total_fused_pts}, timing: {self.timer_pts_fusion.last_elapsed}"
            )

        # reset optimization abort flag
        self.set_opt_abort_flag(False)

        if (self.queue.empty() and not self.is_stop_requested()) or is_single_thread:

            if self.thread_large_BA is not None:
                if (
                    self.thread_large_BA.is_alive()
                ):  # for security, check if large BA thread finished his work
                    self.thread_large_BA.join()
                    self.timer_large_BA.refresh()
                    LocalMapping.print(f"large BA timing: {self.timer_large_BA.last_elapsed}")

            # launch local bundle adjustment
            self.local_BA()

            if (
                kUseLargeWindowBA
                and self.kf_cur.kid >= Parameters.kEveryNumFramesLargeWindowBA
                and self.kid_last_BA != self.kf_cur.kid
                and self.kf_cur.kid % Parameters.kEveryNumFramesLargeWindowBA == 0
            ):
                self.timer_large_BA.start()
                # launch a parallel large window BA of the map
                self.thread_large_BA = Thread(target=self.large_window_BA)
                self.thread_large_BA.start()

            # check redundant local Keyframes
            self.timer_kf_culling.start()
            num_culled_keyframes = self.cull_keyframes()
            self.last_num_culled_keyframes = num_culled_keyframes
            self.total_num_culled_keyframes += num_culled_keyframes
            self.timer_kf_culling.refresh()
            LocalMapping.print(
                f"\t #culled keyframes: {num_culled_keyframes}, timing: {self.timer_kf_culling.last_elapsed}"
            )

        if self.slam.semantic_mapping is not None and self.kf_cur is not None:
            LocalMapping.print("pushing new keyframe to semantic mapping...")
            self.slam.semantic_mapping.push_keyframe(
                self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur
            )
            # self.slam.semantic_mapping.set_do_not_stop(False) # TODO(dvdmc) check if the stop logic is required. I think it's part of LoopClosure or BA methods
            if not Parameters.kSemanticMappingOnSeparateThread:
                self.slam.semantic_mapping.step()

        if self.slam.loop_closing is not None and self.kf_cur is not None:
            LocalMapping.print(f"pushing new keyframe {self.kf_cur.id} to loop closing...")
            self.slam.loop_closing.add_keyframe(self.kf_cur, self.img_cur)

        if self.slam.volumetric_integrator is not None:
            LocalMapping.print("pushing new keyframe to volumetric integrator...")
            if self.kf_cur.img is None:
                self.kf_cur.img = self.img_cur
            if self.kf_cur.img_right is None:
                self.kf_cur.img_right = self.img_cur_right
            if self.kf_cur.depth_img is None:
                self.kf_cur.depth_img = self.depth_cur
            # NOTE: When the keyframe is added to the volumetric integrator, the keyframe data are snapshots
            # of the keyframe data at the time of the push, and are not updated anymore...
            self.slam.volumetric_integrator.add_keyframe(
                self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur
            )

        elapsed_time = time.time() - time_start
        self.time_local_mapping = elapsed_time
        LocalMapping.print(f"local mapping elapsed time: {elapsed_time}")

    def local_BA(self):
        if self.slam.loop_closing is not None and self.slam.loop_closing.is_correcting():
            return
        # local optimization
        self.time_local_opt.start()
        LocalMapping.print(">>>> local optimization (LBA) ...")
        err, num_kf_ref_tracked_points = self.local_mapping_core.local_BA()
        self.mean_ba_chi2_error = err
        self.time_local_opt.refresh()
        LocalMapping.print(
            f"local optimization (LBA) error^2: {err}, timing: {self.time_local_opt.last_elapsed}"
        )
        Printer.green("KF(%d) #points: %d " % (self.kf_cur.id, num_kf_ref_tracked_points))

    def large_window_BA(self):
        Printer.blue("@large BA")
        # large window optimization of the map
        self.time_large_opt.start()
        err = self.local_mapping_core.large_window_BA()
        self.time_large_opt.refresh()
        Printer.blue("large window optimization error^2: %f, KF id: %d" % (err, self.kf_cur.kid))

    def process_new_keyframe(self):
        # associate map points to keyframe observations (only good points)
        # and update normal and descriptor
        LocalMapping.print(f">>>> processing new keyframe ...")
        self.local_mapping_core.process_new_keyframe()

    def cull_map_points(self):
        LocalMapping.print(">>>> culling map points...")
        return self.local_mapping_core.cull_map_points()

    def cull_keyframes(self):
        LocalMapping.print(">>>> culling keyframes...")
        # check redundant keyframes in local keyframes: a keyframe is considered redundant if the 90% of the MapPoints it sees,
        # are seen in at least other 3 keyframes (in the same or finer scale)
        return self.local_mapping_core.cull_keyframes(
            self.use_fov_centers_based_kf_generation, self.max_fov_centers_distance
        )

    # triangulate matched keypoints (without a corresponding map point) amongst recent keyframes
    def create_new_map_points(self):
        LocalMapping.print(">>>> creating new map points")
        total_new_pts = 0

        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular
        else:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo

        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        LocalMapping.print(
            "local map keyframes: ",
            [kf.id for kf in local_keyframes if not kf.is_bad()],
            " + ",
            self.kf_cur.id,
            "...",
        )

        match_idxs = defaultdict(
            lambda: (None, None)
        )  # dictionary of matches  (kf_i, kf_j) -> (idxs_i,idxs_j)
        # precompute keypoint matches
        match_idxs = compute_frame_matches(
            self.kf_cur,
            local_keyframes,
            match_idxs,
            do_parallel=Parameters.kLocalMappingParallelKpsMatching,
            max_workers=Parameters.kLocalMappingParallelKpsMatchingNumWorkers,
            print_fun=LocalMapping.print,
            is_monocular=(self.sensor_type == SensorType.MONOCULAR),
        )

        # LocalMapping.print(f'\t processing local keyframes...')
        idxs_and_kfs = [
            (i, kf)
            for i, kf in enumerate(local_keyframes)
            if kf is not self.kf_cur and not kf.is_bad()
        ]
        # for i,kf in enumerate(local_keyframes):
        #     if kf is self.kf_cur or kf.is_bad():
        #         continue
        for i, kf in idxs_and_kfs:
            if i > 0 and not self.queue.empty():
                LocalMapping.print("creating new map points *** interruption ***")
                return total_new_pts

            # extract matches from precomputed map
            idxs_kf_cur, idxs_kf = match_idxs[(self.kf_cur, kf)]

            # LocalMapping.print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id})...')

            # find keypoint matches between self.kf_cur and kf
            # N.B.: all the matched keypoints computed by EpipolarMatcher.search_frame_for_triangulation() are without a corresponding map point
            idxs_cur, idxs, num_found_matches = EpipolarMatcher.search_frame_for_triangulation(
                self.kf_cur,
                kf,
                idxs_kf_cur,
                idxs_kf,
                max_descriptor_distance=0.5 * self.descriptor_distance_sigma,
                is_monocular=(self.sensor_type == SensorType.MONOCULAR),
            )
            if num_found_matches == 0:
                continue  # Skip if no matches found

            # LocalMapping.print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id}), #potential matches: {num_found_matches}')

            if len(idxs_cur) > 0:
                # try to triangulate the matched keypoints that do not have a corresponding map point
                pts3d, mask_pts3d = triangulate_normalized_points(
                    self.kf_cur.pose(), kf.pose(), self.kf_cur.kpsn[idxs_cur], kf.kpsn[idxs]
                )

                new_pts_count, _, list_added_points = self.map.add_points(
                    pts3d,
                    mask_pts3d,
                    self.kf_cur,
                    kf,
                    idxs_cur,
                    idxs,
                    self.kf_cur.img,
                    do_check=True,
                    far_points_threshold=self.far_points_threshold,
                )
                LocalMapping.print(
                    f"\t #added map points: {new_pts_count} for KFs ({self.kf_cur.id}), ({kf.id})"
                )
                total_new_pts += new_pts_count
                self.local_mapping_core.add_points(list_added_points)
        return total_new_pts

    def create_new_map_points_parallel(self):
        LocalMapping.print(">>>> creating new map points parallel")
        total_new_pts = 0

        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular
        else:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo

        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        LocalMapping.print(
            "local map keyframes: ",
            [kf.id for kf in local_keyframes if not kf.is_bad()],
            " + ",
            self.kf_cur.id,
            "...",
        )

        match_idxs = defaultdict(
            lambda: (None, None)
        )  # dictionary of matches  (kf_i, kf_j) -> (idxs_i,idxs_j)
        # precompute keypoint matches
        match_idxs = compute_frame_matches(
            self.kf_cur,
            local_keyframes,
            match_idxs,
            do_parallel=Parameters.kLocalMappingParallelKpsMatching,
            max_workers=Parameters.kLocalMappingParallelKpsMatchingNumWorkers,
            print_fun=LocalMapping.print,
            is_monocular=(self.sensor_type == SensorType.MONOCULAR),
        )

        tasks = []
        processes = []
        mp_manager = MultiprocessingManager()
        result_queue = mp_manager.Queue()

        idxs_and_kfs = [
            (i, kf)
            for i, kf in enumerate(local_keyframes)
            if kf is not self.kf_cur and not kf.is_bad()
        ]
        # for kf_idx,kf in enumerate(local_keyframes):
        #     if kf is self.kf_cur or kf.is_bad():
        #         continue
        for kf_idx, kf in idxs_and_kfs:
            if kf_idx > 0 and not self.queue.empty():
                LocalMapping.print("creating new map points *** interruption ***")
                return total_new_pts

            idxs_kf_cur, idxs_kf = match_idxs[(self.kf_cur, kf)]
            kfs_data = (
                self.kf_cur,
                kf,
                kf_idx,
                idxs_kf_cur,
                idxs_kf,
                0.5 * self.descriptor_distance_sigma,
                self.sensor_type == SensorType.MONOCULAR,
                result_queue,
            )
            process = mp.Process(target=kf_search_frame_for_triangulation, args=kfs_data)
            processes.append(process)

        for process in processes:
            process.start()

        LocalMapping.print(f"\t waiting for triangulation results...")
        for p in processes:
            p.join()

        LocalMapping.print(f"\t processing triangulation results...")
        while not result_queue.empty():
            result = result_queue.get()
            idxs_cur, idxs, num_found_matches, kf_idx = result
            kf = local_keyframes[kf_idx]
            # LocalMapping.print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id}), #potential matches: {num_found_matches}')
            if len(idxs_cur) > 0:
                # try to triangulate the matched keypoints that do not have a corresponding map point
                pts3d, mask_pts3d = triangulate_normalized_points(
                    self.kf_cur.pose(), kf.pose(), self.kf_cur.kpsn[idxs_cur], kf.kpsn[idxs]
                )

                new_pts_count, _, list_added_points = self.map.add_points(
                    pts3d,
                    mask_pts3d,
                    self.kf_cur,
                    kf,
                    idxs_cur,
                    idxs,
                    self.kf_cur.img,
                    do_check=True,
                    far_points_threshold=self.far_points_threshold,
                )
                LocalMapping.print(
                    f"\t #added map points: {new_pts_count} for KFs ({self.kf_cur.id}), ({kf.id})"
                )
                total_new_pts += new_pts_count
                self.local_mapping_core.add_points(list_added_points)
        return total_new_pts

    # fuse close map points of local keyframes
    def fuse_map_points(self):
        LocalMapping.print(">>>> fusing map points")
        return self.local_mapping_core.fuse_map_points(self.descriptor_distance_sigma)
