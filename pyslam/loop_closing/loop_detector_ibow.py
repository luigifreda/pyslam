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
import numpy as np
import cv2
from enum import Enum

from pyslam.utilities.logging import Printer
from pyslam.utilities.features import transform_float_to_binary_descriptor

from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_types import FeatureInfo
from pyslam.slam.feature_tracker_shared import SlamFeatureManagerInfo
from pyslam.utilities.timer import TimerFps

from .loop_detector_base import (
    LoopDetectorTaskType,
    LoopDetectKeyframeData,
    LoopDetectorTask,
    LoopDetectorOutput,
    LoopDetectorBase,
)

import pyslam.config as config

config.cfg.set_lib("pyibow")
import pyibow as ibow


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# NOTE: It does not need any prebuilt vocubulary. It works with non-binary descriptors by internally converting them to binary ones.
class LoopDetectorIBow(LoopDetectorBase):
    def __init__(self, local_feature_manager=None, slam_info: SlamFeatureManagerInfo | None = None):
        super().__init__()
        self.local_feature_manager = local_feature_manager
        self.slam_info = slam_info
        self.lc_detector_parameters = ibow.LCDetectorParams()
        self.lc_detector_parameters.p = 50  # default in ibow: 250
        LoopDetectorBase.print(
            f"LoopDetectorIBow: min number of images to start detecting loops: {self.lc_detector_parameters.p}"
        )
        self.lc_detector = ibow.LCDetector(self.lc_detector_parameters)

    def reset(self):
        LoopDetectorBase.reset(self)
        self.lc_detector.clear()

    def save(self, path):
        if self.lc_detector.num_pushed_images() < self.lc_detector_parameters.p:
            Printer.red(
                f"LoopDetectorIBow: not enough keyframes ({self.lc_detector.num_pushed_images()}) to save the database. Need at least {self.lc_detector_parameters.p}"
            )
            Printer.red(f"\t You wont be able to relocalize in the saved map!!!")
            return
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorIBow: saving database to {filepath}...")
        self.lc_detector.print_status()
        self.lc_detector.save(filepath)

    def load(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorIBow: loading database from {filepath}...")
        self.lc_detector.load(filepath)
        self.lc_detector.print_status()
        LoopDetectorBase.print(f"LoopDetectorIBow: ...done")

    def run_task(self, task: LoopDetectorTask):
        LoopDetectorBase.print(
            f"LoopDetectorIBow: running task {task.keyframe_data.id}, entry_id = {self.entry_id}, task_type = {task.task_type.name}"
        )
        keyframe = task.keyframe_data
        frame_id = keyframe.id

        if self.loop_detection_imgs is not None:
            self.map_frame_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()

        self.resize_similary_matrix_if_needed()

        kps, des = keyframe.kps, keyframe.des
        # print(f'LoopDetectorIBow: kps = {len(kps)}, des = {des.shape}')
        if len(kps) > 0 and len(kps[0]) > 2:
            kps_ = [
                (kp[0], kp[1], kp[2], kp[3], kp[4], kp[5]) for kp in kps
            ]  # tuple_x_y_size_angle_response_octave
        else:
            # kp.response is not actually used
            # kps_ = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]  # tuple_x_y_size_angle_response_octave
            kps_ = [
                (kp[0], kp[1], keyframe.sizes[i], keyframe.angles[i], 1, keyframe.octaves[i])
                for i, kp in enumerate(kps)
            ]  # tuple_x_y_size_angle_response_octave
        des_ = des

        # if we are not using a binary descriptr then we conver the float descriptors to binary
        if self.slam_info.feature_descriptor_norm_type != cv2.NORM_HAMMING:
            des_ = transform_float_to_binary_descriptor(des)

        candidate_idxs = []
        candidate_scores = []
        g_des = None

        # the img_ids are mapped to entry_ids (entry ids) inside the database management
        self.map_entry_id_to_frame_id[self.entry_id] = frame_id

        detection_output = LoopDetectorOutput(
            task_type=task.task_type, g_des_vec=g_des, frame_id=frame_id, img=keyframe.img
        )

        result = None

        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            LoopDetectorBase.print(f"LoopDetectorIBow: relocalization task")
            if kps_ is None or len(kps_) == 0:
                LoopDetectorBase.print(f"LoopDetectorIBow: relocalization task: no keypoints")
            if des_ is None or des_.shape[0] == 0:
                LoopDetectorBase.print(f"LoopDetectorIBow: relocalization task: no descriptors")
            result = self.lc_detector.process_without_pushing(self.entry_id, kps_, des_)
            other_entry_id = result.train_id
            other_frame_id = self.map_entry_id_to_frame_id[other_entry_id]

            if result.isLoop():
                candidate_idxs.append(other_frame_id)
                candidate_scores.append(result.score)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores

        else:
            LoopDetectorBase.print(f"LoopDetectorIBow: loop closure task: {task.task_type.name}")
            result = self.lc_detector.process(self.entry_id, kps_, des_)
            other_entry_id = result.train_id
            other_frame_id = self.map_entry_id_to_frame_id[other_entry_id]

            self.update_similarity_matrix(
                score=result.score, entry_id=self.entry_id, other_entry_id=other_entry_id
            )

            if result.isLoop():
                if (
                    abs(other_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure
                    and other_frame_id not in task.connected_keyframes_ids
                ):
                    candidate_idxs.append(other_frame_id)
                    candidate_scores.append(result.score)
                    self.update_loop_closure_imgs(score=result.score, other_frame_id=other_frame_id)

            self.draw_loop_detection_imgs(keyframe.img, frame_id, detection_output)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [
                cov_kf.g_des.toVec() if cov_kf.g_des is not None else None
                for cov_kf in task.covisible_keyframes_data
            ]

        if result is not None:
            if result.status == ibow.LCDetectorStatus.LC_DETECTED:
                # NOTE: it's normal to get zero inliers in some cases where the loop is detected, for instance:
                #       consecutive_loops_ > min_consecutive_loops_ and island.overlaps(last_lc_island_)
                LoopDetectorBase.print(
                    f"LoopDetectorIBow: Loop detected: {result.train_id}, #inliers: {result.inliers}, score: {result.score}"
                )
            elif result.status == ibow.LCDetectorStatus.LC_NOT_DETECTED:
                LoopDetectorBase.print("LoopDetectorIBow: No loop found")
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_IMAGES:
                LoopDetectorBase.print(
                    f"LoopDetectorIBow: Not enough images to found a loop, min number of processed images for loop: {self.lc_detector_parameters.p}, number of pushed images: {self.lc_detector.num_pushed_images()}"
                )
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_ISLANDS:
                LoopDetectorBase.print("LoopDetectorIBow: Not enough islands to found a loop")
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_INLIERS:
                LoopDetectorBase.print("LoopDetectorIBow: Not enough inliers")
            elif result.status == ibow.LCDetectorStatus.LC_TRANSITION:
                LoopDetectorBase.print("LoopDetectorIBow: Transitional loop closure")
            else:
                LoopDetectorBase.print("LoopDetectorIBow: No status information")

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # NOTE: with relocalization we don't need to increment the entry_id since we don't add frames to database
            self.entry_id += 1

        return detection_output
