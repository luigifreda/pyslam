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

config.cfg.set_lib("pyobindex2")
import pyobindex2 as obindex2


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# NOTE: It does not need any prebuilt vocubulary. It works with non-binary descriptors by internally converting them to binary ones.
class LoopDetectorOBIndex2(LoopDetectorBase):
    def __init__(
        self,
        local_feature_manager=None,
        slam_info: SlamFeatureManagerInfo = None,
        match_ratio=0.8,
    ):
        super().__init__()
        self.match_ratio = match_ratio
        self.local_feature_manager = local_feature_manager
        self.slam_info = slam_info
        # Creating a new index of images
        self.index = obindex2.ImageIndex(16, 150, 4, obindex2.MERGE_POLICY_AND, True)

    def reset(self):
        LoopDetectorBase.reset(self)
        self.index.clear()

    def save(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorOBIndex2: saving database to {filepath}...")
        self.index.print_status()
        self.index.save(filepath)

    def load(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorOBIndex2: loading database from {filepath}...")
        self.index.load(filepath)
        self.index.print_status()
        LoopDetectorBase.print(f"LoopDetectorOBIndex2: ...done")

    def run_task(self, task: LoopDetectorTask):
        LoopDetectorBase.print(
            f"LoopDetectorOBIndex2: running task {task.keyframe_data.id}, entry_id = {self.entry_id}, task_type = {task.task_type.name}"
        )
        keyframe = task.keyframe_data
        frame_id = keyframe.id

        if self.loop_detection_imgs is not None:
            self.map_frame_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()

        self.resize_similary_matrix_if_needed()

        kps, des = keyframe.kps, keyframe.des
        # print(f'LoopDetectorOBIndex2: kps = {len(kps)}, des = {des.shape}')
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

        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            LoopDetectorBase.print(f"LoopDetectorOBIndex2: relocalization task")
            # Search the query descriptors against the features in the index
            matches_feats = self.index.searchDescriptors(des_, 2, 64)

            # Filter matches according to the ratio test
            matches = []
            for (
                m
            ) in matches_feats:  # vector of pairs of tuples (queryIdx, trainIdx, imgIdx, distance)
                if m[0][3] < m[1][3] * self.match_ratio:
                    matches.append(m[0])

            if len(matches) > 0:
                # Look for similar images according to the good matches found
                image_matches = self.index.searchImages(des_, matches, True)
                count_valid_candidates = 0
                for i, m in enumerate(image_matches):
                    other_frame_id = self.map_entry_id_to_frame_id[m.image_id]
                    other_entry_id = m.image_id
                    candidate_idxs.append(other_frame_id)
                    candidate_scores.append(m.score)
                    count_valid_candidates += 1
                    if count_valid_candidates >= kMaxResultsForLoopClosure:
                        break

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores

            # NOTE: we do not push the relocalization frames into the database, only the keyframes

        elif task.task_type == LoopDetectorTaskType.LOOP_CLOSURE and self.entry_id >= 1:
            LoopDetectorBase.print(
                f"LoopDetectorOBIndex2: loop closure task: {task.task_type.name}"
            )
            # Search the query descriptors against the features in the index
            matches_feats = self.index.searchDescriptors(des_, 2, 64)

            # Filter matches according to the ratio test
            matches = []
            for (
                m
            ) in matches_feats:  # vector of pairs of tuples (queryIdx, trainIdx, imgIdx, distance)
                if m[0][3] < m[1][3] * self.match_ratio:
                    matches.append(m[0])

            if len(matches) > 0:
                # Look for similar images according to the good matches found
                image_matches = self.index.searchImages(des_, matches, True)
                count_valid_candidates = 0
                for i, m in enumerate(image_matches):
                    other_frame_id = self.map_entry_id_to_frame_id[m.image_id]
                    other_entry_id = m.image_id
                    self.update_similarity_matrix(
                        score=m.score, entry_id=self.entry_id, other_entry_id=other_entry_id
                    )
                    if (
                        abs(other_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure
                        and other_frame_id not in task.connected_keyframes_ids
                    ):
                        candidate_idxs.append(other_frame_id)
                        candidate_scores.append(m.score)
                        count_valid_candidates += 1
                        self.update_loop_closure_imgs(score=m.score, other_frame_id=other_frame_id)
                        if count_valid_candidates >= kMaxResultsForLoopClosure:
                            break

            self.draw_loop_detection_imgs(keyframe.img, frame_id, detection_output)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [
                cov_kf.g_des.toVec() if cov_kf.g_des is not None else None
                for cov_kf in task.covisible_keyframes_data
            ]

            # Add the image to the index. Matched descriptors are used
            # to update the index and the remaining ones are added as new visual words
            self.index.addImage(self.entry_id, kps_, des_, matches)

        else:
            LoopDetectorBase.print(
                f"LoopDetectorOBIndex2: loop closure task: {task.task_type.name}"
            )
            # if we just wanted to compute the global descriptor (LoopDetectorTaskType.COMPUTE_GLOBAL_DES), we don't have to do anything
            self.index.addImage(self.entry_id, kps_, des_)

        # Reindex features every 500 images
        if self.entry_id % 250 == 0 and self.entry_id > 0:
            LoopDetectorBase.print("------ Rebuilding indices ------")
            self.index.rebuild()

        if len(candidate_idxs) > 0:
            LoopDetectorBase.print(f"LoopDetectorOBIndex2: candidate_idxs: {candidate_idxs}")

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # NOTE: with relocalization we don't need to increment the entry_id since we don't add frames to database
            self.entry_id += 1

        return detection_output
