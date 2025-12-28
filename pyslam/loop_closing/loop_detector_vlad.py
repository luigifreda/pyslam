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

from typing import List

from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_types import FeatureInfo

from pyslam.utilities.timer import TimerFps

import traceback
from .loop_detector_base import (
    LoopDetectorTaskType,
    LoopDetectKeyframeData,
    LoopDetectorTask,
    LoopDetectorOutput,
    LoopDetectorBase,
)
from .loop_detector_database import (
    Database,
    SimpleDatabase,
    FlannDatabase,
    FaissDatabase,
    SimpleTorchDatabase,
)
from .loop_detector_vocabulary import VocabularyData
from .loop_detector_score import ScoreBase, ScoreSad, ScoreCosine, ScoreTorchCosine

from .vlad import VLAD

kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False
kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# NOTE: Check in the README how to generate an array of descriptors and train your vocabulary.
class LoopDetectorVlad(LoopDetectorBase):
    def __init__(self, vocabulary_data: VocabularyData, local_feature_manager=None):
        super().__init__()

        import torch.multiprocessing as mp

        # NOTE: the following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).
        if mp.get_start_method() != "spawn":
            mp.set_start_method(
                "spawn", force=True
            )  # NOTE: This may generate some pickling problems with multiprocessing
            #       in combination with torch and we need to check it in other places.
            #       This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

        self.local_feature_manager = local_feature_manager
        self.use_torch_vectors = False  # use torch vectors with a simple database implementation

        LoopDetectorBase.print(f"LoopDetectorVlad: checking vocabulary...")
        vocabulary_data.check_download()
        self.vocabulary_data = vocabulary_data

        self.score = ScoreCosine() if not self.use_torch_vectors else ScoreTorchCosine()
        self.global_feature_extractor = None
        self.global_db = None

        self.init()
        time.sleep(2)  # give a bit of time for the process to start and initialize

    def reset(self):
        LoopDetectorBase.reset(self)
        del self.global_feature_extractor
        del self.global_db
        self.global_feature_extractor = None
        self.global_db = None
        self.init()

    def save(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorVlad: saving database to {filepath}...")
        LoopDetectorBase.print(f"\t Database size: {self.global_db.size()}")
        self.global_db.save(filepath)

    def load(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorVlad: loading database from {filepath}...")
        self.global_db.load(filepath)
        LoopDetectorBase.print(f"\t Database size: {self.global_db.size()}")
        LoopDetectorBase.print(f"LoopDetectorVlad: ...done")

    def init(self):
        try:
            if self.global_db is None:
                self.global_db = self.init_db()
            if self.global_feature_extractor is None:
                self.global_feature_extractor = VLAD(
                    desc_dim=self.vocabulary_data.descriptor_dimension, num_clusters=8
                )
                self.global_feature_extractor.load(self.vocabulary_data.vocab_file_path)
        except Exception as e:
            LoopDetectorBase.print(f"LoopDetectorVlad: init: Exception: {e}")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopDetectorBase.print(f"\t traceback details: {traceback_details}")

    def init_db(self):
        LoopDetectorBase.print(f"LoopDetectorVlad: init_db()")
        if self.use_torch_vectors:
            global_db = SimpleTorchDatabase(
                self.score
            )  # simple implementation, not ideal with large datasets
        else:
            # global_db = SimpleDatabase(self.score) # simple implementation, not ideal with large datasets
            # global_db = FlannDatabase(self.score)
            global_db = FaissDatabase(self.score)
        return global_db

    def compute_global_des(self, local_des, img):
        res = self.global_feature_extractor.generate(local_des)
        if self.use_torch_vectors:
            return res
        else:
            return res.detach().cpu().numpy().reshape(1, -1)

    def run_task(self, task: LoopDetectorTask):
        LoopDetectorBase.print(
            f"LoopDetectorVlad: running task {task.keyframe_data.id}, entry_id = {self.entry_id}, task_type = {task.task_type.name}"
        )
        keyframe = task.keyframe_data
        frame_id = keyframe.id

        self.map_frame_id_to_img[keyframe.id] = keyframe.img
        if self.loop_detection_imgs is not None:
            self.loop_detection_imgs.reset()

        self.resize_similary_matrix_if_needed()

        # compute global descriptor
        if keyframe.g_des is None:
            LoopDetectorBase.print(
                f"LoopDetectorVlad: computing global descriptor for keyframe {keyframe.id}"
            )
            keyframe.g_des = self.compute_global_des(
                keyframe.des, keyframe.img
            )  # get global descriptor

        # print(f'LoopDetectorVlad: g_des = {keyframe.g_des}, type: {type(keyframe.g_des)}, shape: {keyframe.g_des.shape}, dim: {keyframe.g_des.dim()}')

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # add image descriptors to global descriptor database
            # NOTE: relocalization works on frames (not keyframes) and we don't need to add them to the database
            self.global_db.add(keyframe.g_des)

            # the img_ids are mapped to entry_ids (entry ids) inside the database management
            self.map_entry_id_to_frame_id[self.entry_id] = frame_id
            # print(f'LoopDetectorVlad: mapping frame_id: {frame_id} to entry_id: {self.entry_id}')

        detection_output = LoopDetectorOutput(
            task_type=task.task_type, g_des_vec=keyframe.g_des, frame_id=frame_id, img=keyframe.img
        )

        candidate_idxs = []
        candidate_scores = []

        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            if self.entry_id >= 1:
                best_idxs, best_scores = self.global_db.query(
                    keyframe.g_des, max_num_results=kMaxResultsForLoopClosure + 1
                )  # we need plus one since we eliminate the best trivial equal to frame_id
                LoopDetectorBase.print(
                    f"LoopDetectorVlad: Relocalization: frame: {frame_id}, candidate keyframes: {best_idxs}"
                )
                for idx, score in zip(best_idxs, best_scores):
                    other_entry_id = idx
                    other_frame_id = self.map_entry_id_to_frame_id[
                        idx
                    ]  # get the image id of the keyframe from it's internal image count
                    candidate_idxs.append(other_frame_id)
                    candidate_scores.append(score)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores

        elif task.task_type == LoopDetectorTaskType.LOOP_CLOSURE:

            # Compute reference BoW similarity score as the lowest score to a connected keyframe in the covisibility graph.
            min_score = self.compute_reference_similarity_score(
                task, type(keyframe.g_des), score_fun=self.score
            )
            LoopDetectorBase.print(f"LoopDetectorVlad: min_score = {min_score}")

            if self.entry_id >= 1:
                best_idxs, best_scores = self.global_db.query(
                    keyframe.g_des, max_num_results=kMaxResultsForLoopClosure + 1
                )  # we need plus one since we eliminate the best trivial equal to frame_id

                for idx, score in zip(best_idxs, best_scores):
                    other_entry_id = idx
                    other_frame_id = self.map_entry_id_to_frame_id[
                        idx
                    ]  # get the image id of the keyframe from it's internal image count
                    self.update_similarity_matrix(
                        score=score, entry_id=self.entry_id, other_entry_id=other_entry_id
                    )
                    if (
                        abs(other_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure
                        and score >= min_score
                        and other_frame_id not in task.connected_keyframes_ids
                    ):
                        candidate_idxs.append(other_frame_id)
                        candidate_scores.append(score)
                        self.update_loop_closure_imgs(score=score, other_frame_id=other_frame_id)

            self.draw_loop_detection_imgs(keyframe.img, frame_id, detection_output)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [
                cov_kf.g_des for cov_kf in task.covisible_keyframes_data
            ]

        else:
            # if we just wanted to compute the global descriptor (LoopDetectorTaskType.COMPUTE_GLOBAL_DES), we don't have to do anything
            pass

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # NOTE: with relocalization we don't need to increment the entry_id since we don't add frames to database
            self.entry_id += 1

        return detection_output
