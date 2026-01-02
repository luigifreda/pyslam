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

from pyslam.config_parameters import Parameters

from .loop_detector_base import (
    LoopDetectorTaskType,
    LoopDetectKeyframeData,
    LoopDetectorTask,
    LoopDetectorOutput,
    LoopDetectorBase,
)
from .loop_detector_vocabulary import VocabularyData

import pyslam.config as config

config.cfg.set_lib("pydbow3")
import pydbow3 as dbow3


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# At present just working with ORB local features.
# NOTE: Under mac, loading the vocabulary is very slow (both from text and from boost archive).
# NOTE: Check in the README how to generate an array of descriptors and train your vocabulary.
class LoopDetectorDBoW3(LoopDetectorBase):
    def __init__(self, vocabulary_data: VocabularyData, local_feature_manager=None):
        super().__init__()
        self.local_feature_manager = local_feature_manager
        self.voc = dbow3.Vocabulary()
        LoopDetectorBase.print(f"LoopDetectorDBoW3: downloading vocabulary...")
        vocabulary_data.check_download()
        LoopDetectorBase.print(
            f"LoopDetectorDBoW3: loading vocabulary {vocabulary_data.vocab_file_path}..."
        )
        use_boost = True if vocabulary_data.vocab_file_path.endswith(".dbow3") else False
        self.voc.load(vocabulary_data.vocab_file_path, use_boost=use_boost)
        LoopDetectorBase.print(f"LoopDetectorDBoW3: ...done")
        self.db = dbow3.Database()
        self.db.setVocabulary(self.voc)

    def reset(self):
        LoopDetectorBase.reset(self)
        self.db.clear()

    def save(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorDBoW3: saving database to {filepath}...")
        LoopDetectorBase.print(f"\t Database size: {self.db.size()}")
        self.db.save(filepath, save_voc=False)

    def load(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorDBoW3: loading database from {filepath}...")
        self.db.setVocabulary(self.voc)
        self.db.load(filepath, load_voc=False)
        self.db.print_status()
        LoopDetectorBase.print(f"LoopDetectorDBoW3: ...done")

    def compute_global_des(self, local_des, img):
        # print(f'computing global descriptors... voc empty: {self.voc.empty()}')
        global_des = self.voc.transform(local_des)  # this returns a bow vector
        return global_des

    # query with global descriptors
    def db_query(self, global_des: dbow3.BowVector, frame_id, max_num_results=5):
        # print(f'LoopDetectorDBoW3: db_query(frame_id={frame_id}, global_des={global_des})')
        results = self.db.query(
            global_des, max_results=max_num_results + 1
        )  # we need plus one to eliminate the best trivial equal to frame_id
        return results

    def run_task(self, task: LoopDetectorTask):
        LoopDetectorBase.print(
            f"LoopDetectorDBoW3: running task {task.keyframe_data.id}, entry_id = {self.entry_id}, task_type = {task.task_type.name}"
        )
        keyframe = task.keyframe_data
        frame_id = keyframe.id

        if self.loop_detection_imgs is not None:
            self.map_frame_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()

        self.resize_similary_matrix_if_needed()

        # compute global descriptor
        if keyframe.g_des is None:
            LoopDetectorBase.print(
                f"LoopDetectorDBoW3: computing global descriptor for keyframe {keyframe.id}"
            )
            keyframe.g_des = self.compute_global_des(keyframe.des, keyframe.img)  # get bow vector
            g_des_vec = (
                keyframe.g_des.toVec()
            )  # transform it to a vector(numpy array) to make it picklable
        else:
            if not isinstance(keyframe.g_des, dbow3.BowVector):
                g_des_vec = keyframe.g_des
                keyframe.g_des = dbow3.BowVector(
                    keyframe.g_des
                )  # transform back from vec to bow vector
            else:
                g_des_vec = keyframe.g_des.toVec()

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # add image descriptors to global_des_database
            # NOTE: relocalization works on frames (not keyframes) and we don't need to add them to the database
            self.db.addBowVector(keyframe.g_des)

            # the img_ids are mapped to entry_ids (entry ids) inside the database management
            self.map_entry_id_to_frame_id[self.entry_id] = frame_id
            # print(f'LoopDetectorDBoW3: mapping frame_id: {frame_id} to entry_id: {self.entry_id}')

        detection_output = LoopDetectorOutput(
            task_type=task.task_type, g_des_vec=g_des_vec, frame_id=frame_id, img=keyframe.img
        )

        candidate_idxs = []
        candidate_scores = []

        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            if self.entry_id >= 1:
                results = self.db_query(
                    keyframe.g_des, frame_id, max_num_results=kMaxResultsForLoopClosure
                )
                LoopDetectorBase.print(
                    f"LoopDetectorDBoW3: Relocalization: frame: {frame_id}, candidate keyframes: {[r.id for r in results]}"
                )
                for r in results:
                    r_frame_id = self.map_entry_id_to_frame_id[
                        r.id
                    ]  # get the image id of the keyframe from it's internal image count
                    candidate_idxs.append(r_frame_id)
                    candidate_scores.append(r.score)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores

        elif task.task_type == LoopDetectorTaskType.LOOP_CLOSURE:

            # Compute reference BoW similarity score as the lowest score to a connected keyframe in the covisibility graph.
            min_score = self.compute_reference_similarity_score(
                task, dbow3.BowVector, score_fun=self.voc.score
            )
            LoopDetectorBase.print(f"LoopDetectorDBoW3: min_score = {min_score}")

            if self.entry_id >= 1:
                results = self.db_query(
                    keyframe.g_des, frame_id, max_num_results=kMaxResultsForLoopClosure
                )
                # print(f'connected keyframes: {[frame_id for frame_id in task.connected_keyframes_ids]}')
                for r in results:
                    r_frame_id = self.map_entry_id_to_frame_id[
                        r.id
                    ]  # get the image id of the keyframe from it's internal image count
                    # print(f'r_frame_id = {r_frame_id}, r.id = {r.id}')
                    self.update_similarity_matrix(
                        score=r.score, entry_id=self.entry_id, other_entry_id=r.id
                    )
                    if (
                        abs(r_frame_id - frame_id) > kMinDeltaFrameForMeaningfulLoopClosure
                        and r.score >= min_score
                        and r_frame_id not in task.connected_keyframes_ids
                    ):
                        candidate_idxs.append(r_frame_id)
                        candidate_scores.append(r.score)
                        self.update_loop_closure_imgs(score=r.score, other_frame_id=r_frame_id)

            self.draw_loop_detection_imgs(keyframe.img, frame_id, detection_output)

            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [
                cov_kf.g_des.toVec() for cov_kf in task.covisible_keyframes_data
            ]

        else:
            # if we just wanted to compute the global descriptor (LoopDetectorTaskType.COMPUTE_GLOBAL_DES), we don't have to do anything
            pass

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # NOTE: with relocalization we don't need to increment the entry_id since we don't add frames to database
            self.entry_id += 1

        return detection_output
