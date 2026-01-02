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

from pyslam.utilities.logging import Logging, Printer
from pyslam.utilities.img_management import float_to_color, LoopCandidateImgs
from pyslam.utilities.serialization import NumpyB64Json

from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_types import FeatureInfo

from pyslam.slam import KeyFrame, Frame

import torch.multiprocessing as mp
import logging
import sys

# import json
import ujson as json

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame


kVerbose = True
kPrintTrackebackDetails = True

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


class LoopDetectorTaskType(Enum):
    NONE = 0
    COMPUTE_GLOBAL_DES = 2
    LOOP_CLOSURE = 3
    RELOCALIZATION = 4
    SAVE = 5
    LOAD = 6


# keyframe (picklable) data that are needed for loop detection
class LoopDetectKeyframeData:
    def __init__(self, keyframe: KeyFrame = None, img=None):
        # keyframe data
        self.id = keyframe.id if keyframe is not None else -1
        self.kps = keyframe.kps if keyframe is not None else []
        self.angles = keyframe.angles if keyframe is not None else []
        self.sizes = keyframe.sizes if keyframe is not None else []
        self.octaves = keyframe.octaves if keyframe is not None else []
        self.des = keyframe.des if keyframe is not None else []

        self.img = img if img is not None else (keyframe.img if keyframe is not None else None)

        # NOTE: The kid is not actually used for the processing in this whole file
        if isinstance(keyframe, KeyFrame):
            self.kid = keyframe.kid if keyframe is not None else -1
            self.g_des = keyframe.g_des if keyframe is not None else None
        else:
            self.kid = -1
            self.g_des = None


class LoopDetectorTask:
    def __init__(
        self,
        keyframe: KeyFrame,
        img,
        task_type=LoopDetectorTaskType.NONE,
        covisible_keyframes=None,
        connected_keyframes=None,
        load_save_path=None,
    ):
        if covisible_keyframes is None:
            covisible_keyframes = []
        if connected_keyframes is None:
            connected_keyframes = []

        self.task_type = task_type
        self.keyframe_data = LoopDetectKeyframeData(keyframe, img)
        self.covisible_keyframes_data = [
            LoopDetectKeyframeData(kf)
            for kf in covisible_keyframes
            if not kf.is_bad() and kf.id != self.keyframe_data.id
        ]
        self.connected_keyframes_ids = [kf.id for kf in connected_keyframes]
        self.load_save_path = load_save_path
        # # for loop closing
        # self.loop_query_id = None
        # self.num_loop_words = 0
        # self.loop_score = 0
        # # for relocalization
        # self.reloc_query_id = None
        # self.num_reloc_words = 0
        # self.reloc_score = 0

    def __str__(self) -> str:
        return f"LoopDetectorTask: img id = {self.keyframe_data.id}, kid = {self.keyframe_data.kid}, task_type = {self.task_type.name}"


class LoopDetectorOutput:
    def __init__(
        self,
        task_type,
        candidate_idxs=None,
        candidate_scores=None,
        g_des_vec=None,
        frame_id=None,
        img=None,
        covisible_ids=None,
        covisible_gdes_vecs=None,
    ):
        if candidate_idxs is None:
            candidate_idxs = []
        if candidate_scores is None:
            candidate_scores = []
        if covisible_ids is None:
            covisible_ids = []
        if covisible_gdes_vecs is None:
            covisible_gdes_vecs = []

        self.task_type = task_type
        # candidates information + input keyframe data
        self.candidate_idxs = candidate_idxs
        self.candidate_scores = candidate_scores
        self.g_des_vec = g_des_vec  # we use g_des_vec instead of g_des since the first can be used with multiprocessing and transported in the queue
        self.frame_id = frame_id
        self.img = img  # for debugging
        # potential g_des updates/computations for covisible keyframes
        self.covisible_ids = covisible_ids
        self.covisible_gdes_vecs = covisible_gdes_vecs
        # potential img output
        self.similarity_matrix = None
        self.loop_detection_img_candidates = None

    def __str__(self) -> str:
        return f"LoopDetectorOutput: task_type = {self.task_type.name}, candidate_idxs = {self.candidate_idxs}, candidate_scores = {self.candidate_scores}, frame_id = {self.frame_id}"


# Base class for loop detectors
class LoopDetectorBase:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(self):
        self.entry_id = 0  # this corresponds to the internal detector entry counter (incremented only when a new keyframe is added to the detector database)
        self.map_entry_id_to_frame_id = {}
        self.map_frame_id_to_img = {}
        self.map_frame_id_to_entry_id = {}  # not always used

        self.global_descriptor_type = None  # to be set by loop_detector_factory
        self.local_descriptor_aggregation_type = None  # to be set by loop_detector_factory
        self.local_feature_manager = None  # to be set by loop_detector_factory
        self.vocabulary_data = None  # to be set by loop_detector_factory

        self.voc = None  # to be set by derived classes

        # init the similarity matrix
        if Parameters.kLoopClosingDebugWithSimmetryMatrix:
            self.max_num_kfs = 200
            self.S_float = np.empty([self.max_num_kfs, self.max_num_kfs], "float32")
            # self.S_color = np.empty([self.max_num_kfs, self.max_num_kfs, 3], 'uint8')
            self.S_color = np.full(
                [self.max_num_kfs, self.max_num_kfs, 3], 0, "uint8"
            )  # loop closures are found with a small score, this will make them disappear
        else:
            self.S_float = None
            self.S_color = None

        # to nicely visualize current loop candidates in a single image
        self.loop_detection_imgs = (
            LoopCandidateImgs() if Parameters.kLoopClosingDebugWithLoopDetectionImages else None
        )
        self.init_print()

    def init_print(self):
        if kVerbose:
            if Parameters.kLoopClosingDebugAndPrintToFile:
                # redirect the prints of local mapping to the file logs/loop_detecting.log (by default)
                # you can watch the output in separate shell by running:
                # $ tail -f logs/loop_detecting.log

                logging_file = Parameters.kLogsFolder + "/loop_detecting.log"
                LoopDetectorBase.local_logger = Logging.setup_file_logger(
                    "loop_detecting_logger", logging_file, formatter=Logging.simple_log_formatter
                )

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return LoopDetectorBase.local_logger.info(message, **kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return print(message, **kwargs)

            LoopDetectorBase.print = staticmethod(print_file)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def init(self):
        pass

    # Load the maps used by the loop detector database
    def load_db_maps(self, path):
        load_path = path + "/loop_closing_db_maps.json"
        if not os.path.exists(load_path):
            LoopDetectorBase.print(f"LoopDetectorBase: database maps do not exist: {load_path}")
            return
        LoopDetectorBase.print(f"LoopDetectorBase: loading database maps from {load_path}...")
        with open(load_path, "rb") as f:
            output = json.load(f)
        self.entry_id = output["entry_id"]

        def convert_dict(d):
            # Convert keys back to integers (they are trasformed to strings by json when saving to a file)
            return {int(k): v for k, v in d.items()}

        self.map_entry_id_to_frame_id = convert_dict(output["map_entry_id_to_frame_id"])
        self.map_frame_id_to_img = NumpyB64Json.map_id2img_from_json(output["map_frame_id_to_img"])
        self.map_frame_id_to_entry_id = convert_dict(output["map_frame_id_to_entry_id"])
        # for k, v in self.map_frame_id_to_img.items():
        #     LoopDetectorBase.print(f'LoopDetectorBase: loaded img id: {k}, shape: {v.shape}, dtype: {v.dtype}')
        LoopDetectorBase.print(
            f"LoopDetectorBase: ...database maps successfully loaded from: {load_path}"
        )

    # Save the maps used by the loop detector database
    def save_db_maps(self, path):
        save_path = path + "/loop_closing_db_maps.json"
        LoopDetectorBase.print(f"LoopDetectorBase: saving database maps to {save_path}...")
        output = {}
        output["entry_id"] = self.entry_id
        output["map_entry_id_to_frame_id"] = self.map_entry_id_to_frame_id
        output["map_frame_id_to_img"] = NumpyB64Json.map_id2img_to_json(self.map_frame_id_to_img)
        output["map_frame_id_to_entry_id"] = self.map_frame_id_to_entry_id
        with open(save_path, "w") as f:
            f.write(json.dumps(output))
        LoopDetectorBase.print(
            f"LoopDetectorBase: ...database maps successfully saved to: {save_path}"
        )

    def reset(self):
        self.entry_id = 0
        self.map_entry_id_to_frame_id.clear()
        self.map_frame_id_to_img.clear()
        self.map_frame_id_to_entry_id.clear()
        if self.S_float is not None:
            self.S_float.fill(0)
            self.S_color.fill(0)
        if self.loop_detection_imgs is not None:
            self.loop_detection_imgs.reset()

    # Check and compute if requested the image local descriptors by using the potentially allocated independent local feature manager.
    # This feature manager may have be allocated since we want to use different local descriptors in the loop detector (different from the extracted ones in the frontend).
    # If the local feature manager is allocated then compute the local descriptors and replace the "keyframe_data.des" field in the task data structure.
    def compute_local_des_if_needed(self, task: LoopDetectorTask):
        if self.local_feature_manager is not None:
            kps, des = self.local_feature_manager.compute(
                task.keyframe_data.img, task.keyframe_data.kps
            )
            task.keyframe_data.des = des
            LoopDetectorBase.print(
                f"LoopDetectorBase: re-computed {des.shape[0]} local descriptors ({self.local_feature_manager.descriptor_type.name}) for keyframe {task.keyframe_data.id}"
            )

    # Compute global descriptors from local descriptors and input image
    def compute_global_des(self, local_des, img):
        return None

    # Run the loop detector task
    def run_task(self, task: LoopDetectorTask):
        return None

    def compute_reference_similarity_score(self, task: LoopDetectorTask, vector_type, score_fun):
        # Compute reference BoW similarity score.
        # This is the lowest score to a connected keyframe in the covisibility graph.
        # Loop candidates must have a higher similarity than this
        keyframe = task.keyframe_data
        min_score = 1
        # print(f'LoopDetectorBase: computing reference similarity score for keyframe {keyframe.id} with covisible keyframes {[cov_kf.id for cov_kf in task.covisible_keyframes_data]}')
        if len(task.covisible_keyframes_data) == 0:
            return -sys.float_info.max
        for cov_kf in task.covisible_keyframes_data:
            if cov_kf.g_des is None:
                try:
                    if cov_kf.img is None:
                        cov_kf.img = self.map_frame_id_to_img[cov_kf.id]
                        # print(f'LoopDetectorBase: covisible keyframe {cov_kf.id} has no img, loaded from map: shape: {cov_kf.img.shape}, dtype: {cov_kf.img.dtype}')
                except:
                    LoopDetectorBase.print(
                        f"LoopDetectorBase: covisible keyframe {cov_kf.id} has no img"
                    )
                # if we don't have the global descriptor yet, we need to compute it
                if cov_kf.img is not None:
                    # print(f'LoopDetectorBase: covisible keyframe {cov_kf.id} has no g_des, computing on img shape: {cov_kf.img.shape}, dtype: {cov_kf.img.dtype}')
                    if cov_kf.img.dtype != np.uint8:
                        LoopDetectorBase.print(
                            f"LoopDetectorBase: covisible keyframe {cov_kf.id} has img dtype: {cov_kf.img.dtype}, converting to uint8"
                        )
                        cov_kf.img = cov_kf.img.astype(np.uint8)
                LoopDetectorBase.print(
                    f"LoopDetectorBase: computing global descriptor for keyframe {cov_kf.id}"
                )
                cov_kf.g_des = self.compute_global_des(cov_kf.des, cov_kf.img)
            if cov_kf.g_des is not None:
                if not isinstance(cov_kf.g_des, vector_type):
                    # print(f'LoopDetectorBase: covisible keyframe {cov_kf.id} converting g_des from {type(cov_kf.g_des)} to type {vector_type}')
                    cov_kf.g_des = vector_type(
                        cov_kf.g_des
                    )  # transform back from vec to specialized vector (this is used for DBOW vectors)
                score = score_fun(cov_kf.g_des, keyframe.g_des)
                min_score = min(min_score, score)
            else:
                LoopDetectorBase.print(
                    f"LoopDetectorBase: covisible keyframe {cov_kf.id} has no g_des"
                )

        return min_score

    def resize_similary_matrix_if_needed(self):
        if self.S_float is None:
            return
        if self.entry_id >= self.max_num_kfs:
            self.max_num_kfs += 100
            # self.S_float.resize([self.max_num_kfs, self.max_num_kfs])
            # self.S_color.resize([self.max_num_kfs, self.max_num_kfs, 3])
            S_float = np.pad(
                self.S_float,
                (
                    (0, self.max_num_kfs - self.S_float.shape[0]),
                    (0, self.max_num_kfs - self.S_float.shape[1]),
                ),
                mode="constant",
                constant_values=0,
            )
            self.S_float = S_float
            S_color = np.pad(
                self.S_color,
                (
                    (0, self.max_num_kfs - self.S_color.shape[0]),
                    (0, self.max_num_kfs - self.S_color.shape[1]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )
            self.S_color = S_color

    def update_similarity_matrix(self, score, entry_id, other_entry_id):
        color_value = float_to_color(score)
        if self.S_float is not None:
            self.S_float[entry_id, other_entry_id] = score
            self.S_float[other_entry_id, entry_id] = score
        if self.S_color is not None:
            self.S_color[entry_id, other_entry_id] = color_value
            self.S_color[other_entry_id, entry_id] = color_value

    def update_loop_closure_imgs(self, score, other_frame_id):
        if self.loop_detection_imgs is not None:
            loop_img = self.map_frame_id_to_img[other_frame_id]
            self.loop_detection_imgs.add(loop_img.copy(), other_frame_id, score)

    def draw_loop_detection_imgs(self, img_cur, frame_id, detection_output: LoopDetectorOutput):
        if self.S_color is not None:
            detection_output.similarity_matrix = self.S_color  # .copy()

        if self.loop_detection_imgs is not None and self.loop_detection_imgs.candidates is not None:
            detection_output.loop_detection_img_candidates = (
                self.loop_detection_imgs.candidates
            )  # .copy()
