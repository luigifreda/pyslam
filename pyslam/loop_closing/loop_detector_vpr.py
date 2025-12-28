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
    ScoreCosine,
    ScoreSad,
    SimpleDatabase,
    FlannDatabase,
    FaissDatabase,
)
from .global_feature_megaloc import GlobalFeatureMegaloc

# import dill

import pyslam.config as config

config.cfg.set_lib("vpr", prepend=True)

kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False
kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# To combine HDC with any feature manager
class HdcAdaptor:
    def __init__(self, feature_manager):
        self.feature_manager = feature_manager
        self.HDC = None

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        from feature_aggregation.hdc import HDC

        D_local = []
        for img in imgs:
            kps, des = self.feature_manager.detectAndCompute(img)
            D_local.append(des)
        D_local = np.array(D_local)
        self.HDC = HDC(D_local)
        D_holistic = self.HDC.compute_holistic()

        return D_holistic

    def compute_features_step(self, img: List[np.ndarray]) -> np.ndarray:
        from feature_aggregation.hdc import HDC

        kps, des = self.feature_manager.detectAndCompute(img)
        if self.HDC is None:
            self.HDC = HDC(des)  # init HDC only once
        D_holistic = self.HDC.compute_holistic(des)
        return D_holistic


# Table of models covered by LoopDetectorVprBase:
# global_descriptor_name = 'HDC-DELF'    # Slow. local DELF descriptor + Hyperdimensional Computing (HDC)), https://www.tu-chemnitz.de/etit/proaut/hdc_desc
# global_descriptor_name = 'SAD'         # Decently fast. Sum of Absolute Differences as an holistic descriptor (SAD). Milford and Wyeth (2012). "SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights".
# global_descriptor_name = 'AlexNet'     # Slow. AlexNetConv3Extractor https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# global_descriptor_name = 'NetVLAD'     # Decently fast. PatchNetVLADFeatureExtractor model. https://www.di.ens.fr/willow/research/netvlad/
# global_descriptor_name = 'CosPlace'    # Decently fast. CosPlaceFeatureExtractor model. https://github.com/gmberton/CosPlace
# global_descriptor_name = 'EigenPlaces' # Decently fast. EigenPlacesFeatureExtractor model. https://github.com/gmberton/EigenPlaces
# global_descriptor_name = 'Megaloc'     # A bit slow. https://github.com/gmberton/MegaLoc   https://github.com/luigifreda/pyslam/issues/178#issue-3108783337
class LoopDetectorVprBase(LoopDetectorBase):
    def __init__(
        self, global_descriptor_name=None, local_feature_manager=None, name="LoopDetectorVprBase"
    ):
        super().__init__()
        self.name = name
        self.local_feature_manager = local_feature_manager
        self.global_descriptor_name = global_descriptor_name  # this is a string used by LoopDetectorVprBase and it is different from self.global_descriptor_type used by the base class

        if global_descriptor_name is None:
            raise ValueError("LoopDetectorVprBase: global_descriptor_name cannot be None")

        self.score = None
        if self.global_descriptor_name.lower() == "sad":
            self.score = ScoreSad()
            self.min_score = -100
            if Parameters.kLoopClosingDebugWithSimmetryMatrix:
                self.S_float = np.full(
                    [self.max_num_kfs, self.max_num_kfs], self.min_score, "float32"
                )
                self.S_color = np.full([self.max_num_kfs, self.max_num_kfs, 3], 0, "uint8")
        else:
            self.score = ScoreCosine()
            self.min_score = 0
            if Parameters.kLoopClosingDebugWithSimmetryMatrix:
                self.S_float = np.full([self.max_num_kfs, self.max_num_kfs], 0, "float32")
                self.S_color = np.full([self.max_num_kfs, self.max_num_kfs, 3], 0, "uint8")

        self.global_feature_extractor = None
        self.global_db = None

        # NOTE: The following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).
        if (
            global_descriptor_name.lower() == "cosplace"
            or global_descriptor_name.lower() == "alexnet"
            or global_descriptor_name.lower() == "netvlad"
            or global_descriptor_name.lower() == "eigenplaces"
            or global_descriptor_name.lower() == "megaloc"
        ):
            import torch.multiprocessing as mp

            if mp.get_start_method() != "spawn":
                mp.set_start_method(
                    "spawn", force=True
                )  # NOTE: This may generate some pickling problems with multiprocessing
                #       in combination with torch and we need to check it in other places.
                #       This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

        # self.init() # NOTE: We call init() in the run_task() method at its first call to
        #       initialize the global feature extractor in the potentially launched parallel process.
        #       This is required to avoid pickling problem when multiprocessing is used in combination
        #       with torch and CUDA.
        LoopDetectorBase.print(
            f"LoopDetectorVprBase: global_descriptor_name: {global_descriptor_name}"
        )

    def reset(self):
        LoopDetectorBase.reset(self)
        del self.global_feature_extractor
        del self.global_db
        self.global_feature_extractor = None
        self.global_db = None

    def save(self, path):
        filepath = path + "/loop_closing.db"
        LoopDetectorBase.print(f"LoopDetectorVprBase: saving database to {filepath}...")
        LoopDetectorBase.print(f"\t Database size: {self.global_db.size()}")
        self.global_db.save(filepath)

    def load(self, path):
        filepath = path + "/loop_closing.db"
        if not os.path.exists(filepath):
            LoopDetectorBase.print(f"LoopDetectorVprBase: database does not exist: {filepath}")
            return
        LoopDetectorBase.print(f"LoopDetectorVprBase: loading database from {filepath}...")
        self.global_db.load(filepath)
        LoopDetectorBase.print(f"\t Database size: {self.global_db.size()}")
        LoopDetectorBase.print(f"LoopDetectorVprBase: ...done")

    def init(self):
        try:
            if self.global_db is None:
                self.global_db = self.init_db()
            if self.global_feature_extractor is None:
                self.global_feature_extractor = self.init_global_feature_extractor(
                    self.global_descriptor_name
                )
        except Exception as e:
            LoopDetectorBase.print(f"LoopDetectorVprBase: init: Exception: {e}")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopDetectorBase.print(f"\t traceback details: {traceback_details}")

    def init_db(self):
        LoopDetectorBase.print(f"LoopDetectorVprBase: init_db()")
        # global_db = SimpleDatabase(self.score)  # simple implementation, not ideal with large datasets
        # global_db = FlannDatabase(self.score)
        global_db = FaissDatabase(self.score)
        return global_db

    def init_global_feature_extractor(self, global_descriptor_name):
        LoopDetectorBase.print(
            f"LoopDetectorVprBase: init_global_feature_extractor: global_descriptor_name: {global_descriptor_name}"
        )
        global_feature_extractor = None
        if global_descriptor_name.lower() == "hdc-delf":
            from feature_extraction.feature_extractor_holistic import HDCDELF

            global_feature_extractor = HDCDELF()
        elif global_descriptor_name.lower() == "alexnet":
            from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor

            global_feature_extractor = AlexNetConv3Extractor()
        elif global_descriptor_name.lower() == "sad":
            from feature_extraction.feature_extractor_holistic import SAD

            global_feature_extractor = SAD()
        elif global_descriptor_name.lower() == "netvlad":
            from feature_extraction.feature_extractor_patchnetvlad import (
                PatchNetVLADFeatureExtractor,
            )
            from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
            import configparser

            LoopDetectorBase.print(f"PatchNetVLADFeatureExtractor: {PATCHNETVLAD_ROOT_DIR}")
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, "configs/netvlad_extract.ini")
            assert os.path.isfile(configfile)
            config = configparser.ConfigParser()
            config.read(configfile)
            global_feature_extractor = PatchNetVLADFeatureExtractor(config)
        elif global_descriptor_name.lower() == "cosplace":
            from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor

            global_feature_extractor = CosPlaceFeatureExtractor()
        elif global_descriptor_name.lower() == "eigenplaces":
            from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor

            global_feature_extractor = EigenPlacesFeatureExtractor()
        elif global_descriptor_name.lower() == "megaloc":
            from global_feature_megaloc import GlobalFeatureMegaloc

            global_feature_extractor = GlobalFeatureMegaloc()
        else:
            raise ValueError("Unknown descriptor: " + global_descriptor_name)
        return global_feature_extractor

    def compute_global_des(self, local_des, img):
        if img is not None:
            # print(f'LoopDetectorVprBase.compute_global_des: img.shape: {img.shape}, img.dtype: {img.dtype}, type(img): {type(img)}')
            g_des = self.global_feature_extractor.compute_features_step(img)
            # print(f'LoopDetectorVprBase.compute_global_des: g_des.shape: {g_des.shape}, g_des.dtype: {g_des.dtype}, type(g_des): {type(g_des)}')
            return g_des
        else:
            message = "LoopDetectorVprBase.compute_global_des: img is None"
            LoopDetectorBase.print(message)
            Printer.orange(message)
            return None

    def run_task(self, task: LoopDetectorTask):
        LoopDetectorBase.print(
            f"{self.name}: running task {task.keyframe_data.id}, entry_id = {self.entry_id}, task_type = {task.task_type.name}"
        )

        self.init()  # initialize from the potentially launched parallel process or thread at the first run_task() call

        keyframe = task.keyframe_data
        frame_id = keyframe.id

        self.map_frame_id_to_img[keyframe.id] = keyframe.img
        if self.loop_detection_imgs is not None:
            self.loop_detection_imgs.reset()

        self.resize_similary_matrix_if_needed()

        # compute global descriptor
        if keyframe.g_des is None:
            LoopDetectorBase.print(
                f"LoopDetectorVprBase: computing global descriptor for keyframe {keyframe.id}"
            )
            keyframe.g_des = self.compute_global_des(
                keyframe.des, keyframe.img
            )  # get global descriptor

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            if keyframe.g_des is not None:
                # add image descriptors to global descriptor database
                # NOTE: relocalization works on frames (not keyframes) and we don't need to add them to the database
                self.global_db.add(keyframe.g_des)

                # the img_ids are mapped to entry_ids (entry ids) inside the database management
                self.map_entry_id_to_frame_id[self.entry_id] = frame_id
                # print(f'LoopDetectorVprBase: mapping frame_id: {frame_id} to entry_id: {self.entry_id}')

        detection_output = LoopDetectorOutput(
            task_type=task.task_type, g_des_vec=keyframe.g_des, frame_id=frame_id, img=keyframe.img
        )

        if keyframe.g_des is None:
            return detection_output

        candidate_idxs = []
        candidate_scores = []

        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            if self.entry_id >= 1:
                best_idxs, best_scores = self.global_db.query(
                    keyframe.g_des, max_num_results=kMaxResultsForLoopClosure + 1
                )  # we need plus one since we eliminate the best trivial equal to frame_id
                LoopDetectorBase.print(
                    f"LoopDetectorVprBase: Relocalization: frame: {frame_id}, candidate keyframes: {best_idxs}"
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
            LoopDetectorBase.print(f"{self.name}: min_score = {min_score}")

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


#
# Table of models covered by LoopDetectorVprBase:
# global_descriptor_name = 'HDC-DELF'    # Slow. local DELF descriptor + Hyperdimensional Computing (HDC)), https://www.tu-chemnitz.de/etit/proaut/hdc_desc
# global_descriptor_name = 'SAD'         # Decently fast. Sum of Absolute Differences as an holistic descriptor (SAD). Milford and Wyeth (2012). "SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights".
# global_descriptor_name = 'AlexNet'     # Slow. AlexNetConv3Extractor https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# global_descriptor_name = 'NetVLAD'     # Decently fast. PatchNetVLADFeatureExtractor model. https://www.di.ens.fr/willow/research/netvlad/
# global_descriptor_name = 'CosPlace'    # Decently fast. CosPlaceFeatureExtractor model. https://github.com/gmberton/CosPlace
# global_descriptor_name = 'EigenPlaces' # Decently fast. EigenPlacesFeatureExtractor model. https://github.com/gmberton/EigenPlaces
# global_descriptor_name = 'Megaloc'     # Slow. https://github.com/gmberton/MegaLoc
class LoopDetectorHdcDelf(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="HDC-DELF",
        local_feature_manager=None,
        name="LoopDetectorHdcDelf",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorSad(LoopDetectorVprBase):
    def __init__(
        self, global_descriptor_name="SAD", local_feature_manager=None, name="LoopDetectorSad"
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorAlexNet(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="AlexNet",
        local_feature_manager=None,
        name="LoopDetectorAlexNet",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorNetVLAD(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="NetVLAD",
        local_feature_manager=None,
        name="LoopDetectorNetVLAD",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorCosPlace(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="CosPlace",
        local_feature_manager=None,
        name="LoopDetectorCosPlace",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorEigenPlaces(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="EigenPlaces",
        local_feature_manager=None,
        name="LoopDetectorEigenPlaces",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)


class LoopDetectorMegaloc(LoopDetectorVprBase):
    def __init__(
        self,
        global_descriptor_name="Megaloc",
        local_feature_manager=None,
        name="LoopDetectorMegaloc",
    ):
        super().__init__(global_descriptor_name, local_feature_manager, name)
