"""
* This file is part of PYSLAM
*
* Adpated from https://github.com/lzx551402/contextdesc/blob/master/image_matching.py, see the license therein.
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

import pyslam.config as config

config.cfg.set_lib("contextdesc", prepend=True)

from threading import RLock

import warnings  # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import cv2
import numpy as np

from .feature_base import BaseFeature2D
from pyslam.utilities.tensorflow import import_tf_compat_v1

if False:
    import tensorflow as tf
else:
    tf = import_tf_compat_v1()


from contextdesc.utils.opencvhelper import MatcherWrapper

# from contextdesc.models import get_model
from contextdesc.models.reg_model import RegModel
from contextdesc.models.loc_model import LocModel
from contextdesc.models.aug_model import AugModel

from pyslam.utilities.tensorflow import set_tf_logging
from pyslam.utilities.logging import Printer


kVerbose = True


# Interface for pySLAM
class ContextDescFeature2D(BaseFeature2D):
    quantize = False  #  Wheter to quantize or not the output descriptor

    def __init__(
        self,
        num_features=2000,
        n_sample=2048,  #  Maximum number of sampled keypoints per octave
        dense_desc=False,  #  Whether to use dense descriptor model
        model_type="pb",
        do_tf_logging=False,
    ):
        print("Using ContextDescFeature2D")
        self.lock = RLock()
        self.model_base_path = config.cfg.root_folder + "/thirdparty/contextdesc/"

        set_tf_logging(do_tf_logging)

        self.num_features = num_features
        self.n_sample = n_sample
        self.model_type = model_type
        self.dense_desc = dense_desc
        self.quantize = ContextDescFeature2D.quantize

        self.loc_model_path = self.model_base_path + "pretrained/contextdesc++"
        self.reg_model_path = self.model_base_path + "pretrained/retrieval_model"

        if self.model_type == "pb":
            reg_model_path = os.path.join(self.reg_model_path, "reg.pb")
            loc_model_path = os.path.join(self.loc_model_path, "loc.pb")
            aug_model_path = os.path.join(self.loc_model_path, "aug.pb")
        elif self.model_type == "ckpt":
            reg_model_path = os.path.join(self.reg_model_path, "model.ckpt-550000")
            loc_model_path = os.path.join(self.loc_model_path, "model.ckpt-400000")
            aug_model_path = os.path.join(self.loc_model_path, "model.ckpt-400000")
        else:
            raise NotImplementedError

        self.keypoint_size = 10  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint

        self.pts = []
        self.kps = []
        self.des = []
        self.scales = []
        self.scores = []
        self.frame = None

        print("==> Loading pre-trained network.")
        self.ref_model = RegModel(
            reg_model_path
        )  # get_model('reg_model')(reg_model_path)  # RegModel(reg_model_path)
        self.loc_model = LocModel(
            loc_model_path,
            **{
                "sift_desc": False,  # compute or not SIFT descriptor (we do not need them here!)
                "n_feature": self.num_features,
                "n_sample": self.n_sample,
                "peak_thld": 0.04,
                "dense_desc": self.dense_desc,
                "upright": False,
            },
        )
        self.aug_model = AugModel(aug_model_path, **{"quantz": self.quantize})
        print("==> Successfully loaded pre-trained network.")

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features
        try:
            self.loc_model.sift_wrapper.n_features = num_features
        except:
            Printer.red("[ContextDescFeature2D] Failed to set number of features for SIFT")

    def __del__(self):
        with self.lock:
            self.ref_model.close()
            self.loc_model.close()
            self.aug_model.close()

    def prep_img(self, img):
        rgb_list = []
        gray_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
        return rgb_list, gray_list

    # extract regional features
    def extract_regional_features(self, rgb_list):
        reg_feat_list = []
        # model = get_model('reg_model')(model_path)
        for _, val in enumerate(rgb_list):
            reg_feat = self.ref_model.run_test_data(val)
            reg_feat_list.append(reg_feat)
        # model.close()
        return reg_feat_list

    # extract local features and keypoint matchability
    def extract_local_features(self, gray_list):
        cv_kpts_list = []
        loc_info_list = []
        loc_feat_list = []
        sift_feat_list = []
        # model = get_model('loc_model')(model_path, **{'sift_desc': True,
        #                                             'n_sample': FLAGS.n_sample,
        #                                             'peak_thld': 0.04,
        #                                             'dense_desc': FLAGS.dense_desc,
        #                                             'upright': False})
        for _, val in enumerate(gray_list):
            loc_feat, kpt_mb, normalized_xy, cv_kpts, sift_desc = self.loc_model.run_test_data(val)
            raw_kpts = [np.array((i.pt[0], i.pt[1], i.size, i.angle, i.response)) for i in cv_kpts]
            raw_kpts = np.stack(raw_kpts, axis=0)
            loc_info = np.concatenate((raw_kpts, normalized_xy, loc_feat, kpt_mb), axis=-1)
            cv_kpts_list.append(cv_kpts)
            loc_info_list.append(loc_info)
            sift_feat_list.append(sift_desc)
            loc_feat_list.append(loc_feat / np.linalg.norm(loc_feat, axis=-1, keepdims=True))
        # model.close()
        return cv_kpts_list, loc_info_list, loc_feat_list, sift_feat_list

    # extract augmented features
    def extract_augmented_features(self, reg_feat_list, loc_info_list):
        aug_feat_list = []
        # model = get_model('aug_model')(model_path, **{'quantz': False})
        assert len(reg_feat_list) == len(loc_info_list)
        for idx, _ in enumerate(reg_feat_list):
            aug_feat, _ = self.aug_model.run_test_data([reg_feat_list[idx], loc_info_list[idx]])
            aug_feat_list.append(aug_feat)
        # model.close()
        return aug_feat_list

    def compute_kps_des(self, frame):
        with self.lock:
            rgb_list, gray_list = self.prep_img(frame)
            # extract regional features.
            reg_feat_list = self.extract_regional_features(rgb_list)
            # extract local features and keypoint matchability.
            cv_kpts_list, loc_info_list, loc_feat_list, sift_feat_list = (
                self.extract_local_features(gray_list)
            )
            # extract augmented features.
            aug_feat_list = self.extract_augmented_features(reg_feat_list, loc_info_list)

            self.kps = cv_kpts_list[0]
            self.des = aug_feat_list[0]

            return self.kps, self.des

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            self.kps, self.des = self.compute_kps_des(frame)
            if kVerbose:
                print(
                    "detector: CONTEXTDESC, descriptor: CONTEXTDESC, #features: ",
                    len(self.kps),
                    ", frame res: ",
                    frame.shape[0:2],
                )
            return self.kps, self.des

    # return keypoints if available otherwise call detectAndCompute()
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            if self.frame is not frame:
                self.detectAndCompute(frame)
            return self.kps

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                # Printer.orange('WARNING: CONTEXTDESC is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
