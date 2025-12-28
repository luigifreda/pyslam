"""
* This file is part of PYSLAM
* Adapted from https://github.com/cvlab-epfl/disk/blob/master/detect.py, see licence therein.
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
import sys
import pyslam.config as config

config.cfg.set_lib("mast3r")

import numpy as np
import torch

import cv2
from threading import RLock
from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from
from pyslam.utilities.dust3r import Dust3rImagePreprocessor


from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy

kVerbose = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kMast3rFolder = kRootFolder + "/thirdparty/mast3r"


# This was an experimental feature descriptor test for the MASt3R model.
# NOTE: The extraction of independent descriptors from a single image does not make sense for the MASt3R/DUST3R model.
#       The model grounds the image matching in a 3D context defined by two images of the same place.
#       You have to use the MAST3R matcher!
class Mast3rFeature2D:
    def __init__(self, num_features=None, device=None):
        self.num_features = num_features

        self.model_name = (
            kMast3rFolder + "/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        )
        self.min_conf_thr = 10  # percentage of the max confidence value
        self.inference_size = 512  # can be 224 or 512
        self.subsample_or_initxy1 = 8  # used in fast_reciprocal_NNs
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if device.type == "cuda":
            print("Mast3rFeature2D: Using CUDA")
        else:
            print("Mast3rFeature2D: Using CPU")

        model = AsymmetricMASt3R.from_pretrained(self.model_name).to(device)
        model = model.to(device).eval()
        self.model = model

    # Compute descriptors for the given input keypoints
    def compute_des(self, img, kps):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgs = [img, img]
        dust3r_preprocessor = Dust3rImagePreprocessor(inference_size=self.inference_size)
        # imgs_preproc = dust3r_preprocess_images(imgs, size=self.inference_size)
        imgs_preproc = dust3r_preprocessor.preprocess_images(imgs)
        output = inference(
            [tuple(imgs_preproc)], self.model, self.device, batch_size=1, verbose=False
        )
        # check test/dust3r/test_mast3r_2images.py
        view1 = output["view1"]
        pred1 = output["pred1"]
        # extract descriptors
        desc = pred1["desc"].squeeze(0).detach()
        # print(f'desc shape: {desc.shape}')
        pts = np.array([kp.pt for kp in kps])
        scaled_down_kps = dust3r_preprocessor.scale_down_keypoints(pts)
        H, W = view1["true_shape"][0]
        H, W = H.item(), W.item()  # Convert tensors to integers if they are tensors
        valid_kps = (
            (scaled_down_kps[:, 0] < W)
            & (scaled_down_kps[:, 0] >= 0)
            & (scaled_down_kps[:, 1] < H)
            & (scaled_down_kps[:, 1] >= 0)
        )
        kps = np.array(kps)[valid_kps]
        scaled_down_kps = scaled_down_kps[valid_kps].astype(int)  # Convert to integers
        des = desc[scaled_down_kps[:, 1], scaled_down_kps[:, 0]]

        # normalize descriptors
        # des = des / np.linalg.norm(des, axis=1, keepdims=True)

        return kps, des

    def compute(self, img, kps, mask=None):  # mask is a fake input
        num_kps = len(kps)
        des = []
        if num_kps > 0:
            kps, des = self.compute_des(img, kps)
        if kVerbose:
            print(
                "descriptor: Mast3rFeature2D, #features: ",
                len(kps),
                ", frame res: ",
                img.shape[0:2],
            )
        return kps, des
