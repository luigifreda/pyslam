"""
* This file is part of PYSLAM
* adapted from https://github.com/cvlab-epfl/log-polar-descriptors/blob/aed70f882cddcfe0c27b65768b9248bf1f2c65cb/example.py, see licence therein.
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

# adapted from https://github.com/cvlab-epfl/log-polar-descriptors/blob/aed70f882cddcfe0c27b65768b9248bf1f2c65cb/example.py

import pyslam.config as config

config.cfg.set_lib("logpolar")

import os
import sys

import torch
import torch.nn as nn

# from modules.ptn.pytorch.models import Transformer

import cv2
import numpy as np
import h5py
from time import time

from configs.defaults import _C as cfg

# from modules.hardnet.models import HardNet  # given some matplotlib backend changes the code is repeated below

from .feature_base import BaseFeature2D
from pyslam.utilities.features import (
    extract_patches_tensor,
    extract_patches_array,
    extract_patches_array_cpp,
)


kVerbose = True
kVerbose2 = True


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


# from modules.hardnet.models
class HardNet(nn.Module):
    def __init__(
        self, transform, coords, patch_size, scale, is_desc256, orientCorrect=True, hard_augm=False
    ):  # <-- added to take care of the possible nonlocal option managed in modules.hardnet.models
        super(HardNet, self).__init__()

        self.transform = transform
        self.transform_layer = Transformer(
            transform=transform, coords=coords, resolution=patch_size, SIFTscale=scale
        )

        self.orientCorrect = orientCorrect
        self.hard_augm = hard_augm

        # model processing patches of size [32 x 32] and giving description vectors of length 2**7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # initialize weights
        self.features.apply(weights_init)
        return

    def input_norm(self, x):

        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    # function to forward-propagate inputs through the network
    def forward(self, img, theta=None, imgIDs=None):

        if theta is None:  # suppose patches are directly given (as e.g. for external test data)
            patches = img
        else:  # extract keypoints from the whole image
            patches = self.transform_layer([img, theta, imgIDs])

        batchSize = patches.shape[0]

        if self.hard_augm:  # args.hard_augm:
            bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))

            if self.transform == "STN":
                # transpose to switch dimensions (only if STN)
                transpose = bernoulli.sample(torch.Size([batchSize]))
                patches = torch.cat(
                    [
                        torch.transpose(patch, 1, 2) if transpose[pdx] else patch
                        for pdx, patch in enumerate(patches)
                    ]
                ).unsqueeze(1)

            # flip the patches' first dimension
            mirrorDim1 = bernoulli.sample(torch.Size([batchSize]))
            patches = torch.cat(
                [
                    torch.flip(patch, [1]) if mirrorDim1[pdx] else patch
                    for pdx, patch in enumerate(patches)
                ]
            ).unsqueeze(1)

        x_features = self.features(self.input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x), patches


def weights_init(m):
    """
    Conv2d module weight initialization method
    """

    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


# Interface for pySLAM
class LogpolarFeature2D(BaseFeature2D):
    def __init__(self, use_log_polar=True, do_cuda=True):
        print("Using LogpolarFeature2D")
        self.model_base_path = config.cfg.root_folder + "/thirdparty/logpolar/"

        if use_log_polar:
            config_path = os.path.join(
                self.model_base_path, "configs", "init_one_example_ptn_96.yml"
            )
            if kVerbose:
                print("-- Using log-polar model")
        else:
            config_path = os.path.join(
                self.model_base_path, "configs", "init_one_example_stn_16.yml"
            )
            if kVerbose:
                print("-- Using cartesian model")
        cfg.merge_from_file(config_path)

        self.model_weights_path = (
            self.model_base_path + cfg.TEST.MODEL_WEIGHTS
        )  # N.B.: this must stay here, after cfg.merge_from_file()
        if kVerbose2:
            print("model_weights_path:", self.model_weights_path)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True

        self.do_cuda = do_cuda & torch.cuda.is_available()
        print("cuda:", self.do_cuda)
        device = torch.device("cuda:0" if self.do_cuda else "cpu")
        self.device = device

        torch.set_grad_enabled(False)

        print("==> Loading pre-trained network.")
        self.model = HardNet(
            transform=cfg.TEST.TRANSFORMER,
            coords=cfg.TEST.COORDS,
            patch_size=cfg.TEST.IMAGE_SIZE,
            scale=cfg.TEST.SCALE,
            is_desc256=cfg.TEST.IS_DESC_256,
            orientCorrect=cfg.TEST.ORIENT_CORRECTION,
        )

        self.checkpoint = torch.load(self.model_weights_path)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        if self.do_cuda:
            self.model.cuda()
            print("Extracting on GPU")
        else:
            print("Extracting on CPU")
            self.model = self.model.cpu()
        self.model.eval()
        print("==> Successfully loaded pre-trained network.")

    def compute_des(self, img, kps):
        h, w = img.shape

        t = time()
        pts = np.array([kp.pt for kp in kps])
        scales = np.array([kp.size for kp in kps])
        oris = np.array([kp.angle for kp in kps])

        # Mirror-pad the image to avoid boundary effects
        if any([s > cfg.TEST.PAD_TO for s in img.shape[:2]]):
            raise RuntimeError(
                "Image exceeds acceptable size ({}x{}), please downsample".format(
                    cfg.TEST.PAD_TO, cfg.TEST.PAD_TO
                )
            )

        fillHeight = cfg.TEST.PAD_TO - img.shape[0]
        fillWidth = cfg.TEST.PAD_TO - img.shape[1]

        padLeft = int(np.round(fillWidth / 2))
        padRight = int(fillWidth - padLeft)
        padUp = int(np.round(fillHeight / 2))
        padDown = int(fillHeight - padUp)

        img = np.pad(img, pad_width=((padUp, padDown), (padLeft, padRight)), mode="reflect")

        # Normalize keypoint locations
        kp_norm = []
        for i, p in enumerate(pts):
            _p = (
                2
                * np.array(
                    [(p[0] + padLeft) / (cfg.TEST.PAD_TO), (p[1] + padUp) / (cfg.TEST.PAD_TO)]
                )
                - 1
            )
            kp_norm.append(_p)

        theta = [
            torch.from_numpy(np.array(kp_norm)).float().squeeze(),
            torch.from_numpy(scales).float(),
            torch.from_numpy(np.array([np.deg2rad(o) for o in oris])).float(),
        ]

        if kVerbose2:
            print(
                "-- Padded image from {}x{} to {}x{} in {} s".format(
                    h, w, img.shape[0], img.shape[1], time() - t
                )
            )

        # Extract descriptors
        t = time()
        device = self.device
        imgs = torch.from_numpy(img).unsqueeze(0).to(device)
        img_keypoints = [theta[0].to(device), theta[1].to(device), theta[2].to(device)]

        descriptors, patches = self.model(
            {"img": imgs}, img_keypoints, ["img"] * len(img_keypoints[0])
        )
        if kVerbose2:
            print(
                "-- Computed {} descriptors in {:0.2f} sec.".format(
                    descriptors.shape[0], time() - t
                )
            )
        return descriptors.cpu().detach().numpy()

    def compute(self, img, kps, mask=None):  # mask is a fake input
        num_kps = len(kps)
        des = []
        if num_kps > 0:
            des = self.compute_des(img, kps)
        if kVerbose:
            print("descriptor: LOGPOLAR, #features: ", len(kps), ", frame res: ", img.shape[0:2])
        return kps, des
