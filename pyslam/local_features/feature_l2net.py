"""
* This file is part of PYSLAM
* Adapted from https://github.com/vcg-uvic/image-matching-benchmark-baselines/blob/master/third_party/l2net_config/l2net_model.py, see licence therein.
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

# adapted from https://github.com/vcg-uvic/image-matching-benchmark-baselines/blob/master/third_party/l2net_config/l2net_model.py

import pyslam.config as config

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy as np

from .feature_base import BaseFeature2D
from pyslam.utilities.features import (
    extract_patches_tensor,
    extract_patches_array,
    extract_patches_array_cpp,
)


kVerbose = True


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x


class L2Net(nn.Module):

    def __init__(self):
        super(L2Net, self).__init__()
        self.eps = 1e-10
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=self.eps),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True),
            nn.BatchNorm2d(128, affine=True, eps=self.eps),
        )
        return

    def input_norm(self, x):
        # matlab norm
        z = x.contiguous().transpose(2, 3).contiguous().view(x.size(0), -1)
        x_minus_mean = z.transpose(0, 1) - z.mean(1)
        sp = torch.std(z, 1).detach()
        norm_inp = x_minus_mean / (sp + 1e-12)
        norm_inp = norm_inp.transpose(0, 1).view(-1, 1, x.size(2), x.size(3)).transpose(2, 3)
        return norm_inp

    def forward(self, input):
        norm_img = self.input_norm(input)
        x_features = self.features(norm_img)
        return nn.LocalResponseNorm(256, 1 * 256, 0.5, 0.5)(x_features).view(input.size(0), -1)


# Interface for pySLAM
class L2NetFeature2D(BaseFeature2D):
    def __init__(self, do_cuda=True):
        print("Using L2NetFeature2D")
        self.model_base_path = config.cfg.root_folder + "/thirdparty/l2net/"
        self.model_weights_path = self.model_base_path + "l2net_ported_weights_lib+.pth"
        # print('model_weights_path:',self.model_weights_path)

        # get pre-trained image mean
        # l2net_weights = sio.loadmat(args.matlab_weights_path)
        # imgMean =  l2net_weights['pixMean']

        self.do_cuda = do_cuda & torch.cuda.is_available()
        print("cuda:", self.do_cuda)
        device = torch.device("cuda:0" if self.do_cuda else "cpu")

        torch.set_grad_enabled(False)

        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint
        self.mag_factor = 1.0

        # inference batch size
        self.batch_size = 512
        self.process_all = True  # process all the patches at once

        print("==> Loading pre-trained network.")
        self.model = L2Net()
        self.checkpoint = torch.load(self.model_weights_path)
        # self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.load_state_dict(self.checkpoint)
        if self.do_cuda:
            self.model.cuda()
            print("Extracting on GPU")
        else:
            print("Extracting on CPU")
            self.model = self.model.cpu()
        self.model.eval()
        print("==> Successfully loaded pre-trained network.")

    def compute_des_batches(self, patches):
        n_batches = int(len(patches) / self.batch_size) + 1
        descriptors_for_net = np.zeros((len(patches), 128))
        for i in range(0, len(patches), self.batch_size):
            data_a = patches[i : i + self.batch_size, :, :, :].astype(np.float32)
            data_a = torch.from_numpy(data_a)
            if self.do_cuda:
                data_a = data_a.cuda()
            data_a = Variable(data_a)
            # compute output
            with torch.no_grad():
                out_a = self.model(data_a)
            descriptors_for_net[i : i + self.batch_size, :] = (
                out_a.data.cpu().numpy().reshape(-1, 128)
            )
        return descriptors_for_net

    def compute_des(self, patches):
        patches = torch.from_numpy(patches).float()
        patches = torch.unsqueeze(patches, 1)
        if self.do_cuda:
            patches = patches.cuda()
        with torch.no_grad():
            descrs = self.model(patches)
        return descrs.detach().cpu().numpy().reshape(-1, 128)

    def compute(self, img, kps, mask=None):  # mask is a fake input
        num_kps = len(kps)
        des = []
        if num_kps > 0:
            if not self.process_all:
                # compute descriptor for each patch
                patches = extract_patches_tensor(
                    img, kps, patch_size=32, mag_factor=self.mag_factor
                )
                des = self.compute_des_batches(patches).astype(np.float32)
            else:
                # compute descriptor by feeeding the full patch tensor to the network
                t = time.time()
                if False:
                    # use python code
                    patches = extract_patches_array(
                        img, kps, patch_size=32, mag_factor=self.mag_factor
                    )
                else:
                    # use faster cpp code
                    patches = extract_patches_array_cpp(
                        img, kps, patch_size=32, mag_factor=self.mag_factor
                    )
                patches = np.asarray(patches)
                if kVerbose:
                    print("patches.shape:", patches.shape)
                if kVerbose:
                    print("patch elapsed: ", time.time() - t)
                des = self.compute_des(patches)
        if kVerbose:
            print("descriptor: L2NET, #features: ", len(kps), ", frame res: ", img.shape[0:2])
        return kps, des
