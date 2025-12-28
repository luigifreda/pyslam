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

import pyslam.config as config

config.cfg.set_lib("l2net_keras")

import numpy as np


from L2_Net import L2Net

from .feature_base import BaseFeature2D
from pyslam.utilities.logging import Printer
from pyslam.utilities.features import extract_patches_array, extract_patches_array_cpp


kVerbose = True


# Interface for pySLAM
class L2NetKerasFeature2D(BaseFeature2D):
    def __init__(self, do_tf_logging=False):
        print("Using L2NetKerasFeature2D")

        #  One of "L2Net-HP", "L2Net-HP+", "L2Net-LIB", "L2Net-LIB+", "L2Net-ND", "L2Net-ND+", "L2Net-YOS", "L2Net-YOS+",
        self.net_name = "L2Net-HP+"

        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint
        self.mag_factor = 3

        # inference batch size
        self.batch_size = 512
        self.process_all = True  # process all the patches at once

        print("==> Loading pre-trained network.")
        self.l2net = L2Net(self.net_name, do_tf_logging=do_tf_logging)
        print("==> Successfully loaded pre-trained network.")

    def compute(self, frame, kps, mask=None):  # mask is a fake input
        # print('kps: ', kps)
        if len(kps) > 0:
            if False:
                # use python code
                patches = extract_patches_array(
                    frame, kps, patch_size=32, mag_factor=self.mag_factor
                )
            else:
                # use faster cpp code
                patches = extract_patches_array_cpp(
                    frame, kps, patch_size=32, mag_factor=self.mag_factor
                )
            patches = np.asarray(patches)
            patches = np.expand_dims(patches, -1)
            self.des = self.l2net.calc_descriptors(patches)
        else:
            self.des = []
        if kVerbose:
            print("descriptor: L2NET, #features: ", len(kps), ", frame res: ", frame.shape[0:2])
        return kps, self.des
