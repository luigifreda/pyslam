
import numpy as np
import torch

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from lietorch import SE3
from .stream import RGBDStream
from .rgbd_utils import loadtum

intrinsics_dict = {
    'freiburg1': [517.3, 516.5, 318.6, 255.3],
    'freiburg2': [520.9, 521.0, 325.1, 249.7],
    'freiburg3': [535.4, 539.2, 320.1, 247.6], 
}

distortion_dict = {
    'freiburg1': [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
    'freiburg2': [0.2312, -0.7849, -0.0033, -0.0001, 0.9172],
    'freiburg3': [0, 0, 0, 0, 0],
}

def as_intrinsics_matrix(intrinsics):
    K = np.eye(3)
    K[0,0] = intrinsics[0]
    K[1,1] = intrinsics[1]
    K[0,2] = intrinsics[2]
    K[1,2] = intrinsics[3]
    return K


class TUMStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TUMStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        images, depths, poses, intrinsics = loadtum(self.datapath, self.frame_rate)
        intrinsic, _ = TUMStream.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        # set first pose to identity
        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        self.images = images
        self.poses = poses
        self.depths = depths
        self.intrinsics = intrinsics

    @staticmethod
    def calib_read(datapath):
        if 'freiburg1' in datapath:
            intrinsic = intrinsics_dict['freiburg1']
            d_coef = distortion_dict['freiburg1']

        elif 'freiburg2' in datapath:
            intrinsic = intrinsics_dict['freiburg2']
            d_coef = distortion_dict['freiburg2']

        elif 'freiburg3' in datapath:
            intrinsic = intrinsics_dict['freiburg3']
            d_coef = distortion_dict['freiburg3']

        return np.array(intrinsic), np.array(d_coef)

    @staticmethod
    def image_read(image_file):
        intrinsics, d_coef = TUMStream.calib_read(image_file)
        K = as_intrinsics_matrix(intrinsics)
        image = cv2.imread(image_file)
        return cv2.undistort(image, K, d_coef)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 5000.0
