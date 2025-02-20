
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .rgbd_utils import *

class RGBDStream(data.Dataset):
    def __init__(self, datapath, frame_rate=-1, crop_size=[384,512]):
        self.datapath = datapath
        self.frame_rate = frame_rate
        self.crop_size = crop_size
        self._build_dataset_index()
    
    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ return training video """
        image = self.__class__.image_read(self.images[index])
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        depth = self.__class__.depth_read(self.depths[index])
        depth = torch.from_numpy(depth).float()

        pose = torch.from_numpy(self.poses[index]).float()
        intrinsic = torch.from_numpy(self.intrinsics[index]).float()

        sx = self.crop_size[1] / depth.shape[1]
        sy = self.crop_size[0] / depth.shape[0]
        image = F.interpolate(image[None], self.crop_size, mode='bilinear', align_corners=True)[0]
        depth = F.interpolate(depth[None,None], self.crop_size, mode='nearest')[0,0]

        image = image[..., 8:-8, 8:-8]
        depth = depth[..., 8:-8, 8:-8]

        fx, fy, cx, cy = intrinsic.unbind(dim=0)
        intrinsic = torch.stack([sx*fx, sy*fy, sx*cx - 8, sy*cy - 8])

        # intrinsic *= torch.as_tensor([sx, sy, sx, sy])
        return index, image, depth, pose, intrinsic



                