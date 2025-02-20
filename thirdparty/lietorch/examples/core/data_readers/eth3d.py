
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

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream
from .augmentation import RGBDAugmentor
from .rgbd_utils import loadtum, all_pairs_distance_matrix

class ETH3D(RGBDDataset):
    def __init__(self, **kwargs):
        super(ETH3D, self).__init__(root='datasets/ETH3D', name='ETH3D', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return False

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building ETH3D dataset")

        scene_info = {}
        dataset_index = []

        for scene in tqdm(os.listdir(self.root)):
            scene_path = osp.join(self.root, scene)
            
            if not osp.isdir(scene_path):
                continue
            
            # don't use scenes with no rgb info
            if 'dark' in scene or 'kidnap' in scene:
                continue

            scene_data, graph = {}, {}
            images, depths, poses, intrinsics = loadtum(scene_path, skip=2)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 5000.0


class ETH3DStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(ETH3DStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        images, depths, poses, intrinsics = loadtum(self.datapath, self.frame_rate)

        # set first pose to identity
        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        self.images = images
        self.poses = poses
        self.depths = depths
        self.intrinsics = intrinsics

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 5000.0
