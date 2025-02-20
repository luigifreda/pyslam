
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

from .augmentation import RGBDAugmentor
from .rgbd_utils import *

class RGBDDataset(data.Dataset):
    def __init__(self, root, name, n_frames=4, crop_size=[384,512], fmin=8.0, fmax=75.0, do_aug=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = root
        self.name = name

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        
        if do_aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        cache_path = osp.join(cur_path, 'cache', '{}.pickle'.format(self.name))

        if not osp.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        if osp.isfile(cache_path):
            scene_info = pickle.load(open(cache_path, 'rb'))[0]
        else:
            scene_info = self._build_dataset()
            with open(cache_path, 'wb') as cachefile:
                pickle.dump((scene_info,), cachefile)

        self.scene_info = scene_info
        self._build_dataset_index()
                
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                for i in graph:
                    if len(graph[i][0]) > self.n_frames:
                        self.dataset_index.append((scene, i))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        # uncomment for nice visualization
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index):
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]

        frame_graph = self.scene_info[scene_id]['graph']
        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold
            k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
            frames = frame_graph[ix][0][k]

            # prefer frames forward in time
            if np.count_nonzero(frames[frames > ix]):
                ix = np.random.choice(frames[frames > ix])
            
            elif np.count_nonzero(frames):
                ix = np.random.choice(frames)

            inds += [ ix ]

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        depths = torch.from_numpy(depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.aug is not None:
            images, poses, depths, intrinsics = \
                self.aug(images, poses, depths, intrinsics)

        return images, poses, depths, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
