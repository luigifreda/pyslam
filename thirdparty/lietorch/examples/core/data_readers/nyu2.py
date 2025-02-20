
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from .base import RGBDDataset
from .augmentation import RGBDAugmentor
from .rgbd_utils import all_pairs_distance_matrix, loadtum

class NYUv2(RGBDDataset):
    def __init__(self, **kwargs):
        super(NYUv2, self).__init__(root='datasets/NYUv2', name='NYUv2', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return False

    def _build_dataset(self):

        from tqdm import tqdm
        print("Building NYUv2 dataset")

        scene_info = {}
        dataset_index = []

        scenes = os.listdir(self.root)
        for scene in tqdm(scenes):
            scene_path = osp.join(self.root, scene)
            images, depths, poses, intrinsics = loadtum(scene_path, frame_rate=10)

            # filter out some errors in dataset
            if images is None or len(images) < 8:
                continue

            intrinsic = NYUv2.calib_read()
            intrinsics = [intrinsic] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02
        return np.array([fx, fy, cx, cy])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 5000.0

