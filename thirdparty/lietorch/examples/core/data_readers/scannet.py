
import numpy as np
import pickle
import torch
import glob
import cv2
import os
import os.path as osp

from .base import RGBDDataset
from .augmentation import RGBDAugmentor
from .rgbd_utils import pose_matrix_to_quaternion
from .rgbd_utils import  all_pairs_distance_matrix


class ScanNet(RGBDDataset):
    def __init__(self, **kwargs):
        super(ScanNet, self).__init__(root='datasets/ScanNet', name='ScanNet', **kwargs)

    @staticmethod
    def is_test_scene(scene):
        scanid = int(re.findall(r'scene(.+?)_', scene)[0])
        return scanid > 660

    def _build_dataset_index(self):
        """ construct scene_info and dataset_index objects """

        from tqdm import tqdm
        print("Building ScanNet dataset")

        scene_info = {}
        dataset_index = []

        for scene in tqdm(os.listdir(self.root)):
            scene_path = osp.join(self.root, scene)
            depth_glob = osp.join(scene_path, 'depth', '*.png')
            depth_list = glob.glob(depth_glob)

            get_indicies = lambda x: int(osp.basename(x).split('.')[0])
            get_images = lambda i: osp.join(scene_path, 'color', '%d.jpg' % i)
            get_depths = lambda i: osp.join(scene_path, 'depth', '%d.png' % i)
            get_poses = lambda i: osp.join(scene_path, 'pose', '%d.txt' % i)

            indicies = sorted(map(get_indicies, depth_list))[::2]
            image_list = list(map(get_images, indicies))
            depth_list = list(map(get_depths, indicies))

            pose_list = map(get_poses, indicies)
            pose_list = list(map(ScanNet.pose_read, pose_list))

            # remove nan poses
            pvecs = np.stack(pose_list, 0)
            keep, = np.where(~np.any(np.isnan(pvecs) | np.isinf(pvecs), axis=1))
            images = [image_list[i] for i in keep]
            depths = [depth_list[i] for i in keep]
            poses = [pose_list[i] for i in keep]

            intrinsic = ScanNet.calib_read(scene_path)
            intrinsics = [intrinsic] * len(images)

            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            for i in range(len(images)):
                if len(graph[i][0]) > 1:
                    dataset_index.append((scene, i))

        return scene_info, dataset_index
            
    @staticmethod
    def calib_read(scene_path):
        intrinsic_file = osp.join(scene_path, 'intrinsic', 'intrinsic_depth.txt')
        K = np.loadtxt(intrinsic_file, delimiter=' ')
        return np.array([K[0,0], K[1,1], K[0,2], K[1,2]])

    @staticmethod
    def pose_read(pose_file):
        pose = np.loadtxt(pose_file, delimiter=' ').astype(np.float64)
        return pose_matrix_to_quaternion(pose)

    @staticmethod
    def image_read(image_file):
        image = cv2.imread(image_file)
        return cv2.resize(image, (640, 480))

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 1000.0