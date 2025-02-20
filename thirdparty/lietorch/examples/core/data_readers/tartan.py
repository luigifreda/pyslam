
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from .base import RGBDDataset
from .stream import RGBDStream

class TartanAir(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0
    TEST_SET = ['westerndesert', 'seasidetown', 'seasonsforest_winter', 'office2', 'gascola']

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(root='datasets/TartanAir', name='TartanAir', **kwargs)

    @staticmethod
    def is_test_scene(scene):
        return scene.split('/')[-3] in TartanAir.TEST_SET

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_left/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))
            
            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


class TartanAirTest(torch.utils.data.Dataset):
    def __init__(self, root='datasets/Tartan'):
        self.root = root
        self.dataset_index = []

        self.scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        
        for scene in sorted(scenes):
            image_glob = osp.join(scene, 'image_left/*.png')
            depth_glob = osp.join(scene, 'depth_left/*.npy')            
            images = sorted(glob.glob(image_glob))
            depths = sorted(glob.glob(depth_glob))

            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            
            self.scene_info[scene] = {'images': images, 
                'depths': depths, 'poses': poses, 'intrinsics': intrinsics}
        
        with open('assets/tartan_test.txt') as f:
            self.dataset_index = f.readlines()
        
    def __getitem__(self, index):
        """ load test example """

        scene_id, ix1, ix2 = self.dataset_index[index].split()
        inds = [int(ix1), int(ix2)]

        images_list = self.scene_info[scene_id]['images']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']

        images, depths, poses, intrinsics = [], [], [], []
        for i in inds:
            images.append(TartanAir.image_read(images_list[i]))
            depths.append(TartanAir.depth_read(depths_list[i]))
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

        return images, poses, depths, intrinsics 

    def __len__(self):
        return len(self.dataset_index)

class TartanAirStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        images, poses, depths, intrinsics = loadtum(self.datapath)
        intrinsic = NYUv2.TUMStream(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images
        self.poses = poses
        self.depths = depths
        self.intrinsics = intrinsics

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        return depth.astype(np.float32) / 5000.0
