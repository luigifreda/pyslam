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


import cv2
import numpy as np
import os

import config

config.cfg.set_lib('mast3r')

import torch

from camera import Camera
from dataset import DatasetEnvironmentType
from utils_dust3r import dust3r_preprocess_images, invert_dust3r_preprocess_depth
from utils_depth import img_from_depth

from depth_estimator_base import DepthEstimator

from mast3r.model import AsymmetricMASt3R

from dust3r.inference import inference
from dust3r.utils.device import to_numpy



kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kMast3rFolder = kRootFolder + '/thirdparty/mast3r'
kResultsFolder = kRootFolder + '/results/mast3r'

model_name = kMast3rFolder + "/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

# Mono/Stereo Depth estimator using the MAST3R model.
# NOTE: See the examples test/dust3r/test_mast3r_2images.py, test/dust3r/test_mast3r.py 
class DepthEstimatorMast3r(DepthEstimator):
    def __init__(self, device=None, camera:Camera=None, 
                 min_depth=0, max_depth=50,
                 inference_size=512,   # choices=[512, 224]
                 min_conf_thr=10,      # percentage of the max confidence value
                 dataset_env_type=DatasetEnvironmentType.OUTDOOR):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('DepthEstimatorMast3r: Using CUDA')
        else:
            print('DepthEstimatorMast3r: Using CPU')    
            
        self.inference_size = inference_size
        self.min_conf_thr = min_conf_thr
            
        # Load model and preprocessing transform
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        model = model.to(device).eval()
        transform = None
        super().__init__(model, transform, device, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type)

    def infer(self, image, image_right=None):
        images = []
        if image_right is None:
            images = [image,image]
        else:
            images = [image,image_right]
        images = dust3r_preprocess_images(images, self.inference_size)
        
        # get inference output 
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)
                
        # extract 3D points
        pts3d = [output['pred1']['pts3d'][0]] # + [output['pred2']['pts3d_in_other_view'][0]]
        pts3d = to_numpy(pts3d) 
        
        # extract rgb images
        rgb_imgs = [output['view1']['img']] # + [output['view2']['img']]
        for i in range(len(rgb_imgs)):
            rgb_imgs[i] = (rgb_imgs[i] + 1) / 2
            rgb_imgs[i] = rgb_imgs[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb_imgs[i] = cv2.cvtColor(rgb_imgs[i], cv2.COLOR_RGB2BGR)
        
        # extract predicted confidence 
        conf = [output['pred1']['conf'][0]] # + [output['pred2']['conf'][0]]
        conf_vec = torch.stack([x.reshape(-1) for x in conf]) # get a monodimensional vector
        conf_sorted = conf_vec.reshape(-1).sort()[0]    
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(self.min_conf_thr) * 0.01)]
        print(f'confidence threshold: {conf_thres}')
        mask = [x >= conf_thres for x in conf]        
        
        # extract first image depth with mask 
        h, w = rgb_imgs[0].shape[0:2]           
        valid_first = mask[0].reshape(h,w)

        if False: 
            # get a depth at the inference size
            depth_map_from_inference = np.zeros(shape=(h,w),dtype=pts3d[0].dtype)
            depth_map_from_inference[valid_first] = pts3d[0][valid_first,2]          
            # get a depth a the original image size by using interpolation  
            depth_prediction_1 = invert_dust3r_preprocess_depth(depth_map_from_inference, image.shape[0:2], self.inference_size)
        
        if True:
            # Project 3D points to 2D image plane and get a
            intrinsics = self.camera.K
            # Extract valid 3D points
            pts3d1_flat = pts3d[0][valid_first].reshape(-1, 3)  # (N, 3)
            # Project to 2D
            pts2d_hom = (intrinsics @ pts3d1_flat.T).T  # (N, 3)
            pts2d = pts2d_hom[:, :2] / pts2d_hom[:, 2:]  # Normalize by depth
            # Create depth map
            depth_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

            # Convert to integer pixel coordinates
            xy = np.rint(pts2d).astype(int)  # Round to nearest integer
            valid_mask = (0 <= xy[:, 0]) & (xy[:, 0] < depth_map.shape[1]) & \
                         (0 <= xy[:, 1]) & (xy[:, 1] < depth_map.shape[0])  # Ensure in bounds

            # Apply depth values using NumPy advanced indexing
            depth_map[xy[valid_mask, 1], xy[valid_mask, 0]] = pts3d1_flat[valid_mask, 2]       
            depth_prediction_2 = depth_map        
            
        if False:
            depth_prediction_1_copy = depth_prediction_1.copy()
            depth_prediction_1_mask = (depth_prediction_2 == 0)
            depth_prediction_1_copy[depth_prediction_1_mask] = 0
            depth_diff = np.abs(depth_prediction_1_copy - depth_prediction_2)
            print(f'depth diff min: {np.min(depth_diff)}, max: {np.max(depth_diff)}, mean: {np.mean(depth_diff)}')
            depth_diff_img = img_from_depth(depth_diff)
            cv2.imshow("depth_diff", depth_diff_img)  
           
        return depth_prediction_2


