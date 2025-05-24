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
import sys
import config

#config.cfg.remove_lib('mast3r')
config.cfg.set_lib('mvdust3r') 

import copy
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as tvf
import trimesh
from scipy.spatial.transform import Rotation

from viewer3D import Viewer3D, VizPointCloud, VizMesh, VizCameraImage
from utils_img import ImageTable
import time

from utils_files import select_image_files 
from utils_dust3r import dust3r_preprocess_images #, invert_dust3r_preprocess_depth
from utils_img import img_from_floats
from utils_depth import img_from_depth, PointCloud, point_cloud_to_depth

from camera import Camera
from dataset_types import DatasetEnvironmentType
from depth_estimator_base import DepthEstimator


#from dust3r.dummy_io import *
os.environ["meta_internal"] = "False"

from mvdust3r.dust3r.inference import inference_mv
from mvdust3r.dust3r.model import AsymmetricCroCo3DStereoMultiView
from mvdust3r.dust3r.utils.device import to_numpy


inf = np.inf
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kMvdust3rFolder = kRootFolder + '/thirdparty/mvdust3r'
kResultsFolder = kRootFolder + '/results/mvdust3r'
kMvdust3rMModelWeightPath = kMvdust3rFolder + '/checkpoints/MVD.pth'

# Mono/Stereo depth estimator using the Mv-DUST3R model.
# NOTE: See the example test/dust3r/test_mvdust3r.py
class DepthEstimatorMvdust3r(DepthEstimator):
    def __init__(self, device=None, camera:Camera=None, 
                 min_depth=0, max_depth=50,
                 inference_size=512,   # choices=[512, 224]
                 min_conf_thr=10,      # percentage of the max confidence value                 
                 dataset_env_type=DatasetEnvironmentType.OUTDOOR,
                 weights_path=kMvdust3rMModelWeightPath,
                 model_name='MVD'):  # name of the model weights, choices=["MVD", "MVDp"]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('DepthEstimatorMvdust3r: Using CUDA')
        else:
            print('DepthEstimatorMvdust3r: Using CPU')    
            
        self.inference_size = inference_size
        self.min_conf_thr = min_conf_thr        
                    
        if model_name is None:
            if "MVDp" in weights_path:
                model_name = "MVDp"
            elif "MVD" in weights_path:
                model_name = "MVD"
            else:
                raise ValueError("model name not found in weights path")
            
        if not os.path.exists(weights_path):
            raise ValueError(f"weights file {weights_path} not found")

        # Load model and preprocessing transform
        if model_name == "MVD":
            model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
            model.to(device)
            model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(device)
            state_dict_loaded = model_loaded.state_dict()
            model.load_state_dict(state_dict_loaded, strict=True)
        elif model_name == "MVDp":
            model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True, n_ref = 4)
            model.to(device)
            model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(weights_path).to(device)
            state_dict_loaded = model_loaded.state_dict()
            model.load_state_dict(state_dict_loaded, strict=True)

        else:
            raise ValueError(f"{model_name} is not supported")
            
        transform = None
        super().__init__(model, transform, device, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type)

    def _infer_mv(self, model, device, imgs, verbose):
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        for img in imgs:
            img['true_shape'] = torch.from_numpy(img['true_shape']).long()

        if len(imgs) < 12:
            if len(imgs) > 3:
                imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
            if len(imgs) > 6:
                imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
        else:
            change_id = len(imgs) // 4 + 1
            imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
            change_id = (len(imgs) * 2) // 4 + 1
            imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
            change_id = (len(imgs) * 3) // 4 + 1
            imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])
        
        output = inference_mv(imgs, model, device, verbose=verbose)

        # print(output['pred1']['rgb'].shape, imgs[0]['img'].shape, 'aha')
        output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
        for x, img in zip(output['pred2s'], imgs[1:]):
            x['rgb'] = img['img'].permute(0,2,3,1)
        
        return output

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        images = []
        if image_right is None:
            images = [image,image]
        else:
            images = [image,image_right]
        images = dust3r_preprocess_images(images, self.inference_size)
        
        # get inference output 
        output = self._infer_mv(self.model, self.device, images, verbose=False)
        
        # at this stage, you have the raw dust3r predictions
        # view1, pred1 = output['view1'], output['pred1']
        # view2, pred2 = output['view2'], output['pred2']        
        
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

        intrinsics = self.camera.K
        # Extract valid 3D points
        pts3d1_flat = pts3d[0][valid_first].reshape(-1, 3)  # (N, 3)
        # Project 3D points to 2D image plane and get a depth map
        depth_prediction_project = point_cloud_to_depth(pts3d1_flat, intrinsics, image.shape[1], image.shape[0])   
           
        pts3d_first = pts3d[0][valid_first]
        color_first = rgb_imgs[0][valid_first]
        
        return depth_prediction_project, PointCloud(pts3d_first, color_first)


