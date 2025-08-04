import os
import sys 


from pyslam.config import Config
config = Config()
config.set_lib('depth_pro') 

from pyslam.utilities.utils_files import gdrive_download_lambda 
from pyslam.utilities.utils_sys import getchar, Printer 
from pyslam.utilities.utils_depth import depth2pointcloud, img_from_depth, filter_shadow_points, PointCloud

import math
import cv2 
import numpy as np

from pyslam.io.dataset_factory import dataset_factory
from pyslam.slam.frame import Frame, FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.config_parameters import Parameters

import torch
import depth_pro

import time

from pyslam.viz.viewer3D import Viewer3D


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000
    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared. 
    FeatureTrackerShared.set_feature_tracker(feature_tracker)     
    
    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
    
    fx = config.cam_settings['Camera.fx']
    fy = config.cam_settings['Camera.fy']
    cx = config.cam_settings['Camera.cx']
    cy = config.cam_settings['Camera.cy']
    
    print(f'fx: {fx}, fy: {fy}')
    
    max_depth = 50   # [m]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Using CUDA')
    else:
        print('Using CPU')    
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model = model.to(device).eval()
        
        
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while True:

        timestamp, img = None, None 
        
        if dataset.is_ok:
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
            
            start_time = time.time()
            
            image = transform(img)
                                                
            prediction = model.infer(image, f_px=torch.tensor(fx))

            # Extract depth and focal length.
            depth_prediction = prediction["depth"]  # Depth in [m].
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.            
            depth_prediction = depth_prediction.squeeze().cpu().numpy()
            
            print(f'inference time: {time.time() - start_time}')

            # Filter depth
            if True:
                depth_filtered = filter_shadow_points(depth_prediction)
            else:
                depth_filtered = depth_prediction

            # Visualize depth map.                        
            depth_img = img_from_depth(depth_prediction, img_min=0, img_max=max_depth)
            depth_filtered_img = img_from_depth(depth_filtered, img_min=0, img_max=max_depth)

            
            # Visualize 3D point cloud.
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             
            point_cloud = depth2pointcloud(depth_filtered, image_rgb, fx, fy, cx, cy, max_depth) 
            viewer3D.draw_dense_geometry(point_cloud=point_cloud)
            
            
            cv2.imshow('img', img)
            cv2.imshow("depth", depth_img)
            cv2.imshow("depth filtered", depth_filtered_img)            
            
            cv2.waitKey(1)
        else: 
            cv2.waitKey(100)
            
        img_id += 1