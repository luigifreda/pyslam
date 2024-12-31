import os
import sys 
sys.path.append("../../")
from config import Config
config = Config()
config.set_lib('depth_anything_v2') 

from utils_depth import depth2pointcloud, img_from_depth, filter_shadow_points, PointCloud

import time
import cv2 
import numpy as np

from dataset import dataset_factory
from frame import Frame
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from config_parameters import Parameters

from utils_depth import depth2pointcloud, img_from_depth, filter_shadow_points, PointCloud

import torch
from depth_anything_v2.dpt import DepthAnythingV2

from viewer3D import Viewer3D



depth_anything_v2_path = '../../thirdparty/depth_anything_v2/metric_depth'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

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
    
    encoder_name = 'vitl' # or 'vits', 'vitb'
    dataset_name = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 50 # 20 for indoor model, 80 for outdoor model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Using CUDA')
    else:
        print('Using CPU')   
        
    # Load model     
    model = DepthAnythingV2(**{**model_configs[encoder_name], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'{depth_anything_v2_path}/checkpoints/depth_anything_v2_metric_{dataset_name}_{encoder_name}.pth', map_location='cpu'))
    model = model.to(device).eval()
        
        
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while True:

        timestamp, img = None, None 
        
        if dataset.isOk():
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
                                                
            start_time = time.time()

            depth_prediction = model.infer_image(img)
            #print(f'depth shape: {depth.shape}')
            
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