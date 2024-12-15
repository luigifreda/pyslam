import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('depth_anything_v2') 

import torch
import numpy as np
import cv2
import torch

from utils_depth import depth2pointcloud, img_from_depth, filter_shadow_points, PointCloud

from depth_anything_v2.dpt import DepthAnythingV2


depth_anything_v2_path = '../../thirdparty/depth_anything_v2/metric_depth'
data_path = '../data/'
image_path = data_path + 'kitti06-12-color.png'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
    
if __name__ == "__main__":

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'{depth_anything_v2_path}/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    
    #print(model) # print model architecture

    raw_img = cv2.imread(image_path)
    print(f'image shape: {raw_img.shape}')
    
    image = raw_img

    # Call infer_image with error handling
    try:
        depth = model.infer_image(image)
        print(f'depth shape: {depth.shape}')
    except Exception as e:
        print(f"Error during inference: {e}")
        exit(1) 
            
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
    
    # Visualize depth map.
    depth = img_from_depth(depth, img_min=0, img_max=max_depth)
    
    cv2.imshow("image", raw_img)
    cv2.imshow("depth", depth)
    cv2.waitKey()    