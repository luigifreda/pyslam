import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('depth_pro') 

import torch
import numpy as np
import cv2
from PIL import Image

import depth_pro

depth_pro_base_path='../../thirdparty/ml_depth_pro/'
image_path = depth_pro_base_path + "/data/example.jpg"

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Using CUDA')
    else:
        print('Using CPU')

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()

    # Load and preprocess an image.
    print(f'Loading image {image_path} ...')
    image, _, f_px = depth_pro.load_rgb(image_path)
    print(f'image shape: {image.shape}')
    print(f'f_px: {f_px}')
    print(f'Transforming image ...')
    image = transform(image)
    #print(f'image: {image}')

    # Run inference.
    print(f'Running inference ...')
    prediction = model.infer(image, f_px=f_px)

    # Extract depth and focal length.
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    
    # Visualize depth map.
    depth = depth.squeeze().cpu().numpy()
    depth = np.clip(depth, 0, np.max(depth))
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
    depth = depth.astype(np.uint8)
    cv2.imshow("Depth", depth)
    cv2.waitKey()