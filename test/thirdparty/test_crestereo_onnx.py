import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('crestereo_onnx') 
import os

import cv2
import numpy as np
import torch
import re

import onnxruntime as onxr
from crestereo import CREStereo, CameraConfig


data_path = '../data'
models_path = '../../data'
crestereo_onnx__base_path='../../thirdparty/crestereo_onnx'



model_names = [
    "crestereo_combined_iter10_240x320.onnx",
    "crestereo_combined_iter10_480x640.onnx",
    "crestereo_combined_iter10_720x1280.onnx",
    "crestereo_combined_iter20_240x320.onnx",
    "crestereo_combined_iter20_480x640.onnx",
    "crestereo_combined_iter20_720x1280.onnx",
    "crestereo_combined_iter2_240x320.onnx",
    "crestereo_combined_iter2_480x640.onnx",
    "crestereo_combined_iter2_720x1280.onnx",
    "crestereo_combined_iter5_240x320.onnx",
    "crestereo_combined_iter5_380x480.onnx",
    "crestereo_combined_iter5_480x640.onnx",
    "crestereo_combined_iter5_720x1280.onnx",
    "crestereo_init_iter10_180x320.onnx",
    "crestereo_init_iter10_240x320.onnx",
    "crestereo_init_iter10_480x640.onnx",
    "crestereo_init_iter10_720x1280.onnx",
    "crestereo_init_iter20_180x320.onnx",
    "crestereo_init_iter20_240x320.onnx",
    "crestereo_init_iter20_480x640.onnx",
    "crestereo_init_iter20_720x1280.onnx",
    "crestereo_init_iter2_180x320.onnx",
    "crestereo_init_iter2_240x320.onnx",
    "crestereo_init_iter2_480x640.onnx",
    "crestereo_init_iter2_720x1280.onnx",
    "crestereo_init_iter5_180x320.onnx",
    "crestereo_init_iter5_240x320.onnx",
    "crestereo_init_iter5_480x640.onnx",
    "crestereo_init_iter5_720x1280.onnx"
]

# Select the best model based on the closest resolution to the input image.
def select_best_model(model_names, image):
   # Extract image height and width
    if isinstance(image, np.ndarray):  # For numpy arrays
        height, width = image.shape[:2]
    elif isinstance(image, torch.Tensor):  # For PyTorch tensors
        height, width = image.size(-2), image.size(-1)
    else:
        raise TypeError("Unsupported image type. Provide a numpy array or a torch tensor.")

    best_model = None
    best_diff = float('inf')
    
    for model_name in model_names:
        # Extract resolution from model name using regex
        match = re.search(r'(\d+)x(\d+)', model_name)
        if match:
            model_height, model_width = map(int, match.groups())
            # Calculate resolution difference
            diff = abs(model_height - height) + abs(model_width - width)
            if diff < best_diff:
                best_diff = diff
                best_model = model_name
                
    return best_model

if __name__ == "__main__":

    # # Initialize video
    # # cap = cv2.VideoCapture("video.mp4")
    # videoUrl = 'https://youtu.be/Yui48w71SG0'
    # start_time = 0 # skip first {start_time} seconds
    # videoPafy = pafy.new(videoUrl)
    # print(videoPafy.streams)
    # cap = cv2.VideoCapture(videoPafy.streams[-1].url)
    # # cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)
    
    
    print(onxr.get_device())
    
    left_img_path = data_path + '/stereo_bicycle/im0.png'
    right_img_path = data_path + '/stereo_bicycle/im1.png'
    
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)    

    # Model options (not all options supported together)
    iters = 5            # Lower iterations are faster, but will lower detail. 
                        # Options: 2, 5, 10, 20 

    #input_shape = (320, 480)   # Input resolution. 
                        # Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

    input_shape = left_img.shape

    version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
                        # Options: "init", "combined"

    # Camera options: baseline (m), focal length (pixel) and max distance
    # TODO: Fix with the values witht the correct configuration for YOUR CAMERA
    camera_config = CameraConfig(0.12, 0.5*input_shape[1]/0.72) 
    max_distance = 10

    model_name = select_best_model(model_names, left_img)
    print(f"Selected model: {model_name}")

    # Initialize model
    #model_path = f'{models_path}/models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
    model_path = f'{models_path}/models/' + model_name
    depth_estimator = CREStereo(model_path, camera_config=camera_config, max_dist=max_distance)

    #cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	


    # Estimate the depth
    disparity_map = depth_estimator(left_img, right_img)
    #color_depth = depth_estimator.draw_depth()

    cv2.imshow("lefgt image", left_img)
    cv2.imshow("disparity map", disparity_map)
    cv2.waitKey(0)

