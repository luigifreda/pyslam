#!/usr/bin/env -S python3 -O
import sys 
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

from pyslam.config import Config

from pyslam.viz.mplot_figure import MPlotFigure
from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType

from pyslam.utilities.utils_img_processing import detect_blur_laplacian
from pyslam.utilities.utils_sys import Printer

# ==================================================================================================
# N.B.: here we test feature manager detectAndCompute()
# ==================================================================================================

if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config)

    des = None 

    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.is_ok:
        
        print('..................................')
             
        img = dataset.getImageColor(img_id)        
        timestamp = dataset.getTimestamp()  
        next_timestamp = dataset.getNextTimestamp() # get next timestamp 
        frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1        
        
        if img is not None:
            time_start = time.time()
            is_blurry, laplacian_var = detect_blur_laplacian(img)
            time_end = time.time()
            
            message = f'img {img_id}, timestamp: {timestamp}, is blurry, laplacian_var: {laplacian_var}, processing time: {time_end-time_start}'
            if is_blurry:
                Printer.red(message)
            else: 
                Printer.green(message)   
            
            cv2.imshow('image', img)
            
            processing_duration = time.time()-time_start
            if(frame_duration > processing_duration):
                time.sleep(frame_duration-processing_duration) 
                
            img_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
