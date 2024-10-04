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


import os
import time
import math 
from multiprocessing import Process, Queue, Value, Condition
from threading import Thread 
import numpy as np
import cv2

from config import Config
config = Config()

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopDetectionCandidateImgs
from utils_features import transform_float_to_binary_descriptor
from timer import TimerFps

from keyframe import KeyFrame
from frame import Frame

from loop_detection import LoopDetection
from parameters import Parameters


kTimerVerbose = False

           
if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detection import print
            
            
class LoopClosing(object):
    def __init__(self, slam):
        self.slam = slam
        self.sensor_type = slam.sensor_type
        
        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings  
        self.timer_loop_closing = TimerFps('LoopClosing', is_verbose = self.timer_verbose)         
        
        self.queued_keyframes_map = {}
        self.last_loop_kf_id = 0
        
        self.loop_detection = LoopDetection(slam) # parallel process
        self.time_loop_detection = self.loop_detection.time_loop_detection
        
        self.is_running = False
        self.stop = False    
        self.work_thread = Thread(target=self.run)    

    def start(self):    
        self.work_thread.start()
        
    def quit(self):
        print('LoopClosing: quitting...')            
        if self.is_running:        
            self.is_running = False           
            if self.stop == False:
                self.stop = True    
                if self.work_thread is not None:
                    self.work_thread.join(timeout=5)
        self.loop_detection.quit()                          
        print('LoopClosing: done')             
      

    def add_keyframe(self, keyframe: KeyFrame, img):
        print(f'LoopClosing: Adding keyframe {keyframe.kid} ({keyframe.id})')
        keyframe.set_not_erase()
        # If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        if keyframe.kid < self.last_loop_kf_id + 10:
            keyframe.set_erase()
            return 
        self.queued_keyframes_map[keyframe.id] = keyframe
        self.loop_detection.add_keyframe(keyframe, img)

    def run(self):
        # thread execution
        print('LoopClosing: starting...')               
        self.is_running = True
        while not self.stop:
            detection_out = self.loop_detection.pop_output() # blocking call
            try:         
                if detection_out is not None:
                    self.timer_loop_closing.start()
                    # retrieve the keyframe corresponding to the output img_id
                    keyframe = self.queued_keyframes_map[detection_out.img_id]
                    # update the keyframe with the detection output
                    keyframe.g_des = detection_out.g_des_vec
                    # remove the keyframe from the map
                    del self.queued_keyframes_map[detection_out.img_id]         
                    if len(detection_out.candidate_idxs) == 0:
                        keyframe.set_erase()
                        print(f'LoopClosing: No loop candidates detected')
                    else:
                        print(f'LoopClosing: Detected loop candidates between {keyframe.kid} and {detection_out.candidate_idxs}, time: {self.timer_loop_closing.last_elapsed}')
                    self.timer_loop_closing.refresh()                
                else: 
                    print(f'LoopClosing: detection_out None - No loop candidates detected')
            except Exception as e:
                print(f'LoopClosing: Exception {e}')

        print('LoopClosing: loop exit...')                       