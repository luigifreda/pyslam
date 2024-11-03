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

#import multiprocessing as mp
import torch.multiprocessing as mp

import numpy as np
import cv2
from enum import Enum

from utils_sys import Printer, set_rlimit
from utils_data import empty_queue

from parameters import Parameters
from feature_types import FeatureInfo

from timer import TimerFps

from keyframe import KeyFrame
from frame import Frame

from loop_detector_configs import LoopDetectorConfigs, loop_detector_factory, loop_detector_check
from loop_detector_base import LoopDetectorTask, LoopDetectorTaskType, LoopDetectorBase
import traceback
import resource

kVerbose = True
kPrintTrackebackDetails = True 

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'


if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detector_base import print
    


# Entry point for loop detection that generates candidates for loop closure. An instance of LoopDetectingProcess is used by LoopClosing. 
# For efficiency, we use multiprocessing to run detection tasks in a parallel process. That means on a different CPU core thanks to multiprocessing.
# This wouldn't be possible with python multithreading that runs threads on the same CPU core (due to the GIL).
# A LoopDetectingProcess instance is owned by LoopClosing. The latter does the full job of managing (1) detection, (2) consistency verification, (3) geometry verification and (4) correction.  
class LoopDetectingProcess:
    def __init__(self, slam, loop_detector_config = LoopDetectorConfigs.DBOW3):
        set_rlimit()          
        self.loop_detector = loop_detector_factory(**loop_detector_config)
        if slam is not None:
            loop_detector_check(self.loop_detector, slam.feature_tracker.feature_manager.descriptor_type)
           
        self.time_loop_detection = mp.Value('d',0.0)       
        
        self.last_input_task = None
        #self.manager = mp.Manager()  # NOTE: this generates pickling problems with torch.multiprocessing
        #self.q_in = self.manager.Queue()
        #self.q_out = self.manager.Queue()
        self.q_in = mp.Queue()
        self.q_out = mp.Queue()
                                        
        self.q_in_condition = mp.Condition()
        self.q_out_condition = mp.Condition()        
        
        self.is_running  = mp.Value('i',1)
        self.process = mp.Process(target=self.run,
                          args=(self.loop_detector, self.q_in, self.q_in_condition, self.q_out, self.q_out_condition, \
                                self.is_running, self.time_loop_detection,))
        
        #self.process.daemon = True
        self.process.start()
        
        if self.loop_detector.using_torch_mp():
            time.sleep(2) # give a bit of time for the process to start and initialize
        
    def using_torch_mp(self):
        return self.loop_detector.using_torch_mp()

    def quit(self):
        if self.is_running.value == 1:
            print('LoopDetectingProcess: quitting...')
            self.is_running.value = 0            
            with self.q_in_condition:
                self.q_in.put(None)  # put a None in the queue to signal we have to exit
                self.q_in_condition.notify_all()       
            with self.q_out_condition:
                self.q_out_condition.notify_all()                           
            self.process.join(timeout=5)
            if self.process.is_alive():
                Printer.orange("Warning: Loop detection process did not terminate in time, forced kill.")  
                self.process.terminate()      
            print('LoopDetectingProcess: done')   
    
    # main loop of the loop detection process
    def run(self, loop_detector: LoopDetectorBase, q_in, q_in_condition, q_out, q_out_condition, is_running, time_loop_detection):
        print('LoopDetectingProcess: starting...')
        loop_detector.init()
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1:
                    print('LoopDetectingProcess: waiting for new task...')
                    q_in_condition.wait()
            if not q_in.empty():            
                self.loop_detecting(loop_detector, q_in, q_out, q_out_condition, is_running, time_loop_detection)
            else: 
                print('LoopDetectingProcess: q_in is empty...')

        empty_queue(q_in) # empty the queue before exiting         
        print('LoopDetectingProcess: loop exit...')         

    def loop_detecting(self, loop_detector, q_in, q_out, q_out_condition, is_running, time_loop_detection):
        #print('LoopDetectingProcess: loop_detecting')        
        timer = TimerFps("LoopDetectingProcess", is_verbose = kTimerVerbose)
        timer.start()        
        try: 
            if is_running.value == 1:
                q_in_size = q_in.qsize()
                if q_in_size >= 10: 
                    warn_msg = f'\n!LoopDetectingProcess: WARNING: q_in size: {q_in_size} is too big!!!\n'
                    print(warn_msg)
                    Printer.red(warn_msg)
                self.last_input_task = q_in.get() # blocking call to get a new input task
                if self.last_input_task is None: # got a None to exit
                    is_running.value = 0
                else:
                    last_output = None
                    try:
                        # check and compute if needed the local descriptors by using the independent local feature manager (if present). 
                        loop_detector.compute_local_des_if_needed(self.last_input_task)
                        # run the loop detection task
                        last_output = loop_detector.run_task(self.last_input_task)
                    except Exception as e:
                        print(f'LoopDetectingProcess: EXCEPTION: {e} !!!')
                        if kPrintTrackebackDetails:
                            traceback_details = traceback.format_exc()
                            print(f'\t traceback details: {traceback_details}')  
                            
                    if is_running.value == 1 and last_output is not None:
                        with q_out_condition:
                            # push the computed output in the output queue
                            q_out.put(last_output)
                            q_out_condition.notify_all()
                            print(f'LoopDetectingProcess: pushed new output to queue_out size: {q_out.qsize()}')
                
        except Exception as e:
            print(f'LoopDetectingProcess: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')

        timer.refresh()
        time_loop_detection.value = timer.last_elapsed
        print(f'LoopDetectingProcess: q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, loop-detection-process elapsed time: {time_loop_detection.value}')


    def add_task(self, task: LoopDetectorTask): 
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(task)
                self.q_in_condition.notify_all()

    def pop_output(self): 
        if self.is_running.value == 0:
            return None
        with self.q_out_condition:        
            while self.q_out.empty() and self.is_running.value == 1:
                ok = self.q_out_condition.wait(timeout=Parameters.kLoopDetectingTimeoutPopKeyframe)
                if not ok: 
                    break # Timeout occurred
                
        if self.q_out.empty():
            return None
               
        try:
            return self.q_out.get(timeout=Parameters.kLoopDetectingTimeoutPopKeyframe)
        except Exception as e:
            print(f'LoopDetectingProcess: pop_output: encountered exception: {e}')
            return None