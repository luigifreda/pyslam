
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

import multiprocessing as mp
import threading 

import numpy as np
import cv2
from enum import Enum
import traceback

from pyslam.config_parameters import Parameters

from pyslam.slam.map import Map
from pyslam.utilities.timer import TimerFps

import g2o
from pyslam.slam import optimizer_gtsam
from pyslam.slam import optimizer_g2o

from pyslam.slam.keyframe_data import KeyFrameData
from pyslam.utilities.utils_sys import Printer, Logging
from pyslam.utilities.utils_mp import MultiprocessingManager
from pyslam.utilities.utils_data import empty_queue, Value

import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyslam.slam.slam import Slam # Only imported when type checking, not at runtime


kVerbose = True
kTimerVerbose = False
kPrintTrackebackDetails = True


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'

    
class GlobalBundleAdjustment:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
        
    def __init__(self, slam: 'Slam', use_multiprocessing=True):
        self.init_print()        
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: starting with use_multiprocessing: {use_multiprocessing}')
        #self.slam = slam
        self.map = slam.map    # type: Map
        self.local_mapping = slam.local_mapping
        
        self.use_multiprocessing = use_multiprocessing                
                        
        # parameters for GBA 
        self.loop_kf_id = -1
        self.rounds = 10
        self.use_robust_kernel = Parameters.kGBAUseRobustKernel
                        
        self.process = None
                
        if use_multiprocessing:
            self.opt_abort_flag = None
            self.mp_opt_abort_flag = mp.Value('i',False)
            self.time_GBA = mp.Value('d',-1)
            self.mean_squared_error = mp.Value('d',-1)
            self._is_running  = mp.Value('i',0)       # True if the child process is running                     
            # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
            self.mp_manager = MultiprocessingManager()
            self.q_message = self.mp_manager.Queue()
            self.result_dict_queue = self.mp_manager.Queue() # we share the dictionary result as an entry of this dedicated multiprocessing queue        
        else:
            self.opt_abort_flag = g2o.Flag(False)
            self.mp_opt_abort_flag = None
            self.time_GBA = Value('d',-1)
            self.mean_squared_error = Value('d',-1)
            self._is_running = Value('i',0)
            self.q_message = []
            self.result_dict_queue = []
    
    def init_print(self):
        if kVerbose:
            if Parameters.kGBADebugAndPrintToFile:
                # redirect the prints of GBA to the file logs/gba.log
                # you can watch the output in separate shell by running:
                # $ tail -f logs/gba.log 
                
                logging_file = Parameters.kLogsFolder + '/gba.log'
                GlobalBundleAdjustment.local_logger = Logging.setup_file_logger('gba_logger', logging_file, formatter=Logging.simple_log_formatter)
                def print_file(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)  # Convert all arguments to strings and join with spaces                
                    return GlobalBundleAdjustment.local_logger.info(message, **kwargs)  
            else: 
                def print_file(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)  # Convert all arguments to strings and join with spaces                
                    return print(message, **kwargs)
            GlobalBundleAdjustment.print = staticmethod(print_file)
    
    def start(self, loop_kf_id):
        if self.is_running():
            Printer.red('GlobalBundleAdjustment: GBA is already running! You can only have one GBA running at a time.')
            return
        
        if self.use_multiprocessing:
            empty_queue(self.q_message) # clear the output queue, prevent messages from previous runs from interfering with the current run
            empty_queue(self.result_dict_queue)
        else:
            self.q_message.clear()
            self.result_dict_queue.clear()
        
        self._is_running.value = 0  # reset it to zero, then it is set to 1 in run()
        self.time_GBA.value = -1
        self.mean_squared_error.value = -1
        self.loop_kf_id = loop_kf_id
        
        if self.use_multiprocessing:
            self.mp_opt_abort_flag.value = False # reset it to False
        else: 
            self.opt_abort_flag.value = False
        
        keyframes = self.map.get_keyframes()
        points = self.map.get_points()
    
        args=(keyframes, points, self.loop_kf_id, self.rounds, self.use_robust_kernel,
              self.q_message, self.result_dict_queue, self._is_running, self.time_GBA,
              self.mean_squared_error, self.opt_abort_flag, self.mp_opt_abort_flag)
                                              
        if self.use_multiprocessing:     
            # launch child process
            GlobalBundleAdjustment.print('GlobalBundleAdjustment: starting child process...')
            self.process = mp.Process(target=self.run, args=args)
            self.process.daemon = True
        else: 
            # launch thread
            GlobalBundleAdjustment.print('GlobalBundleAdjustment: starting thread...')
            self.process = threading.Thread(target=self.run, args=args)
          
        self.process.start()
                    
    def is_running(self):
        return (self._is_running.value == 1)
    
    def abort(self):
        GlobalBundleAdjustment.print('GlobalBundleAdjustment: interrupting GBA...')
        if self.use_multiprocessing:               
            self.mp_opt_abort_flag.value = True   
        else:
            self.opt_abort_flag.value = True
    
    def quit(self):
        if self.is_running():
            GlobalBundleAdjustment.print('GlobalBundleAdjustment: quitting...')
            self.abort()
            if self.process.is_alive():                               
                self.process.join(timeout=1)
            if self.process.is_alive():
                message = 'GlobalBundleAdjustment: WARNING: GBA process did not terminate in time, forced kill.'
                GlobalBundleAdjustment.print(message)
                Printer.orange(message)  
                self.process.terminate()
            if self.use_multiprocessing:
                empty_queue(self.q_message)
            else: 
                self.q_message.clear()
            self._is_running.value = 0      
            GlobalBundleAdjustment.print('GlobalBundleAdjustment: done')  
                   
    def check_GBA_has_finished_and_correct_if_needed(self):            
        if not self.is_running() and (self.q_message.qsize() if self.use_multiprocessing else len(self.q_message)) > 0:
            output = self.q_message.get() if self.use_multiprocessing else self.q_message.pop(0)
            try: 
                return self.correct_after_GBA()
            except Exception as e:
                GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: check_GBA_has_finished_and_correct_if_needed: encountered exception: {e}')
                if kPrintTrackebackDetails:
                    traceback_details = traceback.format_exc()
                    GlobalBundleAdjustment.print(f'\t traceback details: {traceback_details}')
        return False                       
            
    def correct_after_GBA(self):
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: correct after GBA...')
        
        # Send a stop signal to Local Mapping
        # Avoid new keyframes are inserted while correcting the loop
        self.local_mapping.request_stop()
        # wait till local mapping is idle
        self.local_mapping.wait_idle(timeout=1.0, print=print)  
        
        GlobalBundleAdjustment.print('GlobalBundleAdjustment: starting correction ...')
        # get the updates from GBA results and put them in their temporary fields in the map
        loop_kf_id = self.loop_kf_id
        is_result_dict_queue_empty = self.result_dict_queue.empty() if self.use_multiprocessing else len(self.result_dict_queue) == 0
        if is_result_dict_queue_empty:
            Printer.red('GlobalBundleAdjustment: result_dict_queue is empty!')
            raise Exception('GlobalBundleAdjustment: result_dict_queue is empty!')
        result_dict = self.result_dict_queue.get() if self.use_multiprocessing else self.result_dict_queue.pop(0)
        keyframe_updates = result_dict['keyframe_updates'] 
        point_updates = result_dict['point_updates']
        
        keyframes = self.map.get_keyframes()
        points = self.map.get_points()
        
        num_kf_updates = 0
        num_pt_updates = 0
        
        num_kf_without_updates = 0
        num_pt_without_updates = 0
        
        # put frames back
        for kf in keyframes:
            try: 
                T = keyframe_updates[kf.id]
                kf.Tcw_GBA = T
                kf.GBA_kf_id = loop_kf_id
                num_kf_updates += 1
            except:
                #print(f'GlobalBundleAdjustment: keyframe {kf.id} not in keyframe_updates')
                num_kf_without_updates += 1

        # put points back
        for p in points:
            try:
                p.pt_GBA = point_updates[p.id]
                p.GBA_kf_id = loop_kf_id
                num_pt_updates += 1
            except:
                #print(f'GlobalBundleAdjustment: point {p.id} not in point_updates')
                num_pt_without_updates += 1
                        
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: got {num_kf_updates} keyframe updates and {num_pt_updates} point updates after GBA.')
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: {num_kf_without_updates} keyframes without updates and {num_pt_without_updates} points without updates.')
        
        # Update all MapPoints and KeyFrames.
        # Local Mapping was active during BA, that means that there might be new keyframes
        # not included in the Global BA and they are not consistent with the updated map.
        # We need to propagate the correction through the spanning tree.
        try: 
            # Get Map Mutex
            with self.map.update_lock:
                
                # Correct keyframes starting at map first keyframe
                keyframes_to_check = list(self.map.keyframe_origins)
                while keyframes_to_check:
                    keyframe = keyframes_to_check.pop(0)
                    child_keyframes = keyframe.get_children()
                    Twc = keyframe.Twc

                    if keyframe.Tcw_GBA is None:
                        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: WARNING: keyframe {keyframe.id} (is_bad: {keyframe.is_bad}) with empty Tcw_GBA!')              
                        
                    # propagate the correction to children
                    for child in child_keyframes:
                        if child.GBA_kf_id != self.loop_kf_id:
                            if keyframe.Tcw_GBA is not None:                            
                                T_child_c = child.Tcw @ Twc
                                child.Tcw_GBA = T_child_c @ keyframe.Tcw_GBA
                                child.GBA_kf_id = self.loop_kf_id
                        keyframes_to_check.append(child)

                    keyframe.Tcw_before_GBA = keyframe.Tcw
                    if keyframe.Tcw_GBA is not None:
                        keyframe.update_pose(keyframe.Tcw_GBA)

                # Correct MapPoints
                for map_point in self.map.get_points():
                    if map_point.is_bad:
                        continue

                    if map_point.GBA_kf_id == self.loop_kf_id:
                        # If optimized by Global BA, just update
                        map_point.update_position(map_point.pt_GBA)
                    else:
                        # Update according to the correction of its reference keyframe
                        ref_keyframe = map_point.get_reference_keyframe()
                        if ref_keyframe is None:
                            GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: WARNING: MapPoint {map_point.id} has no reference keyframe!')
                            continue

                        if ref_keyframe.GBA_kf_id != self.loop_kf_id:
                            continue

                        # Map to non-corrected camera
                        Rcw = ref_keyframe.Tcw_before_GBA[0:3, 0:3]
                        tcw = ref_keyframe.Tcw_before_GBA[0:3, 3]
                        Xc = Rcw @ map_point.pt + tcw

                        # Backproject using corrected camera
                        Twc = ref_keyframe.Twc
                        Rwc = Twc[0:3, 0:3]
                        twc = Twc[0:3, 3]

                        map_point.update_position(Rwc @ Xc + twc)
                    map_point.update_normal_and_depth()

                self.local_mapping.release()

                GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: map updated!')
                return True
            
        except Exception as e:
            GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f'\t traceback details: {traceback_details}')
                
        return False
          
                    
    def run(self, keyframes, points, loop_kf_id, rounds, use_robust_kernel, 
            q_message, result_dict_queue, is_running, time_GBA, mean_squared_error, 
            opt_abort_flag, mp_opt_abort_flag):
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: starting global bundle adjustment with loop_kf_id {loop_kf_id}...')        
        is_running.value = 1   
             
        timer = TimerFps("GlobalBundleAdjustment", is_verbose = kTimerVerbose)
        timer.start()
        
        task_completed = False
        
        result_dict = {}
        
        try:
            if Parameters.kOptimizationBundleAdjustUseGtsam:
                global_bundle_adjustment_fun = optimizer_gtsam.global_bundle_adjustment
            else: 
                global_bundle_adjustment_fun = optimizer_g2o.global_bundle_adjustment
            mean_squared_error.value, result_dict = \
                global_bundle_adjustment_fun(keyframes=keyframes, points=points, rounds=rounds, \
                                             loop_kf_id=loop_kf_id, use_robust_kernel=use_robust_kernel,\
                                             abort_flag=opt_abort_flag, mp_abort_flag=mp_opt_abort_flag,
                                             result_dict=result_dict, verbose=False, print=print)
            task_completed = True
        except Exception as e:           
            GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                GlobalBundleAdjustment.print(f'\t traceback details: {traceback_details}') 
            mean_squared_error.value = -1
            time_GBA.value = -1
            
        if self.use_multiprocessing:
            result_dict_queue.put(result_dict)
        else:
            result_dict_queue.append(result_dict)
            
        if task_completed:
            timer.refresh()
            time_GBA.value = timer.last_elapsed            
            # push a simple task-completed signal  
            if self.use_multiprocessing:    
                q_message.put("Finished")
            else:
                q_message.append("Finished")
        
        is_running.value = 0
        GlobalBundleAdjustment.print(f'GlobalBundleAdjustment: task completed {task_completed}, mean_squared_error: {mean_squared_error.value}, elapsed time: {time_GBA.value}')