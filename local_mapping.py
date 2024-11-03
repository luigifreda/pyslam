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

from __future__ import print_function # This must be the first statement before other statements 

import sys
import time
import numpy as np
import cv2
import g2o
from enum import Enum

from collections import defaultdict 

from threading import RLock, Lock, Thread, Condition, current_thread
from queue import Queue 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from parameters import Parameters  

from feature_manager import FeatureManager
from feature_tracker import FeatureTracker

from dataset import SensorType
from keyframe import KeyFrame
from frame import Frame, compute_frame_matches
from search_points import search_frame_for_triangulation, search_and_fuse
from map_point import MapPoint
from map import Map
from loop_closing import LoopClosing

from timer import Timer, TimerFps
import optimizer_g2o

from utils_sys import Printer
from utils_geom import triangulate_normalized_points
from utils_data import AtomicCounter, empty_queue

import multiprocessing


kVerbose=True     
kTimerVerbose = False 

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kLocalMappingDebugAndPrintToFile = Parameters.kLocalMappingDebugAndPrintToFile

kUseLargeWindowBA = Parameters.kUseLargeWindowBA
        
kNumMinObsForKeyFrameDefault = 3

kLocalMappingSleepTime = 5e-3  # [s]


if kVerbose:
    if kLocalMappingOnSeparateThread: 
        if kLocalMappingDebugAndPrintToFile:
            # redirect the prints of local mapping to the file local_mapping.log
            # you can watch the output in separate shell by is_running:
            # $ tail -f local_mapping.log 
            import builtins as __builtin__
            logging_file=open('local_mapping.log','w')
            def print(*args, **kwargs):
                return __builtin__.print(*args,**kwargs,file=logging_file,flush=True)
else:
    def print(*args, **kwargs):
        return
    

def kf_search_frame_for_triangulation(kf1, kf2, kf2_idx, idxs1, idxs2, max_descriptor_distance, is_monocular, result_queue):
    idxs1_out, idxs2_out, num_found_matches, _ = search_frame_for_triangulation(kf1, kf2, idxs1, idxs2, max_descriptor_distance, is_monocular)
    print(f'\t kf_search_frame_for_triangulation: found for ({kf1.id}, {kf2.id}), #potential matches: {num_found_matches}')    
    result_queue.put((idxs1_out, idxs2_out, num_found_matches, _, kf2_idx)) 
                
    
class LocalMapping:
    def __init__(self, slam):
        self.feature_tracker = slam.feature_tracker # type: FeatureTracker
        self.map = slam.map  # type: Map
        self.sensor_type = slam.sensor_type
        self.recently_added_points = set()
        
        if hasattr(slam, 'loop_closing'):
            self.loop_closing = slam.loop_closing # type: LoopClosing
        
        self.mean_ba_chi2_error = None
        self.time_local_mapping = None
        
        self.kf_cur = None   # current processed keyframe  
        self.kid_last_BA = -1 # last keyframe id when performed BA  
        
        self.descriptor_distance_sigma = self.feature_tracker.feature_manager.max_descriptor_distance            

        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings  
        self.timer_triangulation = TimerFps('Triangulation', is_verbose = self.timer_verbose)    
        self.timer_pts_culling = TimerFps('Culling points', is_verbose = self.timer_verbose)     
        self.timer_kf_culling = TimerFps('Culling kfs', is_verbose = self.timer_verbose)                  
        self.timer_pts_fusion = TimerFps('Fusing points', is_verbose = self.timer_verbose)  
        self.timer_large_BA = TimerFps('Large window BA', is_verbose = self.timer_verbose)        
        self.time_local_opt = TimerFps('Local optimization', is_verbose = self.timer_verbose)        
        self.time_large_opt = TimerFps('Large window optimization', is_verbose = self.timer_verbose)    
        
        self.queue = Queue()
        self.queue_condition = Condition()
        self.work_thread = Thread(target=self.run)
        self.is_running = False
        
        self._is_idle = True 
        self.idle_codition = Condition()
        
        # use self.set_opt_abort_flag() to manage the following two guys
        self.opt_abort_flag = g2o.Flag(False)                      # for multi-threading
        self.mp_opt_abort_flag = multiprocessing.Value('i',False)  # for multi-processing (when used)
                                              
        self.stop_requested = False
        self.stopped = False
        self.stop_mutex = RLock()
         
        self.log_file = None 
        self.thread_large_BA = None 
        
        self.last_processed_kf_img_id = None        
        self.last_num_triangulated_points = None
        self.last_num_fused_points = None
        self.last_num_culled_points = None
        self.last_num_culled_keyframes = None
        
        
    def set_loop_closing(self, loop_closing):
        self.loop_closing = loop_closing
        
    def start(self):    
        self.work_thread.start()

    def quit(self):
        print('LocalMapping: quitting...')
        if self.is_running:
            self.is_running = False
            self.set_opt_abort_flag(True)      
            self.work_thread.join(timeout=5)
        print('LocalMapping: done')   
        
    def set_opt_abort_flag(self, value):
        if self.opt_abort_flag.value != value:
            self.opt_abort_flag.value = value     # for multi-threading
            self.mp_opt_abort_flag.value = value  # for multi-processing  (when used)      
        
    # push the new keyframe and its image into the queue
    def push_keyframe(self, keyframe, img=None):
        with self.queue_condition:
            self.queue.put((keyframe,img))      
            self.queue_condition.notifyAll() 
            self.set_opt_abort_flag(True)              
        
    # blocking call
    def pop_keyframe(self):
        with self.queue_condition:        
            if self.queue.empty():
                while self.queue.empty() and not self.stop_requested:
                    ok = self.queue_condition.wait(timeout=Parameters.kLocalMappingTimeoutPopKeyframe)
                    if not ok: 
                        break # Timeout occurred
        if self.queue.empty() or self.stop_requested:
            return None
        try:
            return self.queue.get(timeout=Parameters.kLocalMappingTimeoutPopKeyframe)
        except Exception as e:
            print(f'LocalMapping: pop_keyframe: encountered exception: {e}')
            return None
        
    def queue_size(self):
        return self.queue.qsize()              
                
    def is_idle(self):
        with self.idle_codition: 
            return self._is_idle
                    
    def set_idle(self, flag):
        with self.idle_codition: 
            self._is_idle = flag
            self.idle_codition.notifyAll() 
            
    def wait_idle(self, print=print, timeout=None): 
        if self.is_running == False:
            return
        with self.idle_codition:
            while not self._is_idle and self.is_running:
                print('LocalMapping: waiting for idle...')
                ok = self.idle_codition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(f'LocalMapping: timeout {timeout}s reached, quit waiting for idle')
                    return
        
    def interrupt_optimization(self):
        Printer.yellow('interrupting local mapping optimization')        
        self.set_opt_abort_flag(True)
          
    def request_stop(self):
        with self.stop_mutex:
            Printer.yellow('requesting a stop for local mapping optimization') 
            self.stop_requested = True
        with self.queue_condition:
            self.set_opt_abort_flag(True) 
            self.queue_condition.notifyAll() # to unblock self.pop_keyframe()  
        
    def is_stop_requested(self):
        with self.stop_mutex:         
            return self.stop_requested    
    
    def stop(self):
        with self.stop_mutex:        
            if self.stop_requested and not self.stopped:
                self.stopped = True
                print('LocalMapping: stopped...')
                return True
            return False
        
    def is_stopped(self):
        with self.stop_mutex:         
            return self.stopped
    
    def set_not_stop(self, value):
        with self.stop_mutex:              
            if value and self.stopped:
                return False 
            self.stopped = value
            return True
    
    def release(self):   
        if not self.is_running:
            return 
        with self.stop_mutex:              
            self.stopped = False
            self.stop_requested = False
            # emtpy the queue
            while not self.queue.empty():
                self.queue.get()
            print(f'LocalMapping: released...')
        
    def run(self):
        self.is_running = True
        while self.is_running:
            self.step()
        empty_queue(self.queue) # empty the queue before exiting
        print('LocalMapping: loop exit...')
        
    def step(self):
        if not self.map.local_map.is_empty():
            if not self.stop_requested:              
                ret = self.pop_keyframe() # blocking call
                if ret is not None: 
                    self.kf_cur, self.img_cur = ret
                    if self.kf_cur is not None:
                        self.last_processed_kf_img_id = self.kf_cur.img_id
                                                                                    
                        self.set_idle(False) 
                        self.do_local_mapping()    
                        self.set_idle(True)
            elif self.stop():
                self.set_idle(True)
                while self.is_stopped():
                    print(f'LocalMapping: stopped, idle: {self._is_idle} ...')
                    time.sleep(kLocalMappingSleepTime) 
                
        else: 
            #Printer.red('[local mapping] local map is empty')
            time.sleep(kLocalMappingSleepTime)
        
    def do_local_mapping(self):
        print('local mapping: starting...')                        
                          
        Printer.cyan('@local mapping')
        time_start = time.time()
                                        
        if self.kf_cur is None:
            Printer.red('local mapping: no keyframe to process')
            return
                        
        if kLocalMappingOnSeparateThread: 
            print('..................................')
            print('processing KF: ', self.kf_cur.id, ', queue size: ', self.queue_size())   
        
        #print('descriptor_distance_sigma: ', self.descriptor_distance_sigma)
                        
        self.process_new_keyframe()          
                
        # do map points culling 
        self.timer_pts_culling.start()        
        num_culled_points = self.cull_map_points()
        self.last_num_culled_points = num_culled_points
        self.timer_pts_culling.refresh()    
        print(f'#culled points: {num_culled_points}, timing: {self.timer_pts_culling.last_elapsed}')                 
                
        # create new points by triangulation 
        self.timer_triangulation.start()
        total_new_pts = self.create_new_map_points()
        self.last_num_triangulated_points = total_new_pts
        self.timer_triangulation.refresh()
        print(f'#new map points: {total_new_pts}, timing: {self.timer_triangulation.last_elapsed}')   
        
        if self.queue.empty():
            # fuse map points of close keyframes
            self.timer_pts_fusion.start()
            total_fused_pts = self.fuse_map_points()
            self.last_num_fused_points = total_fused_pts
            self.timer_pts_fusion.refresh()
            print(f'#fused map points: {total_fused_pts}, timing: {self.timer_pts_fusion.last_elapsed}') 

        # reset optimization abort flag 
        self.set_opt_abort_flag(False)                
        
        if self.queue.empty() and not self.is_stop_requested():                
             
            if self.thread_large_BA is not None: 
                if self.thread_large_BA.is_alive(): # for security, check if large BA thread finished his work
                    self.thread_large_BA.join()   
                    self.timer_large_BA.refresh()
                    print(f'large BA timing: {self.timer_large_BA.last_elapsed}')
        
            # launch local bundle adjustment 
            self.local_BA()      

            if kUseLargeWindowBA and \
               self.kf_cur.kid >= Parameters.kEveryNumFramesLargeWindowBA and \
               self.kid_last_BA != self.kf_cur.kid and self.kf_cur.kid % Parameters.kEveryNumFramesLargeWindowBA == 0:
                self.timer_large_BA.start()
                # launch a parallel large window BA of the map
                self.thread_large_BA = Thread(target=self.large_window_BA)       
                self.thread_large_BA.start()  
                
            # check redundant local Keyframes
            self.timer_kf_culling.start()
            num_culled_keyframes = self.cull_keyframes()
            self.last_num_culled_keyframes = num_culled_keyframes
            self.timer_kf_culling.refresh() 
            print(f'\t #culled keyframes: {num_culled_keyframes}, timing: {self.timer_kf_culling.last_elapsed}')                       
            
        if self.loop_closing is not None and self.kf_cur is not None:
            print('pushing new keyframe to loop closing...')
            self.loop_closing.add_keyframe(self.kf_cur, self.img_cur)  
                                    
        elapsed_time = time.time() - time_start
        self.time_local_mapping = elapsed_time
        print(f'local mapping elapsed time: {elapsed_time}')
         
         
    def local_BA(self):
        # local optimization 
        self.time_local_opt.start()   
        print('>>>> local optimization (LBA) ...')
        err = self.map.locally_optimize(kf_ref=self.kf_cur, abort_flag=self.opt_abort_flag, mp_abort_flag=self.mp_opt_abort_flag)
        self.mean_ba_chi2_error = err
        self.time_local_opt.refresh()
        print(f'local optimization (LBA) error^2: {err}, timing: {self.time_local_opt.last_elapsed}')     
        num_kf_ref_tracked_points = self.kf_cur.num_tracked_points(kNumMinObsForKeyFrameDefault) # number of tracked points in k_ref
        Printer.green('KF(%d) #points: %d ' %(self.kf_cur.id, num_kf_ref_tracked_points))           
              
          
    def large_window_BA(self):
        Printer.blue('@large BA')
        # large window optimization of the map
        self.kid_last_BA = self.kf_cur.kid   
        self.time_large_opt.start() 
        err = self.map.optimize(local_window=Parameters.kLargeBAWindow, abort_flag=self.opt_abort_flag)  # verbose=True)
        self.time_large_opt.refresh()
        Printer.blue('large window optimization error^2: %f, KF id: %d' % (err,self.kf_cur.kid))                                                                 
        
        
    def process_new_keyframe(self):
        # associate map points to keyframe observations (only good points)
        # and update normal and descriptor
        kf_cur_points = self.kf_cur.get_points()
        print(f'>>>> updating map points ({len(kf_cur_points)})...')
        for idx,p in enumerate(kf_cur_points):
            #print(f'{idx}/{len(kf_cur_points)}', end='\r')
            # if p is not None and not p.is_bad:  
            #     if p.add_observation(self.kf_cur, idx):
            #         p.update_info() 
            #     else: 
            #         self.recently_added_points.add(p)    
            if p is not None:
                   added, is_bad = p.add_observation_if_not_bad(self.kf_cur, idx)
                   if added: 
                       p.update_info()
                   elif not is_bad:
                       self.recently_added_points.add(p)
        
        print('>>>> updating connections ...')     
        self.kf_cur.update_connections()
        #self.map.add_keyframe(self.kf_cur)   # add kf_cur to map        
                        
        
    def cull_map_points(self):
        print('>>>> culling map points...')  
        th_num_observations = 2      
        if self.sensor_type != SensorType.MONOCULAR:
            th_num_observations = 3
        min_found_ratio = 0.25  
        current_kid = self.kf_cur.kid 
        remove_set = set() 
        for p in self.recently_added_points:
            if p.is_bad:
                remove_set.add(p)
            elif p.get_found_ratio() < min_found_ratio:
                p.set_bad()
                self.map.remove_point(p)  
                remove_set.add(p)                  
            elif (current_kid - p.first_kid) >= 2 and p.num_observations <= th_num_observations:  
                p.set_bad()           
                self.map.remove_point(p)  
                remove_set.add(p)        
            elif (current_kid - p.first_kid) >= 3:  # after three keyframes we do not consider the point a recent one         
                remove_set.add(p)   
        self.recently_added_points = self.recently_added_points - remove_set  
        num_culled_points = len(remove_set)                                             
        return num_culled_points           
           
           
    def cull_keyframes(self): 
        print('>>>> culling keyframes...')    
        num_culled_keyframes = 0
        # check redundant keyframes in local keyframes: a keyframe is considered redundant if the 90% of the MapPoints it sees, 
        # are seen in at least other 3 keyframes (in the same or finer scale)
        th_num_observations = 3
        for kf in self.kf_cur.get_covisible_keyframes(): 
            if kf.kid==0:
                continue 
            kf_num_points = 0     # num good points for kf          
            kf_num_redundant_observations = 0   # num redundant observations for kf       
            for i,p in enumerate(kf.get_points()): 
                if p is not None and not p.is_bad:
                    if kf.depths is not None and (kf.depths[i] > kf.camera.depth_threshold or kf.depths[i] < 0.0):
                        continue
                    kf_num_points += 1
                    if p.num_observations>th_num_observations:
                        scale_level = kf.octaves[i]  # scale level of observation in kf 
                        p_num_observations = 0
                        for kf_j,idx in p.observations():
                            if kf_j is kf:
                                continue
                            assert(not kf_j.is_bad)
                            scale_level_i = kf_j.octaves[idx]  # scale level of observation in kfi
                            if scale_level_i <= scale_level+1:  # N.B.1 <- more aggressive culling  (expecially when scale_factor=2)
                            #if scale_level_i <= scale_level:     # N.B.2 <- only same scale or finer                            
                                p_num_observations +=1
                                if p_num_observations >= th_num_observations:
                                    break 
                        if p_num_observations >= th_num_observations:
                            kf_num_redundant_observations += 1
            if (kf_num_redundant_observations > Parameters.kKeyframeCullingRedundantObsRatio * kf_num_points) and \
               (kf_num_points > Parameters.kKeyframeCullingMinNumPoints) and \
               (kf.timestamp - kf.parent.timestamp < Parameters.kKeyframeMaxTimeDistanceInSecForCulling):
                print('setting keyframe ', kf.id,' bad - redundant observations: ', kf_num_redundant_observations/max(kf_num_points,1),'%')
                kf.set_bad()
                num_culled_keyframes += 1
        return num_culled_keyframes

            
    # triangulate matched keypoints (without a corresponding map point) amongst recent keyframes      
    def create_new_map_points(self):
        print('>>>> creating new map points')
        total_new_pts = 0
        
        num_neighbors = 10
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = 20
        
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        print('local map keyframes: ', [kf.id for kf in local_keyframes if not kf.is_bad], ' + ', self.kf_cur.id, '...')            
        
        match_idxs = defaultdict(lambda: (None,None))   # dictionary of matches  (kf_i, kf_j) -> (idxs_i,idxs_j)         
        # precompute keypoint matches 
        match_idxs = compute_frame_matches(self.kf_cur, local_keyframes, match_idxs, \
                                              do_parallel=Parameters.kLocalMappingParallelKpsMatching,\
                                              max_workers=Parameters.kLocalMappingParallelKpsMatchingNumWorkers,
                                              print_fun=print)
                    
        #print(f'\t processing local keyframes...')
        for i,kf in enumerate(local_keyframes):
            if kf is self.kf_cur or kf.is_bad:
                continue 
            if i>0 and not self.queue.empty():
                print('creating new map points *** interruption ***')
                return total_new_pts
            
            # extract matches from precomputed map  
            idxs_kf_cur, idxs_kf = match_idxs[(self.kf_cur,kf)]
            
            #print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id})...')
                                                                
            # find keypoint matches between self.kf_cur and kf
            # N.B.: all the matched keypoints computed by search_frame_for_triangulation() are without a corresponding map point              
            idxs_cur, idxs, num_found_matches, _ = search_frame_for_triangulation(self.kf_cur, kf, idxs_kf_cur, idxs_kf,
                                                                                   max_descriptor_distance=0.5*self.descriptor_distance_sigma,
                                                                                   is_monocular=(self.sensor_type == SensorType.MONOCULAR))
            
            #print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id}), #potential matches: {num_found_matches}')  
                                    
            if len(idxs_cur) > 0:
                # try to triangulate the matched keypoints that do not have a corresponding map point   
                pts3d, mask_pts3d = triangulate_normalized_points(self.kf_cur.pose, kf.pose, self.kf_cur.kpsn[idxs_cur], kf.kpsn[idxs])
                    
                new_pts_count,_,list_added_points = self.map.add_points(pts3d, mask_pts3d, self.kf_cur, kf, idxs_cur, idxs, self.kf_cur.img, do_check=True)
                print(f'\t #added map points: {new_pts_count} for KFs ({self.kf_cur.id}), ({kf.id})')        
                total_new_pts += new_pts_count 
                self.recently_added_points.update(list_added_points)       
        return total_new_pts                
        
                
    def create_new_map_points_parallel(self):
        print('>>>> creating new map points')
        total_new_pts = 0
        
        num_neighbors = 10
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = 20
        
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        print('local map keyframes: ', [kf.id for kf in local_keyframes if not kf.is_bad], ' + ', self.kf_cur.id, '...')            
        
        match_idxs = defaultdict(lambda: (None,None))   # dictionary of matches  (kf_i, kf_j) -> (idxs_i,idxs_j)         
        # precompute keypoint matches 
        match_idxs = self.precompute_kps_matches(match_idxs, local_keyframes)
                            
        tasks = []
        processes = []
        result_queue = multiprocessing.Queue()
        
        for kf_idx,kf in enumerate(local_keyframes):
            if kf is self.kf_cur or kf.is_bad:
                continue
            if kf_idx>0 and not self.queue.empty():
                print('creating new map points *** interruption ***')
                return total_new_pts

            idxs_kf_cur, idxs_kf = match_idxs[(self.kf_cur, kf)]
            kfs_data = (self.kf_cur, kf, kf_idx, idxs_kf_cur, idxs_kf, 0.5*self.descriptor_distance_sigma, self.sensor_type == SensorType.MONOCULAR,result_queue)
            process = multiprocessing.Process(target=kf_search_frame_for_triangulation, args=kfs_data) 
            processes.append(process)
            process.start()
        
        print(f'\t waiting for triangulation results...')
        for p in processes:
            p.join()           
                
        print(f'\t processing triangulation results...')
        while not result_queue.empty():
            result = result_queue.get()    
            idxs_cur, idxs, num_found_matches, _, kf_idx = result   
            kf = local_keyframes[kf_idx]         
            #print(f'\t adding map points for KFs ({self.kf_cur.id}, {kf.id}), #potential matches: {num_found_matches}')
            if len(idxs_cur) > 0:
                # try to triangulate the matched keypoints that do not have a corresponding map point   
                pts3d, mask_pts3d = triangulate_normalized_points(self.kf_cur.pose, kf.pose, self.kf_cur.kpsn[idxs_cur], kf.kpsn[idxs])
                    
                new_pts_count,_,list_added_points = self.map.add_points(pts3d, mask_pts3d, self.kf_cur, kf, idxs_cur, idxs, self.kf_cur.img, do_check=True)
                print(f'\t #added map points: {new_pts_count} for KFs ({self.kf_cur.id}), ({kf.id})')        
                total_new_pts += new_pts_count 
                self.recently_added_points.update(list_added_points)       
        return total_new_pts                
            
        
    # fuse close map points of local keyframes 
    def fuse_map_points(self):
        print('>>>> fusing map points')
        total_fused_pts = 0
        
        num_neighbors = 10
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = 20
        
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        print('local map keyframes: ', [kf.id for kf in local_keyframes if not kf.is_bad], ' + ', self.kf_cur.id, '...')   
                
        # search matches by projection from current KF in close KFs        
        for kf in local_keyframes:
            if kf is self.kf_cur or kf.is_bad:  
                continue      
            num_fused_pts = search_and_fuse(self.kf_cur.get_points(), kf,
                                            max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                                            max_descriptor_distance=0.5*self.descriptor_distance_sigma) # finer search
            print(f'\t #fused map points: {num_fused_pts} for KFs ({self.kf_cur.id}, {kf.id})')  
            total_fused_pts += num_fused_pts    
               
        # search matches by projection from local points in current KF  
        good_local_points = [p for kf in local_keyframes if not kf.is_bad for p in kf.get_points() if (p is not None and not p.is_bad) ]  # all good points in local frames 
        good_local_points = np.array(list(set(good_local_points))) # be sure to get only one instance per point                
        num_fused_pts = search_and_fuse(good_local_points, self.kf_cur,
                                        max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                                        max_descriptor_distance=0.5*self.descriptor_distance_sigma) # finer search
        print(f'\t #fused map points: {num_fused_pts} for local map into KF {self.kf_cur.id}')  
        total_fused_pts += num_fused_pts   
            
        # update points info 
        for p in self.kf_cur.get_points():
            if p is not None and not p.is_bad: 
                p.update_info() 
                
        # update connections in covisibility graph 
        self.kf_cur.update_connections()            
        
        return total_fused_pts               
