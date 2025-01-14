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
import torch.multiprocessing as mp 

import cv2 
import numpy as np

from camera import Camera
from map import Map

from dataset import DatasetEnvironmentType, SensorType

from utils_sys import Printer, set_rlimit, FileLogger, LoggerQueue
from utils_mp import MultiprocessingManager
from utils_data import empty_queue, push_to_front, static_fields_to_dict
from utils_depth import filter_shadow_points
from utils_mt import SimpleTaskTimer

from config_parameters import Parameters

import traceback

from collections import deque

from keyframe import KeyFrame
from frame import Frame

from enum import Enum

import torch
from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
        
import open3d as o3d


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True 

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kDataFolder = kRootFolder + '/data'
kLogsFolder = kRootFolder + '/logs'


kVolumetricIntegratorProcessName = 'VolumetricIntegratorProcess'
logging_manager, logger = None, None

if kVerbose:
    if Parameters.kUseVolumetricIntegration and Parameters.kVolumetricIntegrationDebugAndPrintToFile:
        # redirect the prints of volumetric integration to the file logs/volumetric_integration.log 
        # you can watch the output in separate shell by running:
        # $ tail -f logs/volumetric_integration.log 
                
        logging_file=kLogsFolder + '/volumetric_integrator.log'
        if logging_manager is None:
            logging_manager = LoggerQueue.get_instance(logging_file)
            logger = logging_manager.get_logger("volumetric_integrator_main")
                        
        # if logger is None:
        #     logger = FileLogger(logging_file)
            
        def print(*args, **kwargs):
            if logger is not None:
                message = ' '.join(str(arg) for arg in args)  # Convert all arguments to strings and join with spaces                
                return logger.info(message, **kwargs)
else:
    def print(*args, **kwargs):
        return
    

class VolumetricIntegrationTaskType(Enum):
    NONE            = 0
    INTEGRATE       = 1
    SAVE            = 2
    LOAD            = 3
    RESET           = 4
    UPDATE_OUTPUT   = 5   
    

# keyframe (pickable) data that are needed for volumetric integration
class VolumetricIntegrationKeyframeData:
    def __init__(self, keyframe: KeyFrame, img=None, img_right=None, depth=None):
        # keyframe data
        self.id = keyframe.id if keyframe is not None else -1
        self.kid = keyframe.kid if keyframe is not None else -1
        self.img_id = keyframe.img_id if keyframe is not None else -1
        self.timestamp = keyframe.timestamp if keyframe is not None else -1
        self.pose = keyframe.pose if keyframe is not None else None   # Tcw
        self.camera = keyframe.camera if keyframe is not None else None 
        
        self.img = img if img is not None else (keyframe.img if keyframe is not None else None)
        self.img_right = img_right if img_right is not None else (keyframe.img_right if keyframe is not None else None)
        self.depth = depth if depth is not None else (keyframe.depth_img if keyframe is not None else None)      


class VolumetricIntegrationTask: 
    def __init__(self, keyframe: KeyFrame=None, img=None, img_right=None, depth=None, task_type=VolumetricIntegrationTaskType.NONE, load_save_path=None):
        self.task_type = task_type
        self.keyframe_data = VolumetricIntegrationKeyframeData(keyframe, img, img_right, depth)
        self.load_save_path = load_save_path   
        
        
# pickable point cloud obtained from o3d.geometry.PointCloud
class VolumetricIntegrationPointCloud:
    def __init__(self, point_cloud: o3d.geometry.PointCloud=None, points=None, colors=None):
        if point_cloud is not None:
            self.points = np.asarray(point_cloud.points)
            self.colors = np.asarray(point_cloud.colors)
        else:
            self.points = np.asarray(points) if points is not None else None
            self.colors = np.asarray(colors) if colors is not None else None
        
    def to_o3d(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        pc.colors = o3d.utility.Vector3dVector(self.colors)
        return pc
        
# pickable mesh obtained from o3d.geometry.TriangleMesh
class VolumetricIntegrationMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        
    def to_o3d(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)        
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)        
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        return mesh        

       
class VolumetricIntegrationOutput:
    def __init__(self, task_type, id=-1, point_cloud:VolumetricIntegrationPointCloud =None, mesh:VolumetricIntegrationMesh =None):
        self.task_type = task_type
        self.id = id
        self.point_cloud = point_cloud # type: VolumetricIntegrationPointCloud
        self.mesh = mesh               # type: VolumetricIntegrationMesh
        self.timestamp = time.perf_counter()
        

class VolumetricIntegratorBase:
    def __init__(self, camera, environment_type, sensor_type):        
        import torch.multiprocessing as mp
        # NOTE: The following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).        
        if mp.get_start_method() != 'spawn':
            mp.set_start_method('spawn', force=True) # NOTE: This may generate some pickling problems with multiprocessing 
                                                        #    in combination with torch and we need to check it in other places.
                                                        #    This set start method can be checked with MultiprocessingManager.is_start_method_spawn()
        
        set_rlimit()
         
        self.camera = camera
        self.environment_type = environment_type
        self.sensor_type = sensor_type
        
        
        self.keyframe_queue_timer = SimpleTaskTimer(interval=1, callback=self.flush_keyframe_queue, single_shot=False, name='KeyframeQueueTimer')
        self.keyframe_queue_lock = mp.Lock()
        self.keyframe_queue = deque() # We use a deque to accumulate keyframes for volumetric integration. 
                                      # We integrate only the keyframes that have been processed by LBA at least once. 
        
        self.volume = None
 
        self.time_volumetric_integration = mp.Value('d',0.0)       
        
        self.last_input_task = None
        self.last_output = None
        self.last_integrated_id = -1
        
        self.depth_estimator = None
        self.img_id_to_depth = None
        
        self.reset_mutex = mp.Lock()
        self.reset_requested = mp.Value('i',-1)

        self.load_request_completed = mp.Value('i',-1)
        self.load_request_condition = mp.Condition()
        self.save_request_completed = mp.Value('i',-1)
        self.save_request_condition = mp.Condition()

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.q_in = self.mp_manager.Queue()
        self.q_out = self.mp_manager.Queue()
        
        # NOTE: The Parameters static fields are not sinced in the parallel process launched below (which has its own memory space). 
        #       Here we explicitly copy the current state at _initialization time_ and pass it to the parallel process.
        #       In this way, the parallel process will be able to access the updated prameters (until the parallel process is launched).
        self.parameters_dict = self.mp_manager.Dict()
        self.parameters_dict = static_fields_to_dict(Parameters)
                                        
        self.q_in_condition = mp.Condition()
        self.q_out_condition = mp.Condition()    
        
        self.is_running  = mp.Value('i',0)
        
        self.start()
        
        
    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove from the state the things you don't want to pickle
        if 'keyframe_queue_timer' in state: 
            del state['keyframe_queue_timer']        
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the lock after unpickling
        #self.keyframe_queue_timer = SimpleTaskTimer(interval=1, callback=self.flush_keyframe_queue, single_shot=False)
                            
                            
    def start(self):
        self.is_running.value = 1
        self.process = mp.Process(target=self.run,
                          args=(self.camera, self.environment_type, self.sensor_type, \
                                self.q_in, self.q_in_condition, \
                                self.q_out, self.q_out_condition, \
                                self.is_running, self.reset_mutex, self.reset_requested,
                                self.load_request_completed, self.load_request_condition, \
                                self.save_request_completed, self.save_request_condition, \
                                self.time_volumetric_integration,
                                self.parameters_dict), \
                          name=kVolumetricIntegratorProcessName)
        
        #self.process.daemon = True
        self.process.start()
        self.keyframe_queue_timer.start()


    # This method must be implemented in the derived classes and is called in the quit() method
    def _stop_volume_integrator_implementation(self):
        pass


    def save(self, path):
        print('VolumetricIntegratorBase: saving...')
        try:
            if self.save_request_completed.value == 0:
                print('VolumericIntegratorBase: saving: already saving...')
                return
            filepath = path + '/dense_map.ply'        
            task_type = VolumetricIntegrationTaskType.SAVE
            task = VolumetricIntegrationTask(task_type=task_type, load_save_path=filepath)
            self.save_request_completed.value = 0
            self.add_task(task, front=True)
            with self.save_request_condition:
                while self.save_request_completed.value == 0:
                    self.save_request_condition.wait()
            print('VolumetricIntegratorBase: saving done')
        except Exception as e:
            print(f'VolumetricIntegratorBase: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')
        
    
    def load(self, path):
        if False:
            task_type = VolumetricIntegrationTaskType.LOAD
            task = VolumetricIntegrationTask(None, None, task_type, load_save_path=path)
            self.load_request_completed.value = 0
            self.add_task(task)
            with self.load_request_condition:
                while self.load_request_completed.value == 0:
                    self.load_request_condition.wait()
        #TODO: Clarify how to use this


    def request_reset(self):
        print('VolumetricIntegratorBase: Requesting reset...')
        with self.reset_mutex:
            if self.reset_requested.value == 1:
                print('VolumetricIntegratorBase: reset already requested...')
                return
            self.reset_requested.value = 1
        while True:
            with self.reset_mutex:
                with self.q_in_condition:
                    self.q_in_condition.notify_all() # to unblock q_in_condition.wait() in run() method               
                if self.reset_requested.value == 0:
                    break
            time.sleep(0.1)
        self.keyframe_queue.clear()
        print('VolumetricIntegratorBase: ...Reset done.')
        
        
    def reset_if_requested(self, reset_mutex, reset_requested, \
                            q_in, q_in_condition, \
                            q_out, q_out_condition):
        # acting within the launched process with the passed mp.Value() (received in input)      
        with reset_mutex:
            if reset_requested.value == 1:
                print('VolumetricIntegratorBase: reset_if_requested()...')                
                with q_in_condition:
                    empty_queue(q_in)
                    q_in_condition.notify_all()
                with q_out_condition:
                    empty_queue(q_out)
                    q_out_condition.notify_all()
                # Now reset the volume integrator in the launched parallel process
                try:
                    self.volume.reset()
                except Exception as e:
                    print(f'VolumetricIntegratorBase: reset_if_requested: Exception: {e}')
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        print(f'\t traceback details: {traceback_details}')
                reset_requested.value = 0
        
    
    def quit(self):
        try: 
            if self.is_running.value == 1:
                print('VolumetricIntegratorBase: quitting...')
                self.is_running.value = 0
                self.keyframe_queue_timer.stop()            
                with self.q_in_condition:
                    self.q_in.put(None)  # put a None in the queue to signal we have to exit
                    self.q_in_condition.notify_all()       
                with self.q_out_condition:
                    self.q_out_condition.notify_all()             
                self.process.join(timeout=5)
                if self.process.is_alive():
                    Printer.orange("Warning: Volumetric integration process did not terminate in time, forced kill.")  
                    self.process.terminate()    
                print('VolumetricIntegratorBase: done')    
        except Exception as e:
            print(f'VolumetricIntegratorBase: quit: Exception: {e}')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')
    
    
    def init(self, camera: Camera, environment_type: DatasetEnvironmentType, sensor_type: SensorType, parameters_dict):
        
        self.last_output = None
        self.depth_factor = 1.0 #/camera.depth_factor # Now, depth factor is already included in the set kf depth image
        self.environment_type = environment_type    
        self.sensor_type = sensor_type
    
        self.img_id_to_depth = {}
        
        self.parameters_dict = parameters_dict

        depth_estimator_type_str = parameters_dict['kVolumetricIntegrationDepthEstimatorType']
        depth_estimator_type = DepthEstimatorType.from_string(depth_estimator_type_str)
        use_depth_estimator = parameters_dict['kVolumetricIntegrationUseDepthEstimator']        
        if use_depth_estimator:
            min_depth = 0
            max_depth = 50 if environment_type == DatasetEnvironmentType.OUTDOOR else 10 
            precision = torch.float16  # for depth_pro
            if sensor_type == SensorType.MONOCULAR:
                Printer.red("*************************************************************************************")
                Printer.red('VolumetricIntegratorBase: ERROR: You cannot use a MONOCULAR depth estimator here when')
                Printer.red('you are using a MONOCULAR SLAM system: The scale of the metric depth estimator will')
                Printer.red('conflict with the independent scale of the SLAM system!')
                Printer.red("*************************************************************************************")
            print(f'VolumetricIntegratorBase: init: depth_estimator_type={depth_estimator_type}, min_depth={min_depth}, max_depth={max_depth}, precision={precision}')
            self.depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, 
                                                    min_depth=min_depth, max_depth=max_depth,
                                                    dataset_env_type=environment_type, precision=precision,
                                                    camera=camera)
        else: 
            print(f'VolumetricIntegratorBase: init: depth_estimator=None, depth_estimator_type={depth_estimator_type}')
        
        # Prepare maps to undistort color and depth images
        h, w = camera.height, camera.width
        D = camera.D
        K = camera.K
        #Printer.green(f'VolumetricIntegratorBase: init: h={h}, w={w}, D={D}, K={K}')
        if np.linalg.norm(np.array(D, dtype=float)) <= 1e-10:
            self.new_K = K
            self.calib_map1 = None
            self.calib_map2 = None
        else:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            self.calib_map1, self.calib_map2 = cv2.initUndistortRectifyMap(K, D, None, self.new_K, (w, h), cv2.CV_32FC1)
    
    
    # main loop of the volume integration process
    def run(self, camera, environment_type, sensor_type, \
            q_in, q_in_condition, \
            q_out, q_out_condition, \
            is_running, reset_mutex, reset_requested, \
            load_request_completed, load_request_condition, \
            save_request_completed, save_request_condition, \
            time_volumetric_integration, \
            parameters_dict):
        
        print('VolumetricIntegratorBase: starting...')
        self.init(camera, environment_type, sensor_type, parameters_dict)
        
        # main loop
        while is_running.value == 1:
            try:
                
                with q_in_condition:
                    while q_in.empty() and is_running.value == 1 and reset_requested.value == 0:
                        print('VolumetricIntegratorBase: waiting for new task...')
                        q_in_condition.wait()
                if not q_in.empty():
                    # check q_in size and dump a warn message if it is too big
                    q_in_size = q_in.qsize()
                    if q_in_size >= 10: 
                        warn_msg = f'!VolumetricIntegratorBase: WARNING: q_in size: {q_in_size}'
                        print(warn_msg)
                        Printer.orange(warn_msg)
        
                    self.volume_integration(q_in, q_out, q_out_condition, is_running, \
                                        load_request_completed, load_request_condition, \
                                        save_request_completed, save_request_condition, \
                                        time_volumetric_integration)
                                                        
                else: 
                    print('VolumetricIntegratorBase: q_in is empty...')
                    time.sleep(0.1) # sleep for a bit before checking the queue again
                self.reset_if_requested(reset_mutex, reset_requested, q_in, q_in_condition, q_out, q_out_condition)
            
            except Exception as e:
                print('VolumetricIntegratorBase: Exception: ', e)
                traceback.print_exc()
                
        # Stop the volume integrator: This is expected to be an implementation-specific operation
        self._stop_volume_integrator_implementation()       
            
        empty_queue(q_in) # empty the queue before exiting
        empty_queue(q_out) # empty the queue before exiting   
        print('VolumetricIntegratorBase: loop exit...')         


    def estimate_depth_in_needed_and_rectify(self, keyframe_data: VolumetricIntegrationKeyframeData):
        color = keyframe_data.img
        color_right = keyframe_data.img_right
        depth = keyframe_data.depth    
    
        if depth is None: 
            if self.depth_estimator is None:
                Printer.yellow('VolumetricIntegratorBase: depth_estimator is None, depth is None, skipping the keyframe...')
                return None, None # skip this keyframe
            else:
                inference_start_time = time.time()
                if keyframe_data.id in self.img_id_to_depth:
                    depth = self.img_id_to_depth[keyframe_data.id]
                else:
                    if self.sensor_type == SensorType.MONOCULAR:
                        Printer.error('VolumetricIntegratorBase: You cannot use a MONOCULAR depth estimator with a MONOCULAR SLAM system!')                                    
                    depth = self.depth_estimator.infer(color, color_right)
                    print(f'VolumetricIntegratorBase: depth inference time: {time.time() - inference_start_time}')
                    if self.parameters_dict['kVolumetricIntegrationDepthEstimationFilterShadowPoints']:
                        depth = filter_shadow_points(depth, delta_depth=None)
                    
        if not depth.dtype in [np.uint8, np.uint16, np.float32]:
            depth = depth.astype(np.float32)
        
        if self.calib_map1 is not None and self.calib_map2 is not None:
            color_undistorted = cv2.remap(color, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_LINEAR)
            depth_undistorted = cv2.remap(depth, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_NEAREST)
        else: 
            color_undistorted = color
            depth_undistorted = depth
            
        if self.depth_estimator is not None:
            if not keyframe_data.id in self.img_id_to_depth:
                self.img_id_to_depth[keyframe_data.id] = depth_undistorted
                                
        color_undistorted = cv2.cvtColor(color_undistorted, cv2.COLOR_BGR2RGB)
        return color_undistorted, depth_undistorted


    def volume_integration(self, q_in, q_out, q_out_condition, is_running, \
                             load_request_completed, load_request_condition, \
                             save_request_completed, save_request_condition, \
                             time_volumetric_integration):
        # to be overridden in the derived classes
        pass
        
    
    # called by add_keyframe() and periodically by the keyframe_queue_timer
    def flush_keyframe_queue(self):
        # iterate over the keyframe queue and flush the keyframes into the task queue
        with self.keyframe_queue_lock:
            i = 0  
            while i < len(self.keyframe_queue):
                kf_to_process = self.keyframe_queue[i]
                # We integrate only the keyframes that have been processed by LBA at least once.
                if kf_to_process.lba_count >= Parameters.kVolumetricIntegrationMinNumLBATimes:
                    print(f'VolumetricIntegratorBase: Adding integration task with keyframe id: {kf_to_process.id} (kid: {kf_to_process.kid})')
                    task_type = VolumetricIntegrationTaskType.INTEGRATE
                    task = VolumetricIntegrationTask(kf_to_process, task_type=task_type)      
                    self.add_task(task)
                    del self.keyframe_queue[i]  # Safely remove the item     
                else:
                    i += 1  # Only move forward if no removal to avoid skipping   

    def add_keyframe(self, keyframe: KeyFrame, img, img_right, depth, print=print):
        use_depth_estimator = Parameters.kVolumetricIntegrationUseDepthEstimator          
        if depth is None and not use_depth_estimator:
            print(f'VolumetricIntegratorBase: add_keyframe: depth is None -> skipping frame {keyframe.id}')
            return
        try:
            # We accumulate the keyframe in a queue. 
            # We integrate only the keyframes that have been processed by LBA at least once.
            self.keyframe_queue.append(keyframe)
            self.flush_keyframe_queue()    
            
        except Exception as e:
            print(f'VolumetricIntegratorBase: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')
                        
    def add_task(self, task: VolumetricIntegrationTask, front=True):
        if self.is_running.value == 1:
            with self.q_in_condition:
                if front:
                    push_to_front(self.q_in, task)
                else:
                    self.q_in.put(task)
                self.q_in_condition.notify_all()
                
    def add_update_output_task(self):
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(VolumetricIntegrationTask(task_type=VolumetricIntegrationTaskType.UPDATE_OUTPUT))
                self.q_in_condition.notify_all()
                
    def rebuild(self, map: Map):
        print('')
        print(f'VolumetricIntegratorBase: rebuild() rebuilding volumetric mapping...')
        if self.is_running.value == 1:
            task = VolumetricIntegrationTask(task_type=VolumetricIntegrationTaskType.RESET)      
            self.add_task(task) 
            
            self.keyframe_queue.clear()
            for kf in map.keyframes:
                if not kf.is_bad and kf.lba_count >= Parameters.kVolumetricIntegrationMinNumLBATimes:
                    if kf.depth_img is None:
                        print(f'VolumetricIntegratorBase: rebuild: depth is None -> skipping frame {kf.id}')
                        continue
                    self.keyframe_queue.append(kf)
            self.flush_keyframe_queue()
        
    def pop_output(self, timeout=Parameters.kLoopDetectingTimeoutPopKeyframe) -> VolumetricIntegrationOutput: 
        q_out = self.q_out
        q_out_condition = self.q_out_condition
             
        if self.is_running.value == 0:
            return None
        with q_out_condition:        
            while q_out.empty() and self.is_running.value == 1:
                ok = q_out_condition.wait(timeout=timeout)
                if not ok: 
                    print('VolumetricIntegratorBase: pop_output: timeout')
                    break # Timeout occurred
        if q_out.empty():
            return None
        try:
            return q_out.get(timeout=timeout)
        except Exception as e:
            print(f'VolumetricIntegratorBase: pop_output: encountered exception: {e}')
            return None