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
import torch.multiprocessing as mp 

import cv2 
import numpy as np

from camera import Camera
from map import Map

from dataset import DatasetEnvironmentType, SensorType

from utils_geom import inv_T, align_trajs_with_svd
from utils_sys import Printer, set_rlimit, LoggerQueue
from utils_mp import MultiprocessingManager
from utils_data import empty_queue
from utils_depth import filter_shadow_points

from timer import TimerFps

from parameters import Parameters

import traceback

from collections import deque

from keyframe import KeyFrame
from frame import Frame

from enum import Enum

import logging 

import open3d as o3d

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from slam import Slam, SlamState  # Only imported when type checking, not at runtime


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True 

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
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
        logging_manager = LoggerQueue(logging_file)
        logger = logging_manager.get_logger("volumetric_integrator_main")
        def print(*args, **kwargs):
            return logger.info(*args, **kwargs)
else:
    def print(*args, **kwargs):
        return
    

class VolumetricIntegratorTaskType(Enum):
    NONE = 0
    INTEGRATE = 1
    SAVE = 2
    LOAD = 3
    RESET = 4
    
# keyframe (pickable) data that are needed for volumetric integration
class VolumeIntegratorKeyframeData:
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


class VolumetricIntegratorTask: 
    def __init__(self, keyframe: KeyFrame=None, img=None, img_right=None, depth=None, task_type=VolumetricIntegratorTaskType.NONE, load_save_path=None):
        self.task_type = task_type
        self.keyframe_data = VolumeIntegratorKeyframeData(keyframe, img, img_right, depth)
        self.load_save_path = load_save_path   
        
        
# pickable point cloud obtained from o3d.geometry.PointCloud
class VolumeIntegratorPointCloud:
    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)        
        self.points = points
        self.colors = colors
        
    def to_o3d(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        pc.colors = o3d.utility.Vector3dVector(self.colors)
        return pc
        
# pickable mesh obtained from o3d.geometry.TriangleMesh
class VolumeIntegrationMesh:
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
        
class VolumetricIntegratorOutput:
    def __init__(self, task_type, id=-1, point_cloud:VolumeIntegratorPointCloud =None, mesh:VolumeIntegrationMesh =None):
        self.task_type = task_type
        self.id = id
        self.point_cloud = point_cloud # type: VolumeIntegratorPointCloud
        self.mesh = mesh               # type: VolumeIntegrationMesh
        self.timestamp = time.perf_counter()
        

class VolumetricIntegrator:
    def __init__(self, slam: 'Slam'):        
        import torch.multiprocessing as mp
        # NOTE: The following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).        
        if mp.get_start_method() != 'spawn':
            mp.set_start_method('spawn', force=True) # NOTE: This may generate some pickling problems with multiprocessing 
                                                        #    in combination with torch and we need to check it in other places.
                                                        #    This set start method can be checked with MultiprocessingManager.is_start_method_spawn()
        
        set_rlimit()
         
        self.camera = slam.camera
        self.environment_type = slam.environment_type
        self.sensor_type = slam.sensor_type
        
        self.keyframe_queue = deque() # We use a deque to accumulate keyframes for volumetric integration. 
                                      # We integrate only the keyframes that have been processed by LBA at least once. 
        
        self.volume = None
 
        self.time_volumetric_integration = mp.Value('d',0.0)       
        
        self.last_input_task = None
        self.last_output = None
        
        self.depth_estimator = None
        self.img_id_to_depth = None
        
        self.reset_mutex = mp.Lock()
        self.reset_requested = mp.Value('i',0)

        self.load_request_completed = mp.Value('i',0)
        self.load_request_condition = mp.Condition()
        self.save_request_completed = mp.Value('i',0)
        self.save_request_condition = mp.Condition()

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.q_in = self.mp_manager.Queue()
        self.q_out = self.mp_manager.Queue()
                                        
        self.q_in_condition = mp.Condition()
        self.q_out_condition = mp.Condition()    
        
        self.is_running  = mp.Value('i',0)
        
        self.start()
        
    def start(self):
        self.is_running.value = 1
        self.process = mp.Process(target=self.run,
                          args=(self.camera, self.environment_type, self.sensor_type, \
                                self.q_in, self.q_in_condition, \
                                self.q_out, self.q_out_condition, \
                                self.is_running, self.reset_mutex, self.reset_requested,
                                self.load_request_completed, self.load_request_condition, \
                                self.save_request_completed, self.save_request_condition, \
                                self.time_volumetric_integration,),name=kVolumetricIntegratorProcessName)
        
        #self.process.daemon = True
        self.process.start()

    def save(self, path):
        filepath = path + '/dense_map.ply'        
        task_type = VolumetricIntegratorTaskType.SAVE
        task = VolumetricIntegratorTask(task_type=task_type, load_save_path=filepath)
        self.save_request_completed.value = 0
        self.add_task(task)
        with self.save_request_condition:
            while self.save_request_completed.value == 0:
                self.save_request_condition.wait()
        
    def load(self, path):
        if False:
            task_type = VolumetricIntegratorTaskType.LOAD
            task = VolumetricIntegratorTask(None, None, task_type, load_save_path=path)
            self.load_request_completed.value = 0
            self.add_task(task)
            with self.load_request_condition:
                while self.load_request_completed.value == 0:
                    self.load_request_condition.wait()
        #TODO: Clarify how to use this

    def request_reset(self):
        print('VolumetricIntegrator: Requesting reset...')
        with self.reset_mutex:
            self.reset_requested.value = 1
        while True:
            with self.reset_mutex:
                with self.q_in_condition:
                    self.q_in_condition.notify_all() # to unblock q_in_condition.wait() in run() method               
                if self.reset_requested.value == 0:
                    break
            time.sleep(0.1)
        self.keyframe_queue.clear()
        print('VolumetricIntegrator: ...Reset done.')
            
    def reset_if_requested(self, reset_mutex, reset_requested, volume, \
                            q_in, q_in_condition, \
                            q_out, q_out_condition):
        # acting within the launched process with the passed mp.Value() (received in input)      
        with reset_mutex:
            if reset_requested.value == 1:
                print('VolumetricIntegrator: reset_if_requested()...')                
                with q_in_condition:
                    empty_queue(q_in)
                    q_in_condition.notify_all()
                with q_out_condition:
                    empty_queue(q_out)
                    q_out_condition.notify_all()
                # Now reset the volume integrator in the launched parallel process
                try:
                    volume.reset()
                except Exception as e:
                    print(f'VolumetricIntegrator: reset_if_requested: Exception: {e}')
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        print(f'\t traceback details: {traceback_details}')
                reset_requested.value = 0
        
    def quit(self):
        if self.is_running.value == 1:
            print('VolumetricIntegrator: quitting...')
            self.is_running.value = 0            
            with self.q_in_condition:
                self.q_in.put(None)  # put a None in the queue to signal we have to exit
                self.q_in_condition.notify_all()       
            with self.q_out_condition:
                self.q_out_condition.notify_all()                    
            self.process.join(timeout=5)
            if self.process.is_alive():
                Printer.orange("Warning: Volumetric integration process did not terminate in time, forced kill.")  
                self.process.terminate()      
            print('VolumetricIntegrator: done')   
    
    def init(self, camera: Camera, environment_type: DatasetEnvironmentType, sensor_type: SensorType):
        
        self.last_output = None
        self.depth_factor = 1.0/camera.depth_factor
        self.environment_type = environment_type    
        self.volumetric_integration_depth_trunc = Parameters.kVolumetricIntegrationDepthTruncIndoor if environment_type == DatasetEnvironmentType.INDOOR else Parameters.kVolumetricIntegrationDepthTruncOutdoor
    
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=Parameters.kVolumetricIntegrationVoxelLength,
            sdf_trunc=Parameters.kVolumetricIntegrationSdfTrunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
                
        self.o3d_camera = o3d.camera.PinholeCameraIntrinsic(width=camera.width, height=camera.height, fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)     
        
        # pip install flash_attn
        self.img_id_to_depth = {}
        import torch
        from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
        if Parameters.kVolumetricIntegrationUseDepthEstimator:
            depth_estimator_type = DepthEstimatorType.from_string(Parameters.kVolumetricIntegrationDepthEstimatorType)
            min_depth = 0
            max_depth = 50 if environment_type == DatasetEnvironmentType.OUTDOOR else 10 
            precision = torch.float16  # for depth_pro
            if sensor_type == SensorType.MONOCULAR:
                Printer.red("*************************************************************************************")
                Printer.error('VolumetricIntegrator: You cannot use a MONOCULAR depth estimator with a MONOCULAR SLAM system!')
                Printer.red('The scale of the metric depth estimator will conflict with the scale of the SLAM system!')
                Printer.red("*************************************************************************************")
            self.depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, 
                                                    min_depth=min_depth, max_depth=max_depth,
                                                    dataset_env_type=environment_type, precision=precision,
                                                    camera=camera)
        
        # Prepare maps to undistort color and depth images
        h, w = camera.height, camera.width
        D = camera.D
        K = camera.K
        #Printer.green(f'VolumetricIntegrator: init: h={h}, w={w}, D={D}, K={K}')
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
            time_volumetric_integration):
        print('VolumetricIntegrator: starting...')
        self.init(camera, environment_type, sensor_type)
        self.sensor_type = sensor_type
        # main loop
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1 and reset_requested.value == 0:
                    print('VolumetricIntegrator: waiting for new task...')
                    q_in_condition.wait()
            if not q_in.empty():            
                self.volume_integration(self.volume, q_in, q_out, q_out_condition, is_running, \
                                    load_request_completed, load_request_condition, save_request_completed, save_request_condition, time_volumetric_integration)
            else: 
                print('VolumetricIntegrator: q_in is empty...')
            self.reset_if_requested(reset_mutex, reset_requested, self.volume, q_in, q_in_condition, q_out, q_out_condition)

        empty_queue(q_in) # empty the queue before exiting         
        print('VolumetricIntegrator: loop exit...')         

    def volume_integration(self, volume, q_in, q_out, q_out_condition, is_running, \
                             load_request_completed, load_request_condition, save_request_completed, save_request_condition, time_volumetric_integration):
        #print('VolumetricIntegrator: volume_integration')
        last_output = None        
        timer = TimerFps("VolumetricIntegrator", is_verbose = kTimerVerbose)
        timer.start()        
        try: 
            if is_running.value == 1:
                
                # check q_in size and dump a warn message if it is too big
                q_in_size = q_in.qsize()
                if q_in_size >= 10: 
                    warn_msg = f'!VolumetricIntegrator: WARNING: q_in size: {q_in_size}'
                    print(warn_msg)
                    Printer.orange(warn_msg)
                    
                self.last_input_task = q_in.get() # blocking call to get a new input task for volume integration
                if self.last_input_task is None: 
                    is_running.value = 0 # got a None to exit
                else:
                    
                    if self.last_input_task.task_type == VolumetricIntegratorTaskType.INTEGRATE:
                        keyframe_data = self.last_input_task.keyframe_data
                        color = keyframe_data.img
                        color_right = keyframe_data.img_right
                        depth = keyframe_data.depth
                        if depth is None: 
                            if self.depth_estimator is None:
                                Printer.yellow('VolumetricIntegrator: depth is None, skipping the keyframe...')
                                return # skip this keyframe
                            else:
                                inference_start_time = time.time()
                                if keyframe_data.id in self.img_id_to_depth:
                                    depth = self.img_id_to_depth[keyframe_data.id]
                                else:
                                    if self.sensor_type == SensorType.MONOCULAR:
                                        Printer.error('VolumetricIntegrator: You cannot use a MONOCULAR depth estimator with a MONOCULAR SLAM system!')                                    
                                    depth = self.depth_estimator.infer(color, color_right)
                                    print(f'VolumetricIntegrator: depth inference time: {time.time() - inference_start_time}')
                                    if Parameters.kVolumetricIntegrationDepthEstimationFilterShadowPoints:
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
                                                                        
                        pose = keyframe_data.pose # Tcw
                        #inv_pose = inv_T(pose)   # Twc
                        
                        print(f'VolumetricIntegrator: depth_undistorted: shape: {depth_undistorted.shape}, type: {depth_undistorted.dtype}')
                                                                                
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(color_undistorted), 
                            o3d.geometry.Image(depth_undistorted), 
                            depth_scale=self.depth_factor,
                            depth_trunc=self.volumetric_integration_depth_trunc, 
                            convert_rgb_to_intensity=False)
                        
                        volume.integrate(rgbd, self.o3d_camera, pose)
                        
                        do_output = True
                        if self.last_output is not None:
                            elapsed_time = time.perf_counter() - self.last_output.timestamp
                            if elapsed_time < Parameters.kVolumetricIntegrationOutputTimeInterval:
                                do_output = False
                        
                        if do_output:
                            mesh_out, pc_out = None, None
                            if Parameters.kVolumetricIntegrationExtractMesh:
                                mesh = volume.extract_triangle_mesh()
                                #mesh.compute_vertex_normals()
                                mesh_out = VolumeIntegrationMesh(mesh)
                                print(f'VolumetricIntegrator: id: {keyframe_data.id} -> Mesh: points: {mesh_out.vertices.shape}')
                            else: 
                                point_cloud = volume.extract_point_cloud()
                                pc_out = VolumeIntegratorPointCloud(point_cloud)
                                print(f'VolumetricIntegrator: id: {keyframe_data.id} -> PointCloud: points: {pc_out.points.shape}')                                  

                            last_output = VolumetricIntegratorOutput(self.last_input_task.task_type, 
                                                                    keyframe_data.id, 
                                                                    pc_out, 
                                                                    mesh_out)
                            self.last_output = last_output
                            
                        
                    elif self.last_input_task.task_type == VolumetricIntegratorTaskType.SAVE:
                        save_path = self.last_input_task.load_save_path
                        if Parameters.kVolumetricIntegrationExtractMesh:
                            mesh = volume.extract_triangle_mesh()
                            #mesh.compute_vertex_normals()
                            o3d.io.write_triangle_mesh(save_path, mesh)
                        else: 
                            point_cloud = volume.extract_point_cloud()
                            o3d.io.write_point_cloud(save_path, point_cloud)
                                                        
                        # if self.last_output is not None:
                        #     if self.last_output.mesh is not None:
                        #         mesh = self.last_output.mesh
                        #         o3d.io.write_triangle_mesh(save_path, mesh.to_o3d())
                        #     else: 
                        #         point_cloud = self.last_output.point_cloud
                        #         o3d.io.write_point_cloud(save_path, point_cloud.to_o3d())      
                                
                        last_output = VolumetricIntegratorOutput(self.last_input_task.task_type)      
                        
                    elif self.last_input_task.task_type == VolumetricIntegratorTaskType.RESET:
                        volume.reset()                                           
                        
                    if is_running.value == 1 and last_output is not None:
                        if last_output.task_type == VolumetricIntegratorTaskType.INTEGRATE:
                            with q_out_condition:
                                # push the computed output in the output queue
                                q_out.put(last_output)
                                q_out_condition.notify_all()
                                print(f'VolumetricIntegrator: pushed new output to q_out size: {q_out.qsize()}')
                        elif last_output.task_type == VolumetricIntegratorTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()
            
        except Exception as e:
            print(f'VolumetricIntegrator: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')

        timer.refresh()
        time_volumetric_integration.value = timer.last_elapsed
        id_info = f'id: {self.last_output.id}, ' if last_output is not None else ''
        print(f'VolumetricIntegrator: {id_info} q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, volume-integration-process elapsed time: {time_volumetric_integration.value}')


    # TODO: Add a timer and a mutex to call this periodically
    def flush_keyframe_queue(self):
        # iterate over the keyframe queue
        i = 0  
        while i < len(self.keyframe_queue):
            kf_to_process = self.keyframe_queue[i]
            # We integrate only the keyframes that have been processed by LBA at least once.
            if kf_to_process.lba_count >= Parameters.kVolumetricIntegrationMinNumLBATimes:
                self.keyframe_queue.remove(kf_to_process)  # Remove item
                print(f'VolumetricIntegrator: Adding keyframe with img id: {kf_to_process.id} (kid: {kf_to_process.kid})')
                task_type = VolumetricIntegratorTaskType.INTEGRATE
                task = VolumetricIntegratorTask(kf_to_process, task_type=task_type)      
                self.add_task(task)        
            else:
                i += 1  # Only move forward if no removal to avoid skipping   

    def add_keyframe(self, keyframe: KeyFrame, img, img_right, depth, print=print):
        if depth is None and not Parameters.kVolumetricIntegrationUseDepthEstimator:
            print(f'VolumetricIntegrator: add_keyframe: depth is None -> skipping frame {keyframe.id}')
            return
        try:
            # We accumulate the keyframe in a queue. 
            # We integrate only the keyframes that have been processed by LBA at least once.
            self.keyframe_queue.append(keyframe)
            self.flush_keyframe_queue()    
            
        except Exception as e:
            print(f'VolumetricIntegrator: EXCEPTION: {e} !!!')
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                print(f'\t traceback details: {traceback_details}')
                        
    def add_task(self, task: VolumetricIntegratorTask):
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(task)
                self.q_in_condition.notify_all()
                
    def rebuild(self, map: Map):
        print('')
        print(f'VolumetricIntegrator: rebuild() rebuilding volumetric mapping...')
        if self.is_running.value == 1:
            task = VolumetricIntegratorTask(task_type=VolumetricIntegratorTaskType.RESET)      
            self.add_task(task) 
            
            self.keyframe_queue.clear()
            for kf in map.keyframes:
                if not kf.is_bad and kf.lba_count > 0:
                    if kf.depth_img is None:
                        print(f'VolumetricIntegrator: rebuild: depth is None -> skipping frame {kf.id}')
                        continue
                    self.keyframe_queue.append(kf)
            self.flush_keyframe_queue()
        
    def pop_output(self, timeout=Parameters.kLoopDetectingTimeoutPopKeyframe) -> VolumetricIntegratorOutput: 
        q_out = self.q_out
        q_out_condition = self.q_out_condition
             
        if self.is_running.value == 0:
            return None
        with q_out_condition:        
            while q_out.empty() and self.is_running.value == 1:
                ok = q_out_condition.wait(timeout=timeout)
                if not ok: 
                    print('VolumetricIntegrator: pop_output: timeout')
                    break # Timeout occurred
        if q_out.empty():
            return None
        try:
            return q_out.get(timeout=timeout)
        except Exception as e:
            print(f'VolumetricIntegrator: pop_output: encountered exception: {e}')
            return None