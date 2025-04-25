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

import sys
import os
import numpy as np
import time

#import json
import ujson as json

from config_parameters import Parameters  

from frame import Frame, FeatureTrackerShared, match_frames
from keyframe import KeyFrame
from map_point import MapPoint
from map import Map
from camera import Camera

from search_points import propagate_map_point_matches
from search_points import search_map_by_projection, search_frame_by_projection

from local_mapping import LocalMapping

from loop_closing import LoopClosing

from dataset_types import SensorType, DatasetEnvironmentType

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_tracker import feature_tracker_factory, FeatureTracker, FeatureTrackerTypes 

from utils_serialization import SerializableEnum, SerializationJSON, register_class
from utils_sys import Printer, getchar, Logging
from utils_mp import MultiprocessingManager
from utils_geom import inv_T
from global_bundle_adjustment import GlobalBundleAdjustment

from slam_commons import SlamState
from tracking import Tracking

from volumetric_integrator_base import VolumetricIntegrationOutput
from volumetric_integrator_factory import volumetric_integrator_factory, VolumetricIntegratorType

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Config


kVerbose = True     

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kTrackingWaitForLocalMappingToGetIdle = Parameters.kTrackingWaitForLocalMappingToGetIdle

kLogKFinfoToFile = True 
      
kUseEssentialMatrixFitting = Parameters.kUseEssentialMatrixFitting      
       

if not kVerbose:
    def print(*args, **kwargs):
        pass 


class SlamMode(SerializableEnum):
    SLAM        = 0   # Normal SLAM mode.
    MAP_BROWSER = 1   # Just want to reload a map and view it. Don't need loop closing, volume integrator


# Main slam system class containing all the required modules. 
class Slam(object):
    def __init__(self, camera: Camera, 
                 feature_tracker_config: dict, 
                 loop_detector_config=None, 
                 sensor_type=SensorType.MONOCULAR, 
                 environment_type=DatasetEnvironmentType.OUTDOOR,
                 slam_mode=SlamMode.SLAM,
                 config:'Config' =None,
                 headless=False):
        self.camera = camera 
        self.feature_tracker_config = feature_tracker_config
        self.loop_detector_config = loop_detector_config
        self.sensor_type = sensor_type  
        self.environment_type = environment_type
        self.slam_mode = slam_mode
        self.headless = headless
          
        self.feature_tracker = None
        self.init_feature_tracker(feature_tracker_config)
        
        self.map = Map()
        self.local_mapping = LocalMapping(self)        
        self.loop_closing = None
        self.GBA = None
        self.GBA_on_demand = None    # used independently when pressing "Bundle Adjust" button on GUI
        self.volumetric_integrator = None
        self.reset_requested = False
                    
        if slam_mode == SlamMode.SLAM:
            self.init_volumetric_integrator() 
            self.init_loop_closing(loop_detector_config, headless=headless)     
                
        if kLocalMappingOnSeparateThread:
            self.local_mapping.start()
                     
        self.tracking = Tracking(self) # after all the other initializations
        
        self.set_config_params(config)         
            
    def set_config_params(self, config:'Config'):
        self.config = config
        if config is not None:
            # get params from config
            far_points_threshold = config.far_points_threshold #if self.sensor_type != SensorType.MONOCULAR else None
            if far_points_threshold is not None:
                Printer.green(f'Slam: Using far points threshold from config: {far_points_threshold}')
            use_fov_centers_based_kf_generation = config.use_fov_centers_based_kf_generation 
            max_fov_centers_distance = config.max_fov_centers_distance
            # distribute the read params to submodules
            Frame.is_compute_median_depth = config.use_fov_centers_based_kf_generation
            if use_fov_centers_based_kf_generation:
                Printer.green(f'Slam: Using fov centers based kf generation from config: {use_fov_centers_based_kf_generation}, max fov centers distance: {max_fov_centers_distance}')            
            if self.tracking is not None: 
                self.tracking.far_points_threshold = config.far_points_threshold
                self.tracking.use_fov_centers_based_kf_generation = config.use_fov_centers_based_kf_generation
                self.tracking.max_fov_centers_distance = config.max_fov_centers_distance
            if self.local_mapping is not None:
                self.local_mapping.far_points_threshold = config.far_points_threshold
                self.local_mapping.use_fov_centers_based_kf_generation = config.use_fov_centers_based_kf_generation
                self.local_mapping.max_fov_centers_distance = config.max_fov_centers_distance

    def request_reset(self):
        self.reset_requested = True

    def reset(self):
        self.local_mapping.request_reset()
        if self.loop_closing is not None:
            self.loop_closing.request_reset()
        if self.volumetric_integrator is not None:
            self.volumetric_integrator.request_reset()        
        self.tracking.reset()
        self.map.reset()
        
    def reset_session(self):
        self.local_mapping.request_reset()
        # See the discussion here: https://github.com/luigifreda/pyslam/issues/131 
        # if self.loop_closing is not None:
        #     self.loop_closing.request_reset()
        if self.volumetric_integrator is not None:
            self.volumetric_integrator.request_reset()  
        self.tracking.reset()
        self.map.reset_session()        
        
    def quit(self):
        print('SLAM: quitting ...')
        if kLocalMappingOnSeparateThread:
            self.local_mapping.quit()  
        if self.loop_closing is not None:
            self.loop_closing.quit()
        if self.volumetric_integrator is not None:
            self.volumetric_integrator.quit()                               
        print('SLAM: done')

    def init_feature_tracker(self, feature_tracker_config):
        feature_tracker = feature_tracker_factory(**feature_tracker_config)          
        self.feature_tracker = feature_tracker
        # Set the static field of the class Frame and FeatureTrackerShared.
        # NOTE: Here, we set force=True since we don't care if the feature_tracker is already set.
        FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)
        if self.sensor_type == SensorType.STEREO:
            # In case of stereo, for thread-safety reasons, we create a second feature_tracker that we can use in parallel during stereo image processing 
            feature_tracker_right = feature_tracker_factory(**feature_tracker_config)
            FeatureTrackerShared.set_feature_tracker_right(feature_tracker_right, force=True)
        if kUseEssentialMatrixFitting:
            Printer.orange('SLAM: forcing feature matcher ratio_test to 0.8')
            feature_tracker.matcher.ratio_test = 0.8
        if feature_tracker.tracker_type == FeatureTrackerTypes.LK:
            raise ValueError("SLAM: At present time, you cannot use Lukas-Kanade feature_tracker in this SLAM framework!")  
        
    def init_loop_closing(self, loop_detector_config, headless=False):
        if Parameters.kUseLoopClosing and loop_detector_config is not None:
            if self.loop_closing is not None: 
                self.loop_closing.quit()
            self.loop_closing = LoopClosing(self, loop_detector_config, headless=headless)
            self.GBA = self.loop_closing.GBA
            self.loop_closing.start()
            time.sleep(1)             
        
    def init_volumetric_integrator(self):
        if Parameters.kUseVolumetricIntegration:
            # TODO(@dvdmc): Implement semantics in vol. integrator.
            self.volumetric_integrator_type = VolumetricIntegratorType.from_string(Parameters.kVolumetricIntegrationType)
            self.volumetric_integrator = volumetric_integrator_factory(self.volumetric_integrator_type, self.camera, self.environment_type, self.sensor_type)
        
    # @ main track method @
    def track(self, img, img_right, depth, img_id, timestamp=None, semantic_img=None):
        return self.tracking.track(img, img_right, depth, img_id, timestamp, semantic_img)
    
    def set_tracking_state(self, state: SlamState):
        self.tracking.state = state
        
    def get_dense_map(self) -> VolumetricIntegrationOutput:
        if self.volumetric_integrator is not None:
            q_out_size = self.volumetric_integrator.q_out.qsize()
            if q_out_size > 0:
                print(f'VolumetricIntegrator: getting dense map ... q_out size: {q_out_size} ')                
                return self.volumetric_integrator.pop_output()    
        return None 
    
    def save_system_state(self, path):
        Printer.green(f'\nSLAM: saving the system state into {path}...')
        if not os.path.exists(path):
            os.mkdir(path)
            
        map_out_json= {}      # single output map state json: map + frontend config + backend config
        config_out_json = {}  # let's put redundant human readable config info here 
        
        map_json = self.map.to_json()
        feature_tracker_config_json = SerializationJSON.serialize(self.feature_tracker_config)
        loop_detector_config_json = SerializationJSON.serialize(self.loop_detector_config)
        
        map_out_json['sensor_type'] = SerializationJSON.serialize(self.sensor_type)
        map_out_json['environment_type'] = SerializationJSON.serialize(self.environment_type)
        
        map_out_json['map'] = map_json
        map_out_json['feature_tracker_config'] = feature_tracker_config_json 
        map_out_json['loop_detector_config'] = loop_detector_config_json
        
        config_out_json['feature_tracker_config'] = feature_tracker_config_json
        config_out_json['loop_detector_config'] = loop_detector_config_json
        
        map_file_path = path + '/map.json'       
        with open(map_file_path, 'w') as f:
            f.write(json.dumps(map_out_json))
            
        config_file_path = path + '/config_info.json'  # redundant, not used on reloading, just for human readability
        with open(config_file_path, 'w') as f:
            f.write(json.dumps(config_out_json, indent=4))            
            
        if self.loop_closing is not None:
            self.loop_closing.save(path)
            
        if self.volumetric_integrator is not None:
            self.volumetric_integrator.save(path)
            
        Printer.green(f'SLAM: ...system state successfully saved to: {path}')        
    
    def load_system_state(self, path):
        Printer.green(f'\nSLAM: loading the system state from {path}...')
        self.local_mapping.quit()     
        if not os.path.exists(path):
            Printer.red(f'SLAM: Cannot load system state: {path}')
            sys.exit(0)
        map_file_path = path + '/map.json'
        if not os.path.exists(map_file_path):
            Printer.red(f'SLAM: File does not exist: {map_file_path}')
            return
        with open(map_file_path, 'rb') as f:
            loaded_json = json.loads(f.read())
            print()
            loaded_sensor_type = SerializationJSON.deserialize(loaded_json['sensor_type'])
            if loaded_sensor_type != self.sensor_type and self.slam_mode == SlamMode.SLAM:
                Printer.yellow(f'SLAM: sensor type mismatch on load_system_state(): {loaded_sensor_type} != {self.sensor_type}')
                #sys.exit(0)
            self.sensor_type = loaded_sensor_type
            loaded_environment_type = SerializationJSON.deserialize(loaded_json['environment_type'])
            if loaded_environment_type != self.environment_type:
                Printer.yellow(f'SLAM: environment type mismatch on load_system_state(): {loaded_environment_type} != {self.environment_type}')
                #sys.exit(0)
            
            #loaded_feature_tracker_config = loaded_json['feature_tracker_config']
            #print(f'SLAM: deserializing feature_tracker_config: {loaded_feature_tracker_config} ...')
            feature_tracker_config = SerializationJSON.deserialize(loaded_json['feature_tracker_config'])
            print(f'SLAM: loaded feature feature_tracker config: {feature_tracker_config}')
            print()
            #loaded_loop_detector_config = loaded_json['loop_detector_config']
            #print(f'SLAM: deserializing loop_detector_config: {loaded_loop_detector_config} ...')            
            loop_detector_config = SerializationJSON.deserialize(loaded_json['loop_detector_config'])
            print(f'SLAM: loaded loop detector config: {loop_detector_config}')
            print()
            
            print(f'SLAM: initializing feature tracker: {feature_tracker_config}')
            self.init_feature_tracker(feature_tracker_config)
            
            if self.slam_mode == SlamMode.SLAM:
                print(f'SLAM: initializing loop closing...')
                self.init_loop_closing(loop_detector_config)
                
            if self.loop_closing is not None:
                print(f'SLAM: loading the loop closing state from {path}...')
                self.loop_closing.load(path)
                print(f'SLAM: ... successfully loaded the loop closing state from {path}.')
                        
            print()
            print(f'SLAM: loading map...')   
            map_json = loaded_json['map']                     
            self.map.from_json(map_json)
            print(f'SLAM: map loaded')               
            self.local_mapping.start()
            
        # check if current set camera is the same as the one stored in the map 
        if self.map is not None and len(self.map.keyframes) > 0:
            camera0 = self.map.keyframes[0].camera
            if camera0.width != self.camera.width or camera0.height != self.camera.height:
                if self.camera.width is not None and self.camera.height is not None:
                    Printer.yellow(f'SLAM: camera size mismatch on load_system_state(): {camera0.width}x{camera0.height} != {self.camera.width}x{self.camera.height}')
                # update the camera
                self.camera = camera0 
                if self.volumetric_integrator is not None:
                    self.volumetric_integrator.camera = camera0
        Printer.green(f'SLAM: ...system state successfully loaded from: {path}')
    
    def save_map(self, path):
        Printer.green(f'\nSLAM: saving the map into {path}...')
        self.map.save(path)

    def load_map(self, path):
        Printer.green(f'\nSLAM: loading the map from {path}...')         
        if not os.path.exists(path):
            Printer.red(f'SLAM: Cannot load map: {path}')
            sys.exit(0)
        self.map.load(path)
    
    def bundle_adjust(self):
        # Start an independent global bundle adjustment
        print(f'SLAM: starting global bundle adjustment ...')
        
        if not self.GBA_on_demand:
            # we create a GBA here to allow refining the loaded
            use_multiprocessing = not MultiprocessingManager.is_start_method_spawn() 
            self.GBA_on_demand = GlobalBundleAdjustment(self, use_multiprocessing=use_multiprocessing)
                    
        if self.GBA_on_demand is not None:
            if self.GBA_on_demand.is_running():
                Printer.red('GlobalBundleAdjustment: GBA is already running! You can only have one GBA running at a time.')
                return
            else:
                self.GBA_on_demand.quit()
                self.GBA_on_demand.start(loop_kf_id=0)  # loop_kf_id=0 means that it GBA is run for all keyframes and there is no loop closure involved
                while not self.GBA_on_demand.is_running(): # wait for GBA to start
                    time.sleep(0.1)
                while self.GBA_on_demand.is_running(): # wait for GBA to finish
                    time.sleep(0.1)
                self.GBA_on_demand.correct_after_GBA()
                print(f'SLAM: ...global bundle adjustment finished.')
     
    # For saving and reloading it when needed
    def set_viewer_scale(self, scale):
        self.map.viewer_scale = scale
        
    def viewer_scale(self):
        return self.map.viewer_scale
    
    # Retrieve the final processed trajectory. A pose estimate is returned for each camera frame (not only for keyframes). 
    # If used at the end of dataset playback, this returns an "final" trajectory where each extracted camera pose Twc has been optimized multiple times by LBA and GBA over the multiple window optimizations that cover Twc. 
    # To get the "online" trajectory (where each camera pose Twc is extracted at the end of each tracking iteration) just get (slam.tracking.cur_R, slam.tracking.cur_t, timestamp) at the end of each tracking iteration.
    # NOTE: In the "online" trajectory, pose estimates only depend on the past ones. 
    #       In the "final" trajectory, pose estimates depend on both the past and future ones. 
    def get_final_trajectory(self):
        print(f"\nRetrieving final trajectory ...")

        poses = []
        timestamps = []
        ids = []

        keyframes = self.map.get_keyframes()
        #keyframes.sort(key=lambda kf: kf.id)
        #print(f'keyframes: {[kf.id for kf in keyframes]}')

        # Transform all keyframes so that the first keyframe is at the origin.
        # After a loop closure the first keyframe might not be at the origin.
        Two = keyframes[0].Twc

        # Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        # We need to get first the keyframe pose and then concatenate the relative transformation.
        # Frames not localized (tracking failure) are not saved.
        
        tracking_history = self.tracking.tracking_history

        # For each frame we have a reference keyframe (ref_kf), the timestamp (timestamp) and the SLAM state (state)
        for rel_pose, ref_kf, timestamp, id, state in zip(tracking_history.relative_frame_poses, tracking_history.kf_references, tracking_history.timestamps, tracking_history.ids, tracking_history.slam_states):
            if state != SlamState.OK:
                continue

            keyframe = ref_kf
            Tcr = np.eye(4, dtype=np.float32)
            # If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while keyframe.is_bad:
                Tcr = Tcr @ keyframe.Tcp
                keyframe = keyframe.parent

            Trw = Tcr @ keyframe.Tcw @ Two
                        
            Tcw = rel_pose.matrix() @ Trw
            Twc = inv_T(Tcw)
            
            poses.append(Twc)
            timestamps.append(timestamp)
            ids.append(id)

        return poses, timestamps, ids

