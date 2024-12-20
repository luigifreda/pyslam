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

from frame import Frame, FrameShared, match_frames
from keyframe import KeyFrame
from map_point import MapPoint
from map import Map
from camera import Camera

from search_points import propagate_map_point_matches
from search_points import search_map_by_projection, search_frame_by_projection

from local_mapping import LocalMapping

from loop_closing import LoopClosing

from dataset import SensorType, DatasetEnvironmentType

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_tracker import feature_tracker_factory, FeatureTracker, FeatureTrackerTypes 

from utils_serialization import SerializableEnum, SerializationJSON, register_class
from utils_sys import Printer, getchar, Logging
from utils_draw import draw_feature_matches
from utils_geom import triangulate_points, poseRt, normalize_vector, inv_T, triangulate_normalized_points, estimate_pose_ess_mat
from utils_features import ImageGrid

from slam_commons import SlamState
from tracking import Tracking

from volumetric_integrator import VolumetricIntegrator, VolumetricIntegratorOutput

kVerbose = True     

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kTrackingWaitForLocalMappingToGetIdle = Parameters.kTrackingWaitForLocalMappingToGetIdle

kLogKFinfoToFile = True 
      
kUseEssentialMatrixFitting = Parameters.kUseEssentialMatrixFitting      
       

if not kVerbose:
    def print(*args, **kwargs):
        pass 


# Main slam system class containing all the required modules. 
class Slam(object):
    def __init__(self, camera: Camera, feature_tracker_config: dict, loop_detector_config=None, 
                 sensor_type=SensorType.MONOCULAR, groundtruth=None, environment_type=DatasetEnvironmentType.OUTDOOR):
        self.camera = camera 
        self.feature_tracker_config = feature_tracker_config
        self.loop_detector_config = loop_detector_config
        self.sensor_type = sensor_type  
        self.environment_type = environment_type   
        self.init_feature_tracker(feature_tracker_config)
        self.map = Map()
        self.local_mapping = LocalMapping(self)        
        self.loop_closing = None
        self.GBA = None
        self.volumetric_integrator = None
        self.reset_requested = False
  
        self.init_volumetric_integrator() 
        self.init_loop_closing(loop_detector_config)     
                
        if kLocalMappingOnSeparateThread:
            self.local_mapping.start()
        self.tracking = Tracking(self) # after all the other initializations
        
        #self.groundtruth = groundtruth  # not actually used here; could be used for evaluating performances (at present done in Viewer3D)

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
        if self.loop_closing is not None:
            self.loop_closing.request_reset()
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
        # Set the static field of the class Frame and FrameShared.
        # NOTE: Here, we set force=True since we don't care if the feature_tracker is already set.
        Frame.set_tracker(feature_tracker, force=True) 
        if kUseEssentialMatrixFitting:
            Printer.orange('SLAM: forcing feature matcher ratio_test to 0.8')
            feature_tracker.matcher.ratio_test = 0.8
        if feature_tracker.tracker_type == FeatureTrackerTypes.LK:
            raise ValueError("SLAM: You cannot use Lukas-Kanade feature_tracker in this SLAM approach!")  
        
    def init_loop_closing(self, loop_detector_config):
        if Parameters.kUseLoopClosing and loop_detector_config is not None:
            if self.loop_closing is not None: 
                self.loop_closing.quit()
            self.loop_closing = LoopClosing(self, loop_detector_config)
            self.GBA = self.loop_closing.GBA
            self.loop_closing.start()
            time.sleep(1)             
        
    def init_volumetric_integrator(self):
        if Parameters.kUseVolumetricIntegration:
            self.volumetric_integrator = VolumetricIntegrator(self)
        
    # @ main track method @
    def track(self, img, img_right, depth, img_id, timestamp=None):
        return self.tracking.track(img, img_right, depth, img_id, timestamp)
    
    def set_tracking_state(self, state: SlamState):
        self.tracking.state = state
        
    def get_dense_map(self) -> VolumetricIntegratorOutput:
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
        with open(map_file_path, 'rb') as f:
            loaded_json = json.loads(f.read())
            
            print()
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
            
            print(f'SLAM: initializing feature tracker...')
            self.init_feature_tracker(feature_tracker_config)
            
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
            self.local_mapping.start()
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
        
    # for saving and reloading it when needed
    def set_viewer_scale(self, scale):
        self.map.viewer_scale = scale
        
    def viewer_scale(self):
        return self.map.viewer_scale