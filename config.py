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
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()
    
import os
import yaml
import numpy as np
from utilities.utils_sys import Printer, locally_configure_qt_environment
import math


# N.B.: this file must stay in the root folder of the repository 


# get the folder location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Class for getting libs settings (from config.yaml) and camera settings from a yaml file 
class Config(object):
    '''
    Config is used for getting libs settings (from config.yaml) and camera settings from a yaml file 
    '''
    def __init__(self):
        self.root_folder = __location__
        self.config_file = 'config.yaml'
        self.config_file_path = __location__ + '/' + self.config_file
        #print(f'root folder: {self.root_folder}, config file path: {self.config_file_path}')
        self.config = yaml.load(open(self.config_file_path, 'r'), Loader=yaml.FullLoader)
        self.cam_settings = None
        self.system_settings = None
        self.dataset_settings = None
        self.dataset_type = None
        self.sensor_type = None
        self.system_state_settings = None
        self.system_state_folder_path = None
        self.system_state_load = False
        self.trajectory_settings = None
        self.start_frame_id = 0  
        
        #locally_configure_qt_environment()

        self.set_core_lib_paths()
        self.read_lib_paths()
        
        self.get_dataset_settings()
        self.get_general_system_settings()        
        self.get_system_state_settings()
        self.get_trajectory_settings()


    # read core lib paths from config.yaml and set sys paths
    def set_core_lib_paths(self):
        self.core_lib_paths = self.config['CORE_LIB_PATHS']
        for path in self.core_lib_paths:
            ext_path = __location__ + '/' + self.core_lib_paths[path]
            #print( "importing path: ", ext_path )
            sys.path.append(ext_path)
            
    # read lib paths from config.yaml 
    def read_lib_paths(self):
        self.lib_paths = self.config['LIB_PATHS']
        
    # set sys path of lib 
    def set_lib(self,lib_name,prepend=False):
        if lib_name in self.lib_paths:
            lib_paths = [e.strip() for e in self.lib_paths[lib_name].split(',')]
            #print('setting lib paths:',lib_paths)
            for lib_path in lib_paths:
                ext_path = __location__ + '/' + lib_path
                #print( "importing path: ", ext_path )
                if not prepend: 
                    sys.path.append(ext_path)      
                else: 
                    sys.path.insert(0,ext_path)
        else: 
            print('cannot set lib: ', lib_name)
            
    # get dataset settings
    def get_dataset_settings(self):
        self.dataset_type = self.config['DATASET']['type']
        self.dataset_settings = self.config[self.dataset_type]
        self.sensor_type = self.dataset_settings['sensor_type'].lower()
        self.dataset_path = self.dataset_settings['base_path']
        self.dataset_settings['base_path'] = os.path.join( __location__, self.dataset_path)
        #print('dataset_settings: ', self.dataset_settings)
                    
    # get general system settings
    def get_general_system_settings(self):
        self.system_settings = None
        self.general_settings_filepath = __location__ + '/' + self.config[self.dataset_type]['settings']
        if self.sensor_type == 'stereo' and 'settings_stereo' in self.config[self.dataset_type]:
            self.general_settings_filepath = __location__ + '/' + self.config[self.dataset_type]['settings_stereo']
            Printer.orange('Using stereo settings file: ' + self.general_settings_filepath)
            print('------------------------------------')          
        if(self.general_settings_filepath is not None):
            with open(self.general_settings_filepath, 'r') as stream:
                try:
                    self.system_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)
        self.cam_settings = self.system_settings                                                    

    def get_system_state_settings(self):
        self.system_state_settings = self.config['SYSTEM_STATE']
        self.system_state_load = self.system_state_settings['load_state']
        self.system_state_folder_path = __location__ + '/' + self.system_state_settings['folder_path']
        folder_path_exists = os.path.exists(self.system_state_folder_path)
        folder_path_is_not_empty = os.path.getsize(self.system_state_folder_path) > 0 if folder_path_exists else False
        if self.system_state_load and not(folder_path_exists and folder_path_is_not_empty):
            Printer.red('System state folder does not exist or is empty: ' + self.system_state_folder_path)
            self.system_state_load = False

    # get trajectory save settings
    def get_trajectory_settings(self):
        self.trajectory_settings = self.config['SAVE_TRAJECTORY']


    # calibration matrix
    @property
    def K(self):
        if not hasattr(self, '_K'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._K = np.array([[fx,  0, cx],
                                [ 0, fy, cy],
                                [ 0,  0,  1]])
        return self._K

    # inverse of calibration matrix
    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._Kinv = np.array([[1/fx,    0, -cx/fx],
                                   [   0, 1/fy, -cy/fy],
                                   [   0,    0,      1]])
        return self._Kinv

    # distortion coefficients
    @property
    def DistCoef(self):
        if not hasattr(self, '_DistCoef'):
            k1 = self.cam_settings['Camera.k1']
            k2 = self.cam_settings['Camera.k2']
            p1 = self.cam_settings['Camera.p1']
            p2 = self.cam_settings['Camera.p2']
            k3 = 0
            if 'Camera.k3' in self.cam_settings:
                k3 = self.cam_settings['Camera.k3']
            self._DistCoef = np.array([k1, k2, p1, p2, k3])
            if self.sensor_type == 'stereo':
                self._DistCoef = np.array([0, 0, 0, 0, 0])
                Printer.orange('WARNING: Using stereo camera, images are automatically rectified, and DistCoef is set to [0,0,0,0,0]')
        return self._DistCoef
    
    # baseline times fx
    @property
    def bf(self):
        if not hasattr(self, '_bf'):
            self._bf = self.cam_settings['Camera.bf']
        return self._bf

    # camera width
    @property
    def width(self):
        if not hasattr(self, '_width'):
            self._width = self.cam_settings['Camera.width']
        return self._width

    # camera height
    @property
    def height(self):
        if not hasattr(self, '_height'):
            self._height = self.cam_settings['Camera.height']
        return self._height
    
    # camera fps
    @property
    def fps(self):
        if not hasattr(self, '_fps'):
            self._fps= self.cam_settings['Camera.fps']
        return self._fps    

    # depth factor
    @property
    def depth_factor(self):
        if not hasattr(self, '_depth_factor'):
            if 'Camera.DepthMapFactor' in self.cam_settings:
                self._depth_factor = self.cam_settings['Camera.DepthMapFactor']
            else:
                self._depth_factor = 1.0
        return self._depth_factor
    
    # depth threshold 
    @property
    def depth_threshold(self):
        if not hasattr(self, '_depth_threshold'):
            if 'Camera.ThDepth' in self.cam_settings:
                self._depth_threshold = self.cam_settings['Camera.ThDepth']
            else:
                self._depth_threshold = float('inf')
        return self._depth_threshold

    # num features to extract 
    @property
    def num_features_to_extract(self):
        if not hasattr(self, '_num_features_to_extract'):
            if 'FeatureExtractor.nFeatures' in self.system_settings:
                self._num_features_to_extract = self.system_settings['FeatureExtractor.nFeatures']
            else:
                self._num_features_to_extract = 0
        return self._num_features_to_extract    

    @property 
    def far_points_threshold(self):
        if not hasattr(self, '_far_points_threshold'):
            if 'Matching.farPointsThreshold' in self.system_settings:
                self._far_points_threshold = self.system_settings['Matching.farPointsThreshold']
            else:
                self._far_points_threshold = None
        return self._far_points_threshold
    
    @property
    def use_fov_centers_based_kf_generation(self):
        if not hasattr(self, '_use_fov_centers_based_kf_generation'):
            self._use_fov_centers_based_kf_generation = False
            if 'KeyFrame.useFovCentersBasedGeneration' in self.system_settings:
                self._use_fov_centers_based_kf_generation = bool(self.system_settings['KeyFrame.useFovCentersBasedGeneration'])
        return self._use_fov_centers_based_kf_generation
    
    @property
    def max_fov_centers_distance(self):
        if not hasattr(self, '_max_fov_centers_distance'):
            self._max_fov_centers_distance = -1
            if 'KeyFrame.maxFovCentersDistance' in self.system_settings:
                self._max_fov_centers_distance = self.system_settings['KeyFrame.maxFovCentersDistance']
        return self._max_fov_centers_distance
    
    # stereo settings 
    @property
    def cam_stereo_settings(self):
        if not hasattr(self, '_cam_stereo_settings'):
            self._cam_stereo_settings = None
            left, right = {}, {}
            if 'LEFT.D' in self.cam_settings:
                left_D = self.cam_settings['LEFT.D']
                left_D = np.array(left_D['data'],dtype=float).reshape(left_D['rows'], left_D['cols'])
                left['D'] = left_D
            if 'LEFT.K' in self.cam_settings:
                left_K = self.cam_settings['LEFT.K']
                left_K = np.array(left_K['data'],dtype=float).reshape(left_K['rows'], left_K['cols'])
                left['K'] = left_K
            if 'LEFT.R' in self.cam_settings:
                left_R = self.cam_settings['LEFT.R']
                left_R = np.array(left_R['data'],dtype=float).reshape(left_R['rows'], left_R['cols'])
                left['R'] = left_R
            if 'LEFT.P' in self.cam_settings:
                left_P = self.cam_settings['LEFT.P']
                left_P = np.array(left_P['data'],dtype=float).reshape(left_P['rows'], left_P['cols'])
                left['P'] = left_P
                
            if 'RIGHT.D' in self.cam_settings:
                right_D = self.cam_settings['RIGHT.D']
                right_D = np.array(right_D['data'],dtype=float).reshape(right_D['rows'], right_D['cols'])
                right['D'] = right_D
            if 'RIGHT.K' in self.cam_settings:
                right_K = self.cam_settings['RIGHT.K']
                right_K = np.array(right_K['data'],dtype=float).reshape(right_K['rows'], right_K['cols'])
                right['K'] = right_K
            if 'RIGHT.R' in self.cam_settings:
                right_R = self.cam_settings['RIGHT.R']
                right_R = np.array(right_R['data'],dtype=float).reshape(right_R['rows'], right_R['cols'])
                right['R'] = right_R 
            if 'RIGHT.P' in self.cam_settings:
                right_P = self.cam_settings['RIGHT.P']
                right_P = np.array(right_P['data'],dtype=float).reshape(right_P['rows'], right_P['cols'])
                right['P'] = right_P         
                   
            if len(left) > 0 and len(right) > 0:
                self._cam_stereo_settings = {'left':left, 'right':right}
        #print(f'[config] stereo settings: {self._cam_stereo_settings}')
        return self._cam_stereo_settings

   
if __name__ != "__main__":
    # we automatically read lib path when this file is called via 'import'
    cfg = Config()
