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
    
import configparser
import os
import yaml
import numpy as np


# N.B.: this file must stay in the root folder of the repository 


# get the folder location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Class for getting libs settings (from config.ini) and camera settings from a yaml file 
class Config(object):
    '''
    Config is used for getting libs settings (from config.ini) and camera settings from a yaml file 
    '''
    def __init__(self):
        self.root_folder = __location__
        self.config_file = 'config.ini'
        self.config_parser = configparser.ConfigParser()
        self.cam_settings = None
        self.dataset_settings = None
        self.dataset_type = None
        #self.current_path = os.getcwd()
        #print('current path: ', self.current_path)

        self.config_parser.read(__location__ + '/' + self.config_file)
        self.set_core_lib_paths()
        self.read_lib_paths()
        
        self.get_dataset_settings()
        self.get_cam_settings()

    # read core lib paths from config.ini and set sys paths
    def set_core_lib_paths(self):
        self.core_lib_paths = self.config_parser['CORE_LIB_PATHS']
        for path in self.core_lib_paths:
            ext_path = __location__ + '/' + self.core_lib_paths[path]
            # print( "importing path: ", ext_path )
            sys.path.append(ext_path)
            
    # read lib paths from config.ini 
    def read_lib_paths(self):
        self.lib_paths = self.config_parser['LIB_PATHS']
        
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
        self.dataset_type = self.config_parser['DATASET']['type']
        self.dataset_settings = self.config_parser[self.dataset_type]

        self.dataset_path = self.dataset_settings['base_path'];
        self.dataset_settings['base_path'] = os.path.join( __location__, self.dataset_path)
        #print('dataset_settings: ', self.dataset_settings)

    # get camera settings
    def get_cam_settings(self):
        self.cam_settings = None
        self.settings_doc = __location__ + '/' + self.config_parser[self.dataset_type]['cam_settings']
        if(self.settings_doc is not None):
            with open(self.settings_doc, 'r') as stream:
                try:
                    self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)

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
            # if k3 != 0:
            #     self._DistCoef = np.array([k1,k2,p1,p2,k3])
            # else:
            #     self._DistCoef = np.array([k1,k2,p1,p2])
        return self._DistCoef

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


if __name__ != "__main__":
    # we automatically read lib path when this file is called via 'import'
    cfg = Config()
