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

from enum import Enum
import numpy as np 
import cv2
import math 

#import json
import ujson as json

#import g2o
from utils_geom import add_ones
from utils_sys import Printer


class CameraTypes(Enum):
    NONE = 0
    PINHOLE = 1


# Convert fov [rad] to focal length in pixels
def fov2focal(fov, pixels):
    return float(pixels) / (2 * math.tan(fov / 2.0))

# Convert focal length in pixels to fov [rad]
def focal2fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))
    
        
class CameraBase:
    def __init__(self):
        self.type = CameraTypes.NONE
        self.width, self.height = None, None
        self.fx, self.fy = None, None
        self.cx, self.cy = None, None
        
        self.D = None
        self.is_distorted = None
        
        self.fps = None
        
        self.bf = None
        self.b = None        

        self.u_min = None 
        self.u_max = None
        self.v_min = None
        self.v_max = None
        self.initialized = False   


class Camera(CameraBase): 
    def __init__(self, config):
        super().__init__()
        if config is None:
            return 
        width = config.cam_settings['Camera.width']
        height = config.cam_settings['Camera.height']
        fx = config.cam_settings['Camera.fx']
        fy = config.cam_settings['Camera.fy']
        cx = config.cam_settings['Camera.cx']
        cy = config.cam_settings['Camera.cy']
        D = config.DistCoef # D = [k1, k2, p1, p2, k3]
        fps = config.cam_settings['Camera.fps']

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.fovx = focal2fov(fx, width)
        self.fovy = focal2fov(fy, height)
    
        self.D = np.array(D,dtype=float) # np.array([k1, k2, p1, p2, k3])  distortion coefficients 
        self.is_distorted = np.linalg.norm(self.D) > 1e-10
            
        self.fps = fps 
    
        # If stereo camera => assuming rectified images as input at present (so no need of left-right transformation matrix Tlr)     
        if 'Camera.bf' in config.cam_settings:
            self.bf = config.cam_settings['Camera.bf']
            self.b = self.bf/self.fx
        if config.sensor_type == 'stereo' and self.bf is None:
            raise ValueError('Expecting the field Camera.bf in the camera config file')
        self.depth_factor = 1.0 # Depthmap values factor 
        if 'DepthMapFactor' in config.cam_settings:
            self.depth_factor = 1.0/float(config.cam_settings['DepthMapFactor'])
            #print('Using DepthMapFactor = %f' % self.depth_factor)
        if config.sensor_type == 'rgbd' and self.depth_factor is None:
            raise ValueError('Expecting the field DepthMapFactor in the camera config file')            
        self.depth_threshold = None  # Close/Far threshold. Baseline times.
        if 'ThDepth' in config.cam_settings:
            depth_threshold = float(config.cam_settings['ThDepth'])
            assert(self.bf is not None)
            self.depth_threshold = self.bf * depth_threshold / self.fx 
            #print('Using depth_threshold = %f' % self.depth_threshold)
        if (config.sensor_type == 'rgbd' or config.sensor_type == 'stereo') and self.depth_threshold is None:
            raise ValueError('Expecting the field ThDepth in the camera config file')              
        
  
    def is_stereo(self):
        return self.bf is not None

    def to_json(self):
        return {'type':int(self.type.value),
                'width':int(self.width), 
                'height':int(self.height), 
                'fx':float(self.fx), 
                'fy':float(self.fy), 
                'cx':float(self.cx), 
                'cy':float(self.cy), 
                'D':json.dumps(self.D.astype(float).tolist() if self.D is not None else None), 
                'fps':int(self.fps),
                'bf':float(self.bf),
                'b':float(self.b),
                'depth_factor':float(self.depth_factor),
                'depth_threshold':float(self.depth_threshold),
                'is_distorted':bool(self.is_distorted),
                'u_min':float(self.u_min),
                'u_max':float(self.u_max),
                'v_min':float(self.v_min),
                'v_max':float(self.v_max),
                'initialized':bool(self.initialized)
                }
        
    def init_from_json(self, json_str):
        self.type = CameraTypes(int(json_str['type']))
        self.width = int(json_str['width'])
        self.height = int(json_str['height'])
        self.fx = float(json_str['fx'])
        self.fy = float(json_str['fy'])
        self.cx = float(json_str['cx'])
        self.cy = float(json_str['cy'])
        self.D = np.array(json.loads(json_str['D'])) if json_str['D'] is not None else None
        self.fps = int(json_str['fps'])
        self.bf = float(json_str['bf'])
        self.b = float(json_str['b'])
        self.depth_factor = float(json_str['depth_factor'])
        self.depth_threshold = float(json_str['depth_threshold'])
        self.is_distorted = bool(json_str['is_distorted'])
        self.u_min = float(json_str['u_min'])
        self.u_max = float(json_str['u_max'])
        self.v_min = float(json_str['v_min'])
        self.v_max = float(json_str['v_max'])
        self.initialized = bool(json_str['initialized'])
        if not hasattr(self, 'fovx'):
            self.fovx = focal2fov(self.fx, self.width)
        if not hasattr(self, 'fovy'):   
            self.fovy = focal2fov(self.fy, self.height)
        
    def is_in_image(self, uv, z):
        return (uv[0] > self.u_min) & (uv[0] < self.u_max) & \
               (uv[1] > self.v_min) & (uv[1] < self.v_max) & \
               (z > 0)         
                
    # input: [Nx2] array of uvs, [Nx1] of zs 
    # output: [Nx1] array of visibility flags             
    def are_in_image(self, uvs, zs):
        return (uvs[:, 0] > self.u_min) & (uvs[:, 0] < self.u_max) & \
               (uvs[:, 1] > self.v_min) & (uvs[:, 1] < self.v_max) & \
               (zs > 0 )

    # Get the projection matrix for rendering
    def get_render_projection_matrix(self, znear=0.01, zfar=100.0):
        W, H = self.width, self.height
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        left = ((2 * cx - W) / W - 1.0) * W / 2.0
        right = ((2 * cx - W) / W + 1.0) * W / 2.0
        top = ((2 * cy - H) / H + 1.0) * H / 2.0
        bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
        left = znear / fx * left
        right = znear / fx * right
        top = znear / fy * top
        bottom = znear / fy * bottom
        P = np.zeros((4, 4), dtype=float)
        z_sign = 1.0
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    # Set the camera's horizontal field of view [rad] and the corresponding horizontal focal length
    def set_fovx(self, fovx):
        self.fx = fov2focal(fovx, self.width)
        self.fovx = fovx
        
    # Set the camera's vertical field of view [rad] and the corresponding vertical focal length
    def set_fovy(self, fovy):
        self.fy = fov2focal(fovy, self.height)
        self.fovy = fovy
        

class PinholeCamera(Camera):
    def __init__(self, config=None):
        super().__init__(config)
        self.type = CameraTypes.PINHOLE
        
        if config is None:
            return 
        
        fx = self.fx 
        fy = self.fy
        cx = self.cx
        cy = self.cy
        self.K = np.array([[fx, 0,cx],
                           [ 0,fy,cy],
                           [ 0, 0, 1]])
        self.Kinv = np.array([[1/fx,    0,-cx/fx],
                              [   0, 1/fy,-cy/fy],
                              [   0,    0,    1]])             
        
        #print(f'PinholeCamera: K = {self.K}')
        self.u_min, self.u_max = 0, self.width 
        self.v_min, self.v_max = 0, self.height       
        self.init()    
        
    def to_json(self):
        camera_json = super().to_json()
        camera_json['K'] = json.dumps(self.K.astype(float).tolist())
        camera_json['Kinv'] = json.dumps(self.Kinv.astype(float).tolist())
        return camera_json
    
    @staticmethod
    def from_json(json_str):
        c = PinholeCamera(None)
        c.init_from_json(json_str)
        c.K = np.array(json.loads(json_str['K']))
        c.Kinv = np.array(json.loads(json_str['Kinv']))
        return c
    
    def init(self):
        if not self.initialized:
            self.initialized = True 
            self.undistort_image_bounds()        
                
    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths     
    def project(self, xcs):
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy  
        projs = self.K @ xcs.T
        zs = projs[-1]      
        projs = projs[:2]/ zs   
        return projs.T, zs
    
    # stereo-project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # (assuming rectified stereo images)
    # out: Nx3 image points, [Nx1] array of map point depths     
    def project_stereo(self, xcs):
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy  
        # ur = u - bf//xc[2]
        projs = self.K @ xcs.T     
        zs = projs[-1]      
        projs = projs[:2]/ zs 
        ur = projs[0] - self.bf/zs
        projs = np.concatenate((projs.T,ur[:, np.newaxis]),axis=1)  
        return projs, zs    
        
    # unproject 2D point uv (pixels on image plane) on 
    def unproject(self, uv):
        x = (uv[0] - self.cx)/self.fx
        y = (uv[1] - self.cy)/self.fy
        return x,y

    # in:  uvs [Nx2]
    # out: xcs array [Nx2] of normalized coordinates     
    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]        

    # in:  uvs [Nx2], depths [Nx1]
    # out: xcs array [Nx3] of normalized coordinates     
    def unproject_points_3d(self, uvs, depths):
        return np.dot(self.Kinv, add_ones(uvs).T * depths).T[:, 0:3]    

    # in:  uvs [Nx2]
    # out: uvs_undistorted array [Nx2] of undistorted coordinates  
    def undistort_points(self, uvs):
        if self.is_distorted:
            #uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)   # =>  Error: while undistorting the points error: (-215:Assertion failed) src.isContinuous() 
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)            
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs 
        
    # update image bounds     
    def undistort_image_bounds(self):
        uv_bounds = np.array([[self.u_min, self.v_min],
                                [self.u_min, self.v_max],
                                [self.u_max, self.v_min],
                                [self.u_max, self.v_max]], dtype=np.float32).reshape(4,2)
        #print('uv_bounds: ', uv_bounds)
        if self.is_distorted:
                uv_bounds_undistorted = cv2.undistortPoints(np.expand_dims(uv_bounds, axis=1), self.K, self.D, None, self.K)      
                uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(uv_bounds_undistorted.shape[0], 2)
        else:
            uv_bounds_undistorted = uv_bounds 
        #print('uv_bounds_undistorted: ', uv_bounds_undistorted)                
        self.u_min = min(uv_bounds_undistorted[0][0],uv_bounds_undistorted[1][0])
        self.u_max = max(uv_bounds_undistorted[2][0],uv_bounds_undistorted[3][0])        
        self.v_min = min(uv_bounds_undistorted[0][1],uv_bounds_undistorted[2][1])    
        self.v_max = max(uv_bounds_undistorted[1][1],uv_bounds_undistorted[3][1])  
        # print('camera u_min: ', self.u_min)
        # print('camera u_max: ', self.u_max)
        # print('camera v_min: ', self.v_min)         
        # print('camera v_max: ', self.v_max)      