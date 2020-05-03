"""
* This file is part of PYSLAM 
* This file contains a revised and fixed version of the class in https://github.com/uoip/stereo_ptam/blob/master/motion.py
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

import numpy as np 
import g2o


class MotionModelBase(object):
    def __init__(self, 
            timestamp=None, 
            initial_position=None, 
            initial_orientation=None, 
            initial_covariance=None):

        self.timestamp = timestamp
        if initial_position is not None: 
            self.position = initial_position
        else:
            self.position = np.zeros(3)
        if initial_orientation is not None: 
            self.orientation = initial_orientation
        else:
            self.orientation = g2o.Quaternion()            
        self.orientation = initial_orientation
        self.covariance = initial_covariance    # pose covariance

        self.is_ok = False 
        self.initialized = False

    def current_pose(self):
        '''
        Get the current camera pose.
        '''
        return (g2o.Isometry3d(self.orientation, self.position), self.covariance)

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        return None 

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        return None 

    # correction= Tcw_old.inverse() * Tcw_new  (transform from world_new to worl_old)
    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        return None 


# simple kinematic motion model without damping (does not actually use timestamps)
class MotionModel(MotionModelBase):
    def __init__(self, 
            timestamp=None, 
            initial_position=None, 
            initial_orientation=None, 
            initial_covariance=None):
        super().__init__(timestamp, initial_position, initial_orientation, initial_covariance)
        
        self.delta_position = np.zeros(3)    # delta translation 
        self.delta_orientation = g2o.Quaternion()

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        '''
        Predict the next camera pose.
        '''
        if prev_position is not None:
            self.position = prev_position
        if prev_orientation is not None:
            self.orientation = prev_orientation            
        
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), self.covariance)
        
        orientation = self.delta_orientation * self.orientation 
        position = self.position + self.delta_orientation * self.delta_position        

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            self.delta_position    = new_position - self.position            
            self.delta_orientation = new_orientation * self.orientation.inverse()
            self.delta_orientation.normalize()
                        
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation                        
        self.covariance = new_covariance
        self.initialized = True

    # correction= Tcw_corrected * Tcw_uncorrected.inverse()  (transform from camera_uncorrected to camera_corrected)
    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        '''
        Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)
        '''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)

        current = g2o.Isometry3d(self.orientation, self.position)
        current = correction * current

        self.position = current.position()
        self.orientation = current.orientation()

        # correction= Tcw_corrected * Tcw_uncorrected.inverse()  (transform from camera_uncorrected to camera_corrected)
        self.delta_orientation = correction.orientation() * self.delta_orientation    
        self.delta_position = correction.orientation() * self.delta_position


# motion model with damping 
class MotionModelDamping(MotionModelBase):
    def __init__(self, 
            timestamp=None, 
            initial_position=None, 
            initial_orientation=None, 
            initial_covariance=None,
            damping=0.95):
        super().__init__(timestamp, initial_position, initial_orientation, initial_covariance)        

        self.v_linear = np.zeros(3)    # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])
                
        self.damp = damping         # damping factor

    def predict_pose(self, timestamp, prev_position=None, prev_orientation=None):
        '''
        Predict the next camera pose.
        '''
        if prev_position is not None:
            self.position = prev_position
        if prev_orientation is not None:
            self.orientation = prev_orientation            
                
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), self.covariance)
        
        dt = timestamp - self.timestamp

        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp, 
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)

        orientation = delta_orientation * self.orientation 
        position = self.position + delta_orientation * (self.v_linear * dt * self.damp)        

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            self.v_linear = v_linear

            delta_q = new_orientation * self.orientation.inverse()
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt
        
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation                        
        self.covariance = new_covariance
        self.initialized = True        

    # correction= Tcw_corrected * Tcw_uncorrected.inverse()  (transform from camera_uncorrected to camera_corrected)
    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        '''
        Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)
        '''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)

        current = g2o.Isometry3d(self.orientation, self.position)
        current = correction * current  

        self.position = current.position()
        self.orientation = current.orientation()

        # correction= Tcw_corrected * Tcw_uncorrected.inverse()  (transform from camera_uncorrected to camera_corrected)
        self.v_angular_axis = correction.orientation() * self.v_angular_axis
        self.v_linear = correction.orientation() * self.v_linear