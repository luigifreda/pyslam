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

import numpy as np
import g2o


# camera pose representation by using g2o.Isometry3d()
class CameraPose(object):
    def __init__(self, pose=None):
        if pose is None:
            pose = g2o.Isometry3d()
        self.set(pose)
        self.covariance = np.identity(6, dtype=np.float64)  # pose covariance

    def copy(self):
        return CameraPose(self._pose.copy())

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Replace the unpickable g2o.Isometry3d with a picklable matrix
        if "_pose" in state:
            state["_pose"] = self._pose.matrix()
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        self._pose = g2o.Isometry3d(self._pose)  # set back to g2o.Isometry3d

    # input pose_cw is expected to be an g2o.Isometry3d
    def set(self, pose):
        if isinstance(pose, g2o.SE3Quat) or isinstance(pose, g2o.Isometry3d):
            self._pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self._pose = g2o.Isometry3d(pose)  # g2o.Isometry3d
        self.set_mat(self._pose.matrix())

    # input pose_cw is expected to be an g2o.Isometry3d
    def update(self, pose):
        self.set(pose)

    def set_mat(self, Tcw):
        self.Tcw = np.ascontiguousarray(
            Tcw
        )  # homogeneous transformation matrix: (4, 4)   pc_ = Tcw * pw_
        self.Rcw = np.ascontiguousarray(self.Tcw[:3, :3])
        self.tcw = np.ascontiguousarray(self.Tcw[:3, 3])  #  pc = Rcw * pw + tcw
        self.Rwc = np.ascontiguousarray(self.Rcw.T)
        self.Ow = np.ascontiguousarray(-(self.Rwc @ self.tcw))  # origin of camera frame w.r.t world

    def update_mat(self, Tcw):
        self.set_matrix(Tcw)

    @property
    def isometry3d(self):  # pose as g2o.Isometry3d
        return self._pose

    @property
    def quaternion(self):  # g2o.Quaternion(),  quaternion_cw
        return self._pose.orientation()

    @property
    def orientation(self):  # g2o.Quaternion(),  quaternion_cw
        return self._pose.orientation()

    @property
    def position(self):  # 3D vector tcw (world origin w.r.t. camera frame)
        return self._pose.position()

    def get_rotation_matrix(self):
        return self._pose.rotation_matrix()

    def get_rotation_angle_axis(self):
        angle_axis = g2o.AngleAxis(self._pose.orientation())
        # angle = angle_axis.angle()
        # axis = angle_axis.axis()
        return angle_axis

    def get_matrix(self):
        return self._pose.matrix()

    def get_inverse_matrix(self):
        return self._pose.inverse().matrix()

    # set from orientation (g2o.Quaternion()) and position (3D vector)
    def set_from_quaternion_and_position(self, quaternion, position):
        self.set(g2o.Isometry3d(quaternion, position))

    # set from 4x4 homogeneous transformation matrix Tcw  (pc_ = Tcw * pw_)
    def set_from_matrix(self, Tcw):
        self.set(g2o.Isometry3d(Tcw))

    def set_from_rotation_and_translation(self, Rcw, tcw):
        self.set(g2o.Isometry3d(g2o.Quaternion(Rcw), tcw))

    def set_quaternion(self, quaternion):
        self.set(g2o.Isometry3d(quaternion, self._pose.position()))

    def set_rotation_matrix(self, Rcw):
        self.set(g2o.Isometry3d(g2o.Quaternion(Rcw), self._pose.position()))

    def set_translation(self, tcw):
        self.set(g2o.Isometry3d(self._pose.orientation(), tcw))
