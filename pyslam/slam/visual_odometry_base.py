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
import cv2
import platform
from enum import Enum

from pyslam.utilities.geometry import poseRt, inv_poseRt, xyzq2Tmat

from pyslam.utilities.timer import TimerFps
from pyslam.io.ground_truth import GroundTruth
from pyslam.io.dataset_types import SensorType
from pyslam.slam import Camera


# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .camera import Camera


class VoState(Enum):
    NO_IMAGES_YET = 0  # no image received
    GOT_FIRST_IMAGE = 1  # got first image, we can proceed in a normal way (match current image with previous image)


kVerbose = True


class VisualOdometryBase:
    def __init__(self, cam: Camera, groundtruth: GroundTruth):
        self.state = VoState.NO_IMAGES_YET
        self.cam = cam
        self.sensor_type = cam.sensor_type if cam is not None else SensorType.MONOCULAR

        self.cur_image = None  # current image
        self.cur_image_right = None  # current right image (if stereo)
        self.cur_depth = None  # current depth image
        self.cur_timestamp = None

        self.prev_image = None  # previous/reference image
        self.prev_image_right = None  # previous/reference right image (if stereo)
        self.prev_depth = None  # previous/reference depth image
        self.prev_timestamp = None

        self.cur_R = np.eye(3, 3)  # current rotation Rwc
        self.cur_t = np.zeros((3, 1))  # current translation twc

        self.gt_x, self.gt_y, self.gt_z = None, None, None
        self.gt_qx, self.gt_qy, self.gt_qz, self.gt_qw = None, None, None, None
        self.gt_T = None
        self.gt_timestamp, self.gt_scale = None, None
        self.groundtruth = groundtruth

        self.track_result = None

        self.mask_match = None  # mask of matched keypoints used for drawing
        self.draw_img = None

        self.num_matched_kps = 0  # current number of matched keypoints
        self.num_inliers = 0  # current number of inliers

        self.init_history = True
        self.poses = []  # history of poses
        self.pose_timestamps = []  # history of pose timestamps
        self.t0_est = None  # history of estimated translations
        self.t0_gt = None  # history of ground truth translations (if available)
        self.traj3d_est = []  # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = (
            []
        )  # history of estimated ground truth translations centered w.r.t. first one

        self.timer_verbose = False  # set this to True if you want to print timings
        self.timer_main = TimerFps("VO", is_verbose=self.timer_verbose)

    # get current translation scale from ground-truth if groundtruth is not None
    def update_gt_data(self, frame_id):
        if self.groundtruth is not None:
            (
                self.gt_timestamp,
                self.gt_x,
                self.gt_y,
                self.gt_z,
                self.gt_qx,
                self.gt_qy,
                self.gt_qz,
                self.gt_qw,
                self.gt_scale,
            ) = self.groundtruth.getTimestampPoseAndAbsoluteScale(frame_id)
            self.gt_T = xyzq2Tmat(
                self.gt_x, self.gt_y, self.gt_z, self.gt_qx, self.gt_qy, self.gt_qz, self.gt_qw
            )
        else:
            self.gt_x = 0
            self.gt_y = 0
            self.gt_z = 0

    def process_first_frame(self, frame_id) -> None:
        pass

    def process_frame(self, frame_id) -> None:
        pass

    def track(self, img, img_right, depth, frame_id, timestamp) -> None:
        if kVerbose:
            print("..................................")
            print(f"frame: {frame_id}, timestamp: {timestamp}")
        # check coherence of image size with camera settings
        assert (
            img.shape[0] == self.cam.height and img.shape[1] == self.cam.width
        ), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.cur_image = img
        self.cur_image_right = img_right
        self.cur_depth = depth
        self.cur_timestamp = timestamp
        # manage and check stage
        if self.state == VoState.GOT_FIRST_IMAGE:
            self.process_frame(frame_id)
            self.update_history()
        elif self.state == VoState.NO_IMAGES_YET:
            self.process_first_frame(frame_id)
            self.state = VoState.GOT_FIRST_IMAGE
        self.prev_image = self.cur_image
        self.prev_image_right = self.cur_image_right
        self.prev_depth = self.cur_depth
        self.prev_timestamp = self.cur_timestamp
        # update main timer (for profiling)
        self.timer_main.refresh()

    def update_history(self) -> None:
        if self.init_history and (self.gt_x is not None):
            self.t0_est = np.array(
                [self.cur_t[0], self.cur_t[1], self.cur_t[2]]
            )  # starting translation
            self.t0_gt = np.array([self.gt_x, self.gt_y, self.gt_z])  # starting translation
            self.T0_inv_est = inv_poseRt(self.cur_R, self.cur_t.ravel())
            self.T0_inv_gt = inv_poseRt(self.gt_T[:3, :3], self.gt_T[:3, 3])
            self.init_history = False
        if (self.t0_est is not None) and (self.t0_gt is not None):

            # p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            cur_T = self.T0_inv_est @ poseRt(self.cur_R, self.cur_t.ravel())
            p = cur_T[:3, 3].ravel()
            self.traj3d_est.append(p)

            # pg = [self.gt_x-self.t0_gt[0], self.gt_y-self.t0_gt[1], self.gt_z-self.t0_gt[2]]  # the groudtruth traj starts at 0
            gt_T = self.T0_inv_gt @ self.gt_T
            pg = gt_T[:3, 3].ravel()
            self.traj3d_gt.append(pg)

            self.poses.append(poseRt(self.cur_R, np.array(p).ravel()))
            self.pose_timestamps.append(self.cur_timestamp)
            # self.poses.append(poseRt(self.cur_R, p[0]))
