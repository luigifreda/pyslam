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
import open3d as o3d
import open3d.core as o3c
import cv2

from pyslam.utilities.timer import TimerFps
from pyslam.utilities.logging import Printer
from pyslam.utilities.geometry import poseRt, inv_poseRt

from pyslam.io.ground_truth import GroundTruth
from .visual_odometry_base import VoState, VisualOdometryBase
from pyslam.slam import Camera
from pyslam.config_parameters import Parameters

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .camera import Camera

kVerbose = True


class VisualOdometryRgbdBase:
    def __init__(self, cam: Camera):
        self.depth_factor = cam.depth_factor

        self.prev_rgbd = None
        self.cur_rgbd = None

        # Prepare maps to undistort color and depth images
        h, w = cam.height, cam.width
        D = cam.D
        K = cam.K
        # Printer.green(f'VisualOdometryRgbdBase: init: h={h}, w={w}, D={D}, K={K}')

        # Ensure D is a numpy array with proper shape for OpenCV
        if D is not None:
            D = np.array(D, dtype=np.float64).flatten()
        else:
            D = np.array([0, 0, 0, 0, 0], dtype=np.float64)

        # Ensure K is a numpy array
        K = np.array(K, dtype=np.float64)

        if np.linalg.norm(D) <= 1e-10:
            self.new_K = K
            self.calib_map1 = None
            self.calib_map2 = None
        else:
            # print(f'VolumetricIntegratorBase: init: D={D} => undistort-rectify maps')
            if Parameters.kDepthImageUndistortionUseOptimalNewCameraMatrixWithAlphaScale:
                alpha = Parameters.kDepthImageUndistortionOptimalNewCameraMatrixWithAlphaScaleValue
                self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            else:
                self.new_K = K
            self.calib_map1, self.calib_map2 = cv2.initUndistortRectifyMap(
                K, D, None, self.new_K, (w, h), cv2.CV_32FC1
            )

        # Store rectified camera intrinsics for use when depth is rectified
        # Extract fx, fy, cx, cy from new_K
        self.rectified_fx = float(self.new_K[0, 0])
        self.rectified_fy = float(self.new_K[1, 1])
        self.rectified_cx = float(self.new_K[0, 2])
        self.rectified_cy = float(self.new_K[1, 2])

    def rectify_in_needed(self, color, depth):
        if self.depth_factor != 1.0:
            depth = depth * self.depth_factor

        if color.ndim == 2:
            color = cv2.cvtColor(self.cur_image, cv2.COLOR_GRAY2RGB)

        if not depth.dtype in [np.uint8, np.uint16, np.float32]:
            depth = depth.astype(np.float32)

        if self.calib_map1 is not None and self.calib_map2 is not None:
            color_undistorted = cv2.remap(
                color, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_LINEAR
            )
            depth_undistorted = cv2.remap(
                depth, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_NEAREST
            )
        else:
            color_undistorted = color
            depth_undistorted = depth

        color_undistorted = cv2.cvtColor(color_undistorted, cv2.COLOR_BGR2RGB)
        return color_undistorted, depth_undistorted


# Open3D has implemented two RGBD odometries: [Steinbrucker2011] and [Park2017].
# F.Steinbrucker, J. Sturm, and D. Cremers, "Real-time visual odometry from dense RGB-D images", In ICCV Workshops, 2011.
# J.Park, Q.-Y. Zhou, and V. Koltun, "Colored Point Cloud Registration Revisited", ICCV, 2017.
class VisualOdometryRgbd(VisualOdometryBase, VisualOdometryRgbdBase):
    def __init__(self, cam, groundtruth: GroundTruth):
        VisualOdometryBase.__init__(self, cam, groundtruth)
        VisualOdometryRgbdBase.__init__(self, cam)

        self.option = o3d.pipelines.odometry.OdometryOption()
        print(f"VisualOdometryRgbd: option: {self.option}")

        # Use rectified intrinsics if depth rectification is enabled
        if self.calib_map1 is not None and self.calib_map2 is not None:
            fx, fy, cx, cy = (
                self.rectified_fx,
                self.rectified_fy,
                self.rectified_cx,
                self.rectified_cy,
            )
        else:
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

        self.o3d_camera = o3d.camera.PinholeCameraIntrinsic(
            width=cam.width, height=cam.height, fx=fx, fy=fy, cx=cx, cy=cy
        )
        self.timer_pose_est = TimerFps("PoseEst", is_verbose=self.timer_verbose)

    def process_first_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()

        if self.cur_depth is None:
            message = "Depth image is None, are you using a dataset with depth images?"
            Printer.error(message)
            raise ValueError(message)
        color_undistorted, depth_undistorted = self.rectify_in_needed(
            self.cur_image, self.cur_depth
        )
        self.prev_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_undistorted),
            o3d.geometry.Image(depth_undistorted),
            depth_scale=1.0,
        )

    def process_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()

        self.update_gt_data(frame_id)

        color_undistorted, depth_undistorted = self.rectify_in_needed(
            self.cur_image, self.cur_depth
        )
        self.cur_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_undistorted),
            o3d.geometry.Image(depth_undistorted),
            depth_scale=1.0,
        )

        init_transform = np.eye(4)

        self.timer_pose_est.start()
        success, rel_transform, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.prev_rgbd,
            self.cur_rgbd,
            self.o3d_camera,
            init_transform,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            self.option,
        )
        self.prev_rgbd = self.cur_rgbd
        self.timer_pose_est.refresh()

        print(
            f"VisualOdometryRgbd: success: {success}"
        )  # , rel_transform: {rel_transform}') #, info: {info}')

        # pose = poseRt(rel_transform[:3, :3], rel_transform[:3, 3])
        inv_pose = inv_poseRt(rel_transform[:3, :3], rel_transform[:3, 3])

        R, t = inv_pose[:3, :3], inv_pose[:3, 3]
        t = np.array(t).reshape(3, 1)

        self.cur_R = self.cur_R @ R
        self.cur_t = self.cur_t + self.cur_R @ t


# Open3D has implemented two RGBD odometries: [Steinbrucker2011] and [Park2017].
# F.Steinbrucker, J. Sturm, and D. Cremers, "Real-time visual odometry from dense RGB-D images", In ICCV Workshops, 2011.
# J.Park, Q.-Y. Zhou, and V. Koltun, "Colored Point Cloud Registration Revisited", ICCV, 2017.
# This version implements the odometry using the tensor version of Open3D
class VisualOdometryRgbdTensor(VisualOdometryBase, VisualOdometryRgbdBase):
    def __init__(self, cam, groundtruth: GroundTruth, method_name="hybrid", device="cuda"):
        VisualOdometryBase.__init__(self, cam, groundtruth)
        VisualOdometryRgbdBase.__init__(self, cam)

        device = "CUDA:0" if device == "cuda" and o3d.core.cuda.is_available() else "CPU:0"
        print(f"VisualOdometryRgbdTensor: device: {device}")
        self.device = o3c.Device(device)

        # Use rectified intrinsics if depth rectification is enabled
        if self.calib_map1 is not None and self.calib_map2 is not None:
            # Create rectified K matrix from rectified intrinsics
            rectified_K = np.array(
                [
                    [self.rectified_fx, 0, self.rectified_cx],
                    [0, self.rectified_fy, self.rectified_cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            self.intrinsics = o3d.core.Tensor(rectified_K, o3d.core.Dtype.Float64)
        else:
            self.intrinsics = o3d.core.Tensor(cam.K, o3d.core.Dtype.Float64)
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(max_iteration=20),
        ]
        self.setup_method(method_name)
        self.max_depth = 10.0
        self.depth_scale = 1.0

        self.timer_pose_est = TimerFps("PoseEst", is_verbose=self.timer_verbose)

    def setup_method(self, method_name: str) -> None:
        if method_name == "hybrid":
            self.method = o3d.t.pipelines.odometry.Method.Hybrid
        elif method_name == "point_to_plane":
            self.method = o3d.t.pipelines.odometry.Method.PointToPlane
        else:
            raise ValueError("Odometry method does not exist!")

    def process_first_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()

        if self.cur_depth is None:
            message = "Depth image is None, are you using a dataset with depth images?"
            Printer.error(message)
            raise ValueError(message)
        color_undistorted, depth_undistorted = self.rectify_in_needed(
            self.cur_image, self.cur_depth
        )
        self.prev_rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color_undistorted).to(self.device),
            o3d.t.geometry.Image(depth_undistorted).to(self.device),
        )

    def process_frame(self, frame_id) -> None:
        self.draw_img = self.cur_image.copy()

        self.update_gt_data(frame_id)

        color_undistorted, depth_undistorted = self.rectify_in_needed(
            self.cur_image, self.cur_depth
        )
        self.cur_rgbd = o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color_undistorted).to(self.device),
            o3d.t.geometry.Image(depth_undistorted).to(self.device),
        )

        init_transform = np.eye(4)

        self.timer_pose_est.start()
        rel_transform = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            self.prev_rgbd,
            self.cur_rgbd,
            self.intrinsics,
            o3c.Tensor(init_transform),
            self.depth_scale,
            self.max_depth,
            self.criteria_list,
            self.method,
        )

        print(f"VisualOdometryRgbd: rel_transform: {rel_transform}")
        rel_transform = rel_transform.transformation.cpu().numpy()

        self.prev_rgbd = self.cur_rgbd.clone()
        self.timer_pose_est.refresh()

        inv_pose = inv_poseRt(rel_transform[:3, :3], rel_transform[:3, 3])

        R, t = inv_pose[:3, :3], inv_pose[:3, 3]
        t = np.array(t).reshape(3, 1)

        self.cur_R = self.cur_R @ R
        self.cur_t = self.cur_t + self.cur_R @ t
