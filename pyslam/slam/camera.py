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

from numba import njit

# import json
import ujson as json

from pyslam.config import Config
from pyslam.utilities.geometry import add_ones, add_ones_numba
from pyslam.utilities.serialization import deserialize_array_flexible
from pyslam.utilities.logging import Printer
from pyslam.io.dataset_types import SensorType, get_sensor_type

from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from pyslam.config import Config


class CameraType(Enum):  # keep it consistent with C++ CameraType
    NONE = 0
    PINHOLE = 1


# Convert fov [rad] to focal length in pixels
def fov2focal(fov, pixels):
    return float(pixels) / (2 * math.tan(fov / 2.0))


# Convert focal length in pixels to fov [rad]
def focal2fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))


class CameraUtils:

    # Backproject 2d image points (pixels) into 3D points by using depth and intrinsic K
    # Input:
    #   uv: array [N,2]
    #   depth: array [N]
    #   K: array [3,3]
    # Output:
    #   xyz: array [N,3]
    @staticmethod
    def backproject_3d(uv, depth, K):
        uv1 = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        p3d = depth.reshape(-1, 1) * (np.linalg.inv(K) @ uv1.T).T
        return p3d.reshape(-1, 3)

    @njit(cache=True)
    def backproject_3d_numba(uv, depth, Kinv):
        N = uv.shape[0]
        uv1 = np.ones((N, 3), dtype=np.float64)
        uv1[:, 0:2] = uv
        p3d = np.empty((N, 3), dtype=np.float64)
        for i in range(N):
            p = Kinv @ uv1[i]
            p3d[i, :] = depth[i] * p
        return p3d

    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths
    def project(xcs, K):  # python version
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy
        projs = K @ xcs.T
        zs = projs[-1]
        projs = projs[:2] / zs
        return projs.T, zs

    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths
    @njit(cache=True)
    def project_numba(xcs, K):  # numba-optimized version
        N = xcs.shape[0]
        projs = K @ xcs.T  # shape (3, N)
        zs = projs[2, :]  # shape (N,)
        u = projs[0, :] / zs
        v = projs[1, :] / zs
        uv = np.empty((N, 2), dtype=np.float64)
        for i in range(N):
            uv[i, 0] = u[i]
            uv[i, 1] = v[i]
        return uv, zs

    # stereo-project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # (assuming rectified stereo images)
    # out: Nx3 image points, [Nx1] array of map point depths
    def project_stereo(xcs, K, bf):  # python version
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy
        # ur = u - bf/xc[2]
        projs = K @ xcs.T
        zs = projs[-1]
        projs = projs[:2] / zs
        ur = projs[0] - bf / zs
        projs = np.concatenate((projs.T, ur[:, np.newaxis]), axis=1)
        return projs, zs

    # stereo-project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # (assuming rectified stereo images)
    # out: Nx3 image points, [Nx1] array of map point depths
    @njit(cache=True)
    def project_stereo_numba(xcs, K, bf):  # numba-optimized version
        N = xcs.shape[0]
        projs = K @ xcs.T  # shape (3, N)
        zs = projs[2, :]  # shape (N,)
        u = projs[0, :] / zs
        v = projs[1, :] / zs
        ur = u - bf / zs
        out = np.empty((N, 3), dtype=np.float64)
        for i in range(N):
            out[i, 0] = u[i]
            out[i, 1] = v[i]
            out[i, 2] = ur[i]
        return out, zs

    # in:  uvs [Nx2]
    # out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points on z=1 plane)
    def unproject_points(uvs, Kinv):  # python version
        return np.dot(Kinv, add_ones(uvs).T).T[:, 0:2]

    # in:  uvs [Nx2]
    # out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points on z=1 plane)
    @njit(cache=True)
    def unproject_points_numba(uvs, Kinv):  # numba-optimized version
        N = uvs.shape[0]
        uv1 = add_ones_numba(uvs)
        out = np.empty((N, 2), dtype=uvs.dtype)
        for i in range(N):
            p = Kinv @ uv1[i]
            out[i, 0] = p[0]
            out[i, 1] = p[1]
        return out

    # in:  uvs [Nx2], depths [Nx1]
    # out: xcs array [Nx3] of backprojected 3D points
    def unproject_points_3d(uvs, depths, Kinv):  # python version
        return np.dot(Kinv, add_ones(uvs).T * depths).T[:, 0:3]

    # in:  uvs [Nx2], depths [Nx1]
    # out: xcs array [Nx3] of backprojected 3D points
    @njit(cache=True)
    def unproject_points_3d_numba(uvs, depths, Kinv):  # numba-optimized version
        N = uvs.shape[0]
        uv1 = add_ones_numba(uvs)
        out = np.empty((N, 3), dtype=uvs.dtype)
        for i in range(N):
            p = Kinv @ (uv1[i] * depths[i])
            out[i, 0] = p[0]
            out[i, 1] = p[1]
            out[i, 2] = p[2]
        return out

    # input: [Nx2] array of uvs, [Nx1] of zs
    # output: [Nx1] array of visibility flags
    @njit(cache=True)
    def are_in_image_numba(uvs, zs, u_min, u_max, v_min, v_max):
        N = uvs.shape[0]
        out = np.empty(N, dtype=np.bool_)
        for i in range(N):
            out[i] = (
                (uvs[i, 0] >= u_min)
                & (uvs[i, 0] < u_max)
                & (uvs[i, 1] >= v_min)
                & (uvs[i, 1] < v_max)
                & (zs[i] > 0)
            )
        return out


class CameraBase:
    def __init__(self):
        self.type = CameraType.NONE
        self.width, self.height = None, None
        self.fx, self.fy = None, None
        self.cx, self.cy = None, None
        self.K, self.Kinv = None, None

        self.D: np.ndarray | None = None
        self.is_distorted = None

        self.fps = None

        self.bf = None
        self.b = None
        self.depth_factor = None
        self.depth_threshold = None

        self.u_min = None
        self.u_max = None
        self.v_min = None
        self.v_max = None
        self.initialized = False


class Camera(CameraBase):
    def __init__(self, config: "Config"):
        super().__init__()
        if config is None:
            return
        if isinstance(config, dict):
            # convert a possibly dict input into Config
            config_ = Config()
            config_.from_json(config)
            config = config_

        width = (
            config.cam_settings["Camera.width"]
            if "Camera.width" in config.cam_settings
            else config.cam_settings["Camera.w"]
        )
        height = (
            config.cam_settings["Camera.height"]
            if "Camera.height" in config.cam_settings
            else config.cam_settings["Camera.h"]
        )
        fx = config.cam_settings["Camera.fx"]
        fy = config.cam_settings["Camera.fy"]
        cx = config.cam_settings["Camera.cx"]
        cy = config.cam_settings["Camera.cy"]
        D = config.DistCoef  # D = [k1, k2, p1, p2, k3]
        fps = config.cam_settings["Camera.fps"]

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if not self.width or not self.height:
            raise ValueError(
                "Camera: Expecting the fields Camera.width and Camera.height in the camera config file"
            )

        self.K: np.ndarray | None = None
        self.Kinv: np.ndarray | None = None
        self.set_intrinsic_matrices()

        self.fovx = focal2fov(fx, width)
        self.fovy = focal2fov(fy, height)

        self.D = np.array(D, dtype=float)  # np.array([k1, k2, p1, p2, k3])  distortion coefficients
        self.is_distorted = np.linalg.norm(self.D) > 1e-10

        self.fps = fps

        sensor_type = config.sensor_type if hasattr(config, "sensor_type") else "mono"
        self.sensor_type = get_sensor_type(sensor_type)
        print(f"Camera: sensor_type = {self.sensor_type}")

        # If stereo camera => assuming rectified images as input at present (so no need of left-right transformation matrix Tlr)
        if "Camera.bf" in config.cam_settings and self.sensor_type != SensorType.MONOCULAR:
            self.bf = config.cam_settings["Camera.bf"]
            self.b = self.bf / self.fx
        if config.sensor_type == SensorType.STEREO and self.bf is None:
            raise ValueError("Camera: Expecting the field Camera.bf in the camera config file")

        self.depth_factor = 1.0  # Depthmap values factor
        if "DepthMapFactor" in config.cam_settings:
            self.depth_factor = 1.0 / float(config.cam_settings["DepthMapFactor"])
            # print("Using DepthMapFactor = %f" % self.depth_factor)
        if config.sensor_type == SensorType.RGBD and self.depth_factor <= 0:
            raise ValueError("Camera: Expecting the field DepthMapFactor in the camera config file")

        self.depth_threshold = float("inf")  # Close/Far threshold.
        if "ThDepth" in config.cam_settings and self.sensor_type != SensorType.MONOCULAR:
            depth_threshold = float(config.cam_settings["ThDepth"])  # Baseline times.
            assert self.bf is not None
            self.depth_threshold = self.bf * depth_threshold / self.fx  # Depth threshold in meters
            print("Camera: Using depth_threshold = %f" % self.depth_threshold)
        if (
            config.sensor_type == SensorType.RGBD or config.sensor_type == SensorType.STEREO
        ) and self.depth_threshold is None:
            raise ValueError("Camera: Expecting the field ThDepth in the camera config file")
        print(f"Camera: is_stereo = {self.is_stereo()}")

    def set_intrinsic_matrices(self):
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.Kinv = np.array(
            [[1.0 / fx, 0.0, -cx / fx], [0.0, 1.0 / fy, -cy / fy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def is_stereo(self):
        return self.bf is not None and self.sensor_type != SensorType.MONOCULAR

    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths
    def project(self, xcs):
        pass

    # stereo-project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # (assuming rectified stereo images)
    # out: Nx3 image points, [Nx1] array of map point depths
    def project_stereo(self, xcs):
        pass

    def to_json(self):
        return {
            "type": int(self.type.value),
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "D": json.dumps(self.D.astype(float).tolist() if self.D is not None else None),
            "fps": int(self.fps) if self.fps is not None else None,
            "bf": float(self.bf) if self.bf is not None else None,
            "b": float(self.b) if self.b is not None else None,
            "depth_factor": float(self.depth_factor) if self.depth_factor is not None else None,
            "depth_threshold": (
                float(self.depth_threshold) if self.depth_threshold is not None else None
            ),
            "is_distorted": bool(self.is_distorted if self.is_distorted is not None else None),
            "u_min": float(self.u_min) if self.u_min is not None else None,
            "u_max": float(self.u_max) if self.u_max is not None else None,
            "v_min": float(self.v_min) if self.v_min is not None else None,
            "v_max": float(self.v_max) if self.v_max is not None else None,
            "initialized": bool(self.initialized if self.initialized is not None else None),
            "K": json.dumps(self.K.astype(float).tolist() if self.K is not None else None),
            "Kinv": json.dumps(self.Kinv.astype(float).tolist() if self.Kinv is not None else None),
            "sensor_type": (
                self.sensor_type.to_json()
                if hasattr(self, "sensor_type") and self.sensor_type is not None
                else None
            ),
        }

    def init_from_json(self, json_str):
        # Handle both string and dict inputs (C++ saves as dict, Python saves as string)
        if isinstance(json_str, str):
            json_str = json.loads(json_str)
        self.type = CameraType(int(json_str["type"]))
        self.width = int(json_str["width"])
        self.height = int(json_str["height"])
        self.fx = float(json_str["fx"])
        self.fy = float(json_str["fy"])
        self.cx = float(json_str["cx"])
        self.cy = float(json_str["cy"])
        # Handle D (distortion coefficients) - can be a string (Python) or array (C++)
        self.D = deserialize_array_flexible(json_str["D"])
        self.fps = int(json_str["fps"])
        bf_str = json_str["bf"]
        b_str = json_str["b"]
        self.bf = float(bf_str) if bf_str is not None else None
        self.b = float(b_str) if b_str is not None else None
        self.depth_factor = (
            float(json_str["depth_factor"]) if json_str["depth_factor"] is not None else None
        )
        self.depth_threshold = (
            float(json_str["depth_threshold"]) if json_str["depth_threshold"] is not None else None
        )
        self.is_distorted = bool(json_str["is_distorted"])
        self.u_min = float(json_str["u_min"])
        self.u_max = float(json_str["u_max"])
        self.v_min = float(json_str["v_min"])
        self.v_max = float(json_str["v_max"])
        self.initialized = bool(json_str["initialized"])
        if not hasattr(self, "fovx"):
            self.fovx = focal2fov(self.fx, self.width)
        if not hasattr(self, "fovy"):
            self.fovy = focal2fov(self.fy, self.height)
        # Handle K and Kinv - can be strings (Python) or arrays (C++)
        self.K = deserialize_array_flexible(json_str["K"])
        self.Kinv = deserialize_array_flexible(json_str["Kinv"])

        self.sensor_type = SensorType.MONOCULAR
        if "sensor_type" in json_str and json_str["sensor_type"] is not None:
            try:
                self.sensor_type = SensorType.from_json(json_str["sensor_type"])
            except (ValueError, KeyError, AttributeError):
                # If deserialization fails, default to MONOCULAR with warning
                Printer.red(
                    "Camera: sensor_type not found or invalid in JSON, defaulting to MONOCULAR. "
                    "Please ensure sensor_type is explicitly saved in future maps."
                )
        else:
            # Backward compatibility: default to MONOCULAR with warning
            Printer.red(
                "Camera: sensor_type not found in JSON, defaulting to MONOCULAR. "
                "Please ensure sensor_type is explicitly saved in future maps."
            )

    def is_in_image(self, uv, z):
        return (
            (uv[0] >= self.u_min)
            & (uv[0] < self.u_max)
            & (uv[1] >= self.v_min)
            & (uv[1] < self.v_max)
            & (z > 0)
        )

    # input: [Nx2] array of uvs, [Nx1] of zs
    # output: [Nx1] array of visibility flags
    def are_in_image(self, uvs, zs):
        return CameraUtils.are_in_image_numba(
            uvs, zs, self.u_min, self.u_max, self.v_min, self.v_max
        )

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
        self.type = CameraType.PINHOLE

        if config is None:
            return

        # print(f'PinholeCamera: K = {self.K}')
        if self.width is None or self.height is None:
            raise ValueError(
                "Camera: Expecting the fields Camera.width and Camera.height in the camera config file"
            )
        self.u_min, self.u_max = 0, self.width
        self.v_min, self.v_max = 0, self.height
        self.init()

    def to_json(self):
        camera_json = super().to_json()
        return camera_json

    @staticmethod
    def from_json(json_str):
        c = PinholeCamera(None)
        c.init_from_json(json_str)
        return c

    def init(self):
        if not self.initialized:
            self.initialized = True
            self.undistort_image_bounds()

    # project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # out: Nx2 image points, [Nx1] array of map point depths
    def project(self, xcs):  # numba version
        # Ensure xcs is always 2D
        if xcs.ndim == 1:
            xcs = xcs.reshape(1, 3)
        xcs = xcs.astype(np.float64)
        return CameraUtils.project_numba(xcs, self.K)

    # stereo-project a 3D point or an array of 3D points (w.r.t. camera frame), of shape [Nx3]
    # (assuming rectified stereo images)
    # out: Nx3 image points, [Nx1] array of map point depths
    def project_stereo(self, xcs):  # numba version
        # Ensure xcs is always 2D
        if xcs.ndim == 1:
            xcs = xcs.reshape(1, 3)
        xcs = xcs.astype(np.float64)
        return CameraUtils.project_stereo_numba(xcs, self.K, self.bf)

    # unproject single 2D point uv (pixels on image plane) to 2D normalized point (representing 3D point on z=1 plane)
    def unproject(self, uv):
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        return x, y

    # unproject single 2D point uv (pixels on image plane) to a 3D point on z=depth plane
    def unproject_3d(self, u, v, depth):
        x = depth * (u - self.cx) / self.fx
        y = depth * (v - self.cy) / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64).reshape(3, 1)

    # in:  uvs [Nx2]
    # out: xcs array [Nx2] of 2D normalized coordinates (representing 3D points on z=1 plane)
    def unproject_points(self, uvs):  # numba version
        uvs = uvs.astype(np.float64)
        return CameraUtils.unproject_points_numba(uvs, self.Kinv)

    # in:  uvs [Nx2], depths [Nx1]
    # out: xcs array [Nx3] of backprojected 3D points
    def unproject_points_3d(self, uvs, depths):  # numba version
        uvs = uvs.astype(np.float64)
        depths = depths.astype(np.float64)
        return CameraUtils.unproject_points_3d_numba(uvs, depths, self.Kinv)

    # in:  uvs [Nx2]
    # out: uvs_undistorted array [Nx2] of undistorted coordinates
    def undistort_points(self, uvs):
        if self.is_distorted:
            # uvs_undistorted = cv2.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)   # =>  Error: while undistorting the points error: (-215:Assertion failed) src.isContinuous()
            uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
            uvs_undistorted = cv2.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)
            return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        else:
            return uvs

    # update image bounds
    def undistort_image_bounds(self):
        uv_bounds = np.array(
            [
                [self.u_min, self.v_min],
                [self.u_min, self.v_max],
                [self.u_max, self.v_min],
                [self.u_max, self.v_max],
            ],
            dtype=np.float64,
        ).reshape(4, 2)
        # print('uv_bounds: ', uv_bounds)
        if self.is_distorted:
            uv_bounds_undistorted = cv2.undistortPoints(
                np.expand_dims(uv_bounds, axis=1), self.K, self.D, None, self.K
            )
            uv_bounds_undistorted = uv_bounds_undistorted.ravel().reshape(
                uv_bounds_undistorted.shape[0], 2
            )
        else:
            uv_bounds_undistorted = uv_bounds
        # print('uv_bounds_undistorted: ', uv_bounds_undistorted)
        self.u_min = min(uv_bounds_undistorted[0][0], uv_bounds_undistorted[1][0])
        self.u_max = max(uv_bounds_undistorted[2][0], uv_bounds_undistorted[3][0])
        self.v_min = min(uv_bounds_undistorted[0][1], uv_bounds_undistorted[2][1])
        self.v_max = max(uv_bounds_undistorted[1][1], uv_bounds_undistorted[3][1])
        # print('camera u_min: ', self.u_min)
        # print('camera u_max: ', self.u_max)
        # print('camera v_min: ', self.v_min)
        # print('camera v_max: ', self.v_max)
