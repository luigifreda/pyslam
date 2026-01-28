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

from __future__ import annotations

import math
import os
import subprocess
import time
from typing import Optional

import cv2
import numpy as np
import psutil
import rerun as rr  # pip install rerun-sdk

import pyslam.utilities.geometry as utils_geom
from pyslam.slam import Camera
from pyslam.utilities.logging import Printer


def check_command_start(command: str) -> bool:
    """Try to start a command and check if a process with that name is running."""
    try:
        _ = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)
        for proc in psutil.process_iter(attrs=["name"]):
            if proc.info["name"] == command and proc.is_running():
                Printer.green("INFO: " + command + " running")
                return True
        Printer.orange("WARNING: " + command + " not running")
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class Rerun:
    """
    Rerun logging interface for PYSLAM. NEW VERSION for the latest rerun-sdk version.
    It uses the new blueprint API with EyeControls3D.

    Key design choices in this version:
      - Use ONLY absolute entity paths (always starting with '/').
      - Separate two scene roots:
          * /world          : "global" scene (camera moves through the world)
          * /view/world     : "follow" scene (world moves around the camera)
        The 3D viewport is rooted at /view by default so the follow-mode works
        even on older rerun-sdk versions (no EyeControls3D).
      - You can still log global stuff under /world if you want.
    """

    # -----------------------------------------------------------------------------------
    # Static config
    # -----------------------------------------------------------------------------------
    blueprint = None

    img_compress: bool = False
    img_compress_jpeg_quality: int = 85
    camera_img_resize_factors = None  # e.g. [0.5, 0.5] -> [h_factor, w_factor]

    is_initialized: bool = False
    is_z_up: bool = False

    # Scene roots
    VIEW_ROOT: str = "/view"  # The 3D view origin points here
    FOLLOW_WORLD: str = "/view/world"  # The "follow" world we move with inv(camera_pose)
    GLOBAL_WORLD: str = "/world"  # Optional global world

    # Visual tuning
    frustum_size: float = 0.5  # Controls custom wireframe frustum size
    frustum_line_radius: float = 0.01
    pinhole_plane_distance_base: float = 20.0  # image plane distance multiplier base
    trajectory_radius: float = 0.2
    point_radius: float = 0.005

    # Follow-camera mode
    enable_follow: bool = True  # If True, update FOLLOW_WORLD transform each frame
    follow_pan_deg: float = 0.0  # yaw around camera up (degrees)
    follow_tilt_deg: float = -20.0  # pitch around camera right (degrees)
    follow_distance: float = 100.0  # meters behind camera (along -Z in camera frame)

    def __init__(self) -> None:
        self.init()

    @staticmethod
    def is_ok() -> bool:
        command = "rerun"
        try:
            return check_command_start(command)
        except Exception as e:
            Printer.orange("ERROR: " + str(e))
            return False

    # ===================================================================================
    # Init
    # ===================================================================================

    @staticmethod
    def init(img_compress: bool = False) -> None:
        Rerun.img_compress = img_compress

        if Rerun.blueprint is not None:
            rr.init("pyslam", spawn=True, default_blueprint=Rerun.blueprint)
        else:
            rr.init("pyslam", spawn=True)

        Rerun.is_initialized = True

    @staticmethod
    def init3d(img_compress: bool = False) -> None:
        Rerun.init(img_compress)

        # World coordinate system for the FOLLOW scene root
        if Rerun.is_z_up:
            rr.log(Rerun.FOLLOW_WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        else:
            rr.log(
                Rerun.FOLLOW_WORLD, rr.ViewCoordinates.RDF, static=True
            )  # X=Right, Y=Down, Z=Forward

        # Optional: also define for global world if you log there
        if Rerun.is_z_up:
            rr.log(Rerun.GLOBAL_WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        else:
            rr.log(Rerun.GLOBAL_WORLD, rr.ViewCoordinates.RDF, static=True)

        # Static grid in BOTH scenes (handy for debugging)
        Rerun.log_3d_grid_plane(root=Rerun.FOLLOW_WORLD)
        Rerun.log_3d_grid_plane(root=Rerun.GLOBAL_WORLD)

    @staticmethod
    def init_vo(img_compress: bool = False) -> None:
        import rerun.blueprint as rrb

        # NOTE: We root the 3D view at /view so the follow-world trick works.
        # The follow world lives at /view/world.
        Rerun.blueprint = rrb.Vertical(
            rrb.Horizontal(
                # rrb.Spatial3DView(name="3D", origin=Rerun.VIEW_ROOT),
                rrb.Spatial3DView(
                    name="3D",
                    origin=Rerun.GLOBAL_WORLD,
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.FirstPerson,
                        tracking_entity=f"{Rerun.GLOBAL_WORLD}/camera/image",
                        speed=5.0,  # optional
                    ),
                ),
                rrb.Spatial2DView(name="Camera", origin=f"{Rerun.GLOBAL_WORLD}/camera/image"),
            ),
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/trajectory_error"),
                    rrb.TimeSeriesView(origin="/trajectory_stats"),
                    column_shares=[1, 1],
                ),
                rrb.Spatial2DView(name="Trajectory 2D", origin="/trajectory_img/2d"),
                column_shares=[3, 2],
            ),
            row_shares=[3, 2],
        )

        Rerun.init3d(img_compress)

        # Ensure follow root exists
        rr.log(
            Rerun.VIEW_ROOT, rr.Transform3D(translation=[0, 0, 0], from_parent=False), static=True
        )
        rr.log(
            Rerun.FOLLOW_WORLD,
            rr.Transform3D(translation=[0, 0, 0], from_parent=False),
            static=True,
        )

    # ===================================================================================
    # Helpers
    # ===================================================================================

    @staticmethod
    def _camera_paths(root: str) -> tuple[str, str, str]:
        """
        Return (camera_path, image_path, depth_path) under a given root.
        """
        cam = f"{root}/camera"
        img = f"{root}/camera/image"
        dep = f"{root}/camera/depth"
        return cam, img, dep

    # ===================================================================================
    # 3D logging
    # ===================================================================================

    @staticmethod
    def log_3d_camera_img_seq(
        frame_id: int,
        img_bgr,
        depth: Optional[np.ndarray],
        camera: Camera,
        camera_pose: np.ndarray,
        *,
        # You can override the following per-call if you want:
        root_follow: Optional[str] = None,
        root_global: Optional[str] = None,
        log_global: bool = False,
        log_follow: bool = True,
        frustum_size: Optional[float] = None,
    ) -> None:
        """
        Log a camera image + intrinsics + (optional) depth + frustum.

        - If log_follow=True: logs under /view/world (by default) so the 3D view shows it.
        - If log_global=True: logs under /world as well (useful for debugging).
        - If enable_follow=True: also updates /view/world transform via inv(camera_pose),
          so the view behaves like it follows the camera.

        Important: this does NOT require EyeControls3D.
        """

        if camera_pose.shape != (4, 4):
            raise ValueError("camera_pose should have shape (4, 4)")

        rr.set_time("frame_id", sequence=frame_id)

        # Update follow transform (moves the *world* so the camera appears fixed / view follows)
        if Rerun.enable_follow:
            Rerun.log_3d_follow_world_from_camera_pose(frame_id, camera_pose)

        fs = frustum_size if frustum_size is not None else Rerun.frustum_size

        # Decide roots
        rf = root_follow if root_follow is not None else Rerun.FOLLOW_WORLD
        rg = root_global if root_global is not None else Rerun.GLOBAL_WORLD

        # Log to follow and/or global
        if log_follow:
            Rerun._log_camera_bundle_under_root(
                frame_id, rf, img_bgr, depth, camera, camera_pose, fs
            )

        if log_global:
            Rerun._log_camera_bundle_under_root(
                frame_id, rg, img_bgr, depth, camera, camera_pose, fs
            )

    @staticmethod
    def _log_camera_bundle_under_root(
        frame_id: int,
        root: str,
        img_bgr,
        depth: Optional[np.ndarray],
        camera: Camera,
        camera_pose: np.ndarray,
        frustum_size: float,
    ) -> None:
        cam_path, img_path, depth_path = Rerun._camera_paths(root)

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        # Camera pose node
        rr.log(cam_path, rr.Transform3D(translation=t, mat3x3=R, from_parent=False))

        if Rerun.is_z_up:
            rr.log(cam_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        else:
            rr.log(cam_path, rr.ViewCoordinates.RDF, static=True)

        # Image node (child of camera)
        rr.log(img_path, rr.Transform3D(translation=[0, 0, 0], from_parent=True))

        # Intrinsics (Pinhole)
        rr.log(
            img_path,
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=[camera.fx, camera.fy],
                principal_point=[camera.cx, camera.cy],
                image_plane_distance=Rerun.pinhole_plane_distance_base * float(frustum_size),
            ),
        )

        # Image data
        if Rerun.camera_img_resize_factors is not None:
            new_w = int(float(img_bgr.shape[1]) * float(Rerun.camera_img_resize_factors[1]))
            new_h = int(float(img_bgr.shape[0]) * float(Rerun.camera_img_resize_factors[0]))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h))
            if depth is not None:
                depth = cv2.resize(depth, (new_w, new_h))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if Rerun.img_compress:
            rr.log(
                img_path, rr.Image(img_rgb).compress(jpeg_quality=Rerun.img_compress_jpeg_quality)
            )
        else:
            rr.log(img_path, rr.Image(img_rgb))

        if depth is not None:
            rr.log(depth_path, rr.DepthImage(depth, meter=1.0, colormap="viridis"))

        # Frustum wireframe (custom)
        Rerun.log_3d_camera_frustum_wireframe(
            root=root,
            frame_id=frame_id,
            camera=camera,
            pose=camera_pose,
            size=frustum_size,
            color=[0, 255, 0],
            line_radius=Rerun.frustum_line_radius,
        )

    @staticmethod
    def log_3d_follow_world_from_camera_pose(
        frame_id: int, camera_pose: np.ndarray, invert_pose: bool = True
    ) -> None:
        """
        Move the FOLLOW world root so the view behaves like it follows the camera.

        This updates /view/world transform each frame.
        """
        view_pose = utils_geom.inv_T(camera_pose) if invert_pose else camera_pose

        # Optional shoulder offset relative to camera frame, while *always* looking at camera.
        # Camera frame is RDF (x=right, y=down, z=forward).
        if (
            Rerun.follow_pan_deg != 0.0
            or Rerun.follow_tilt_deg != 0.0
            or Rerun.follow_distance != 0.0
        ):
            yaw = math.radians(Rerun.follow_pan_deg)
            pitch = math.radians(Rerun.follow_tilt_deg)

            cy, sy = math.cos(yaw), math.sin(yaw)
            cp, sp = math.cos(pitch), math.sin(pitch)

            # Yaw about camera Y (down), then pitch about camera X (right)
            R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            R_offset = R_yaw @ R_pitch

            # View position in camera frame (behind camera by default)
            p_c = R_offset @ np.array([0.0, 0.0, -float(Rerun.follow_distance)])

            # Build view->camera rotation so view's +Z points to the camera.
            z_axis = -p_c
            z_norm = float(np.linalg.norm(z_axis))
            if z_norm < 1e-9:
                # Degenerate (distance ~0): fall back to identity.
                z_axis = np.array([0.0, 0.0, 1.0])
            else:
                z_axis = z_axis / z_norm

            down_ref = np.array([0.0, 1.0, 0.0])  # camera +Y is "down"
            x_axis = np.cross(down_ref, z_axis)
            x_norm = float(np.linalg.norm(x_axis))
            if x_norm < 1e-9:
                # If z_axis parallel to down, pick a different reference.
                down_ref = np.array([0.0, 0.0, 1.0])
                x_axis = np.cross(down_ref, z_axis)
                x_axis /= float(np.linalg.norm(x_axis))
            else:
                x_axis = x_axis / x_norm
            y_axis = np.cross(z_axis, x_axis)

            R_cv = np.stack([x_axis, y_axis, z_axis], axis=1)

            T_cv = np.eye(4)
            T_cv[:3, :3] = R_cv
            T_cv[:3, 3] = p_c

            # Compute view pose in world and invert for /view/world transform.
            T_wc = camera_pose if invert_pose else utils_geom.inv_T(camera_pose)
            T_wv = T_wc @ T_cv
            view_pose = utils_geom.inv_T(T_wv)
        R = view_pose[:3, :3]
        t = view_pose[:3, 3]

        rr.set_time("frame_id", sequence=frame_id)
        rr.log(Rerun.FOLLOW_WORLD, rr.Transform3D(translation=t, mat3x3=R, from_parent=False))

    @staticmethod
    def log_3d_grid_plane(root: str, num_divs: int = 30, div_size: float = 10.0) -> None:
        rr.set_time("frame_id", sequence=0)

        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size

        lines = []
        for n in range(2 * num_divs + 1):
            x = minx + div_size * n
            z = minz + div_size * n
            lines.append([[x, 0, minz], [x, 0, maxz]])
            lines.append([[minx, 0, z], [maxx, 0, z]])

        rr.log(
            f"{root}/grid",
            rr.LineStrips3D(
                lines,
                radii=0.01,
                colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
            ),
        )

    @staticmethod
    def log_3d_camera_frustum_wireframe(
        *,
        root: str,
        frame_id: int,
        camera: Camera,
        pose: np.ndarray,
        size: float,
        color=[0, 255, 0],
        line_radius: float = 0.01,
        name: str = "camera_frustum",
    ) -> None:
        """
        Draw a simple wireframe frustum attached to the camera pose under:
          {root}/debug/{name}

        This is independent from the rr.Pinhole visualization.
        """
        rr.set_time("frame_id", sequence=frame_id)

        topic = f"{root}/debug/{name}"

        R = pose[:3, :3]
        t = pose[:3, 3]
        rr.log(topic, rr.Transform3D(translation=t, mat3x3=R, from_parent=False))

        a = float(camera.width) / float(camera.height)
        w = a * size
        h = size

        # If you want "purely visual" frustum depth, just use z = size.
        # If you want it to scale a bit with intrinsics, keep the following:
        z = size * 0.5 * float(camera.fx + camera.fy) / float(camera.height)

        lines = [
            [[0, 0, 0], [w, h, z]],
            [[0, 0, 0], [w, -h, z]],
            [[0, 0, 0], [-w, -h, z]],
            [[0, 0, 0], [-w, h, z]],
            [[w, h, z], [w, -h, z]],
            [[-w, h, z], [-w, -h, z]],
            [[-w, h, z], [w, h, z]],
            [[-w, -h, z], [w, -h, z]],
        ]

        rr.log(
            topic,
            rr.LineStrips3D(
                lines,
                radii=line_radius,
                colors=color,
            ),
        )

    @staticmethod
    def log_3d_trajectory(
        frame_id: int,
        points: np.ndarray,
        trajectory_string: str = "trajectory",
        color=[255, 0, 0],
        radius: Optional[float] = None,
        root: Optional[str] = None,
    ) -> None:
        rr.set_time("frame_id", sequence=frame_id)

        r = float(radius) if radius is not None else float(Rerun.trajectory_radius)
        base = root if root is not None else Rerun.FOLLOW_WORLD

        pts = np.asarray(points).reshape(-1, 3)
        rr.log(
            f"{base}/{trajectory_string}",
            rr.LineStrips3D([pts], radii=r, colors=color),
        )

    @staticmethod
    def log_3d_box(
        timestamp: float,
        *,
        root: Optional[str] = None,
        color=[255, 0, 0],
        center=[0, 0, 0],
        quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
        half_size=[1.0, 1.0, 1.0],
        label=None,
        box_name_string="bbox",
        box_id: int = 0,
        fill_mode=None,
        base_topic: str = "bboxes",
    ) -> None:
        base = root if root is not None else Rerun.FOLLOW_WORLD
        rr.set_time("time", timestamp=timestamp)

        rr.log(
            f"{base}/{base_topic}/{box_name_string}{box_id}",
            rr.Boxes3D(
                half_sizes=half_size,
                centers=center,
                quaternions=quaternion,
                colors=color,
                labels=label,
                fill_mode=fill_mode,
            ),
        )

    @staticmethod
    def log_3d_pointcloud(
        timestamp: float,
        points: np.ndarray,
        *,
        pose: Optional[np.ndarray] = None,
        root: Optional[str] = None,
        topic: str = "pointcloud",
        colors: Optional[np.ndarray] = None,
        point_radius: Optional[float] = None,
    ) -> None:
        if points.shape[1] != 3:
            raise ValueError("Points should have shape (N, 3)")

        rr.set_time("time", timestamp=timestamp)

        base = root if root is not None else Rerun.FOLLOW_WORLD
        pr = float(point_radius) if point_radius is not None else float(Rerun.point_radius)

        if pose is not None:
            R = pose[:3, :3]
            t = pose[:3, 3]
            pts = (R @ points.T).T + t
        else:
            pts = points

        rr.log(
            f"{base}/{topic}",
            rr.Points3D(
                pts,
                colors=colors if colors is not None else [255, 255, 255],
                radii=pr,
            ),
        )

    # ===================================================================================
    # 2D logging
    # ===================================================================================

    @staticmethod
    def log_2d_seq_scalar(topic: str, frame_id: int, scalar_data) -> None:
        rr.set_time("frame_id", sequence=frame_id)
        rr.log(topic, rr.Scalars(scalar_data))

    @staticmethod
    def log_2d_time_scalar(topic: str, frame_time_ns: int, scalar_data) -> None:
        rr.set_time_nanos("time", frame_time_ns)
        rr.log(topic, rr.Scalars(scalar_data))

    @staticmethod
    def log_img_seq(topic: str, frame_id: int, img_bgr, adjust_rgb: bool = True) -> None:
        rr.set_time("frame_id", sequence=frame_id)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if adjust_rgb else img_bgr
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))

    @staticmethod
    def log_img_time(topic: str, frame_time_ns: int, img_bgr, adjust_rgb: bool = True) -> None:
        rr.set_time_nanos("time", frame_time_ns)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if adjust_rgb else img_bgr
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))
