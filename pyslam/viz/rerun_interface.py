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
import rerun as rr  # pip install rerun-sdk
import math as math
import psutil
import time
import os
import subprocess

import pyslam.utilities.geometry as utils_geom
from pyslam.slam import Camera

from pyslam.utilities.logging import Printer


def check_command_start(command):
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(1)
        for proc in psutil.process_iter(attrs=["name"]):
            # print(f'found process: {proc.info["name"]}')
            if proc.info["name"] == command and proc.is_running():
                Printer.green("INFO: " + command + " running")
                return True
        Printer.orange("WARNING: " + command + " not running")
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class Rerun:
    # static parameters
    blueprint = None
    img_compress = False  # set to true if you want to compress the data
    img_compress_jpeg_quality = 85
    camera_img_resize_factors = None  # [0.1, 0.1]
    current_camera_view_scale = 0.3
    camera_poses_view_size = 0.5
    is_initialized = False
    is_z_up = False

    def __init__(self) -> None:
        self.init()

    @staticmethod
    def is_ok() -> bool:
        command = "rerun"
        result = False
        try:
            result = check_command_start(command)
        except Exception as e:
            Printer.orange("ERROR: " + str(e))
        return result

    # ===================================================================================
    # Init
    # ===================================================================================

    @staticmethod
    def init(img_compress=False) -> None:
        Rerun.img_compress = img_compress

        if Rerun.blueprint:
            rr.init("pyslam", spawn=True, default_blueprint=Rerun.blueprint)
        else:
            rr.init("pyslam", spawn=True)
        # rr.connect()  # Connect to a remote viewer
        Rerun.is_initialized = True

    @staticmethod
    def init3d(img_compress=False) -> None:
        Rerun.init(img_compress)
        if Rerun.is_z_up:
            rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("/world", rr.Transform3D(translation=[0, 0, 0], from_parent=True))
        else:
            rr.log("/world", rr.ViewCoordinates.RDF, static=True)  # X=Right, Y=Down, Z=Forward
            Rerun.log_3d_grid_plane()

    @staticmethod
    def init_vo(img_compress=False) -> None:
        import rerun.blueprint as rrb

        # Setup the blueprint
        Rerun.blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D", origin="/world"),
                # rrb.Spatial3DView(
                #     name="3D",
                #     origin="/world",
                #     eye_controls=rrb.EyeControls3D(
                #         kind=rrb.Eye3DKind.FirstPerson,
                #         tracking_entity="/world/camera",  # <- try this first
                #     ),
                # ),
                rrb.Spatial2DView(name="Camera", origin="/world/camera/image"),
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
            row_shares=[3, 2],  # 3 "parts" in the first Horizontal, 2 in the second
        )
        # Init rerun
        Rerun.init3d(img_compress)

    # ===================================================================================
    # 3D logging
    # ===================================================================================

    @staticmethod
    def log_3d_camera_img_seq(
        frame_id: int, img, depth, camera: Camera, camera_pose, size=0.5
    ) -> None:
        """
        Log a camera image and depth map.

        Args:
            frame_id: The frame ID of the camera image.
            img: The camera image.
            depth: The depth map.
            camera: The camera object.
            camera_pose: The camera pose.
        """

        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

        rr.set_time("frame_id", sequence=frame_id)

        if Rerun.is_z_up:
            # Log camera pose in Z-up coordinates
            rr.log("/world/camera", rr.Transform3D(translation=t, mat3x3=R, from_parent=False))
            rr.log("/world/camera", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        else:
            # Log camera pose in RDF coordinates
            rr.log(
                "/world/camera",
                rr.Transform3D(
                    translation=t,
                    mat3x3=R,  # R * Rerun.current_camera_view_scale,
                    from_parent=False,
                ),
            )
            rr.log(
                "/world/camera", rr.ViewCoordinates.RDF, static=True
            )  # X=Right, Y=Down, Z=Forward

        # Attach image to camera in the scene graph
        rr.log(
            "/world/camera/image",
            rr.Transform3D(translation=[0, 0, 0], from_parent=True),
        )

        # Log camera intrinsics
        rr.log(
            "/world/camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=[camera.fx, camera.fy],
                principal_point=[camera.cx, camera.cy],
                image_plane_distance=20.0 * size,
            ),
        )

        if Rerun.camera_img_resize_factors:
            new_width = int(float(img.shape[1]) * Rerun.camera_img_resize_factors[1])
            new_height = int(float(img.shape[0]) * Rerun.camera_img_resize_factors[0])
            bgr = cv2.resize(img, (new_width, new_height))
            if depth is not None:
                depth = cv2.resize(depth, (new_width, new_height))
        else:
            bgr = img
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if Rerun.img_compress:
            rr.log(
                "/world/camera/image",
                rr.Image(rgb).compress(jpeg_quality=Rerun.img_compress_jpeg_quality),
            )
        else:
            rr.log("/world/camera/image", rr.Image(rgb))

        if depth is not None:
            rr.log("/world/camera/depth", rr.DepthImage(depth, meter=1.0, colormap="viridis"))

        Rerun.log_3d_camera_pose(
            frame_id, camera, camera_pose, color=[0, 255, 0], size=Rerun.camera_poses_view_size
        )

    @staticmethod
    def log_3d_grid_plane(num_divs=30, div_size=10):
        """
        Log a 3D grid plane.

        Args:
            num_divs: The number of divisions.
            div_size: The size of each division.
        """
        rr.set_time("frame_id", sequence=0)
        # Plane parallel to x-z at origin with normal -y
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size
        lines = []
        for n in range(2 * num_divs):
            lines.append([[minx + div_size * n, 0, minz], [minx + div_size * n, 0, maxz]])
            lines.append([[minx, 0, minz + div_size * n], [maxx, 0, minz + div_size * n]])
        rr.log(
            "world/grid",
            rr.LineStrips3D(
                lines,
                # rr.Radius.ui_points produces radii that the viewer interprets as given in ui points.
                radii=0.01,
                colors=[0.7 * 255, 0.7 * 255, 0.7 * 255],
            ),
        )

    @staticmethod
    def log_3d_box(
        timestamp: float,
        color=[255, 0, 0],
        center=[0, 0, 0],
        quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
        half_size=[1.0, 1.0, 1.0],
        label=None,
        box_name_string="bbox",
        box_id: int = 0,
        fill_mode=None,
        base_topic="/world/bboxes",
    ) -> None:
        """
        Log a 3D box.

        Args:
            timestamp: The timestamp of the box.
            color: The color of the box.
            center: The center of the box.
            quaternion: The quaternion of the box.
            half_size: The half size of the box.
            label: The label of the box.
            box_name_string: The name of the box.
            box_id: The ID of the box.
            fill_mode: The fill mode of the box.
            base_topic: The base topic of the box.
        """
        # rr.set_time("frame_id", sequence=frame_id)
        rr.log(
            base_topic + "/" + box_name_string + str(box_id),
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
    def log_3d_trajectory(
        frame_id: int,
        points: np.ndarray,
        trajectory_string: str = "trajectory",
        color=[255, 0, 0],
        size=0.2,
    ) -> None:
        """
        Log a 3D trajectory.

        Args:
            frame_id: The frame ID of the trajectory.
            points: The points of the trajectory.
            trajectory_string: The name of the trajectory.
            color: The color of the trajectory.
            size: The size of the trajectory.
        """
        # rr.set_time("frame_id", sequence=frame_id)
        points = np.array(points).reshape(-1, 3)
        rr.log(
            "world/" + trajectory_string,
            rr.LineStrips3D(
                [points],
                # rr.Radius.ui_points produces radii that the viewer interprets as given in ui points.
                radii=size,
                colors=color,
            ),
        )

    @staticmethod
    def log_3d_camera_pose(frame_id: int, camera: Camera, pose, color=[0, 255, 0], size=1.0):
        """
        Log a 3D camera pose.

        Args:
            frame_id: The frame ID of the camera pose.
            camera: The camera object.
            pose: The camera pose.
            color: The color of the camera pose.
            size: The size of the camera pose.
        """
        topic_name = "world/camara_poses/camera_" + str(frame_id)
        R = pose[:3, :3]
        t = pose[:3, 3]
        rr.log(topic_name, rr.Transform3D(translation=t, mat3x3=R, from_parent=False))

        a = camera.width / camera.height
        w = a * size
        h = size
        z = size * 0.5 * (camera.fx + camera.fy) / camera.height

        lines = []
        lines.append([[0, 0, 0], [w, h, z]])
        lines.append([[0, 0, 0], [w, -h, z]])
        lines.append([[0, 0, 0], [-w, -h, z]])
        lines.append([[0, 0, 0], [-w, h, z]])
        lines.append([[w, h, z], [w, -h, z]])
        lines.append([[-w, h, z], [-w, -h, z]])
        lines.append([[-w, h, z], [w, h, z]])
        lines.append([[-w, -h, z], [w, -h, z]])
        rr.log(
            topic_name,
            rr.LineStrips3D(
                lines,
                # rr.Radius.ui_points produces radii that the viewer interprets as given in ui points.
                radii=0.01,
                colors=color,
            ),
        )

    @staticmethod
    def log_3d_pointcloud(
        timestamp: float,
        points: np.ndarray,  # shape (N, 3)
        pose: np.ndarray = None,  # 4x4 transformation matrix
        topic: str = "/world/pointcloud",
        colors: np.ndarray = None,  # shape (N, 3)
        point_radius: float = 0.005,  # default radius in world units
    ):
        """
        Log a 3D pointcloud.

        Args:
            timestamp: The timestamp of the pointcloud.
            points: The points of the pointcloud.
            pose: The pose of the pointcloud.
            topic: The topic of the pointcloud.
            colors: The colors of the pointcloud.
            point_radius: The radius of the pointcloud.
        """
        if points.shape[1] != 3:
            raise ValueError("Points should have shape (N, 3)")

        rr.set_time("time", timestamp=timestamp)

        if pose is not None:
            # Apply pose transformation
            R = pose[:3, :3]
            t = pose[:3, 3]
            transformed_points = (R @ points.T).T + t
        else:
            # No transformation, use points as is
            transformed_points = points

        rr.log(
            topic,
            rr.Points3D(
                transformed_points,
                colors=colors if colors is not None else [255, 255, 255],
                radii=point_radius,  # Set the visual size of each point
            ),
        )

    @staticmethod
    def log_3d_view_from_camera_pose(
        frame_id: int,
        camera_pose: np.ndarray,
        invert_pose: bool = True,
        world_path: str = "/world",
    ) -> None:
        """
        Move the current 3D view by applying a transform to the scene root.

        Note:
            Rerun 0.23.x does not expose a direct API to set the viewer camera
            pose, so we move the view by updating the transform of the world
            entity. Passing a camera pose here will make the view follow the
            camera (by applying the inverse transform).

        Args:
            frame_id: The frame ID to log the view update on.
            camera_pose: 4x4 pose of the camera in world coordinates.
            invert_pose: If True, apply the inverse pose so the view follows
                the camera.
            world_path: The scene entity path to move (defaults to "/view").
        """
        if camera_pose.shape != (4, 4):
            raise ValueError("camera_pose should have shape (4, 4)")

        view_pose = utils_geom.inv_T(camera_pose) if invert_pose else camera_pose
        R = view_pose[:3, :3]
        t = view_pose[:3, 3]

        rr.set_time("frame_id", sequence=frame_id)
        rr.log(
            world_path,
            rr.Transform3D(translation=t, mat3x3=R, from_parent=False),
        )

    # ===================================================================================
    # 2D logging
    # ===================================================================================

    @staticmethod
    def log_2d_seq_scalar(topic: str, frame_id: int, scalar_data) -> None:
        """
        Log a 2D scalar at a specific frame ID.

        Args:
            topic: The topic of the scalar sequence.
            frame_id: The frame ID of the scalar sequence.
            scalar_data: The scalar data.
        """
        rr.set_time("frame_id", sequence=frame_id)
        rr.log(topic, rr.Scalars(scalar_data))

    @staticmethod
    def log_2d_time_scalar(topic: str, frame_time_ns, scalar_data) -> None:
        """
        Log a 2D scalar sequence at a specific time.

        Args:
            topic: The topic of the scalar sequence.
            frame_time_ns: The timestamp of the scalar sequence.
            scalar_data: The scalar data.
        """
        rr.set_time_nanos("time", frame_time_ns)
        rr.log(topic, rr.Scalars(scalar_data))

    @staticmethod
    def log_img_seq(topic: str, frame_id: int, img, adjust_rgb=True) -> None:
        """
        Log a 2D image sequence at a specific frame ID.

        Args:
            topic: The topic of the image sequence.
            frame_id: The frame ID of the image sequence.
            img: The image.
            adjust_rgb: Whether to adjust the RGB color space.
        """
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rr.set_time("frame_id", sequence=frame_id)
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))

    @staticmethod
    def log_img_time(topic: str, frame_time_ns, img, adjust_rgb=True) -> None:
        """
        Log a 2D image at a specific time.

        Args:
            topic: The topic of the image.
            frame_time_ns: The timestamp of the image.
            img: The image.
            adjust_rgb: Whether to adjust the RGB color space.
        """
        if adjust_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rr.set_time_nanos("time", frame_time_ns)
        if Rerun.img_compress:
            rr.log(topic, rr.Image(img).compress(jpeg_quality=Rerun.img_compress_jpeg_quality))
        else:
            rr.log(topic, rr.Image(img))
