#!/usr/bin/env -S python3 -O
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

import cv2
import time
import os
import sys
import numpy as np

import platform

import sys


from pyslam.config import Config

from pyslam.slam.slam import Slam, SlamState
from pyslam.slam import PinholeCamera
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType

from pyslam.viz.mplot_thread import Mplot2d
import matplotlib.colors as mcolors


from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.system import getchar, Printer
from pyslam.utilities.img_management import ImgWriter
from pyslam.utilities.geometry import inv_T

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

from pyslam.config_parameters import Parameters

from pyslam.viz.rerun_interface import Rerun

import open3d as o3d

import traceback


# Function to create a grid
def create_grid(size=10, divisions=10):
    lines = []
    points = []
    step = size / divisions

    # Generate lines in X-Z plane
    for i in range(divisions + 1):
        coord = -size / 2 + i * step
        # Lines parallel to X-axis
        lines.append([len(points), len(points) + 1])
        points.append([coord, 0, -size / 2])
        points.append([coord, 0, size / 2])

        # Lines parallel to Z-axis
        lines.append([len(points), len(points) + 1])
        points.append([-size / 2, 0, coord])
        points.append([size / 2, 0, coord])

    # Create LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config)
    if dataset.sensor_type != SensorType.RGBD:
        Printer.red(
            "This example only supports RGBD datasets. Please change your config.yaml to use an RGBD dataset"
        )
        sys.exit(0)

    groundtruth = groundtruth_factory(config.dataset_settings)
    gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()

    camera = PinholeCamera(config)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    o3d_camera = o3d.camera.PinholeCameraIntrinsic(
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
    )

    # Prepare maps to undistort color and depth images
    h, w = camera.height, camera.width
    D = camera.D
    K = camera.K
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    calib_map1, calib_map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)

    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Volume Integration", width=800, height=600)

    # Get render options and set point size
    render_option = vis.get_render_option()
    render_option.point_size = 0.01  # Adjust the point size (e.g., 5.0 for larger points)

    # Adjust the FOV using the view control
    view_control = vis.get_view_control()
    view_control.set_constant_z_near(0.1)
    view_control.set_constant_z_far(1000)
    # view_control.set_zoom(0.1)
    # view_control.set_lookat([0, 0, 0])
    # view_control.scale(1000)

    # Initial trajectory points and lines
    trajectory_points = []  # Starting point
    trajectory_lines = []  # No connections initially
    trajectory_colors = []  # Line color (red)
    # Create LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory_points),
        lines=o3d.utility.Vector2iVector(trajectory_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(trajectory_colors)
    # Add the LineSet to the visualizer
    vis.add_geometry(line_set)

    # Create the initial point cloud object
    point_cloud = o3d.geometry.PointCloud()
    # Initialize with the first frame's data
    if False:
        points, colors = generate_point_cloud_data(0)
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # Add the point cloud to the visualizer
    vis.add_geometry(point_cloud)

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    vis.add_geometry(mesh)

    # Add a coordinate frame to the visualizer
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_frame)

    # Add a grid to the visualizer
    grid = create_grid(size=1000, divisions=100)
    vis.add_geometry(grid)

    key_cv = None

    count_gt = 0

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    while dataset.is_ok:

        print("..................................")
        img = dataset.getImageColor(img_id)
        depth = dataset.getDepth(img_id)
        img_right = (
            dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
        )

        if img is None:
            print("image is empty")
            # getchar()
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)
            continue

        timestamp = dataset.getTimestamp()  # get current timestamp
        next_timestamp = dataset.getNextTimestamp()  # get next timestamp
        frame_duration = (
            next_timestamp - timestamp
            if (timestamp is not None and next_timestamp is not None)
            else -1
        )

        gt_pose = groundtruth.getClosestPose(timestamp)  # Twc
        gt_inv_pose = inv_T(gt_pose)  # Tcw
        gt_x, gt_y, gt_z = gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3]

        print(f"image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}")

        time_start = None
        if img is not None:
            time_start = time.time()

            color_undistorted = cv2.remap(
                img, calib_map1, calib_map2, interpolation=cv2.INTER_LINEAR
            )
            depth_undistorted = cv2.remap(
                depth, calib_map1, calib_map2, interpolation=cv2.INTER_NEAREST
            )

            if False:
                cv2.imshow("color_undistorted", color_undistorted)
                cv2.imshow("depth_undistorted", depth_undistorted)

            color_undistorted = cv2.cvtColor(color_undistorted, cv2.COLOR_BGR2RGB)

            new_gt_point = np.array([[gt_x, gt_y, gt_z]])
            if count_gt == 0:
                trajectory_points = new_gt_point
            else:
                trajectory_points = np.vstack((trajectory_points, new_gt_point))

            # Add a new line segment connecting the last two points
            if count_gt > 0:
                trajectory_lines.append([count_gt - 1, count_gt])
                trajectory_colors = [[1, 0, 0] for _ in trajectory_lines]
            else:
                trajectory_colors = [[1, 0, 0]]

            # Update LineSet geometry
            line_set.points = o3d.utility.Vector3dVector(trajectory_points)
            line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)
            line_set.colors = o3d.utility.Vector3dVector(trajectory_colors)  # Keep color red
            # Update the visualizer
            vis.update_geometry(line_set)

            if img_id % 10 == 0:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_undistorted),
                    o3d.geometry.Image(depth_undistorted),
                    depth_scale=5000.0,
                    depth_trunc=4.0,
                    convert_rgb_to_intensity=False,
                )

                volume.integrate(rgbd, o3d_camera, gt_inv_pose)

                if False:
                    pc_out = volume.extract_point_cloud()
                    # points = np.asarray(pc_out.points)
                    # colors = np.asarray(pc_out.colors)
                    # print(f'points: {points.shape}, colors: {colors.shape}')
                    point_cloud.points = pc_out.points
                    point_cloud.colors = pc_out.colors

                    # Update the visualizer
                    vis.update_geometry(point_cloud)
                else:
                    m_out = volume.extract_triangle_mesh()
                    m_out.compute_vertex_normals()

                    # Manually copy vertex colors from the volume
                    if m_out.has_vertex_colors():
                        m_out.vertex_colors = (
                            m_out.vertex_colors
                        )  # Already correctly set during integration

                    mesh.vertices = m_out.vertices
                    mesh.triangles = m_out.triangles
                    mesh.vertex_normals = m_out.vertex_normals
                    mesh.vertex_colors = (
                        m_out.vertex_colors
                    )  # Ensure vertex colors are correctly assigned

                    # Update the visualizer
                    vis.update_geometry(mesh)

            # Adjust the FOV using the view control
            # view_control.set_lookat(new_gt_point[0])

            # Render the visualizer
            vis.poll_events()
            vis.update_renderer()

            cv2.imshow("Camera", img)

        img_id += 1
        count_gt += 1

        # get keys
        key_cv = cv2.waitKey(1) & 0xFF
