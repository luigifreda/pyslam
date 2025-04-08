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
sys.path.append("../../")
from config import Config

from slam import Slam, SlamState
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import SensorType

from mplot_thread import Mplot2d
import matplotlib.colors as mcolors


from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_img import ImgWriter
from utils_geom import inv_T

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from config_parameters import Parameters  

from rerun_interface import Rerun

import open3d as o3d

import traceback



if __name__ == "__main__":
                               
    config = Config()

    dataset = dataset_factory(config)
    if dataset.sensor_type != SensorType.RGBD:
        Printer.red('This example only supports RGBD datasets. Please change your config.yaml to use an RGBD dataset')
        sys.exit(0)

    groundtruth = groundtruth_factory(config.dataset_settings)
    gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        
    camera = PinholeCamera(config)
    
    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
    viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=dataset.sensor_type==SensorType.MONOCULAR)
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    o3d_camera = o3d.camera.PinholeCameraIntrinsic(width=camera.width, height=camera.height, fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)     
    
    # Prepare maps to undistort color and depth images
    h, w = camera.height, camera.width
    D = camera.D
    K = camera.K
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    calib_map1, calib_map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
    
                
    key_cv = None
            
    count_gt = 0
    
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.isOk():
            
        print('..................................')               
        img = dataset.getImageColor(img_id)
        depth = dataset.getDepth(img_id)
        img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
        
        if img is None:
            print('image is empty')
            #getchar()
            time.sleep(0.03)
            continue
            
        timestamp = dataset.getTimestamp()          # get current timestamp 
        next_timestamp = dataset.getNextTimestamp() # get next timestamp 
        frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1

        gt_pose = groundtruth.getClosestPose(timestamp)  # Twc
        gt_inv_pose = inv_T(gt_pose)                     # Tcw
        gt_x, gt_y, gt_z = gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3]
            
        print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
        
        time_start = None 
        if img is not None:
            time_start = time.time()                  
                            
            # 3D display (map display)
            # if viewer3D is not None:
            #     viewer3D.draw_slam_map(slam)

            color_undistorted = cv2.remap(img, calib_map1, calib_map2, interpolation=cv2.INTER_LINEAR)
            color_undistorted = cv2.cvtColor(color_undistorted, cv2.COLOR_BGR2RGB)
            depth_undistorted = cv2.remap(depth, calib_map1, calib_map2, interpolation=cv2.INTER_NEAREST)
            
            if img_id % 10 == 0:               
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_undistorted), 
                    o3d.geometry.Image(depth_undistorted), 
                    depth_scale=5000.0,
                    depth_trunc=4.0, convert_rgb_to_intensity=False)
                
                volume.integrate(rgbd, o3d_camera, gt_inv_pose)
                
                if False:                            
                    pc_out = volume.extract_point_cloud()
                    # points = np.asarray(pc_out.points)
                    # colors = np.asarray(pc_out.colors)
                    # print(f'points: {points.shape}, colors: {colors.shape}')                
                    # point_cloud.points = pc_out.points
                    # point_cloud.colors = pc_out.colors
                                        
                    # Update the visualizer
                    viewer3D.draw_dense_geometry(pointcloud=pc_out)

                else:
                    m_out = volume.extract_triangle_mesh()
                    m_out.compute_vertex_normals()
                    
                    # Manually copy vertex colors from the volume
                    # if m_out.has_vertex_colors():
                    #     m_out.vertex_colors = m_out.vertex_colors  # Already correctly set during integration

                    # mesh.vertices = m_out.vertices
                    # mesh.triangles = m_out.triangles
                    # mesh.vertex_normals = m_out.vertex_normals
                    # mesh.vertex_colors = m_out.vertex_colors  # Ensure vertex colors are correctly assigned
                    
                    # Update the visualizer
                    viewer3D.draw_dense_geometry(mesh=m_out)
                        
            cv2.imshow('Camera', img)
                            
        img_id += 1     
        count_gt += 1                       
        
        # get keys 
        key_cv = cv2.waitKey(1) & 0xFF   
            
     
        if viewer3D is not None:
            is_paused = viewer3D.is_paused()    
            is_map_save = viewer3D.is_map_save() and is_map_save == False 
            do_step = viewer3D.do_step() and do_step == False  
            do_reset = viewer3D.reset() and do_reset == False  
                                  
        if key_cv == ord('q'):
            if viewer3D is not None:
                viewer3D.quit()           
            break
            
