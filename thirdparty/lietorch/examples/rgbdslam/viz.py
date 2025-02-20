import time
import argparse
import torch
import scipy
import numpy as np
import open3d as o3d

from queue import Empty
from multiprocessing import Queue, Process
from scipy.spatial.transform import Rotation

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

def create_camera_actor(is_gt=False, scale=0.05):
    """ build open3d camera polydata """

    cam_points = scale * np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1],
        [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])

    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_points),
        lines=o3d.utility.Vector2iVector(cam_lines))

    color = (0.0, 0.0, 0.0) if is_gt else (0.0, 0.8, 0.8)
    camera_actor.paint_uniform_color(color)

    return camera_actor

def create_point_cloud_actor(points, colors):
    """ open3d point cloud from numpy array """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def draw_trajectory(queue):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 8

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()
                if data[0] == 'pose':
                    i, pose, is_gt = data[1:]
                    
                    # convert to 4x4 matrix
                    pose = pose_matrix_from_quaternion(pose)

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)
                        
                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]
                            pc.transform(pose_change)
                            vis.update_geometry(pc)

                    else:
                        cam_actor = create_camera_actor(is_gt)
                        cam_actor.transform(pose)
                        vis.add_geometry(cam_actor)

                    if not is_gt:
                        draw_trajectory.cameras[i] = (cam_actor, pose)

                elif data[0] == 'points':
                    i, points, colors = data[1:]
                    point_actor = create_point_cloud_actor(points, colors)

                    pose = draw_trajectory.cameras[i][1]
                    point_actor.transform(pose)
                    vis.add_geometry(point_actor)

                    draw_trajectory.points[i] = point_actor

                elif data[0] == 'reset':
                    draw_trajectory.warmup = -1
                    
                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()

    vis = o3d.visualization.Visualizer()

    vis.register_animation_callback(animation_callback)
    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("assets/renderoption.json")

    vis.run()
    vis.destroy_window()


class SLAMFrontend:
    def __init__(self):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, args=(self.queue, ))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        self.queue.put_nowait(('pose', index, pose, gt))

    def update_points(self, index, points, colors):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        self.queue.put_nowait(('points', index, points, colors))
    
    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()



