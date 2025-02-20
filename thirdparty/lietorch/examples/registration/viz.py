import sys
sys.path.append('../core')

import argparse
import torch
import scipy
import numpy as np

import geom.projective_ops as pops
import open3d as o3d

def make_point_cloud(image, depth, intrinsics, max_depth=5.0):
    """ create a point cloud """
    colors = image.permute(1,2,0).view(-1,3)
    colors = colors[...,[2,1,0]] / 255.0
    clr = colors.cpu().numpy()

    inv_depth = 1.0 / depth[None,None]
    points = pops.iproj(inv_depth, intrinsics[None,None])
    points = (points[..., :3] / points[..., 3:]).view(-1,3)
    pts = points.cpu().numpy()

    # open3d point cloud
    pc = o3d.geometry.PointCloud()

    keep = pts[:,2] < max_depth
    pc.points = o3d.utility.Vector3dVector(pts[keep])
    pc.colors = o3d.utility.Vector3dVector(clr[keep])

    return pc

def set_camera_pose(vis):
    """ set initial camera position """
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    cam.extrinsic = np.array(
        [[ 0.91396544,  0.1462376,  -0.37852575, 0.94374719],
         [-0.13923432,  0.98919177,  0.04597225,  1.01177687],
         [ 0.38115743,  0.01068673,  0.92444838,  3.35964868],
         [ 0.,          0.,          0.,          1.        ]])
    
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)


def sim3_visualization(T, images, depths, intrinsics):
    """ convert depth to open3d point clouds """

    images = images.squeeze(0)
    depths = depths.squeeze(0)
    intrinsics = intrinsics.squeeze(0)

    pc1 = make_point_cloud(images[0], depths[0], intrinsics[0])
    pc2 = make_point_cloud(images[1], depths[1], intrinsics[1])

    sim3_visualization.index = 1
    sim3_visualization.pc2 = pc2

    NUM_STEPS = 100
    dt = scipy.linalg.logm(T) / NUM_STEPS
    dT = scipy.linalg.expm(dt)
    sim3_visualization.transform = dT

    def animation_callback(vis):
        sim3_visualization.index += 1

        pc2 = sim3_visualization.pc2
        if sim3_visualization.index >= NUM_STEPS and \
                sim3_visualization.index < 2*NUM_STEPS:
            pc2.transform(sim3_visualization.transform)

            vis.update_geometry(pc2)
            vis.poll_events()
            vis.update_renderer()

    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(animation_callback)
    vis.create_window(height=540, width=960)

    vis.add_geometry(pc1)
    vis.add_geometry(pc2)

    vis.get_render_option().load_from_json("assets/renderoption.json")
    set_camera_pose(vis)

    print("Press q to move to next example")
    vis.run()
    vis.destroy_window()
