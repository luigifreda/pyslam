import numpy as np
import os.path as osp

import torch
from lietorch import SE3

import geom.projective_ops as pops
from scipy.spatial.transform import Rotation


def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
        
            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))
            
    return associations

def loadtum(datapath, frame_rate=-1):
    """ read video data in tum-rgbd format """
    if osp.isfile(osp.join(datapath, 'groundtruth.txt')):
        pose_list = osp.join(datapath, 'groundtruth.txt')
    elif osp.isfile(osp.join(datapath, 'pose.txt')):
        pose_list = osp.join(datapath, 'pose.txt')

    image_list = osp.join(datapath, 'rgb.txt')
    depth_list = osp.join(datapath, 'depth.txt')

    calib_path = osp.join(datapath, 'calibration.txt')
    intrinsic = None
    if osp.isfile(calib_path):
        intrinsic = np.loadtxt(calib_path, delimiter=' ')
        intrinsic = intrinsic.astype(np.float64)

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:,1:].astype(np.float64)

    tstamp_image = image_data[:,0].astype(np.float64)
    tstamp_depth = depth_data[:,0].astype(np.float64)
    tstamp_pose = pose_data[:,0].astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    indicies = [ 0 ]
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [ i ]

    images, poses, depths, intrinsics = [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [ osp.join(datapath, image_data[i,1]) ]
        depths += [ osp.join(datapath, depth_data[j,1]) ]
        poses += [ pose_vecs[k] ]
        
        if intrinsic is not None:
            intrinsics += [ intrinsic ]

    return images, depths, poses, intrinsics


def all_pairs_distance_matrix(poses, beta=2.5):
    """ compute distance matrix between all pairs of poses """
    poses = np.array(poses, dtype=np.float32)
    poses[:,:3] *= beta # scale to balence rot + trans
    poses = SE3(torch.from_numpy(poses))

    r = (poses[:,None].inv() * poses[None,:]).log()
    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose):
    """ convert 4x4 pose matrix to (t, q) """
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)

def compute_distance_matrix_flow(poses, disps, intrinsics):
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N = poses.shape[1]
    
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW = 100.0
    matrix = np.zeros((N, N), dtype=np.float32)

    s = 2048
    for i in range(0, ii.shape[0], s):
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])
        
        flow = torch.stack([flow1, flow2], dim=2)
        val = torch.stack([val1, val2], dim=2)
        
        mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1 = ii[i:i+s].cpu().numpy()
        j1 = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix
