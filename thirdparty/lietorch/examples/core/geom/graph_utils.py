
import torch
import numpy as np
from collections import OrderedDict

import lietorch
from data_readers.rgbd_utils import compute_distance_matrix_flow


def graph_to_edge_list(graph):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii).cuda()
    jj = torch.as_tensor(jj).cuda()
    kk = torch.as_tensor(kk).cuda()
    return ii, jj, kk

def keyframe_indicies(graph):
    return torch.as_tensor([u for u in graph]).cuda()


def meshgrid(m, n, device='cuda'):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)


class KeyframeGraph:
    def __init__(self, images, poses, depths, intrinsics):
        self.images = images.cpu()
        self.depths = depths.cpu()
        self.poses = poses
        self.intrinsics = intrinsics

        depths = depths[..., 3::8, 3::8].float().cuda()
        disps = torch.where(depths>0.1, 1.0/depths, depths)

        N = poses.shape[1]
        d = compute_distance_matrix_flow(poses, disps, intrinsics / 8.0)

        i, j = 0, 0
        ixs = [ i ]

        while j < N-1:
            if d[i, j+1] > 7.5:
                ixs += [ j ]
                i = j
            j += 1

        # indicies of keyframes
        self.distance_matrix = d[ixs][:,ixs]
        self.ixs = np.array(ixs)
        self.frame_graph = {}

        for i in range(N):
            k = np.argmin(np.abs(i - self.ixs))
            j = self.ixs[k]
            self.frame_graph[i] = (k, poses[:,i] * poses[:,j].inv())

    def get_keyframes(self):
        ix = torch.as_tensor(self.ixs).cuda()
        return self.images[:,ix], self.poses[:,ix], self.depths[:,ix], self.intrinsics[:,ix]

    def get_graph(self, num=-1, thresh=24.0, r=2):
        d = self.distance_matrix.copy()
        
        N = d.shape[0]
        if num < 0:
            num = N

        graph = OrderedDict()
        for i in range(N):
            graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

        for i in range(N):
            for j in range(i-r, i+r+1):
                if j >= 0 and j < N:
                    d[i,j] = np.inf

        for _ in range(num):
            ix = np.argmin(d)
            i, j = ix // N, ix % N

            if d[i,j] < thresh:
                graph[i].append(j)
                for ii in range(i-r, i+r+1):
                    for jj in range(j-r, j+r+1):
                        if ii>=0 and jj>=0 and ii<N and jj<N:
                            d[ii,jj] = np.inf
            else:
                break
        
        return graph

    def get_poses(self, keyframe_poses):
        
        poses_list = []
        for i in range(self.poses.shape[1]):
            k, dP = self.frame_graph[i]
            poses_list += [dP * keyframe_poses[:,k]]

        return lietorch.stack(poses_list, 1)