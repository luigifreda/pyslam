import numpy as np
import torch
from collections import OrderedDict
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
import lietorch
from lietorch import SE3, Sim3

from geom.ba import MoBA
from .modules.corr import CorrBlock, AltCorrBlock
from .rslam import RaftSLAM

import geom.projective_ops as pops
from geom.sampler_utils import bilinear_sampler
from geom.graph_utils import KeyframeGraph, graph_to_edge_list

def meshgrid(m, n, device='cuda'):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)

def normalize_images(images):
    images = images[:, :, [2,1,0]]
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])

class FactorGraph:
    def __init__(self, hidden=None, inputs=None, residu=None, ii=None, jj=None):
        self.hidden = hidden
        self.inputs = inputs
        self.residu = residu
        self.ii = ii
        self.jj = jj

    def __iadd__(self, other):
        if self.hidden is None:
            self.hidden = other.hidden
            self.inputs = other.inputs
            self.residu = other.residu
            self.ii = other.ii
            self.jj = other.jj
        else:
            self.hidden = torch.cat([self.hidden, other.hidden], 1)
            self.inputs = torch.cat([self.inputs, other.inputs], 1)
            self.residu = torch.cat([self.residu, other.residu], 1)
            self.ii = torch.cat([self.ii, other.ii], 0)
            self.jj = torch.cat([self.jj, other.jj], 0)

        return self

    def rm(self, keep):
        self.hidden = self.hidden[:,keep]
        self.inputs = self.inputs[:,keep]
        self.residu = self.residu[:,keep]
        self.ii = self.ii[keep]
        self.jj = self.jj[keep]


class SLAMSystem(RaftSLAM):
    def __init__(self, args):
        super(SLAMSystem, self).__init__(args)
        self.mem = 32
        self.num_keyframes = 5

        self.frontend = None
        self.factors = FactorGraph()
        self.count = 0
        self.fixed_poses = 1

        self.images_list = []
        self.depths_list = []
        self.intrinsics_list = []
        
    def initialize(self, ht, wd):
        """ initialize slam buffers """

        self.ht, self.wd = ht, wd
        ht, wd = ht // 8, wd // 8

        self.fmaps = torch.zeros(1, self.mem, 128, ht, wd, device='cuda', dtype=torch.half)
        self.nets = torch.zeros(1, self.mem, 128, ht, wd, device='cuda', dtype=torch.half)
        self.inps = torch.zeros(1, self.mem, 128, ht, wd, device='cuda', dtype=torch.half)

        self.poses = SE3.Identity(1, 2048, device='cuda')
        self.disps = torch.ones(1, 2048, ht, wd, device='cuda')
        self.intrinsics = torch.zeros(1, 2048, 4, device='cuda')
        self.tstamps = torch.zeros(2048, dtype=torch.long)

    def set_frontend(self, frontend):
        self.frontend = frontend

    def add_point_cloud(self, index, image, pose, depth, intrinsics, s=8):
        """ add point cloud to visualization """

        if self.frontend is None:
            return -1

        image = image[...,s//2::s,s//2::s]
        depth = depth[...,s//2::s,s//2::s]
        intrinsics = intrinsics / s

        # backproject
        points = pops.iproj(1.0/depth[None], intrinsics[None])
        points = points[...,:3] / points[...,[3]]
        
        points = points.reshape(-1, 3)
        valid = (depth > 0).reshape(-1)
        colors = image.reshape(3,-1).t() / 255.0
        
        point_data = points[valid].cpu().numpy()
        color_data = colors[valid].cpu().numpy()
        color_data = color_data[:, [2,1,0]]

        pose_data = pose.inv()[0].data
        self.frontend.update_pose(index, pose_data)
        self.frontend.update_points(index, point_data, color_data)

    def get_keyframes(self):
        """ return keyframe poses and timestamps """
        return self.poses[0, :self.count], self.tstamps[:self.count]

    def raw_poses(self):
        return self.poses[0, :self.count].inv().data

    def add_keyframe(self, tstamp, image, depth, intrinsics):
        """ add keyframe to factor graph """

        if self.count == 0:
            ht, wd = image.shape[3:]
            self.initialize(ht, wd)

        inputs = normalize_images(image)
        with autocast(enabled=True):
            fmaps, net, inp = self.extract_features(inputs)

        ix = self.count % self.mem
        self.fmaps[:, ix] = fmaps.squeeze(1)
        self.nets[:, ix] = net.squeeze(1)
        self.inps[:, ix] = inp.squeeze(1)

        self.tstamps[self.count] = tstamp
        self.intrinsics[:, self.count] = intrinsics / 8.0

        disp = torch.where(depth > 0, 1.0/depth, depth)
        self.disps[:, self.count] = disp[:,3::8,3::8]

        pose = self.poses[:, self.count-1]
        self.add_point_cloud(self.count, image, pose, depth, intrinsics)
        self.count += 1
        
    def get_node_attributes(self, index):
        index = index % self.mem
        return self.fmaps[:, index], self.nets[:, index], self.inps[:, index]

    def add_factors(self, ii, jj):
        """ add factors to slam graph """
        fmaps, hidden, inputs = self.get_node_attributes(ii)
        residu_shape = (1, ii.shape[0], self.ht//8, self.wd//8, 2)
        residu = torch.zeros(*residu_shape).cuda()
        self.factors += FactorGraph(hidden, inputs, residu, ii, jj)

    def transform_project(self, ii, jj, **kwargs):
        """ helper function, compute project transform """
        return pops.projective_transform(self.poses, self.disps, self.intrinsics, ii, jj, **kwargs)

    def moba(self, num_steps=5, is_init=False):
        """ motion only bundle adjustment """

        ii, jj = self.factors.ii, self.factors.jj
        ixs = torch.cat([ii, jj], 0)

        with autocast(enabled=True):
            fmap1 = self.fmaps[:, ii % self.mem]
            fmap2 = self.fmaps[:, jj % self.mem]
            poses = self.poses[:, :jj.max()+1]

            corr_fn = CorrBlock(fmap1, fmap2, num_levels=4, radius=3)
            mask = (self.disps[:,ii] > 0.01).float()

            with autocast(enabled=False):
                coords, valid_mask = pops.projective_transform(poses, self.disps, self.intrinsics, ii, jj)

            for i in range(num_steps):
                corr = corr_fn(coords[...,:2])
                corr = torch.cat([corr, mask[:,:,None]], dim=2)

                with autocast(enabled=False):
                    flow = self.factors.residu.permute(0,1,4,2,3).clamp(-32.0, 32.0)
                    flow = torch.cat([flow, mask[:,:,None]], dim=2)
                
                self.factors.hidden, delta, weight = \
                    self.update(self.factors.hidden, self.factors.inputs, corr, flow)

                with autocast(enabled=False):
                    target = coords + delta
                    weight[...,2] = 0.0

                    for i in range(3):
                        poses = MoBA(target, weight, poses, self.disps, 
                            self.intrinsics, ii, jj, self.fixed_poses)

                    coords, valid_mask = pops.projective_transform(poses, self.disps, self.intrinsics, ii, jj)
                    self.factors.residu = (target - coords)[...,:2]

        self.poses[:, :jj.max()+1] = poses

        # update visualization
        if self.frontend is not None:
            for ix in ixs.cpu().numpy():
                self.frontend.update_pose(ix, self.poses[:,ix].inv()[0].data)


    def track(self, tstamp, image, depth, intrinsics):
        """ main thread """

        self.images_list.append(image)
        self.depths_list.append(depth)
        self.intrinsics_list.append(intrinsics)

        # collect frames for initialization
        if self.count < self.num_keyframes:
            self.add_keyframe(tstamp, image, depth, intrinsics)

            if self.count == self.num_keyframes:
                ii, jj = meshgrid(self.num_keyframes, self.num_keyframes)
                keep = ((ii - jj).abs() > 0) & (((ii - jj).abs() <= 3))

                self.add_factors(ii[keep], jj[keep])
                self.moba(num_steps=8, is_init=True)

        else:
            self.poses[:,self.count] = self.poses[:,self.count-1]
            self.add_keyframe(tstamp, image, depth, intrinsics)

            N = self.count
            ii = torch.as_tensor([N-3, N-2, N-1, N-1, N-1], device='cuda')
            jj = torch.as_tensor([N-1, N-1, N-2, N-3, N-4], device='cuda')
            
            self.add_factors(ii, jj)
            self.moba(num_steps=4)

            self.fixed_poses += 1
            self.factors.rm(self.factors.ii + 2 >= self.fixed_poses)


    def forward(self, poses, images, depths, intrinsics, num_steps=12):
        """ Estimates SE3 or Sim3 between pair of frames """

        keyframe_graph = KeyframeGraph(images, poses, depths, intrinsics)
        images, Gs, depths, intrinsics = keyframe_graph.get_keyframes()

        images = images.cuda()
        depths = depths.cuda()

        if self.frontend is not None:
            self.frontend.reset()
            for i, ix in enumerate(keyframe_graph.ixs):
                self.add_point_cloud(ix, images[:,i], Gs[:,i], depths[:,i], intrinsics[:,i], s=4)
            for i in range(poses.shape[1]):
                self.frontend.update_pose(i, poses[:,i].inv()[0].data)

        graph = keyframe_graph.get_graph()
        ii, jj, kk = graph_to_edge_list(graph)
        ixs = torch.cat([ii, jj], 0)

        images = normalize_images(images.cuda())
        depths = depths[:, :, 3::8, 3::8].cuda()
        mask = (depths > 0.1).float()
        disps = torch.where(depths>0.1, 1.0/depths, depths)
        intrinsics = intrinsics / 8

        with autocast(True):

            fmaps, net, inp = self.extract_features(images)
            net = net[:,ii]
            
            # alternate corr implementation uses less memory but 4x slower
            corr_fn = AltCorrBlock(fmaps.float(), (ii, jj), num_levels=4, radius=3)
            # corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

            with autocast(False):
                coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
                residual = torch.zeros_like(coords[...,:2])

            for step in range(num_steps):
                print("Global refinement iteration #{}".format(step))                
                net_list = []
                targets_list = []
                weights_list = []

                s = 64
                for i in range(0, ii.shape[0], s):
                    ii1 = ii[i:i+s]
                    jj1 = jj[i:i+s]

                    corr1 = corr_fn(coords[:,i:i+s,:,:,:2], ii1, jj1)
                    flow1 = residual[:, i:i+s].permute(0,1,4,2,3).clamp(-32.0, 32.0)
                    
                    corr1 = torch.cat([corr1, mask[:,ii1,None]], dim=2)
                    flow1 = torch.cat([flow1, mask[:,ii1,None]], dim=2)

                    net1, delta, weight = self.update(net[:,i:i+s], inp[:,ii1], corr1, flow1)
                    net[:,i:i+s] = net1

                    targets_list += [ coords[:,i:i+s] + delta.float() ]
                    weights_list += [ weight.float() * torch.as_tensor([1.0, 1.0, 0.0]).cuda() ]

                target = torch.cat(targets_list, 1)
                weight = torch.cat(weights_list, 1)

                with autocast(False):
                    for i in range(3):
                        Gs = MoBA(target, weight, Gs, disps, intrinsics, ii, jj, lm=0.00001, ep=.01)

                    coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
                    residual = (target - coords)[...,:2]

                poses = keyframe_graph.get_poses(Gs)
                if self.frontend is not None:
                    for i in range(poses.shape[1]):
                        self.frontend.update_pose(i, poses[:,i].inv()[0].data)

        return poses
        
    def global_refinement(self):
        """ run global refinement """

        poses = self.poses[:, :self.count]
        images = torch.cat(self.images_list, 1).cpu()
        depths = torch.stack(self.depths_list, 1).cpu()
        intrinsics = torch.stack(self.intrinsics_list, 1)

        poses = self.forward(poses, images, depths, intrinsics, num_steps=16)
        self.poses[:, :self.count] = poses





