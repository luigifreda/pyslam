import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from networks.modules.extractor import BasicEncoder
from networks.modules.corr import CorrBlock
from networks.modules.gru import ConvGRU
from networks.modules.clipping import GradientClip

from lietorch import SE3, Sim3
from geom.ba import MoBA

import geom.projective_ops as pops
from geom.sampler_utils import bilinear_sampler, sample_depths
from geom.graph_utils import graph_to_edge_list, keyframe_indicies


class UpdateModule(nn.Module):
    def __init__(self, args):
        super(UpdateModule, self).__init__()
        self.args = args

        cor_planes = 4 * (2*3 + 1)**2 + 1

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)

    def forward(self, net, inp, corr, flow):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape
        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2).contiguous()
        weight = weight.permute(0,1,3,4,2).contiguous()

        net = net.view(*output_dim)
        return net, delta, weight


class RaftSLAM(nn.Module):
    def __init__(self, args):
        super(RaftSLAM, self).__init__()
        self.args = args
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule(args)

    def extract_features(self, images):
        """ run feeature extraction networks """
        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp

    def forward(self, Gs, images, depths, intrinsics, graph=None, num_steps=12):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        depths = depths[:, :, 3::8, 3::8]
        intrinsics = intrinsics / 8
        mask = (depths > 0.1).float()
        disps = torch.where(depths>0.1, 1.0/depths, depths)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        residual = torch.zeros_like(coords[...,:2])

        Gs_list, coords_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            coords = coords.detach()
            residual = residual.detach()

            corr = corr_fn(coords[...,:2])
            flow = residual.permute(0,1,4,2,3).clamp(-32.0, 32.0)
            
            corr = torch.cat([corr, mask[:,ii,None]], dim=2)
            flow = torch.cat([flow, mask[:,ii,None]], dim=2)
            net, delta, weight = self.update(net, inp, corr, flow)

            target = coords + delta
            weight[...,2] = 0.0

            for i in range(3):
                Gs = MoBA(target, weight, Gs, disps, intrinsics, ii, jj)

            coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords)[...,:2]

            Gs_list.append(Gs)
            coords_list.append(target)

            valid_mask = valid_mask * mask[:,ii].unsqueeze(-1)
            residual_list.append(valid_mask * residual)

        return Gs_list, residual_list
