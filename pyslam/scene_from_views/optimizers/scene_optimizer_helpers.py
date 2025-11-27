"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* The Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.

Unified auxiliary functions and classes for scene optimizers (dense and sparse).
Part of the code is adapted from the original code by Naver Corporation.
Original code Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

This file has been refactored to import from modular submodules while maintaining
backward compatibility. See REFACTORING_SUMMARY.md for details.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from functools import cache, lru_cache

import cv2
import roma
import tqdm
import os
import hashlib
import math
from collections import namedtuple
from typing import Dict, Any, List, Optional

import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

import torch.nn.functional as F
import copy


from .learning_rate_schedules import LearningRateSchedules

from pyslam.utilities.torch import to_numpy, to_cpu, to_device


# ============================================================================
# Point Cloud Optimizer
# ============================================================================


# ============================================================================
# Initialization Functions
# ============================================================================


class PoseInitialization:
    """Functions for initializing camera poses from various sources."""

    @staticmethod
    @torch.no_grad()
    def init_from_known_poses(self, niter_PnP=10, min_conf_thr=3):
        """Initialize camera poses and depthmaps from known ground truth poses.

        This function assumes all camera poses are known and uses them to initialize
        pairwise poses and depthmaps. It aligns predicted pairwise poses with ground
        truth poses and selects the best depthmap for each image.

        Args:
            self: Scene optimizer instance with known poses
            niter_PnP: Number of iterations for PnP solver (default: 10)
            min_conf_thr: Minimum confidence threshold for correspondences (default: 3)
        """
        device = self.device

        # indices of known poses
        nkp, known_poses_msk, known_poses = PoseUtils.get_known_poses(self)
        assert nkp == self.n_imgs, "not all poses are known"

        # get all focals
        nkf, _, im_focals = PoseUtils.get_known_focals(self)
        assert nkf == self.n_imgs
        im_pp = self.get_principal_points()

        best_depthmaps = {}
        # init all pairwise poses
        for e, (i, j) in enumerate(tqdm.tqdm(self.edges, disable=not self.verbose)):
            i_j = EdgeUtils.edge_str(i, j)

            # find relative pose for this pair
            P1 = torch.eye(4, device=device)
            msk = self.conf_i[i_j] > min(min_conf_thr, self.conf_i[i_j].min() - 0.1)
            _, P2 = GeometryUtils.fast_pnp(
                self.pred_j[i_j],
                float(im_focals[i].mean()),
                pp=im_pp[i],
                msk=msk,
                device=device,
                niter_PnP=niter_PnP,
            )

            # align the two predicted camera with the two gt cameras
            s, R, T = GeometryUtils.align_multiple_poses(torch.stack((P1, P2)), known_poses[[i, j]])
            # normally we have known_poses[i] ~= sRT_to_4x4(s,R,T,device) @ P1
            # and GeometryUtils.geom_transform(sRT_to_4x4(1,R,T,device), s*P2[:3,3])
            self._set_pose(self.pw_poses, e, R, T, scale=s)

            # remember if this is a good depthmap
            score = float(self.conf_i[i_j].mean())
            if score > best_depthmaps.get(i, (0,))[0]:
                best_depthmaps[i] = score, i_j, s

        # init all image poses
        for n in range(self.n_imgs):
            assert known_poses_msk[n]
            _, i_j, scale = best_depthmaps[n]
            depth = self.pred_i[i_j][:, :, 2]
            self._set_depthmap(n, depth * scale)

    @staticmethod
    @torch.no_grad()
    def init_minimum_spanning_tree(self, **kw):
        """Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
        """
        device = self.device
        pts3d, _, im_focals, im_poses = PoseInitialization.minimum_spanning_tree(
            self.imshapes,
            self.edges,
            self.pred_i,
            self.pred_j,
            self.conf_i,
            self.conf_j,
            self.im_conf,
            self.min_conf_thr,
            device,
            has_im_poses=self.has_im_poses,
            verbose=self.verbose,
            **kw,
        )

        return PoseInitialization.init_from_pts3d(self, pts3d, im_focals, im_poses)

    @staticmethod
    def init_from_pts3d(self, pts3d, im_focals, im_poses):
        """Initialize camera poses and depthmaps from 3D points and initial poses.

        This function initializes the scene optimizer from pre-computed 3D points,
        focal lengths, and initial camera poses. It performs global alignment if
        multiple known poses are available and sets up pairwise poses.

        Args:
            self: Scene optimizer instance
            pts3d: List of 3D point clouds, one per image
            im_focals: List of focal lengths, one per image
            im_poses: Tensor of initial camera-to-world poses (N, 4, 4)
        """
        # init poses
        nkp, known_poses_msk, known_poses = PoseUtils.get_known_poses(self)
        if nkp == 1:
            raise NotImplementedError(
                "Would be simpler to just align everything afterwards on the single known pose"
            )
        elif nkp > 1:
            # global rigid SE3 alignment
            s, R, T = GeometryUtils.align_multiple_poses(
                im_poses[known_poses_msk], known_poses[known_poses_msk]
            )
            trf = GeometryUtils.sRT_to_4x4(s, R, T, device=known_poses.device)

            # rotate everything
            im_poses = trf @ im_poses
            im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
            for img_pts3d in pts3d:
                img_pts3d[:] = GeometryUtils.geom_transform(trf, img_pts3d)

        # set all pairwise poses
        for e, (i, j) in enumerate(self.edges):
            i_j = EdgeUtils.edge_str(i, j)
            # compute transform that goes from cam to world
            s, R, T = GeometryUtils.rigid_points_registration(
                self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j]
            )
            self._set_pose(self.pw_poses, e, R, T, scale=s)

        # take into account the scale normalization
        s_factor = self.get_pw_norm_scale_factor()
        im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
        for img_pts3d in pts3d:
            img_pts3d *= s_factor

        # init all image poses
        if self.has_im_poses:
            for i in range(self.n_imgs):
                cam2world = im_poses[i]
                depth = GeometryUtils.geom_transform(GeometryUtils.inv(cam2world), pts3d[i])[..., 2]
                self._set_depthmap(i, depth)
                self._set_pose(self.im_poses, i, cam2world)
                if im_focals[i] is not None:
                    self._set_focal(i, im_focals[i])

        if self.verbose:
            print(" init loss =", float(self()))

    @staticmethod
    def minimum_spanning_tree(
        imshapes,
        edges,
        pred_i,
        pred_j,
        conf_i,
        conf_j,
        im_conf,
        min_conf_thr,
        device,
        has_im_poses=True,
        niter_PnP=10,
        verbose=True,
    ):
        """Build a minimum spanning tree from pairwise predictions and initialize poses.

        This function constructs a minimum spanning tree from the pairwise graph based
        on edge scores, then incrementally builds 3D point clouds and camera poses
        by traversing the MST. It starts with the strongest edge and adds cameras
        one by one, aligning new predictions with existing point clouds.

        Args:
            imshapes: List of (H, W) tuples for each image
            edges: List of (i, j) tuples representing image pairs
            pred_i: Dict mapping edge keys to point clouds for view i
            pred_j: Dict mapping edge keys to point clouds for view j
            conf_i: Dict mapping edge keys to confidence maps for view i
            conf_j: Dict mapping edge keys to confidence maps for view j
            im_conf: List of confidence maps, one per image
            min_conf_thr: Minimum confidence threshold
            device: Device for tensors
            has_im_poses: Whether to compute image poses (default: True)
            niter_PnP: Number of iterations for PnP solver (default: 10)
            verbose: Whether to print progress (default: True)

        Returns:
            pts3d: List of 3D point clouds, one per image
            msp_edges: List of MST edges used for initialization
            im_focals: List of focal lengths, one per image
            im_poses: Tensor of camera-to-world poses (N, 4, 4) or None
        """
        n_imgs = len(imshapes)
        sparse_graph = -PoseInitialization.dict_to_sparse_graph(
            EdgeUtils.compute_edge_scores(map(EdgeUtils.i_j_ij, edges), conf_i, conf_j)
        )
        msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

        # temp variable to store 3d points
        pts3d = [None] * len(imshapes)

        todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
        im_poses = [None] * n_imgs
        im_focals = [None] * n_imgs

        # init with strongest edge
        score, i, j = todo.pop()
        if verbose:
            print(f" init edge ({i}*,{j}*) {score=}")
        i_j = EdgeUtils.edge_str(i, j)
        pts3d[i] = pred_i[i_j].clone()
        pts3d[j] = pred_j[i_j].clone()
        done = {i, j}
        if has_im_poses:
            im_poses[i] = torch.eye(4, device=device)
            im_focals[i] = CameraUtils.estimate_focal(pred_i[i_j])

        # set initial pointcloud based on pairwise graph
        msp_edges = [(i, j)]
        while todo:
            # each time, predict the next one
            score, i, j = todo.pop()

            if im_focals[i] is None:
                im_focals[i] = CameraUtils.estimate_focal(pred_i[i_j])

            if i in done:
                if verbose:
                    print(f" init edge ({i},{j}*) {score=}")
                assert j not in done
                # align pred[i] with pts3d[i], and then set j accordingly
                i_j = EdgeUtils.edge_str(i, j)
                s, R, T = GeometryUtils.rigid_points_registration(
                    pred_i[i_j], pts3d[i], conf=conf_i[i_j]
                )
                trf = GeometryUtils.sRT_to_4x4(s, R, T, device)
                pts3d[j] = GeometryUtils.geom_transform(trf, pred_j[i_j])
                done.add(j)
                msp_edges.append((i, j))

                if has_im_poses and im_poses[i] is None:
                    im_poses[i] = GeometryUtils.sRT_to_4x4(1, R, T, device)

            elif j in done:
                if verbose:
                    print(f" init edge ({i}*,{j}) {score=}")
                assert i not in done
                i_j = EdgeUtils.edge_str(i, j)
                s, R, T = GeometryUtils.rigid_points_registration(
                    pred_j[i_j], pts3d[j], conf=conf_j[i_j]
                )
                trf = GeometryUtils.sRT_to_4x4(s, R, T, device)
                pts3d[i] = GeometryUtils.geom_transform(trf, pred_i[i_j])
                done.add(i)
                msp_edges.append((i, j))

                if has_im_poses and im_poses[i] is None:
                    im_poses[i] = GeometryUtils.sRT_to_4x4(1, R, T, device)
            else:
                # let's try again later
                todo.insert(0, (score, i, j))

        if has_im_poses:
            # complete all missing informations
            pair_scores = list(sparse_graph.values())  # already negative scores: less is best
            edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
            for i, j in edges_from_best_to_worse.tolist():
                if im_focals[i] is None:
                    im_focals[i] = CameraUtils.estimate_focal(pred_i[EdgeUtils.edge_str(i, j)])

            for i in range(n_imgs):
                if im_poses[i] is None:
                    msk = im_conf[i] > min_conf_thr
                    res = GeometryUtils.fast_pnp(
                        pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP
                    )
                    if res:
                        im_focals[i], im_poses[i] = res
                if im_poses[i] is None:
                    im_poses[i] = torch.eye(4, device=device)
            im_poses = torch.stack(im_poses)
        else:
            im_poses = im_focals = None

        return pts3d, msp_edges, im_focals, im_poses

    @staticmethod
    def dict_to_sparse_graph(dic):
        """Convert a dictionary of edge values to a sparse graph matrix.

        Args:
            dic: Dictionary mapping (i, j) edge tuples to values

        Returns:
            Sparse matrix (scipy.sparse.dok_array) representing the graph
        """
        n_imgs = max(max(e) for e in dic) + 1
        res = sp.dok_array((n_imgs, n_imgs))
        for edge, value in dic.items():
            res[edge] = value
        return res


class CameraUtils:

    @staticmethod
    def estimate_focal(pts3d_i, pp=None):
        """Estimate camera focal length from 3D points and principal point.

        Args:
            pts3d_i: 3D points (H, W, 3)
            pp: Principal point (2,) tensor or None (default: image center)

        Returns:
            Estimated focal length (float)
        """
        if pp is None:
            H, W, THREE = pts3d_i.shape
            assert THREE == 3
            pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device)
        focal = CameraUtils.estimate_focal_knowing_depth(
            pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode="weiszfeld"
        ).ravel()
        return float(focal)

    @staticmethod
    def estimate_focal_knowing_depth(
        pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
    ):
        """Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
        """
        B, H, W, THREE = pts3d.shape
        assert THREE == 3

        # centered pixel grid
        pixels = GeometryUtils.xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(
            -1, 1, 2
        )  # B,HW,2
        pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

        if focal_mode == "median":
            with torch.no_grad():
                # direct estimation of focal
                u, v = pixels.unbind(dim=-1)
                x, y, z = pts3d.unbind(dim=-1)
                fx_votes = (u * z) / x
                fy_votes = (v * z) / y

                # assume square pixels, hence same focal for X and Y
                f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
                focal = torch.nanmedian(f_votes, dim=-1).values

        elif focal_mode == "weiszfeld":
            # init focal with l2 closed form
            # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
            xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
                posinf=0, neginf=0
            )  # homogeneous (x,y,1)

            dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
            dot_xy_xy = xy_over_z.square().sum(dim=-1)

            focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

            # iterative re-weighted least-squares
            for iter in range(10):
                # re-weighting by inverse of distance
                dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
                # print(dis.nanmean(-1))
                w = dis.clip(min=1e-8).reciprocal()
                # update the scaling with the new weights
                focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
        else:
            raise ValueError(f"bad {focal_mode=}")

        focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
        focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
        # print(focal)
        return focal

    @staticmethod
    def calibrate_camera_pnpransac(pointclouds, img_points, masks, intrinsics):
        """
        Input:
            pointclouds: (bs, N, 3)
            img_points: (bs, N, 2)
        Return:
            rotations: (bs, 3, 3)
            translations: (bs, 3, 1)
            c2ws: (bs, 4, 4)
        """
        bs = pointclouds.shape[0]

        camera_matrix = intrinsics.cpu().numpy()  # (bs, 3, 3)

        dist_coeffs = np.zeros((5, 1))

        rotations = []
        translations = []

        for i in range(bs):
            obj_points = pointclouds[i][masks[i]].cpu().numpy()
            img_pts = img_points[i][[masks[i]]].cpu().numpy()

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points, img_pts, camera_matrix[i], dist_coeffs
            )

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                rotations.append(torch.tensor(rotation_matrix, dtype=torch.float32))
                translations.append(torch.tensor(tvec, dtype=torch.float32))
            else:
                rotations.append(torch.eye(3))
                translations.append(torch.ones(3, 1))

        rotations = torch.stack(rotations).to(pointclouds.device)
        translations = torch.stack(translations).to(pointclouds.device)
        w2cs = torch.eye(4).repeat(bs, 1, 1).to(pointclouds.device)
        w2cs[:, :3, :3] = rotations
        w2cs[:, :3, 3:] = translations
        return torch.linalg.inv(w2cs)


class PointCloudUtils:

    @staticmethod
    def clean_pointcloud(im_confs, K, cams, depthmaps, all_pts3d, tol=0.001, bad_conf=0, dbg=()):
        """
        Clean pointcloud by reducing confidence for inconsistent points.

        This function identifies points that are occluded or inconsistent across
        multiple views and reduces their confidence scores.

        Method:
        1) Express all 3D points in each camera coordinate frame
        2) If they're in front of a depthmap but less confident, lower their confidence

        Args:
            im_confs: List of [H, W] confidence maps
            K: Camera intrinsics [n, 3, 3]
            cams: Camera poses (world-to-cam) [n, 4, 4]
            depthmaps: List of [H, W] depth maps
            all_pts3d: List of [H, W, 3] 3D point clouds
            tol: Tolerance for depth comparison (default: 0.001)
            bad_conf: Confidence value to assign to bad points (default: 0)
            dbg: Debug flags (unused)

        Returns:
            List of cleaned confidence maps
        """
        assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
        assert 0 <= tol < 1
        res = [c.clone() for c in im_confs]

        # Reshape appropriately
        all_pts3d = [p.view(*c.shape, 3) for p, c in zip(all_pts3d, im_confs)]
        depthmaps = [d.view(*c.shape) for d, c in zip(depthmaps, im_confs)]

        for i, pts3d in enumerate(all_pts3d):
            for j in range(len(all_pts3d)):
                if i == j:
                    continue

                # Project 3D points into other view
                proj = GeometryUtils.geom_transform(cams[j], pts3d)
                proj_depth = proj[:, :, 2]
                u, v = (
                    GeometryUtils.geom_transform(K[j], proj, norm=1, ncol=2)
                    .round()
                    .long()
                    .unbind(-1)
                )

                # Check which points are actually in the visible cone
                H, W = im_confs[j].shape
                msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
                msk_j = v[msk_i], u[msk_i]

                # Find bad points = those in front but less confident
                bad_points = (proj_depth[msk_i] < (1 - tol) * depthmaps[j][msk_j]) & (
                    res[i][msk_i] < res[j][msk_j]
                )

                bad_msk_i = msk_i.clone()
                bad_msk_i[msk_i] = bad_points
                res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

        return res

    @staticmethod
    def fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
        """
        Fast conversion from depth map to 3D points.

        Args:
            depth: Depth map tensor
            pixel_grid: Pixel grid coordinates
            focal: Focal length
            pp: Principal point

        Returns:
            3D points tensor
        """
        pp = pp.unsqueeze(1)
        focal = focal.unsqueeze(1)
        assert focal.shape == (len(depth), 1, 1)
        assert pp.shape == (len(depth), 1, 2)
        assert pixel_grid.shape == depth.shape + (2,)
        depth = depth.unsqueeze(-1)
        return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)

    @staticmethod
    def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
        """
        Args:
            - depthmap (BxHxW array):
            - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
        Returns:
            pointmap of absolute coordinates (BxHxWx3 array)
        """

        if len(depth.shape) == 4:
            B, H, W, n = depth.shape
        else:
            B, H, W = depth.shape
            n = None

        if len(pseudo_focal.shape) == 3:  # [B,H,W]
            pseudo_focalx = pseudo_focaly = pseudo_focal
        elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
            pseudo_focalx = pseudo_focal[:, 0]
            if pseudo_focal.shape[1] == 2:
                pseudo_focaly = pseudo_focal[:, 1]
            else:
                pseudo_focaly = pseudo_focalx
        else:
            raise NotImplementedError("Error, unknown input focal shape format.")

        assert pseudo_focalx.shape == depth.shape[:3]
        assert pseudo_focaly.shape == depth.shape[:3]
        grid_x, grid_y = GeometryUtils.xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

        # set principal point
        if pp is None:
            grid_x = grid_x - (W - 1) / 2
            grid_y = grid_y - (H - 1) / 2
        else:
            grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
            grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

        if n is None:
            pts3d = torch.empty((B, H, W, 3), device=depth.device)
            pts3d[..., 0] = depth * grid_x / pseudo_focalx
            pts3d[..., 1] = depth * grid_y / pseudo_focaly
            pts3d[..., 2] = depth
        else:
            pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
            pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
            pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
            pts3d[..., 2, :] = depth
        return pts3d

    @staticmethod
    def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
        """
        Args:
            - depthmap (HxW array):
            - camera_intrinsics: a 3x3 matrix
        Returns:
            pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
        """
        camera_intrinsics = np.float32(camera_intrinsics)
        H, W = depthmap.shape

        # Compute 3D ray associated with each pixel
        # Strong assumption: there are no skew terms
        assert camera_intrinsics[0, 1] == 0.0
        assert camera_intrinsics[1, 0] == 0.0
        if pseudo_focal is None:
            fu = camera_intrinsics[0, 0]
            fv = camera_intrinsics[1, 1]
        else:
            assert pseudo_focal.shape == (H, W)
            fu = fv = pseudo_focal
        cu = camera_intrinsics[0, 2]
        cv = camera_intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z_cam = depthmap
        x_cam = (u - cu) * z_cam / fu
        y_cam = (v - cv) * z_cam / fv
        X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

        # Mask for valid coordinates
        valid_mask = depthmap > 0.0
        return X_cam, valid_mask

    @staticmethod
    def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
        """
        Args:
            - depthmap (HxW array):
            - camera_intrinsics: a 3x3 matrix
            - camera_pose: a 4x3 or 4x4 cam2world matrix
        Returns:
            pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
        X_cam, valid_mask = PointCloudUtils.depthmap_to_camera_coordinates(
            depthmap, camera_intrinsics
        )

        X_world = X_cam  # default
        if camera_pose is not None:
            # R_cam2world = np.float32(camera_params["R_cam2world"])
            # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
            R_cam2world = camera_pose[:3, :3]
            t_cam2world = camera_pose[:3, 3]

            # Express in absolute coordinates (invalid depth values)
            X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

        return X_world, valid_mask


class GeometryUtils:
    """Geometric transformation and utility functions."""

    @staticmethod
    def inv(mat):
        """Invert a torch or numpy matrix"""
        if isinstance(mat, torch.Tensor):
            return torch.linalg.inv(mat)
        if isinstance(mat, np.ndarray):
            return np.linalg.inv(mat)
        raise ValueError(f"bad matrix type = {type(mat)}")

    @staticmethod
    def get_med_dist_between_poses(poses):
        return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))

    @staticmethod
    def rigid_points_registration(pts1, pts2, conf):
        """Compute rigid transformation between two point sets with confidence weights.

        Args:
            pts1: First set of 3D points (H, W, 3) or (N, 3)
            pts2: Second set of 3D points (H, W, 3) or (N, 3)
            conf: Confidence weights (H, W) or (N,)

        Returns:
            scale: Scale factor
            R: Rotation matrix (3, 3)
            T: Translation vector (3,)
        """
        R, T, s = roma.rigid_points_registration(
            pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True
        )
        return s, R, T  # return un-scaled (R, T)

    @staticmethod
    def sRT_to_4x4(scale, R, T, device):
        """Convert scale, rotation, and translation to a 4x4 transformation matrix.

        Args:
            scale: Scale factor (scalar)
            R: Rotation matrix (3, 3)
            T: Translation vector (3,)
            device: Device for the output tensor

        Returns:
            4x4 transformation matrix (torch.Tensor)
        """
        trf = torch.eye(4, device=device)
        trf[:3, :3] = R * scale
        trf[:3, 3] = T.ravel()  # doesn't need scaling
        return trf

    @staticmethod
    @cache
    def pixel_grid(H, W):
        """Generate a cached pixel coordinate grid.

        Args:
            H: Image height
            W: Image width

        Returns:
            Pixel coordinates array (H, W, 2) with (x, y) coordinates
        """
        return np.mgrid[:W, :H].T.astype(np.float32)

    @staticmethod
    def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
        """Extract camera pose using RANSAC-based PnP solver.

        This function uses OpenCV's solvePnPRansac to estimate camera pose from
        3D-2D correspondences. If focal length is unknown, it tries multiple
        candidate focal lengths and selects the best one.

        Args:
            pts3d: 3D points (H, W, 3)
            focal: Focal length (float) or None to search for it
            msk: Boolean mask indicating valid points (H, W)
            device: Device for output tensors
            pp: Principal point (2,) or None (default: image center)
            niter_PnP: Number of RANSAC iterations (default: 10)

        Returns:
            Tuple of (focal_length, cam2world_pose) or None if PnP fails
        """
        # extract camera poses and focals with RANSAC-PnP
        if msk.sum() < 4:
            return None  # we need at least 4 points for PnP
        pts3d, msk = map(to_numpy, (pts3d, msk))

        H, W, THREE = pts3d.shape
        assert THREE == 3
        pixels = GeometryUtils.pixel_grid(H, W)

        if focal is None:
            S = max(W, H)
            tentative_focals = np.geomspace(S / 2, S * 3, 21)
        else:
            tentative_focals = [focal]

        if pp is None:
            pp = (W / 2, H / 2)
        else:
            pp = to_numpy(pp)

        best = (0,)
        for focal in tentative_focals:
            K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

            success, R, T, inliers = cv2.solvePnPRansac(
                pts3d[msk],
                pixels[msk],
                K,
                None,
                iterationsCount=niter_PnP,
                reprojectionError=5,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success:
                continue

            score = len(inliers)
            if success and score > best[0]:
                best = score, R, T, focal

        if not best[0]:
            return None

        _, R, T, best_focal = best
        R = cv2.Rodrigues(R)[0]  # world to cam
        R, T = map(torch.from_numpy, (R, T))
        return best_focal, GeometryUtils.inv(
            GeometryUtils.sRT_to_4x4(1, R, T, device)
        )  # cam to world

    @staticmethod
    def align_multiple_poses(src_poses, target_poses):
        """Compute rigid transformation to align source poses to target poses.

        This function computes a scale, rotation, and translation that best aligns
        a set of source camera poses to target poses. It uses the camera centers
        and forward directions to establish correspondences.

        Args:
            src_poses: Source camera-to-world poses (N, 4, 4)
            target_poses: Target camera-to-world poses (N, 4, 4)

        Returns:
            Tuple of (scale, rotation, translation) where:
                scale: Scale factor (scalar)
                rotation: Rotation matrix (3, 3)
                translation: Translation vector (3,)
        """
        N = len(src_poses)
        assert src_poses.shape == target_poses.shape == (N, 4, 4)

        def center_and_z(poses):
            eps = GeometryUtils.get_med_dist_between_poses(poses) / 100
            return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps * poses[:, :3, 2]))

        R, T, s = roma.rigid_points_registration(
            center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True
        )
        return s, R, T

    @staticmethod
    def xy_grid(
        W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw
    ):
        """Output a (H,W,2) array of int32
        with output[j,i,0] = i + origin[0]
            output[j,i,1] = j + origin[1]
        """
        if device is None:
            # numpy
            arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
        else:
            # torch
            arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
            meshgrid, stack = torch.meshgrid, torch.stack
            ones = lambda *a: torch.ones(*a, device=device)

        tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
        grid = meshgrid(tw, th, indexing="xy")
        if homogeneous:
            grid = grid + (ones((H, W)),)
        if unsqueeze is not None:
            grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
        if cat_dim is not None:
            grid = stack(grid, cat_dim)
        return grid

    @staticmethod
    def geom_transform(Trf, pts, ncol=None, norm=False):
        """Apply a geometric transformation to a list of 3-D points.

        Trf: 3x3 or 4x4 projection matrix (typically a Homography)
        pts: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

        ncol: int. number of columns of the result (2 or 3)
        norm: float. if != 0, the resut is projected on the z=norm plane.

        Returns an array of projected 2d points.
        """
        assert Trf.ndim >= 2
        if isinstance(Trf, np.ndarray):
            pts = np.asarray(pts)
        elif isinstance(Trf, torch.Tensor):
            pts = torch.as_tensor(pts, dtype=Trf.dtype)

        # adapt shape if necessary
        output_reshape = pts.shape[:-1]
        ncol = ncol or pts.shape[-1]

        # optimized code
        if (
            isinstance(Trf, torch.Tensor)
            and isinstance(pts, torch.Tensor)
            and Trf.ndim == 3
            and pts.ndim == 4
        ):
            d = pts.shape[3]
            if Trf.shape[-1] == d:
                pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
            elif Trf.shape[-1] == d + 1:
                pts = (
                    torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                    + Trf[:, None, None, :d, d]
                )
            else:
                raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
        else:
            if Trf.ndim >= 3:
                n = Trf.ndim - 2
                assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
                Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

                if pts.ndim > Trf.ndim:
                    # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                    pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
                elif pts.ndim == 2:
                    # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                    pts = pts[:, None, :]

            if pts.shape[-1] + 1 == Trf.shape[-1]:
                Trf = Trf.swapaxes(-1, -2)  # transpose Trf
                pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
            elif pts.shape[-1] == Trf.shape[-1]:
                Trf = Trf.swapaxes(-1, -2)  # transpose Trf
                pts = pts @ Trf
            else:
                pts = Trf @ pts.T
                if pts.ndim >= 2:
                    pts = pts.swapaxes(-1, -2)

        if norm:
            pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
            if norm != 1:
                pts *= norm

        res = pts[..., :ncol].reshape(*output_reshape, ncol)
        return res


class PoseUtils:
    """Utilities for working with camera poses."""

    @staticmethod
    def get_known_poses(self):
        """Get known (non-optimizable) camera poses.

        Args:
            self: Scene optimizer instance

        Returns:
            Tuple of (count, mask, poses) where:
                count: Number of known poses
                mask: Boolean tensor indicating which poses are known
                poses: Tensor of camera-to-world poses (N, 4, 4)
        """
        if self.has_im_poses:
            known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
            known_poses = self.get_im_poses()
            return known_poses_msk.sum(), known_poses_msk, known_poses
        else:
            return 0, None, None

    @staticmethod
    def get_known_focals(self):
        """Get known (non-optimizable) focal lengths.

        Args:
            self: Scene optimizer instance

        Returns:
            Tuple of (count, mask, focals) where:
                count: Number of known focals
                mask: Boolean tensor indicating which focals are known
                focals: Tensor of focal lengths (N,)
        """
        if self.has_im_poses:
            known_focal_msk = self.get_known_focal_mask()
            known_focals = self.get_focals()
            return known_focal_msk.sum(), known_focal_msk, known_focals
        else:
            return 0, None, None


# ============================================================================
# Sparse Scene Optimizer Helper Functions
# ============================================================================


class FileUtils:
    """File and path utility functions."""

    @staticmethod
    def mkdir_for(f):
        """Create directory for a file path if it doesn't exist.

        Args:
            f: File path

        Returns:
            The file path (for chaining)
        """
        os.makedirs(os.path.dirname(f), exist_ok=True)
        return f

    @staticmethod
    def hash_md5(s):
        """Compute MD5 hash of a string.

        Args:
            s: Input string

        Returns:
            MD5 hash as hexadecimal string
        """
        return hashlib.md5(s.encode("utf-8")).hexdigest()


class EdgeUtils:
    """Edge and graph edge utility functions."""

    @staticmethod
    def edge_str(i, j):
        """Convert edge indices to string representation.

        Args:
            i: First image index
            j: Second image index

        Returns:
            String representation of edge "i_j"
        """
        return f"{i}_{j}"

    @staticmethod
    def i_j_ij(ij):
        """Convert edge tuple to string and return both.

        Args:
            ij: Edge tuple (i, j)

        Returns:
            Tuple of (edge_string, edge_tuple)
        """
        return EdgeUtils.edge_str(*ij), ij

    @staticmethod
    def edge_conf(conf_i, conf_j, edge):
        """Compute edge confidence from two confidence maps.

        Args:
            conf_i: Confidence map for view i
            conf_j: Confidence map for view j
            edge: Edge string identifier

        Returns:
            Combined confidence score (float)
        """
        return float(conf_i[edge].mean() * conf_j[edge].mean())

    @staticmethod
    def compute_edge_scores(edges, conf_i, conf_j):
        """Compute confidence scores for all edges.

        Args:
            edges: Iterable of (edge_string, (i, j)) tuples
            conf_i: Dictionary of confidence maps for view i
            conf_j: Dictionary of confidence maps for view j

        Returns:
            Dictionary mapping (i, j) tuples to confidence scores
        """
        return {(i, j): EdgeUtils.edge_conf(conf_i, conf_j, e) for e, (i, j) in edges}


class ParameterUtils:
    """Parameter manipulation and stacking utilities."""

    @staticmethod
    def NoGradParamDict(x):
        """Create a ParameterDict with requires_grad=False.

        Args:
            x: Dictionary of parameters

        Returns:
            nn.ParameterDict with requires_grad=False
        """
        assert isinstance(x, dict)
        return nn.ParameterDict(x).requires_grad_(False)

    @staticmethod
    def ParameterStack(params, keys=None, is_param=None, fill=0):
        """Stack parameters into a single tensor.

        Args:
            params: List or dict of parameters
            keys: Optional list of keys to select from dict
            is_param: Whether to create nn.Parameter
            fill: Padding size for raveled tensors

        Returns:
            Stacked parameter tensor
        """
        if keys is not None:
            params = [params[k] for k in keys]

        if fill > 0:
            params = [ParameterUtils.ravel_hw(p, fill) for p in params]

        requires_grad = params[0].requires_grad
        assert all(p.requires_grad == requires_grad for p in params)

        params = torch.stack(list(params)).float().detach()
        if is_param or requires_grad:
            params = nn.Parameter(params)
            params.requires_grad_(requires_grad)
        return params

    @staticmethod
    def ravel_hw(tensor, fill=0):
        """Ravel H,W dimensions and optionally pad.

        Args:
            tensor: Input tensor with H, W dimensions
            fill: Padding size (default: 0)

        Returns:
            Ravelled tensor, optionally padded
        """
        # ravel H,W
        tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

        if len(tensor) < fill:
            tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),) + tensor.shape[1:])))
        return tensor


class ImageUtils:
    """Image shape and dimension utilities."""

    @staticmethod
    def rgb(ftensor, true_shape=None):
        if isinstance(ftensor, list):
            return [ImageUtils.rgb(x, true_shape=true_shape) for x in ftensor]
        if isinstance(ftensor, torch.Tensor):
            ftensor = ftensor.detach().cpu().numpy()  # H,W,3
        if ftensor.ndim == 3 and ftensor.shape[0] == 3:
            ftensor = ftensor.transpose(1, 2, 0)
        elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
            ftensor = ftensor.transpose(0, 2, 3, 1)
        if true_shape is not None:
            H, W = true_shape
            ftensor = ftensor[:H, :W]
        if ftensor.dtype == np.uint8:
            img = np.float32(ftensor) / 255
        else:
            img = (ftensor * 0.5) + 0.5
        return img.clip(min=0, max=1)

    @staticmethod
    def get_imshapes(edges, pred_i, pred_j):
        """Extract image shapes from edge predictions.

        Args:
            edges: List of edge tuples (i, j)
            pred_i: Dictionary of predictions for view i
            pred_j: Dictionary of predictions for view j

        Returns:
            List of image shapes (H, W) for each image
        """
        n_imgs = max(max(e) for e in edges) + 1
        imshapes = [None] * n_imgs
        for e, (i, j) in enumerate(edges):
            shape_i = tuple(pred_i[e].shape[0:2])
            shape_j = tuple(pred_j[e].shape[0:2])
            if imshapes[i]:
                assert imshapes[i] == shape_i, f"incorrect shape for image {i}"
            if imshapes[j]:
                assert imshapes[j] == shape_j, f"incorrect shape for image {j}"
            imshapes[i] = shape_i
            imshapes[j] = shape_j
        return imshapes


class ConfidenceUtils:
    """Confidence transformation utilities."""

    @staticmethod
    def get_conf_trf(mode):
        """Get confidence transformation function.

        Args:
            mode: Transformation mode ('log', 'sqrt', 'm1', 'id', 'none')

        Returns:
            Transformation function
        """
        if mode == "log":

            def conf_trf(x):
                return x.log()

        elif mode == "sqrt":

            def conf_trf(x):
                return x.sqrt()

        elif mode == "m1":

            def conf_trf(x):
                return x - 1

        elif mode in ("id", "none"):

            def conf_trf(x):
                return x

        else:
            raise ValueError(f"bad mode for {mode=}")
        return conf_trf


class LossFunctions:
    """Loss computation functions."""

    @staticmethod
    def l05_loss(x, y):
        """Compute L0.5 loss (square root of L2 norm).

        Args:
            x: First tensor (..., D)
            y: Second tensor (..., D)

        Returns:
            L0.5 loss per element (...,)
        """
        return torch.linalg.norm(x - y, dim=-1).sqrt()

    @staticmethod
    def l1_loss(x, y):
        """Compute L1 loss (L2 norm).

        Args:
            x: First tensor (..., D)
            y: Second tensor (..., D)

        Returns:
            L1 loss per element (...,)
        """
        return torch.linalg.norm(x - y, dim=-1)

    @staticmethod
    def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
        """Create a gamma-parameterized loss function.

        This creates a loss function that applies a gamma power to the L1 distance,
        with an optional offset to ensure smooth gradients. When gamma=1, it reduces
        to standard L1 loss.

        Args:
            gamma: Power parameter (typically 0.5-2.0)
            mul: Multiplicative factor (default: 1)
            offset: Offset value (default: None, auto-computed)
            clip: Maximum value to clip distances (default: np.inf)

        Returns:
            Loss function that takes (x, y) and returns loss values
        """
        if offset is None:
            if gamma == 1:
                return LossFunctions.l1_loss
            # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
            offset = (1 / gamma) ** (1 / (gamma - 1))

        def loss_func(x, y):
            return (
                mul * LossFunctions.l1_loss(x, y).clip(max=clip) + offset
            ) ** gamma - offset**gamma

        return loss_func

    @staticmethod
    def meta_gamma_loss():
        """Create a meta loss function that takes alpha as a parameter.

        Returns:
            Function that takes alpha and returns a gamma_loss function
        """
        return lambda alpha: LossFunctions.gamma_loss(alpha)

    @staticmethod
    def l2_dist(a, b, weight):
        """Compute L2 distance between two point sets with weights.

        Args:
            a: First point set (..., D)
            b: Second point set (..., D)
            weight: Weight tensor (...,)

        Returns:
            Weighted L2 distance (...,)
        """
        return (a - b).square().sum(dim=-1) * weight

    @staticmethod
    def l1_dist(a, b, weight):
        """Compute L1 distance between two point sets with weights.

        Args:
            a: First point set (..., D)
            b: Second point set (..., D)
            weight: Weight tensor (...,)

        Returns:
            Weighted L1 distance (...,)
        """
        return (a - b).norm(dim=-1) * weight

    @staticmethod
    def get_all_dists():
        """Get dictionary of all distance functions.

        Returns:
            Dictionary mapping distance names to distance functions
        """
        return dict(l1=LossFunctions.l1_dist, l2=LossFunctions.l2_dist)


class MatchingUtils:
    """Nearest neighbor and correspondence matching utilities."""

    @staticmethod
    @torch.no_grad()
    def bruteforce_reciprocal_nns(A, B, device="cuda", block_size=None, dist="l2"):
        """Compute reciprocal nearest neighbors between two point sets.

        For each point in A, find its nearest neighbor in B, and vice versa.
        Only pairs that are mutual nearest neighbors are returned. Supports
        block-wise processing for large point sets.

        Args:
            A: First point set (N, D)
            B: Second point set (M, D)
            device: Device for computation (default: "cuda")
            block_size: Block size for memory-efficient processing (default: None)
            dist: Distance metric ("l2" or "dot") (default: "l2")

        Returns:
            Tuple of (nn_A, nn_B) where:
                nn_A: Indices of nearest neighbors in B for each point in A (N,)
                nn_B: Indices of nearest neighbors in A for each point in B (M,)
        """
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).to(device)
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).to(device)

        A = A.to(device)
        B = B.to(device)

        if dist == "l2":
            dist_func = torch.cdist
            argmin = torch.min
        elif dist == "dot":

            def dist_func(A, B):
                return A @ B.T

            def argmin(X, dim):
                sim, nn = torch.max(X, dim=dim)
                return sim.neg_(), nn

        else:
            raise ValueError(f"Unknown {dist=}")

        if block_size is None or len(A) * len(B) <= block_size**2:
            dists = dist_func(A, B)
            _, nn_A = argmin(dists, dim=1)
            _, nn_B = argmin(dists, dim=0)
        else:
            dis_A = torch.full((A.shape[0],), float("inf"), device=device, dtype=A.dtype)
            dis_B = torch.full((B.shape[0],), float("inf"), device=device, dtype=B.dtype)
            nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
            nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
            number_of_iteration_A = math.ceil(A.shape[0] / block_size)
            number_of_iteration_B = math.ceil(B.shape[0] / block_size)

            for i in range(number_of_iteration_A):
                A_i = A[i * block_size : (i + 1) * block_size]
                for j in range(number_of_iteration_B):
                    B_j = B[j * block_size : (j + 1) * block_size]
                    dists_blk = dist_func(A_i, B_j)
                    min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                    min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                    col_mask = min_A_i < dis_A[i * block_size : (i + 1) * block_size]
                    line_mask = min_B_j < dis_B[j * block_size : (j + 1) * block_size]

                    dis_A[i * block_size : (i + 1) * block_size][col_mask] = min_A_i[col_mask]
                    dis_B[j * block_size : (j + 1) * block_size][line_mask] = min_B_j[line_mask]

                    nn_A[i * block_size : (i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (
                        j * block_size
                    )
                    nn_B[j * block_size : (j + 1) * block_size][line_mask] = argmin_B_j[
                        line_mask
                    ] + (i * block_size)
        nn_A = nn_A.cpu().numpy()
        nn_B = nn_B.cpu().numpy()
        return nn_A, nn_B

    @staticmethod
    def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
        """Merge correspondences and convert from linear indices to pixel coordinates.

        This function takes two arrays of linear indices and merges them into
        unique correspondences, optionally converting them to (x, y) pixel coordinates.

        Args:
            idx1: Linear indices for first image (N,)
            idx2: Linear indices for second image (N,)
            shape1: Shape (H, W) of first image (required if ret_xy=True)
            shape2: Shape (H, W) of second image (required if ret_xy=True)
            ret_xy: Whether to return pixel coordinates (default: True)
            ret_index: Whether to return indices of unique correspondences (default: False)

        Returns:
            If ret_index=False: Tuple of (xy1, xy2) pixel coordinates
            If ret_index=True: Tuple of (xy1, xy2, indices)
        """
        assert idx1.dtype == idx2.dtype == np.int32

        # unique and sort along idx1
        corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
        if ret_index:
            corres, indices = corres
        xy2, xy1 = corres[:, None].view(np.int32).T

        if ret_xy:
            assert shape1 and shape2
            xy1 = np.unravel_index(xy1, shape1)
            xy2 = np.unravel_index(xy2, shape2)
            if ret_xy != "y_x":
                xy1 = xy1[0].base[:, ::-1]
                xy2 = xy2[0].base[:, ::-1]

        if ret_index:
            return xy1, xy2, indices
        return xy1, xy2

    @staticmethod
    def fast_reciprocal_NNs(
        pts1,
        pts2,
        subsample_or_initxy1=8,
        ret_xy=True,
        pixel_tol=0,
        ret_basin=False,
        device="cuda",
        **matcher_kw,
    ):
        """Find reciprocal nearest neighbors between two point sets iteratively.

        This function performs iterative reciprocal nearest neighbor search,
        starting from a subsampled grid or initial correspondences. It converges
        to stable correspondences through multiple iterations.

        Args:
            pts1: First point set (H1, W1, D)
            pts2: Second point set (H2, W2, D)
            subsample_or_initxy1: Subsampling factor (int) or initial (x, y) coordinates
            ret_xy: Whether to return pixel coordinates (default: True)
            pixel_tol: Pixel tolerance for convergence (default: 0)
            ret_basin: Whether to return basin of attraction map (default: False)
            device: Device for computation (default: "cuda")
            **matcher_kw: Additional keyword arguments for matcher

        Returns:
            Tuple of (xy1, xy2) correspondences, optionally with basin map
        """
        from scipy.spatial import KDTree

        H1, W1, DIM1 = pts1.shape
        H2, W2, DIM2 = pts2.shape
        assert DIM1 == DIM2

        pts1 = pts1.reshape(-1, DIM1)
        pts2 = pts2.reshape(-1, DIM2)

        if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
            S = subsample_or_initxy1
            y1, x1 = np.mgrid[S // 2 : H1 : S, S // 2 : W1 : S].reshape(2, -1)
            max_iter = 10
        else:
            x1, y1 = subsample_or_initxy1
            if isinstance(x1, torch.Tensor):
                x1 = x1.cpu().numpy()
            if isinstance(y1, torch.Tensor):
                y1 = y1.cpu().numpy()
            max_iter = 1

        xy1 = np.int32(np.unique(x1 + W1 * y1))
        xy2 = np.full_like(xy1, -1)
        old_xy1 = xy1.copy()
        old_xy2 = xy2.copy()

        if (
            "dist" in matcher_kw
            or "block_size" in matcher_kw
            or (isinstance(device, str) and device.startswith("cuda"))
            or (isinstance(device, torch.device) and device.type.startswith("cuda"))
        ):
            pts1 = pts1.to(device)
            pts2 = pts2.to(device)
            tree1 = NearestNeighborMatcher(pts1, device=device)
            tree2 = NearestNeighborMatcher(pts2, device=device)
        else:
            pts1, pts2 = to_numpy((pts1, pts2))
            tree1 = KDTree(pts1)
            tree2 = KDTree(pts2)

        notyet = np.ones(len(xy1), dtype=bool)
        basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

        niter = 0
        while notyet.any():
            _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
            if not ret_basin:
                notyet &= old_xy2 != xy2

            _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
            if ret_basin:
                basin[old_xy1[notyet]] = xy1[notyet]
            notyet &= old_xy1 != xy1

            niter += 1
            if niter >= max_iter:
                break

            old_xy2[:] = xy2
            old_xy1[:] = xy1

        if pixel_tol > 0:
            old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
            new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
            dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
            converged = dis < pixel_tol
            if not isinstance(subsample_or_initxy1, int):
                xy1 = old_xy1
        else:
            converged = ~notyet

        xy1, xy2 = MatchingUtils.merge_corres(
            xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy
        )
        if ret_basin:
            return xy1, xy2, basin
        return xy1, xy2


class NearestNeighborMatcher:
    """Matcher class for finding nearest neighbors using brute-force distance computation.

    This class provides a simple interface for nearest neighbor search using
    GPU-accelerated distance computation. It wraps bruteforce_reciprocal_nns
    for query operations.

    Args:
        db_pts: Database points (N, D) tensor
        device: Device for computation (default: "cuda")
    """

    def __init__(self, db_pts, device="cuda"):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = MatchingUtils.bruteforce_reciprocal_nns(
            queries, self.db_pts, device=self.device, **kw
        )
        dis = None
        return dis, nnA


# MASt3R Source Code - Sparse Global Alignment (from sparse_ga.py)
class SparseGA:
    """Sparse Global Alignment class for managing sparse scene optimization results.

    This class stores and provides access to optimized camera poses, depthmaps,
    and 3D points from sparse scene optimization. It can also generate dense
    point clouds from cached canonical views.

    Args:
        img_paths: List of image file paths
        pairs_in: List of image pairs with their data
        res_fine: Dictionary with optimization results (intrinsics, cam2w, depthmaps, pts3d)
        anchors: List of anchor points for each image
        canonical_paths: Optional list of paths to cached canonical views
    """

    def __init__(self, img_paths, pairs_in, res_fine, anchors, canonical_paths=None):
        def torgb(x):
            return (x[0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(min=0.0, max=1.0)

        # Build a mapping from image instance to image tensor
        # This ensures each image is fetched correctly and uniquely
        img_dict = {}
        for im1, im2 in pairs_in:
            if im1["instance"] not in img_dict:
                img_dict[im1["instance"]] = torgb(im1["img"])
            if im2["instance"] not in img_dict:
                img_dict[im2["instance"]] = torgb(im2["img"])

        self.canonical_paths = canonical_paths
        self.img_paths = img_paths
        # Fetch images in order, ensuring each img_path gets its corresponding image
        self.imgs = []
        missing = []
        for img_path in img_paths:
            img = img_dict.get(img_path, None)
            if img is None:
                missing.append(img_path)
            self.imgs.append(img)

        # Verify all images were found
        if missing:
            raise ValueError(f"Could not find images for: {missing} in pairs_in")
        self.intrinsics = res_fine["intrinsics"]
        self.cam2w = res_fine["cam2w"]
        self.depthmaps = res_fine["depthmaps"]
        self.pts3d = res_fine["pts3d"]
        self.pts3d_colors = []
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            # Convert to integers for array indexing
            x = x.astype(np.int64)
            y = y.astype(np.int64)
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        """Get focal lengths for all cameras.

        Returns:
            Tensor of focal lengths (N,)
        """
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        """Get principal points for all cameras.

        Returns:
            Tensor of principal points (N, 2)
        """
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        """Get camera-to-world poses for all cameras.

        Returns:
            Tensor of camera poses (N, 4, 4)
        """
        return self.cam2w

    def get_sparse_pts3d(self):
        """Get sparse 3D points for all images.

        Returns:
            List of 3D point clouds, one per image
        """
        return self.pts3d

    def get_dense_pts3d(self, clean_depth=True, subsample=8):
        """Generate dense 3D points from cached canonical views.

        This function loads canonical views from cache and generates dense
        point clouds by densifying the sparse depthmaps.

        Args:
            clean_depth: Whether to clean the point cloud (default: True)
            subsample: Subsampling factor for anchor points (default: 8)

        Returns:
            Tuple of (pts3d, depthmaps, confs) where:
                pts3d: List of dense 3D point clouds
                depthmaps: List of dense depth maps
                confs: List of confidence maps
        """
        assert self.canonical_paths, "cache_path is required for dense 3d points"
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = CanonicalViewUtils.anchor_depth_offsets(
                canon2, {i: (pixels, None)}, subsample=subsample
            )
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = ProjectionUtils.make_pts3d(
            anchors,
            self.intrinsics,
            self.cam2w,
            [d.ravel() for d in self.depthmaps],
            base_focals=base_focals,
            ret_depth=True,
        )

        if clean_depth:
            confs = PointCloudUtils.clean_pointcloud(
                confs, self.intrinsics, GeometryUtils.inv(self.cam2w), depthmaps, pts3d
            )

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        """Get colors for sparse 3D points.

        Returns:
            List of color arrays, one per image
        """
        return self.pts3d_colors

    def get_depthmaps(self):
        """Get sparse depthmaps for all images.

        Returns:
            List of depth maps, one per image
        """
        return self.depthmaps

    def get_masks(self):
        """Get masks for all images (currently returns full masks).

        Returns:
            List of slice objects, one per image
        """
        return [slice(None, None) for _ in range(len(self.imgs))]


class ProjectionUtils:
    """3D/2D projection and transformation utilities."""

    @staticmethod
    @lru_cache
    def mask110(device, dtype):
        """Create a cached mask tensor (1, 1, 0) for projection operations.

        Args:
            device: Device for the tensor
            dtype: Data type for the tensor

        Returns:
            Cached tensor with values (1, 1, 0)
        """
        return torch.tensor((1, 1, 0), device=device, dtype=dtype)

    @staticmethod
    def proj3d(inv_K, pixels, z):
        """Project 2D pixels to 3D points using inverse intrinsics and depth.

        Args:
            inv_K: Inverse camera intrinsics matrix (3, 3)
            pixels: Pixel coordinates (N, 2) or (N, 3) with homogeneous coordinate
            z: Depth values (N,)

        Returns:
            3D points (N, 3)
        """
        if pixels.shape[-1] == 2:
            pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
        return z.unsqueeze(-1) * (
            pixels * inv_K.diag() + inv_K[:, 2] * ProjectionUtils.mask110(z.device, z.dtype)
        )

    @staticmethod
    def make_pts3d(anchors, K, cam2w, depthmaps, base_focals=None, ret_depth=False):
        """Generate 3D points from anchors, camera parameters, and depthmaps.

        This function projects anchor pixels to 3D space using camera intrinsics
        and depthmaps, then transforms them to world coordinates. It can optionally
        compensate for focal length differences using base_focals.

        Args:
            anchors: Dict mapping image index to (pixels, idxs, offsets) tuples
            K: Camera intrinsics (N, 3, 3)
            cam2w: Camera-to-world poses (N, 4, 4)
            depthmaps: List of depth values, one per image
            base_focals: Optional list of base focal lengths for compensation
            ret_depth: Whether to return depth values (default: False)

        Returns:
            List of 3D points (N, 3) per image, optionally with depth values
        """
        focals = K[:, 0, 0]
        invK = GeometryUtils.inv(K)
        all_pts3d = []
        depth_out = []

        for img, (pixels, idxs, offsets) in anchors.items():
            # from depthmaps to 3d points
            if base_focals is None:
                pass
            else:
                # compensate for focal
                offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])

            pts3d = ProjectionUtils.proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
            if ret_depth:
                depth_out.append(pts3d[..., 2])

            # rotate to world coordinate
            pts3d = GeometryUtils.geom_transform(cam2w[img], pts3d)
            all_pts3d.append(pts3d)

        if ret_depth:
            return all_pts3d, depth_out
        return all_pts3d

    @staticmethod
    def reproj2d(Trf, pts3d):
        """Reproject 3D points to 2D image coordinates.

        Args:
            Trf: Transformation matrix (3, 4) or (4, 4)
            pts3d: 3D points (N, 3)

        Returns:
            2D pixel coordinates (N, 2), clipped to reasonable range
        """
        res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
        clipped_z = res[:, 2:3].clip(min=1e-3)
        uv = res[:, 0:2] / clipped_z
        return uv.clip(min=-1000, max=2000)

    @staticmethod
    def backproj(K, depthmap, subsample):
        """Backproject depthmap to 3D points using camera intrinsics.

        Args:
            K: Camera intrinsics (3, 3)
            depthmap: Depth map (H, W)
            subsample: Subsampling factor for pixel grid

        Returns:
            3D points (H, W, 3)
        """
        H, W = depthmap.shape
        uv = np.mgrid[
            subsample // 2 : subsample * W : subsample, subsample // 2 : subsample * H : subsample
        ].T.reshape(H, W, 2)
        xyz = depthmap.unsqueeze(-1) * GeometryUtils.geom_transform(
            GeometryUtils.inv(K), to_device(uv, K.device), ncol=3
        )
        return xyz


class GraphUtils:
    """Graph algorithms and utilities."""

    @staticmethod
    def bfs(tree, start_node):
        """Perform breadth-first search on a graph.

        Args:
            tree: Sparse graph matrix
            start_node: Starting node index

        Returns:
            Tuple of (ranks, predecessors) where:
                ranks: Node ranks in BFS order
                predecessors: Predecessor array for each node
        """
        order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
        ranks = np.arange(len(order))
        ranks[order] = ranks.copy()
        return ranks, predecessors

    @staticmethod
    def compute_min_spanning_tree(pws):
        """Compute minimum spanning tree from pairwise scores and find root node.

        This function builds a minimum spanning tree from pairwise scores, then
        finds the central node (farthest from any leaf) to use as the root.
        Returns the root index and ordered list of MST edges.

        Args:
            pws: Pairwise scores matrix (N, N)

        Returns:
            Tuple of (root_idx, edges) where:
                root_idx: Index of the root node
                edges: List of (parent, child) tuples in BFS order
        """
        sparse_graph = sp.dok_array(pws.shape)
        for i, j in pws.nonzero().cpu().tolist():
            sparse_graph[i, j] = -float(pws[i, j])
        msp = sp.csgraph.minimum_spanning_tree(sparse_graph)

        # now reorder the oriented edges, starting from the central point
        ranks1, _ = GraphUtils.bfs(msp, 0)
        ranks2, _ = GraphUtils.bfs(msp, ranks1.argmax())
        ranks1, _ = GraphUtils.bfs(msp, ranks2.argmax())
        # this is the point farther from any leaf
        root = np.minimum(ranks1, ranks2).argmax()

        # find the ordered list of edges that describe the tree
        order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
        order = order[1:]
        edges = [(predecessors[i], i) for i in order]

        return root, edges


class CanonicalViewUtils:
    """Canonical view processing and correspondence utilities."""

    @staticmethod
    def convert_dust3r_pairs_naming(imgs, pairs_in):
        """Convert Dust3r pair naming convention to use image instances.

        This function updates the "instance" field in pairs_in to use the actual
        image paths from the imgs list, converting from index-based to path-based naming.

        Args:
            imgs: List of image paths
            pairs_in: List of image pairs, each containing dicts with "idx" and "instance" fields

        Returns:
            Modified pairs_in list with updated "instance" fields
        """
        for pair_id in range(len(pairs_in)):
            for i in range(2):
                pairs_in[pair_id][i]["instance"] = imgs[pairs_in[pair_id][i]["idx"]]
        return pairs_in

    @staticmethod
    def load_corres(path_corres, device, min_conf_thr):
        """Load correspondences from a cached file and filter by confidence.

        Args:
            path_corres: Path to correspondence cache file
            device: Device for loaded tensors
            min_conf_thr: Minimum confidence threshold for filtering

        Returns:
            Tuple of (score, (xy1, xy2, confs)) where:
                score: Pairwise matching score
                xy1: Corresponding points in first image (N, 2)
                xy2: Corresponding points in second image (N, 2)
                confs: Confidence values (N,)
        """
        score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
        valid = confs > min_conf_thr if min_conf_thr else slice(None)
        return score, (xy1[valid], xy2[valid], confs[valid])

    @staticmethod
    def canonical_view(ptmaps11, confs11, subsample, mode="avg-angle"):
        """Compute canonical view from multiple point maps and confidence maps.

        This function aggregates multiple point maps into a canonical representation
        using weighted averaging. It supports two modes: "avg-angle" (angle-based)
        and "avg-reldepth" (relative depth-based).

        Args:
            ptmaps11: List of point maps (N, H, W, 3)
            confs11: List of confidence maps (N, H, W)
            subsample: Subsampling factor for anchor points
            mode: Aggregation mode ("avg-angle" or "avg-reldepth") (default: "avg-angle")

        Returns:
            Tuple of (canon, canon2, confs) where:
                canon: Canonical point map (H, W, 3)
                canon2: Canonical depth offset map (H, W)
                confs: Aggregated confidence map (H, W)
        """
        assert len(ptmaps11) == len(confs11) > 0, "not a single view1 for img={i}"

        # canonical pointmap is just a weighted average
        confs11 = confs11.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

        canon_depth = ptmaps11[..., 2].unsqueeze(1)
        S = slice(subsample // 2, None, subsample)
        center_depth = canon_depth[:, :, S, S]
        center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

        stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
        stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

        if mode == "avg-reldepth":
            rel_depth = stacked_depth / center_depth
            stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
            canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

        elif mode == "avg-angle":
            xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
            stacked_xy = F.pixel_unshuffle(xy, subsample)
            B, _, H, W = stacked_xy.shape
            stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
            stacked_radius.clip_(min=1e-8)

            stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
            avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

            # back to depth
            stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

            canon2 = F.pixel_shuffle(
                (1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample
            ).squeeze()
        else:
            raise ValueError(f"bad {mode=}")

        confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
        return canon, canon2, confs

    @staticmethod
    def anchor_depth_offsets(canon_depth, pixels, subsample=8):
        """Compute depth offsets for pixels relative to anchor points.

        This function creates a grid of anchor points and computes relative depth
        offsets for given pixel coordinates. The anchors are placed on a subsampled grid.

        Args:
            canon_depth: Canonical depth map (H, W)
            pixels: Dict mapping image index to (pixel_coords, confs) tuples
            subsample: Subsampling factor for anchor grid (default: 8)

        Returns:
            Tuple of (core_idxs, core_offs) where:
                core_idxs: Dict mapping image index to anchor indices
                core_offs: Dict mapping image index to depth offset ratios
        """
        device = canon_depth.device

        # create a 2D grid of anchor 3D points
        H1, W1 = canon_depth.shape
        yx = np.mgrid[subsample // 2 : H1 : subsample, subsample // 2 : W1 : subsample]
        H2, W2 = yx.shape[1:]
        cy, cx = yx.reshape(2, -1)
        core_depth = canon_depth[cy, cx]
        assert (core_depth > 0).all()

        # slave 3d points (attached to core 3d points)
        core_idxs = {}
        core_offs = {}

        for img2, (xy1, _confs) in pixels.items():
            px, py = xy1.long().T

            # find nearest anchor == block quantization
            core_idx = (py // subsample) * W2 + (px // subsample)
            core_idxs[img2] = core_idx.to(device)

            # compute relative depth offsets w.r.t. anchors
            ref_z = core_depth[core_idx]
            pts_z = canon_depth[py, px]
            offset = pts_z / ref_z
            core_offs[img2] = offset.detach().to(device)

        return core_idxs, core_offs


PairOfSlices = namedtuple(
    "ImgPair",
    "img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum",
)


class CanonicalViewUtils:
    """Canonical view processing and correspondence utilities."""

    @staticmethod
    def convert_dust3r_pairs_naming(imgs, pairs_in):
        """Convert Dust3r pair naming convention to use image instances.

        This function updates the "instance" field in pairs_in to use the actual
        image paths from the imgs list, converting from index-based to path-based naming.

        Args:
            imgs: List of image paths
            pairs_in: List of image pairs, each containing dicts with "idx" and "instance" fields

        Returns:
            Modified pairs_in list with updated "instance" fields
        """
        for pair_id in range(len(pairs_in)):
            for i in range(2):
                pairs_in[pair_id][i]["instance"] = imgs[pairs_in[pair_id][i]["idx"]]
        return pairs_in

    @staticmethod
    def load_corres(path_corres, device, min_conf_thr):
        """Load correspondences from a cached file and filter by confidence.

        Args:
            path_corres: Path to correspondence cache file
            device: Device for loaded tensors
            min_conf_thr: Minimum confidence threshold for filtering

        Returns:
            Tuple of (score, (xy1, xy2, confs)) where:
                score: Pairwise matching score
                xy1: Corresponding points in first image (N, 2)
                xy2: Corresponding points in second image (N, 2)
                confs: Confidence values (N,)
        """
        score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
        valid = confs > min_conf_thr if min_conf_thr else slice(None)
        return score, (xy1[valid], xy2[valid], confs[valid])

    @staticmethod
    def canonical_view(ptmaps11, confs11, subsample, mode="avg-angle"):
        """Compute canonical view from multiple point maps and confidence maps.

        This function aggregates multiple point maps into a canonical representation
        using weighted averaging. It supports two modes: "avg-angle" (angle-based)
        and "avg-reldepth" (relative depth-based).

        Args:
            ptmaps11: List of point maps (N, H, W, 3)
            confs11: List of confidence maps (N, H, W)
            subsample: Subsampling factor for anchor points
            mode: Aggregation mode ("avg-angle" or "avg-reldepth") (default: "avg-angle")

        Returns:
            Tuple of (canon, canon2, confs) where:
                canon: Canonical point map (H, W, 3)
                canon2: Canonical depth offset map (H, W)
                confs: Aggregated confidence map (H, W)
        """
        assert len(ptmaps11) == len(confs11) > 0, "not a single view1 for img={i}"

        # canonical pointmap is just a weighted average
        confs11 = confs11.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

        canon_depth = ptmaps11[..., 2].unsqueeze(1)
        S = slice(subsample // 2, None, subsample)
        center_depth = canon_depth[:, :, S, S]
        center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

        stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
        stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

        if mode == "avg-reldepth":
            rel_depth = stacked_depth / center_depth
            stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
            canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

        elif mode == "avg-angle":
            xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
            stacked_xy = F.pixel_unshuffle(xy, subsample)
            B, _, H, W = stacked_xy.shape
            stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
            stacked_radius.clip_(min=1e-8)

            stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
            avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

            # back to depth
            stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

            canon2 = F.pixel_shuffle(
                (1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample
            ).squeeze()
        else:
            raise ValueError(f"bad {mode=}")

        confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
        return canon, canon2, confs

    @staticmethod
    def anchor_depth_offsets(canon_depth, pixels, subsample=8):
        """Compute depth offsets for pixels relative to anchor points.

        This function creates a grid of anchor points and computes relative depth
        offsets for given pixel coordinates. The anchors are placed on a subsampled grid.

        Args:
            canon_depth: Canonical depth map (H, W)
            pixels: Dict mapping image index to (pixel_coords, confs) tuples
            subsample: Subsampling factor for anchor grid (default: 8)

        Returns:
            Tuple of (core_idxs, core_offs) where:
                core_idxs: Dict mapping image index to anchor indices
                core_offs: Dict mapping image index to depth offset ratios
        """
        device = canon_depth.device

        # create a 2D grid of anchor 3D points
        H1, W1 = canon_depth.shape
        yx = np.mgrid[subsample // 2 : H1 : subsample, subsample // 2 : W1 : subsample]
        H2, W2 = yx.shape[1:]
        cy, cx = yx.reshape(2, -1)
        core_depth = canon_depth[cy, cx]
        assert (core_depth > 0).all()

        # slave 3d points (attached to core 3d points)
        core_idxs = {}
        core_offs = {}

        for img2, (xy1, _confs) in pixels.items():
            px, py = xy1.long().T

            # find nearest anchor == block quantization
            core_idx = (py // subsample) * W2 + (px // subsample)
            core_idxs[img2] = core_idx.to(device)

            # compute relative depth offsets w.r.t. anchors
            ref_z = core_depth[core_idx]
            pts_z = canon_depth[py, px]
            offset = pts_z / ref_z
            core_offs[img2] = offset.detach().to(device)

        return core_idxs, core_offs

    @staticmethod
    @torch.no_grad()
    def prepare_canonical_data(
        imgs,
        tmp_pairs,
        subsample,
        order_imgs=False,
        min_conf_thr=0,
        cache_path=None,
        device="cuda",
        **kw,
    ):
        """Prepare canonical views and correspondence data for sparse optimization.

        This function processes pairwise predictions to create canonical views for
        each image, computes focal lengths, and extracts depth offsets. It can
        cache results to disk for faster subsequent runs.

        Args:
            imgs: List of image paths
            tmp_pairs: Dict mapping (img1, img2) to ((path1, path2), path_corres)
            subsample: Subsampling factor for canonical views
            order_imgs: Whether to order images (default: False)
            min_conf_thr: Minimum confidence threshold (default: 0)
            cache_path: Optional path for caching canonical views
            device: Device for tensors (default: "cuda")
            **kw: Additional keyword arguments passed to canonical_view

        Returns:
            Tuple of (tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21)
        """
        canonical_views = {}
        pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
        canonical_paths = []
        preds_21 = {}

        for img in tqdm.tqdm(imgs):
            if cache_path:
                cache = os.path.join(
                    cache_path, "canon_views", FileUtils.hash_md5(img) + f"_{subsample=}_{kw=}.pth"
                )
                canonical_paths.append(cache)
            try:
                (canon, canon2, cconf), focal = torch.load(cache, map_location=device)
            except IOError:
                canon = focal = None

            # collect all pred1
            n_pairs = sum((img in pair) for pair in tmp_pairs)

            ptmaps11 = None
            pixels = {}
            n = 0
            for (img1, img2), ((path1, path2), path_corres) in tmp_pairs.items():
                score = None
                if img == img1:
                    X, C, X2, C2 = torch.load(path1, map_location=device)
                    score, (xy1, xy2, confs) = CanonicalViewUtils.load_corres(
                        path_corres, device, min_conf_thr
                    )
                    pixels[img2] = xy1, confs
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img2] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].ravel(),
                    )

                if img == img2:
                    if path2 is None:
                        # No reverse pair available - try to use path1 with swapped data
                        # This can happen with Dust3r when symmetrize=True doesn't create reverse pairs
                        # Load path1 and swap X1/X2 and C1/C2 to get reverse direction
                        X1_loaded, C1_loaded, X2_loaded, C2_loaded = torch.load(
                            path1, map_location=device
                        )
                        # For reverse: X = X2 (pts3d for img2), C = C2, X2 = X1 (pts3d for img1), C2 = C1
                        X, C, X2, C2 = X2_loaded, C2_loaded, X1_loaded, C1_loaded
                        # Try to find reverse correspondence file by swapping indices in filename
                        # Extract base directory and filename
                        corres_dir = os.path.dirname(path_corres)
                        corres_filename = os.path.basename(path_corres)
                        # Try to extract indices from filename (format: "idx1-idx2.pth")
                        # If filename contains "-", try swapping
                        if "-" in corres_filename:
                            parts = corres_filename.rsplit(".", 1)[0].split("-", 1)
                            if len(parts) == 2:
                                idx1_str, idx2_str = parts
                                corres_path_rev = os.path.join(
                                    corres_dir, f"{idx2_str}-{idx1_str}.pth"
                                )
                                if os.path.exists(corres_path_rev):
                                    score, (xy1_rev, xy2_rev, confs_rev) = (
                                        CanonicalViewUtils.load_corres(
                                            corres_path_rev, device, min_conf_thr
                                        )
                                    )
                                    pixels[img1] = (
                                        xy2_rev,
                                        confs_rev,
                                    )  # For img2, pixels[img1] uses xy2_rev
                                else:
                                    # Use forward correspondences but swap xy1 and xy2
                                    score, (xy1_fwd, xy2_fwd, confs_fwd) = (
                                        CanonicalViewUtils.load_corres(
                                            path_corres, device, min_conf_thr
                                        )
                                    )
                                    pixels[img1] = (
                                        xy2_fwd,
                                        confs_fwd,
                                    )  # Swap: use xy2_fwd for img1 when viewing from img2
                            else:
                                # Fallback: use forward correspondences with swapped xy
                                score, (xy1_fwd, xy2_fwd, confs_fwd) = (
                                    CanonicalViewUtils.load_corres(
                                        path_corres, device, min_conf_thr
                                    )
                                )
                                pixels[img1] = xy2_fwd, confs_fwd
                        else:
                            # Fallback: use forward correspondences with swapped xy
                            score, (xy1_fwd, xy2_fwd, confs_fwd) = CanonicalViewUtils.load_corres(
                                path_corres, device, min_conf_thr
                            )
                            pixels[img1] = xy2_fwd, confs_fwd
                    else:
                        X, C, X2, C2 = torch.load(path2, map_location=device)
                        score, (xy1, xy2, confs) = CanonicalViewUtils.load_corres(
                            path_corres, device, min_conf_thr
                        )
                        pixels[img1] = xy2, confs
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img1] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].ravel(),
                    )

                if score is not None:
                    i, j = imgs.index(img1), imgs.index(img2)
                    score = score[2]
                    pairwise_scores[i, j] = score
                    pairwise_scores[j, i] = score

                    if canon is not None:
                        continue
                    if ptmaps11 is None:
                        H, W = C.shape
                        ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                        confs11 = torch.empty((n_pairs, H, W), device=device)

                    ptmaps11[n] = X
                    confs11[n] = C
                    n += 1

            if canon is None:
                # Filter out verbose from kw as canonical_view doesn't accept it
                kw_filtered = {k: v for k, v in kw.items() if k != "verbose"}
                canon, canon2, cconf = CanonicalViewUtils.canonical_view(
                    ptmaps11, confs11, subsample, **kw_filtered
                )
                del ptmaps11
                del confs11

            # compute focals
            H, W = canon.shape[:2]
            pp = torch.tensor([W / 2, H / 2], device=device)
            if focal is None:
                focal = CameraUtils.estimate_focal_knowing_depth(
                    canon[None], pp, focal_mode="weiszfeld", min_focal=0.5, max_focal=3.5
                )
                if cache:
                    torch.save(to_cpu(((canon, canon2, cconf), focal)), FileUtils.mkdir_for(cache))

            # extract depth offsets with correspondences
            core_depth = canon[subsample // 2 :: subsample, subsample // 2 :: subsample, 2]
            idxs, offsets = CanonicalViewUtils.anchor_depth_offsets(
                canon2, pixels, subsample=subsample
            )

            # Debug: check depth initialization
            if kw.get("verbose", False):
                print(
                    f"[prepare_canonical_data] Image {img}: core_depth shape={core_depth.shape}, median={core_depth.median().item():.6f}, min={core_depth.min().item():.6f}, max={core_depth.max().item():.6f}"
                )
                print(
                    f"  canon shape={canon.shape}, canon2 shape={canon2.shape if canon2 is not None else None}"
                )

            canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

        return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21

    @staticmethod
    def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32):
        """Condense and aggregate canonical data into optimization-ready format.

        This function aggregates principal points, shapes, focals, core depths,
        anchors, and correspondences from canonical views into structured data
        suitable for sparse scene optimization.

        Args:
            imgs: List of image paths
            tmp_paths: List of (img1, img2) tuples representing pairs
            canonical_views: Dict mapping image paths to canonical view data
            preds_21: Dict mapping image paths to predictions
            dtype: Data type for tensors (default: torch.float32)

        Returns:
            Tuple of (imsizes, principal_points, focals, core_depth, img_anchors,
                      corres, corres2d, subsamp_preds_21)
        """
        # aggregate all data properly
        set_imgs = set(imgs)

        principal_points = []
        shapes = []
        focals = []
        core_depth = []
        img_anchors = {}
        tmp_pixels = {}

        for idx1, img1 in enumerate(imgs):
            # load stuff
            pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views[img1]

            principal_points.append(pp)
            shapes.append(shape)
            focals.append(focal)
            core_depth.append(anchors)

            img_uv1 = []
            img_idxs = []
            img_offs = []
            cur_n = [0]

            for img2, (pixels, match_confs) in pixels_confs.items():
                if img2 not in set_imgs:
                    continue
                assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
                img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
                img_idxs.append(idxs[img2])
                img_offs.append(offsets[img2])
                cur_n.append(cur_n[-1] + len(pixels))
                tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
            img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))

        all_confs = []
        imgs_slices = []
        corres2d = {img: [] for img in range(len(imgs))}

        for img1, img2 in tmp_paths:
            try:
                pix1, confs1, slice1 = tmp_pixels[img1, img2]
                pix2, confs2, slice2 = tmp_pixels[img2, img1]
            except KeyError:
                continue
            img1 = imgs.index(img1)
            img2 = imgs.index(img2)
            confs = (confs1 * confs2).sqrt()

            # prepare for loss_3d
            all_confs.append(confs)
            anchor_idxs1 = canonical_views[imgs[img1]][5][imgs[img2]]
            anchor_idxs2 = canonical_views[imgs[img2]][5][imgs[img1]]
            imgs_slices.append(
                PairOfSlices(
                    img1,
                    slice1,
                    pix1,
                    anchor_idxs1,
                    img2,
                    slice2,
                    pix2,
                    anchor_idxs2,
                    confs,
                    float(confs.sum()),
                )
            )

            # prepare for loss_2d
            corres2d[img1].append((pix1, confs, img2, slice2))
            corres2d[img2].append((pix2, confs, img1, slice1))

        all_confs = torch.cat(all_confs)
        corres = (all_confs, float(all_confs.sum()), imgs_slices)

        def aggreg_matches(img1, list_matches):
            pix1, confs, img2, slice2 = zip(*list_matches)
            all_pix1 = torch.cat(pix1).to(dtype)
            all_confs = torch.cat(confs).to(dtype)
            return (
                img1,
                all_pix1,
                all_confs,
                float(all_confs.sum()),
                [(j, sl2) for j, sl2 in zip(img2, slice2)],
            )

        corres2d = [aggreg_matches(img, m) for img, m in corres2d.items()]

        imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)
        principal_points = torch.stack(principal_points)
        focals = torch.cat(focals)

        # Subsample preds_21
        subsamp_preds_21 = {}
        for imk, imv in preds_21.items():
            subsamp_preds_21[imk] = {}
            for im2k, (pred, conf) in preds_21[imk].items():
                idxs = img_anchors[imgs.index(im2k)][1]
                subsamp_preds_21[imk][im2k] = (pred[idxs], conf[idxs])

        return (
            imsizes,
            principal_points,
            focals,
            core_depth,
            img_anchors,
            corres,
            corres2d,
            subsamp_preds_21,
        )


def _extract_initial_poses_from_pairwise(
    filelist,
    pairs_output,
    imsizes,
    base_focals,
    pps,
    mst,
    subsample,
    cache_dir,
    device,
    dtype,
    verbose=True,
):
    """
    Extract initial camera poses from pairwise predictions using MST.

    Args:
        filelist: List of image file names
        pairs_output: Dictionary mapping (img1, img2) -> ((path1, path2), path_corres) with cached pair data
        imsizes: Tensor of image sizes (W, H) for each image
        base_focals: Base focal lengths
        pps: Principal points
        mst: Minimum spanning tree (root_idx, [(parent, child), ...])
        subsample: Subsampling factor (not used here, but kept for compatibility)
        cache_dir: Cache directory containing pair data files
        device: Device for tensors
        dtype: Data type for tensors
        verbose: Whether to print debug messages

    Returns:
        rel_quats: List of relative quaternions for each camera (along MST)
        rel_trans: List of relative translations for each camera (along MST)
        abs_cam2w: Absolute camera-to-world poses (for reference)
    """
    n_imgs = len(filelist)

    # Build edges from pairs_output - load full-resolution point clouds from cache
    edges = []
    pred_i_dict = {}
    pred_j_dict = {}
    conf_i_dict = {}
    conf_j_dict = {}

    if pairs_output is None:
        if verbose:
            print("Warning: pairs_output is None, cannot extract initial poses")
        return None, None, None

    for pair_key, (pair_paths, corres_path) in pairs_output.items():
        img1_name, img2_name = pair_key
        try:
            img1_idx = filelist.index(img1_name)
            img2_idx = filelist.index(img2_name)
        except ValueError:
            # Image not in filelist, skip
            continue

        # Load cached pair data: (X1, C1, X2, C2)
        # X1: pts3d for view1 (full resolution), C1: conf for view1
        # X2: pts3d for view2 (in view1's frame), C2: conf for view2
        try:
            cache_path1, cache_path2 = pair_paths
            X1, C1, X2, C2 = torch.load(cache_path1, map_location=device)

            # Ensure tensors are on correct device and dtype
            if isinstance(X1, torch.Tensor):
                X1 = X1.to(device=device, dtype=dtype)
            else:
                X1 = torch.tensor(X1, device=device, dtype=dtype)

            if isinstance(C1, torch.Tensor):
                C1 = C1.to(device=device, dtype=dtype)
            else:
                C1 = torch.tensor(C1, device=device, dtype=dtype)

            if isinstance(X2, torch.Tensor):
                X2 = X2.to(device=device, dtype=dtype)
            else:
                X2 = torch.tensor(X2, device=device, dtype=dtype)

            if isinstance(C2, torch.Tensor):
                C2 = C2.to(device=device, dtype=dtype)
            else:
                C2 = torch.tensor(C2, device=device, dtype=dtype)

            edge_key = (img1_idx, img2_idx)
            if edge_key not in edges:
                edges.append(edge_key)

            # X1 is the point cloud for view1 (img1), X2 is for view2 (img2) in view1's frame
            pred_i_dict[edge_key] = X1
            pred_j_dict[edge_key] = X2
            conf_i_dict[edge_key] = C1
            conf_j_dict[edge_key] = C2

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load pair data for {img1_name}-{img2_name}: {e}")
            continue

    if len(edges) == 0:
        if verbose:
            print("Warning: No pairwise predictions found, using identity poses")
        return None, None, None

    # Use minimum_spanning_tree to extract initial poses
    # Convert to format expected by minimum_spanning_tree
    # Note: imsizes is (W, H), but imshapes expects (H, W)
    imshapes_full = [(int(h), int(w)) for w, h in imsizes]

    # Build pred_i and pred_j dictionaries in the format expected by minimum_spanning_tree
    # The function expects EdgeUtils.edge_str(i, j) format for keys and (H, W, 3) point clouds
    pred_i = {}
    pred_j = {}
    conf_i = {}
    conf_j = {}
    imshapes_used = []

    for i, j in edges:
        # Use EdgeUtils.edge_str function format: "i_j" string
        key = EdgeUtils.edge_str(i, j)

        # Point clouds from cache are full resolution (H, W, 3) tensors
        pts3d_i_tensor = pred_i_dict[(i, j)]
        pts3d_j_tensor = pred_j_dict.get((i, j), pred_i_dict[(i, j)])
        conf_i_tensor = conf_i_dict[(i, j)]
        conf_j_tensor = conf_j_dict.get((i, j), conf_i_dict[(i, j)])

        # Determine actual dimensions from point cloud shape
        if len(pts3d_i_tensor.shape) == 3:
            # Already in (H, W, 3) format - use directly
            H_used, W_used = pts3d_i_tensor.shape[:2]
            pred_i[key] = pts3d_i_tensor
            pred_j[key] = pts3d_j_tensor

            # Reshape confidence maps if needed
            if len(conf_i_tensor.shape) == 2:
                conf_i_reshaped = conf_i_tensor
                conf_j_reshaped = conf_j_tensor
            else:
                conf_i_reshaped = conf_i_tensor.reshape(H_used, W_used)
                conf_j_reshaped = conf_j_tensor.reshape(H_used, W_used)

            conf_i[key] = conf_i_reshaped
            conf_j[key] = conf_j_reshaped

            # Store dimensions for this image
            if i < len(imshapes_used):
                if H_used * W_used > imshapes_used[i][0] * imshapes_used[i][1]:
                    imshapes_used[i] = (H_used, W_used)
            else:
                while len(imshapes_used) <= i:
                    H_full, W_full = imshapes_full[len(imshapes_used)]
                    imshapes_used.append((H_full, W_full))
                imshapes_used[i] = (H_used, W_used)

            continue  # Skip to next edge since we've already processed this one
        else:
            # Flattened point cloud - need to reshape
            pts3d_i_flat = pts3d_i_tensor
            pts3d_j_flat = pts3d_j_tensor
            conf_i_flat = conf_i_tensor
            conf_j_flat = conf_j_tensor

            H_full, W_full = imshapes_full[i]
            num_points = pts3d_i_flat.shape[0]

            # Check if it matches full resolution
            if num_points == H_full * W_full:
                H_used, W_used = H_full, W_full
            # Check if it matches subsampled resolution
            elif num_points == (H_full // subsample) * (W_full // subsample):
                H_used, W_used = H_full // subsample, W_full // subsample
            else:
                # Try to infer dimensions from the point cloud size
                # Find factors that could represent H and W
                # Common aspect ratios: 4:3, 16:9, 1:1
                if verbose:
                    print(
                        f"Warning: Point cloud size {num_points} doesn't match expected dimensions for edge {i}-{j}"
                    )
                    print(f"  Full resolution: {H_full}x{W_full} = {H_full * W_full}")
                    print(
                        f"  Subsampled ({subsample}x): {H_full // subsample}x{W_full // subsample} = {(H_full // subsample) * (W_full // subsample)}"
                    )

                # Try to find reasonable H, W that multiply to num_points
                # Use the full resolution aspect ratio as a guide
                aspect_ratio = W_full / H_full if H_full > 0 else 1.0

                # Find H such that H * W = num_points and W/H ≈ aspect_ratio
                # Start with a guess based on aspect ratio
                H_guess = int((num_points / aspect_ratio) ** 0.5)
                W_guess = int(num_points / H_guess) if H_guess > 0 else int(num_points**0.5)

                # Adjust to make sure H * W == num_points exactly
                # Try nearby values to find exact factors
                best_H, best_W = None, None
                min_aspect_diff = float("inf")

                # Try values around the guess
                for H_test in range(
                    max(1, H_guess - 10), min(H_guess + 10, int(num_points**0.5) + 1)
                ):
                    if num_points % H_test == 0:
                        W_test = num_points // H_test
                        aspect_diff = abs(W_test / H_test - aspect_ratio)
                        if aspect_diff < min_aspect_diff:
                            min_aspect_diff = aspect_diff
                            best_H, best_W = H_test, W_test

                if best_H is not None and best_W is not None:
                    H_used, W_used = best_H, best_W
                    if verbose:
                        print(
                            f"  Inferred dimensions: {H_used}x{W_used} (aspect ratio: {W_used/H_used:.2f}, target: {aspect_ratio:.2f})"
                        )
                else:
                    # Last resort: try to find any factors
                    # Find the largest factor close to sqrt(num_points)
                    sqrt_n = int(num_points**0.5)
                    for H_test in range(sqrt_n, 0, -1):
                        if num_points % H_test == 0:
                            W_test = num_points // H_test
                            H_used, W_used = H_test, W_test
                            if verbose:
                                print(f"  Inferred dimensions (fallback): {H_used}x{W_used}")
                            break
                    else:
                        # Skip this edge if we can't determine dimensions
                        if verbose:
                            print(f"  Skipping edge {i}-{j} - cannot factor {num_points}")
                        continue

                # Reshape to (H, W, 3) format expected by minimum_spanning_tree
                try:
                    pred_i[key] = pts3d_i_flat.reshape(H_used, W_used, 3)
                    pred_j[key] = pts3d_j_flat.reshape(H_used, W_used, 3)
                    conf_i[key] = conf_i_flat.reshape(H_used, W_used)
                    conf_j[key] = conf_j_flat.reshape(H_used, W_used)
                except RuntimeError as e:
                    if verbose:
                        print(f"Warning: Failed to reshape point cloud for edge {i}-{j}: {e}")
                    continue

            # Store the dimensions used for this image
            if i < len(imshapes_used):
                # Update if we have a better match
                if H_used * W_used > imshapes_used[i][0] * imshapes_used[i][1]:
                    imshapes_used[i] = (H_used, W_used)
            else:
                # Extend list if needed
                while len(imshapes_used) <= i:
                    H_full, W_full = imshapes_full[len(imshapes_used)]
                    imshapes_used.append((H_full, W_full))
                imshapes_used[i] = (H_used, W_used)

    if len(pred_i) == 0:
        if verbose:
            print("Warning: No valid edges after reshaping, using identity poses")
        return None, None, None

    # Use the determined dimensions for im_conf
    # If we didn't determine dimensions for some images, use full resolution
    imshapes_final = []
    for idx in range(n_imgs):
        if idx < len(imshapes_used):
            imshapes_final.append(imshapes_used[idx])
        else:
            H_full, W_full = imshapes_full[idx]
            imshapes_final.append((H_full, W_full))

    # Create dummy im_conf (not used for pose extraction)
    im_conf = [torch.ones((h, w), device=device) for h, w in imshapes_final]

    # Extract initial poses using minimum_spanning_tree
    try:
        pts3d, msp_edges, im_focals, im_poses = minimum_spanning_tree(
            imshapes=imshapes_final,
            edges=edges,
            pred_i=pred_i,
            pred_j=pred_j,
            conf_i=conf_i,
            conf_j=conf_j,
            im_conf=im_conf,
            min_conf_thr=0.0,
            device=device,
            has_im_poses=True,
            niter_PnP=10,
            verbose=verbose,
        )

        if im_poses is None:
            return None, None, None

        # Convert absolute poses to relative poses along MST
        # mst is (root_idx, [(parent, child), ...])
        root_idx = mst[0]
        mst_edges = mst[1]

        # Initialize relative poses
        rel_quats = []
        rel_trans = []

        # Root camera has identity relative pose
        for i in range(n_imgs):
            if i == root_idx:
                rel_quats.append(torch.tensor([0, 0, 0, 1], dtype=dtype, device=device))
                rel_trans.append(torch.zeros(3, dtype=dtype, device=device))
            else:
                # Will be set based on MST
                rel_quats.append(torch.tensor([0, 0, 0, 1], dtype=dtype, device=device))
                rel_trans.append(torch.zeros(3, dtype=dtype, device=device))

        # Build parent map for MST
        parent_map = {root_idx: None}
        for parent, child in mst_edges:
            parent_map[child] = parent

        # Convert absolute poses to relative poses along MST
        abs_cam2w = im_poses.clone()

        if verbose:
            print("\n[Initial Pose Extraction] Absolute camera poses from MST:")
            for idx in range(n_imgs):
                pose = abs_cam2w[idx]
                trans = pose[:3, 3]
                rot_first_row = pose[0, :3]
                print(
                    f"  Camera {idx}: translation={trans.cpu().numpy()}, rotation[0]={rot_first_row.cpu().numpy()}"
                )
            print(f"  MST root: {root_idx}, edges: {mst_edges}")

        # For each camera, compute relative pose w.r.t. its parent
        for child_idx in range(n_imgs):
            if child_idx == root_idx:
                continue

            parent_idx = parent_map.get(child_idx)
            if parent_idx is None:
                # No parent found, use identity
                if verbose:
                    print(
                        f"  Warning: Camera {child_idx} has no parent in MST, using identity relative pose"
                    )
                continue

            # Compute relative transform: child_pose = parent_pose @ rel_pose
            # So: rel_pose = GeometryUtils.inv(parent_pose) @ child_pose
            parent_pose = abs_cam2w[parent_idx]
            child_pose = abs_cam2w[child_idx]

            rel_pose = torch.inverse(parent_pose) @ child_pose

            # Extract rotation and translation
            rel_rot = rel_pose[:3, :3]
            rel_t = rel_pose[:3, 3]

            # Convert to quaternion
            rel_quat = roma.rotmat_to_unitquat(rel_rot)

            rel_quats[child_idx] = rel_quat
            rel_trans[child_idx] = rel_t

            if verbose:
                print(
                    f"  Camera {child_idx} (parent={parent_idx}): rel_trans={rel_t.cpu().numpy()}, rel_quat={rel_quat.cpu().numpy()}"
                )

        if verbose:
            print("[Initial Pose Extraction] Relative poses summary:")
            for idx in range(n_imgs):
                if idx == root_idx:
                    print(f"  Camera {idx} (root): identity relative pose")
                else:
                    print(
                        f"  Camera {idx}: rel_trans={rel_trans[idx].cpu().numpy()}, rel_quat={rel_quats[idx].cpu().numpy()}"
                    )

        return rel_quats, rel_trans, abs_cam2w

    except Exception as e:
        if verbose:
            print(f"Warning: Failed to extract initial poses: {e}")
        return None, None, None


def sparse_scene_optimizer(
    imgs,
    subsample,
    imsizes,
    pps,
    base_focals,
    core_depth,
    anchors,
    corres,
    corres2d,
    preds_21,
    canonical_paths,
    mst,
    cache_path,
    lr1=0.07,
    niter1=300,
    loss1=None,
    lr2=0.01,
    niter2=300,
    loss2=None,
    lossd=None,
    opt_pp=True,
    opt_depth=True,
    schedule=LearningRateSchedules.cosine_schedule,
    depth_mode="add",
    exp_depth=False,
    lora_depth=False,
    shared_intrinsics=False,
    init={},
    device="cuda",
    dtype=torch.float32,
    matching_conf_thr=5.0,
    loss_dust3r_w=0.01,
    verbose=False,
    dbg=(),
    pairs_output=None,  # Optional: full-resolution pair data for initial pose extraction
):
    """Optimize sparse scene reconstruction using two-stage optimization.

    This function performs sparse scene optimization in two stages:
    1. Coarse optimization: Optimizes 3D point correspondences
    2. Fine optimization: Refines with 2D reprojection loss

    The optimization uses a kinematic chain representation where cameras are
    parameterized relative to their parent in the minimum spanning tree.

    Args:
        imgs: List of image paths
        subsample: Subsampling factor
        imsizes: Tensor of image sizes (N, 2) as (W, H)
        pps: List of principal points (N, 2)
        base_focals: List of base focal lengths (N,)
        core_depth: List of core depth maps
        anchors: Dict mapping image index to anchor data
        corres: Tuple of (all_confs, confs_sum, imgs_slices)
        corres2d: List of 2D correspondences per image
        preds_21: Dict mapping image paths to predictions
        canonical_paths: List of paths to cached canonical views
        mst: Tuple of (root_idx, edges) for minimum spanning tree
        cache_path: Path for caching intermediate results
        lr1: Learning rate for coarse stage (default: 0.07)
        niter1: Number of iterations for coarse stage (default: 300)
        loss1: Loss function for coarse stage (default: gamma_loss(1.5))
        lr2: Learning rate for fine stage (default: 0.01)
        niter2: Number of iterations for fine stage (default: 300)
        loss2: Loss function for fine stage (default: gamma_loss(0.5))
        lossd: Loss function for Dust3r predictions (default: gamma_loss(1.1))
        opt_pp: Whether to optimize principal points (default: True)
        opt_depth: Whether to optimize depthmaps (default: True)
        schedule: Learning rate schedule function (default: LearningRateSchedules.cosine_schedule)
        depth_mode: Depth parameterization mode ("add" or "mul") (default: "add")
        exp_depth: Whether to use exponential depth parameterization (default: False)
        lora_depth: Whether to use LoRA depth parameterization (default: False)
        shared_intrinsics: Whether to share intrinsics across images (default: False)
        init: Dict of initialization values per image (default: {})
        device: Device for computation (default: "cuda")
        dtype: Data type for tensors (default: torch.float32)
        matching_conf_thr: Confidence threshold for matching (default: 5.0)
        loss_dust3r_w: Weight for Dust3r loss term (default: 0.01)
        verbose: Whether to print debug information (default: False)
        dbg: Debug tuple (default: ())
        pairs_output: Optional full-resolution pair data for initial pose extraction

    Returns:
        Tuple of (imgs, res_coarse, res_fine) where:
            imgs: List of image paths
            res_coarse: Dict with coarse optimization results
            res_fine: Dict with fine optimization results or None
    """
    # Set default loss functions if not provided
    if loss1 is None:
        loss1 = LossFunctions.gamma_loss(1.5)
    if loss2 is None:
        loss2 = LossFunctions.gamma_loss(0.5)
    if lossd is None:
        lossd = LossFunctions.gamma_loss(1.1)

    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # Try to extract initial poses from pairwise predictions if not provided in init
    rel_quats_init = None
    rel_trans_init = None
    # Check if init already contains pose information
    has_init_poses = False
    if init:
        for img_key, init_val in init.items():
            if isinstance(init_val, dict):
                if "cam2w" in init_val or "rel_quat" in init_val:
                    has_init_poses = True
                    break

    if not has_init_poses:
        # Extract initial poses from pairwise predictions
        # Use pairs_output if available (full-resolution point clouds from cache)
        rel_quats_init = None
        rel_trans_init = None

        if pairs_output is not None:
            rel_quats_init, rel_trans_init, abs_cam2w = _extract_initial_poses_from_pairwise(
                filelist=imgs,
                pairs_output=pairs_output,
                imsizes=imsizes,
                base_focals=base_focals,
                pps=pps,
                mst=mst,
                subsample=subsample,
                cache_dir=cache_path,
                device=device,
                dtype=dtype,
                verbose=verbose,
            )

        if rel_quats_init is not None and verbose:
            print(
                f" >> Extracted initial poses from pairwise predictions for {len(rel_quats_init)} cameras"
            )

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f" >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}")

        K = init_values.get("intrinsics")
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]

        depth = init_values.get("depthmap")
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        if verbose:
            print(
                f"  Camera {idx}: core_depth shape={core_depth[idx].shape}, median={med_depth.item():.6f}, min={core_depth[idx].min().item():.6f}, max={core_depth[idx].max().item():.6f}"
            )

        # Check for relative pose initialization (preferred for kinematic chain)
        rel_quat = init_values.get("rel_quat")
        rel_t = init_values.get("rel_trans")
        if rel_quat is not None and rel_t is not None:
            quats[idx].data[:] = rel_quat.detach().to(dtype).to(device)
            trans[idx].data[:] = rel_t.detach().to(dtype).to(device)
        elif (
            rel_quats_init is not None and rel_trans_init is not None and idx < len(rel_quats_init)
        ):
            # Use extracted initial relative poses
            quats[idx].data[:] = rel_quats_init[idx].to(dtype).to(device)
            trans[idx].data[:] = rel_trans_init[idx].to(dtype).to(device)
            if verbose:
                print(f"  Camera {idx}: Initialized from extracted relative pose")
                print(f"    rel_trans={rel_trans_init[idx].cpu().numpy()}")
                print(f"    rel_quat={rel_quats_init[idx].cpu().numpy()}")
        else:
            # Fall back to absolute pose initialization (if provided)
            cam2w = init_values.get("cam2w")
            if cam2w is not None:
                rot = cam2w[:3, :3].detach()
                cam_center = cam2w[:3, 3].detach()
                quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
                trans_offset = med_depth * torch.cat(
                    (imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0])
                )
                trans[idx].data[:] = cam_center + rot @ trans_offset
                del rot
                if verbose:
                    print(
                        f"Warning: Initializing from absolute pose for camera {idx}, but kinematic chain conversion not fully implemented"
                    )
                # Note: This will initialize the root camera correctly, but child cameras
                # will need to be converted to relative poses along the MST

    # Debug: print initial quats and trans after initialization
    if verbose:
        print("\n[Sparse Optimizer] Initial quats and trans after initialization:")
        for idx in range(len(imgs)):
            print(
                f"  Camera {idx}: quat={quats[idx].data.detach().cpu().numpy()}, trans={trans[idx].data.detach().cpu().numpy()}"
            )

    # intrinsics parameters
    if shared_intrinsics:
        confs = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags
    max_focals = 10 * diags

    assert len(mst[1]) == len(pps) - 1

    # Track if this is the first call to make_K_cam_depth for debugging
    make_K_cam_depth_call_count = [0]  # Use list to allow modification in nested function

    # Store initial relative translations for scale regularization
    initial_rel_trans = None
    if rel_trans_init is not None:
        # rel_trans_init is a list of tensors, convert to stacked tensor
        if isinstance(rel_trans_init, list):
            initial_rel_trans = torch.stack(rel_trans_init).clone().detach()
        else:
            initial_rel_trans = rel_trans_init.clone().detach()

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        nonlocal make_K_cam_depth_call_count
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        # Root camera's absolute pose is its relative transform (it's the reference)
        # For child cameras, absolute pose = parent's absolute pose @ child's relative transform
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # Debug: print initial absolute poses computed from kinematic chain (only first time)
        if (
            verbose
            and trans is not None
            and quats is not None
            and make_K_cam_depth_call_count[0] == 0
        ):
            make_K_cam_depth_call_count[0] += 1
            print("\n[Sparse Optimizer] Initial absolute poses from kinematic chain:")
            for idx in range(len(K)):
                pose = tmp_cam2w[idx]
                trans_val = pose[:3, 3]
                rot_first_row = pose[0, :3]
                print(
                    f"  Camera {idx}: translation={trans_val.detach().cpu().numpy()}, rotation[0]={rot_first_row.detach().cpu().numpy()}"
                )
            print(f"  Root camera: {mst[0]}, MST edges: {mst[1]}")

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat(
            (imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1
        )
        new_trans = global_scaling * (
            tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1)
        )

        # Enforce root camera at origin with identity rotation
        # Root camera has identity relative pose, so its absolute pose should be identity
        root_idx = mst[0]
        if trans is not None and quats is not None:
            # Root camera should be at origin with identity rotation
            # Use clone() to avoid inplace modification that breaks gradients
            new_trans = new_trans.clone()
            new_trans[root_idx] = torch.zeros(3, 1, dtype=new_trans.dtype, device=new_trans.device)
            # Create a new rotation matrix tensor for root camera (avoid inplace modification)
            root_rot = torch.eye(3, dtype=tmp_cam2w.dtype, device=tmp_cam2w.device)
            # Clone tmp_cam2w and modify the clone to avoid breaking gradients
            tmp_cam2w_rot = tmp_cam2w[:, :3, :3].clone()
            tmp_cam2w_rot[root_idx] = root_rot
        else:
            tmp_cam2w_rot = tmp_cam2w[:, :3, :3]

        cam2w = torch.cat(
            (
                torch.cat((tmp_cam2w_rot, new_trans), dim=2),
                vec0001.view(1, 1, 4).expand(len(K), 1, 4),
            ),
            dim=1,
        )

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == "add":
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == "mul":
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f"Bad {depth_mode=}")
            depthmaps.append(global_scaling * core_depth_img)

        return K, (GeometryUtils.inv(cam2w), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics:
        print("init focal (shared) = ", to_numpy(K[0, 0, 0]).round(2))
    else:
        print("init focals =", to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = SpectralUtils.spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth
        )
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    core_depth = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres

    # Define which pairs are fine to use with matching
    def matching_check(x):
        return x.max() > matching_conf_thr

    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]

    if verbose:
        print(f"\n[Optimization Setup] Correspondence slices:")
        print(f"  Total slices: {len(imgs_slices)}")
        print(f"  loss3d_slices (matching OK): {len(loss3d_slices)}")
        print(f"  dust3r_slices (matching not OK): {len(dust3r_slices)}")
        if len(loss3d_slices) > 0:
            print(f"  loss3d_slices pairs: {[(s.img1, s.img2) for s in loss3d_slices]}")
        if len(loss3d_slices) == 0:
            print("  WARNING: No loss3d_slices! This may cause optimization issues.")
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    def loss_dust3r(cam2w, pts3d, pix_loss):
        loss = 0.0
        cf_sum = 0.0
        for s in dust3r_slices:
            if init[imgs[s.img1]].get("freeze") and init[imgs[s.img2]].get("freeze"):
                continue
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = GeometryUtils.geom_transform(GeometryUtils.inv(cam2w[s.img2]), tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0.0 else 0.0

    # Track loss_3d calls for debugging
    loss_3d_call_count = [0]

    def loss_3d(K, w2cam, pts3d, pix_loss):
        nonlocal loss_3d_call_count
        if any(v.get("freeze") for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get("freeze") and init[imgs[s.img2]].get("freeze"):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.0
            cf_sum = 1.0

        # Debug: check if loss is zero or very small (might indicate no correspondences)
        loss_3d_call_count[0] += 1
        if verbose and loss_3d_call_count[0] == 1:
            print(
                f"\n[loss_3d] First call: loss={loss.item() if isinstance(loss, torch.Tensor) else loss:.6f}, cf_sum={cf_sum.item() if isinstance(cf_sum, torch.Tensor) else cf_sum:.6f}, num_slices={len(loss3d_slices)}"
            )
            if len(pts3d_1) == 0:
                print("  WARNING: No point correspondences in loss_3d!")
            else:
                # Check if correspondences have reasonable distances
                pts3d_1_cat = torch.cat(pts3d_1) if isinstance(pts3d_1, list) else pts3d_1
                pts3d_2_cat = torch.cat(pts3d_2) if isinstance(pts3d_2, list) else pts3d_2
                if len(pts3d_1_cat) > 0 and len(pts3d_2_cat) > 0:
                    dists = torch.norm(pts3d_1_cat - pts3d_2_cat, dim=-1)
                    print(
                        f"  Correspondence distances: mean={dists.mean().item():.6f}, std={dists.std().item():.6f}, min={dists.min().item():.6f}, max={dists.max().item():.6f}"
                    )
                    print(
                        f"  pts3d_1 shape: {pts3d_1_cat.shape}, pts3d_2 shape: {pts3d_2_cat.shape}"
                    )

        return loss / cf_sum

    def loss_2d(K, w2cam, pts3d, pix_loss):
        proj_matrix = K @ w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get("freeze", 0) >= 1:
                continue
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(
                    pix1_filtered, ProjectionUtils.reproj2d(proj_matrix[img1], pts3d_in_img1)
                )
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.0

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        # Exclude root camera's quat, trans, log_sizes, and log_focals from optimization
        root_idx = mst[0]
        params = (
            pps
            + [f for i, f in enumerate(log_focals) if i != root_idx]
            + [q for i, q in enumerate(quats) if i != root_idx]
            + [t for i, t in enumerate(trans) if i != root_idx]
            + [s for i, s in enumerate(log_sizes) if i != root_idx]
            + core_depth
        )
        if verbose:
            print(f"\n[Optimize Loop] Root camera {root_idx} excluded from optimization")
            print(
                f"  Total params: {len(params)} (excluding root camera quat/trans/log_sizes/log_focals)"
            )
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if "meta" in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm.tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(
                    log_focals, pps, trans, quats, log_sizes, core_depth
                )
                pts3d = ProjectionUtils.make_pts3d(
                    anchors, K, cam2w, depthmaps, base_focals=base_focals
                )

                # Debug: check depth maps and 3D points at key iterations
                if verbose and (
                    iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0)
                ):
                    print(f"\n[Optimization iter {iter}/{niter}] Depth maps and 3D points:")
                    for idx in range(len(imgs)):
                        depth_val = depthmaps[idx]
                        if isinstance(depth_val, torch.Tensor):
                            depth_mean = depth_val.mean().item()
                            depth_std = depth_val.std().item()
                            depth_min = depth_val.min().item()
                            depth_max = depth_val.max().item()
                            print(
                                f"  Camera {idx}: depth mean={depth_mean:.6f}, std={depth_std:.6f}, min={depth_min:.6f}, max={depth_max:.6f}"
                            )
                        if idx < len(pts3d):
                            pts3d_val = pts3d[idx]
                            if isinstance(pts3d_val, torch.Tensor):
                                pts3d_mean = pts3d_val.mean(dim=0).detach().cpu().numpy()
                                pts3d_std = pts3d_val.std(dim=0).detach().cpu().numpy()
                                print(f"  Camera {idx}: pts3d mean={pts3d_mean}, std={pts3d_std}")

                if niter == 0:
                    break

                alpha = iter / niter
                lr = schedule(alpha, lr_base, lr_end)
                LearningRateSchedules.adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(
                    cam2w, pts3d, lossd
                )

                # Add scale regularization to prevent camera collapse
                # Penalize cameras getting too close to each other
                # Use initial relative translations as reference to maintain scale
                root_idx = mst[0]
                scale_reg = 0.0
                if len(imgs) > 1:
                    # Compute pairwise distances between cameras
                    cam_positions = cam2w[:, :3, 3]  # [n_cameras, 3]

                    # Get initial distances from initial relative translations
                    initial_distances = {}
                    if initial_rel_trans is not None:
                        for i in range(len(imgs)):
                            if i == root_idx:
                                continue
                            if i < len(initial_rel_trans):
                                initial_dist = torch.norm(initial_rel_trans[i])
                                initial_distances[i] = initial_dist

                    for i in range(len(imgs)):
                        if i == root_idx:
                            continue
                        # Distance from root camera
                        dist_to_root = torch.norm(cam_positions[i] - cam_positions[root_idx])

                        # Use initial distance as reference if available
                        if i in initial_distances:
                            initial_dist = initial_distances[i]
                            # Penalize if current distance is much smaller than initial (collapse)
                            # Use a threshold of 70% of initial distance to allow some refinement
                            if dist_to_root < initial_dist * 0.7:  # More than 30% collapse
                                collapse_ratio = dist_to_root / (initial_dist + 1e-8)
                                # Strong penalty that increases as collapse gets worse
                                scale_reg += (
                                    (initial_dist * 0.7 - dist_to_root) ** 2
                                    * 200.0
                                    * (1.0 - collapse_ratio)
                                )
                        else:
                            # Fallback: penalize if distance is too small (< 0.05)
                            if dist_to_root < 0.05:
                                scale_reg += (0.05 - dist_to_root) ** 2 * 100.0  # Increased penalty

                    # Also penalize cameras getting too close to each other
                    for i in range(len(imgs)):
                        for j in range(i + 1, len(imgs)):
                            if i == root_idx or j == root_idx:
                                continue
                            dist = torch.norm(cam_positions[i] - cam_positions[j])
                            if dist < 0.05:
                                scale_reg += (0.05 - dist) ** 2 * 50.0  # Increased penalty

                loss = loss + scale_reg

                # Debug: print scale regularization at key iterations
                if (
                    verbose
                    and scale_reg > 0
                    and (iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0))
                ):
                    print(f"  [Scale Reg] iter {iter}: scale_reg={scale_reg.item():.6f}")

                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                # Also enforce root camera identity relative pose
                root_idx = mst[0]
                for i in range(len(imgs)):
                    if i == root_idx:
                        # Root camera: enforce identity relative pose (don't normalize first)
                        quats[i].data[:] = torch.tensor(
                            [0, 0, 0, 1], dtype=quats[i].dtype, device=quats[i].device
                        )
                        trans[i].data[:] = torch.zeros(
                            3, dtype=trans[i].dtype, device=trans[i].device
                        )
                    else:
                        # Normalize quaternions for non-root cameras
                        quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss

                # Debug: print poses at key iterations
                if verbose and (
                    iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0)
                ):
                    print(f"\n[Optimization iter {iter}/{niter}] Camera poses:")
                    for idx in range(len(imgs)):
                        pose = cam2w[idx]
                        trans_val = pose[:3, 3]
                        rot_first_row = pose[0, :3]
                        print(
                            f"  Camera {idx}: translation={trans_val.detach().cpu().numpy()}, rotation[0]={rot_first_row.detach().cpu().numpy()}"
                        )
                    print(f"  Loss: {loss:.6f}, LR: {lr:.6f}")

                bar.set_postfix_str(f"{lr=:.4f}, {loss=:.3f}")
                bar.update(1)

        if niter:
            print(f">> final loss = {loss}")
        return dict(
            intrinsics=K.detach(),
            cam2w=cam2w.detach(),
            depthmaps=[d.detach() for d in depthmaps],
            pts3d=[p.detach() for p in pts3d],
        )

    # at start, don't optimize 3d points
    # Freeze root camera's relative pose (it should remain identity)
    root_idx = mst[0]
    for i, img in enumerate(imgs):
        trainable = not (init[img].get("freeze"))
        # Root camera should have identity relative pose, so freeze its quat, trans, and log_sizes
        # (log_sizes affects absolute pose computation through z_cameras)
        is_root = i == root_idx
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable and not is_root)  # Freeze root camera's relative pose
        trans[i].requires_grad_(trainable and not is_root)  # Freeze root camera's relative pose
        log_sizes[i].requires_grad_(trainable and not is_root)  # Freeze root camera's log_sizes too
        core_depth[i].requires_grad_(False)

        if verbose and is_root:
            print(
                f"\n[Optimization Setup] Freezing root camera {i} relative pose (identity) and log_sizes"
            )

    res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

    res_fine = None
    if niter2:
        # now we can optimize 3d points
        # Keep root camera frozen
        root_idx = mst[0]
        if verbose:
            print(f"\n[Fine Optimization Setup] Root camera {root_idx} will remain frozen")
        for i, img in enumerate(imgs):
            if init[img].get("freeze", 0) >= 1:
                continue
            is_root = i == root_idx
            pps[i].requires_grad_(bool(opt_pp))
            # Keep root camera's log_focals frozen too (affects absolute pose through z_cameras)
            log_focals[i].requires_grad_(True and not is_root)
            # Keep root camera's relative pose and log_sizes frozen
            if not is_root:
                quats[i].requires_grad_(True)
                trans[i].requires_grad_(True)
                log_sizes[i].requires_grad_(True)
            else:
                # Explicitly freeze root camera (quat, trans, log_sizes, and log_focals)
                # log_sizes and log_focals affect absolute pose through z_cameras calculation
                quats[i].requires_grad_(False)
                trans[i].requires_grad_(False)
                log_sizes[i].requires_grad_(False)
                log_focals[i].requires_grad_(False)
                if verbose:
                    print(f"  Camera {i} (root): quat, trans, log_sizes, and log_focals frozen")
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)
    if shared_intrinsics:
        print("Final focal (shared) = ", to_numpy(K[0, 0, 0]).round(2))
    else:
        print("Final focals =", to_numpy(K[:, 0, 0]))

    return imgs, res_coarse, res_fine


class SpectralUtils:
    """Spectral clustering and projection utilities."""

    @staticmethod
    def spectral_clustering(graph, k=None, normalized_cuts=False):
        """Perform spectral clustering on a graph.

        This function computes the graph Laplacian and returns its k smallest
        eigenvalues and corresponding eigenvectors, which can be used for
        clustering or dimensionality reduction.

        Args:
            graph: Adjacency matrix (N, N)
            k: Number of eigenvectors to return (default: None, returns all)
            normalized_cuts: Whether to use normalized Laplacian (default: False)

        Returns:
            Tuple of (eigenvalues, eigenvectors) where:
                eigenvalues: k smallest eigenvalues (k,)
                eigenvectors: Corresponding eigenvectors (N, k)
        """
        graph.fill_diagonal_(0)
        degrees = graph.sum(dim=-1)
        laplacian = torch.diag(degrees) - graph
        if normalized_cuts:
            i_inv = torch.diag(degrees.sqrt().reciprocal())
            laplacian = i_inv @ laplacian @ i_inv
        eigval, eigvec = torch.linalg.eigh(laplacian)
        return eigval[:k], eigvec[:, :k]

    @staticmethod
    def spectral_projection_of_depthmaps(
        imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw
    ):
        """Apply spectral projection to depthmaps for low-rank parameterization.

        This function applies spectral clustering to depthmaps to create a low-rank
        representation (LoRA) that can be used for efficient optimization.

        Args:
            imgs: List of image paths
            intrinsics: Camera intrinsics (N, 3, 3)
            depthmaps: List of depth maps (N,)
            subsample: Subsampling factor
            cache_path: Optional path for caching projections
            **kw: Additional keyword arguments for spectral_projection_depth

        Returns:
            Tuple of (core_depth, lora_proj) where:
                core_depth: List of depth coefficients
                lora_proj: List of projection matrices
        """
        core_depth = []
        lora_proj = []
        for i, img in enumerate(tqdm.tqdm(imgs)):
            cache = (
                os.path.join(cache_path, "lora_depth", FileUtils.hash_md5(img))
                if cache_path
                else None
            )
            depth, proj = SpectralUtils.spectral_projection_depth(
                intrinsics[i], depthmaps[i], subsample, cache_path=cache, **kw
            )
            core_depth.append(depth)
            lora_proj.append(proj)
        return core_depth, lora_proj

    @staticmethod
    def spectral_projection_depth(
        K, depthmap, subsample, k=64, cache_path="", normalized_cuts=True, gamma=7, min_norm=5
    ):
        """Compute spectral projection of a single depthmap for low-rank representation.

        This function backprojects the depthmap to 3D, builds a similarity graph,
        performs spectral clustering, and encodes the depthmap using the resulting
        low-rank basis.

        Args:
            K: Camera intrinsics (3, 3)
            depthmap: Depth map (H, W)
            subsample: Subsampling factor
            k: Number of basis vectors (default: 64)
            cache_path: Optional path for caching projection matrix
            normalized_cuts: Whether to use normalized Laplacian (default: True)
            gamma: Similarity function parameter (default: 7)
            min_norm: Minimum norm for coefficient normalization (default: 5)

        Returns:
            Tuple of (coeffs, lora_proj) where:
                coeffs: Depth coefficients (k,)
                lora_proj: Projection matrix (H*W, k)
        """
        try:
            if cache_path:
                cache_path = cache_path + f"_{k=}_norm={normalized_cuts}_{gamma=}.pth"
            lora_proj = torch.load(cache_path, map_location=K.device)
        except IOError:
            xyz = ProjectionUtils.backproj(K, depthmap, subsample)
            xyz = xyz.reshape(-1, 3)
            graph = SpectralUtils.sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
            _, lora_proj = SpectralUtils.spectral_clustering(
                graph, k, normalized_cuts=normalized_cuts
            )
            if cache_path:
                torch.save(lora_proj.cpu(), FileUtils.mkdir_for(cache_path))
        lora_proj, coeffs = SpectralUtils.lora_encode_normed(
            lora_proj, depthmap.ravel(), min_norm=min_norm
        )
        return coeffs, lora_proj

    @staticmethod
    def sim_func(p1, p2, gamma):
        """Compute similarity between two point sets using exponential distance.

        The similarity is computed as exp(-gamma * (relative_distance)^2), where
        relative_distance is the Euclidean distance normalized by average depth.

        Args:
            p1: First point set (N1, 1, 3) or (N1, 3)
            p2: Second point set (1, N2, 3) or (N2, 3)
            gamma: Scaling parameter for exponential decay

        Returns:
            Similarity matrix (N1, N2)
        """
        diff = (p1 - p2).norm(dim=-1)
        avg_depth = p1[:, :, 2] + p2[:, :, 2]
        rel_distance = diff / avg_depth
        sim = torch.exp(-gamma * rel_distance.square())
        return sim

    @staticmethod
    def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
        """Encode a signal using a low-rank projection with normalized coefficients.

        This function projects a signal onto a low-rank basis and normalizes the
        projection matrix to ensure minimum norm constraints on coefficients.

        Args:
            lora_proj: Projection matrix (M, k)
            x: Signal to encode (M,)
            min_norm: Minimum norm for coefficient normalization
            global_norm: Whether to use global normalization (default: False)

        Returns:
            Tuple of (lora_proj_normalized, coeffs) where:
                lora_proj_normalized: Normalized projection matrix (M, k)
                coeffs: Encoded coefficients (k,)
        """
        coeffs = torch.linalg.pinv(lora_proj) @ x
        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]
        if global_norm:
            lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
        elif min_norm:
            lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
        coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()
        return lora_proj.detach(), coeffs.detach()
