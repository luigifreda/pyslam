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

TSDF (Truncated Signed Distance Function) post-processing for scene optimizers.

This module provides TSDF-based depth refinement that can be applied to any
scene optimizer output (dense or sparse) to improve depth map quality.

Original code from MASt3r's tsdf_optimizer module.
Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""

import torch
import torch.nn as nn
from typing import List, Optional
from tqdm import tqdm


from .scene_optimizer_helpers import GeometryUtils, PointCloudUtils


class TSDFPostProcess:
    """
    Optimizes a signed distance function to improve depthmaps.

    This class applies TSDF (Truncated Signed Distance Function) fusion to refine
    depth maps from any scene optimizer. It can be used with both dense and sparse
    optimizers to improve depth quality by leveraging multi-view consistency.

    The TSDF post-processing:
    1. Projects 3D points from all views into each view
    2. Computes signed distance values for each pixel
    3. Finds the zero-crossing of the TSDF along each ray
    4. Refines depth values based on multi-view consistency

    Args:
        optimizer: Scene optimizer object (must have get_dense_pts3d, get_im_poses,
                   get_focals, get_principal_points methods)
        subsample: Subsampling factor for canonical views (default: 8)
        TSDF_thresh: TSDF threshold for filtering (default: 0.0, disabled)
        TSDF_batchsize: Batch size for TSDF queries (default: 1e7)
    """

    def __init__(self, optimizer, subsample=8, TSDF_thresh=0.0, TSDF_batchsize=int(1e7)):
        self.TSDF_thresh = TSDF_thresh
        self.TSDF_batchsize = TSDF_batchsize
        self.optimizer = optimizer

        # Handle subsample parameter - sparse optimizer supports it, dense doesn't
        if hasattr(optimizer, "get_dense_pts3d"):
            # Try with subsample first (for sparse optimizer)
            import inspect

            sig = inspect.signature(optimizer.get_dense_pts3d)
            if "subsample" in sig.parameters:
                pts3d, depthmaps, confs = optimizer.get_dense_pts3d(
                    clean_depth=False, subsample=subsample
                )
            else:
                # Dense optimizer doesn't support subsample parameter
                pts3d, depthmaps, confs = optimizer.get_dense_pts3d(clean_depth=False)
        else:
            raise ValueError("Optimizer must have get_dense_pts3d method")

        pts3d, depthmaps = self._TSDF_postprocess_or_not(pts3d, depthmaps, confs)
        self.pts3d = pts3d
        self.depthmaps = depthmaps
        self.confs = confs

    def _get_depthmaps(self, TSDF_filtering_thresh=None):
        if TSDF_filtering_thresh:
            self._refine_depths_with_TSDF(TSDF_filtering_thresh)
        dms = self.TSDF_im_depthmaps if TSDF_filtering_thresh else self.im_depthmaps
        return [d.exp() for d in dms]

    @torch.no_grad()
    def _refine_depths_with_TSDF(self, TSDF_filtering_thresh, niter=1, nsamples=1000):
        """
        Leverage TSDF to post-process estimated depths.

        For each pixel, find zero level of TSDF along ray (or closest to 0).

        Args:
            TSDF_filtering_thresh: Threshold for TSDF filtering
            niter: Number of refinement iterations (default: 1)
            nsamples: Number of depth samples per pixel (default: 1000)
        """
        print("Post-Processing Depths with TSDF fusion.")
        self.TSDF_im_depthmaps = []
        alldepths, allposes, allfocals, allpps, allimshapes = (
            self._get_depthmaps(),
            self.optimizer.get_im_poses(),
            self.optimizer.get_focals(),
            self.optimizer.get_principal_points(),
            self.imshapes,
        )
        for vi in tqdm(range(self.optimizer.n_imgs)):
            dm, pose, focal, pp, imshape = (
                alldepths[vi],
                allposes[vi],
                allfocals[vi],
                allpps[vi],
                allimshapes[vi],
            )
            minvals = torch.full(dm.shape, 1e20)

            for it in range(niter):
                H, W = dm.shape
                curthresh = (niter - it) * TSDF_filtering_thresh
                dm_offsets = (torch.randn(H, W, nsamples).to(dm) - 1.0) * curthresh
                newdm = dm[..., None] + dm_offsets
                curproj = self._backproj_pts3d(
                    in_depths=[newdm],
                    in_im_poses=pose[None],
                    in_focals=focal[None],
                    in_pps=pp[None],
                    in_imshapes=[imshape],
                )[0]
                curproj = curproj.view(-1, 3)
                tsdf_vals = []
                valids = []
                for batch in range(0, len(curproj), self.TSDF_batchsize):
                    values, valid = self._TSDF_query(
                        curproj[batch : min(batch + self.TSDF_batchsize, len(curproj))], curthresh
                    )
                    tsdf_vals.append(values)
                    valids.append(valid)
                tsdf_vals = torch.cat(tsdf_vals, dim=0)
                valids = torch.cat(valids, dim=0)

                tsdf_vals = tsdf_vals.view([H, W, nsamples])
                valids = valids.view([H, W, nsamples])

                tsdf_vals[~valids] = torch.inf
                tsdf_vals = tsdf_vals.abs()
                mins = torch.argmin(tsdf_vals, dim=-1, keepdim=True)
                allbad = (tsdf_vals == curthresh).sum(dim=-1) == nsamples
                dm[~allbad] = torch.gather(newdm, -1, mins)[..., 0][~allbad]

            self.TSDF_im_depthmaps.append(dm.log())

    def _TSDF_query(self, qpoints, TSDF_filtering_thresh, weighted=True):
        """
        TSDF query call: returns the weighted TSDF value for each query point [N, 3].

        Args:
            qpoints: Query 3D points [N, 3]
            TSDF_filtering_thresh: TSDF threshold
            weighted: Whether to weight by confidence (default: True)

        Returns:
            Tuple of (TSDF values, validity mask)
        """
        N, three = qpoints.shape
        assert three == 3
        qpoints = qpoints[None].repeat(self.optimizer.n_imgs, 1, 1)
        coords_and_depth = self._proj_pts3d(
            pts3d=qpoints,
            cam2worlds=self.optimizer.get_im_poses(),
            focals=self.optimizer.get_focals(),
            pps=self.optimizer.get_principal_points(),
        )
        image_coords = coords_and_depth[..., :2].round().to(int)
        proj_depths = coords_and_depth[..., -1]
        pred_depths, pred_confs, valids = self._get_pixel_depths(image_coords)
        all_SDF_scores = pred_depths - proj_depths
        unseen = all_SDF_scores < -TSDF_filtering_thresh
        all_TSDF_scores = all_SDF_scores.clip(-TSDF_filtering_thresh, 1e20)
        all_TSDF_weights = (~unseen).float() * valids.float()
        if weighted:
            all_TSDF_weights = pred_confs.exp() * all_TSDF_weights
        TSDF_weights = all_TSDF_weights.sum(dim=0)
        valids = TSDF_weights != 0.0
        TSDF_wsum = (all_TSDF_weights * all_TSDF_scores).sum(dim=0)
        TSDF_wsum[valids] /= TSDF_weights[valids]
        return TSDF_wsum, valids

    def _get_pixel_depths(self, image_coords, TSDF_filtering_thresh=None, with_normals_conf=False):
        """
        Recover depth value for each input pixel coordinate, along with OOB validity mask.

        Args:
            image_coords: Image coordinates [B, N, 2]
            TSDF_filtering_thresh: Optional TSDF filtering threshold
            with_normals_conf: Whether to use normal-based confidence (default: False)

        Returns:
            Tuple of (depths, confidences, validity mask)
        """
        B, N, two = image_coords.shape
        assert B == self.optimizer.n_imgs and two == 2
        depths = torch.zeros([B, N], device=image_coords.device)
        valids = torch.zeros([B, N], dtype=bool, device=image_coords.device)
        confs = torch.zeros([B, N], device=image_coords.device)
        curconfs = self._get_confs_with_normals() if with_normals_conf else self.im_conf
        for ni, (imc, depth, conf) in enumerate(
            zip(image_coords, self._get_depthmaps(TSDF_filtering_thresh), curconfs)
        ):
            H, W = depth.shape
            valids[ni] = torch.logical_and(0 <= imc[:, 1], imc[:, 1] < H) & torch.logical_and(
                0 <= imc[:, 0], imc[:, 0] < W
            )
            imc[~valids[ni]] = 0
            depths[ni] = depth[imc[:, 1], imc[:, 0]]
            confs[ni] = conf.cuda()[imc[:, 1], imc[:, 0]]
        return depths, confs, valids

    def _get_confs_with_normals(self):
        """Compute confidence maps weighted by depth gradient (normal-based confidence)."""
        outconfs = []

        class Sobel(nn.Module):
            """Sobel filter for computing depth gradients."""

            def __init__(self):
                super().__init__()
                self.filter = nn.Conv2d(
                    in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False
                )
                Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
                Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
                G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
                G = G.unsqueeze(1)
                self.filter.weight = nn.Parameter(G, requires_grad=False)

            def forward(self, img):
                x = self.filter(img)
                x = torch.mul(x, x)
                x = torch.sum(x, dim=1, keepdim=True)
                x = torch.sqrt(x)
                return x

        grad_op = Sobel().to(self.im_depthmaps[0].device)
        for conf, depth in zip(self.im_conf, self.im_depthmaps):
            grad_confs = (1.0 - grad_op(depth[None, None])[0, 0]).clip(0)
            outconfs.append(conf * grad_confs.to(conf))
        return outconfs

    def _proj_pts3d(self, pts3d, cam2worlds, focals, pps):
        """
        Projection operation: from 3D points to 2D coordinates + depths.

        Args:
            pts3d: 3D points [B, N, 3]
            cam2worlds: Camera-to-world poses [B, 4, 4]
            focals: Focal lengths [B] or [B, 2]
            pps: Principal points [B, 2]

        Returns:
            Projected coordinates and depths [B, N, 3] (u, v, depth)
        """
        B = pts3d.shape[0]
        assert pts3d.shape[0] == cam2worlds.shape[0]
        R, t = cam2worlds[:, :3, :3], cam2worlds[:, :3, -1]
        Rinv = R.transpose(-2, -1)
        tinv = -Rinv @ t[..., None]
        intrinsics = torch.eye(3).to(cam2worlds)[None].repeat(focals.shape[0], 1, 1)
        if len(focals.shape) == 1:
            focals = torch.stack([focals, focals], dim=-1)
        intrinsics[:, 0, 0] = focals[:, 0]
        intrinsics[:, 1, 1] = focals[:, 1]
        intrinsics[:, :2, -1] = pps
        projpts = intrinsics @ (Rinv @ pts3d.transpose(-2, -1) + tinv)
        projpts = projpts.transpose(-2, -1)
        projpts[..., :2] /= projpts[..., [-1]]
        return projpts

    def _backproj_pts3d(
        self, in_depths=None, in_im_poses=None, in_focals=None, in_pps=None, in_imshapes=None
    ):
        """
        Backprojection operation: from image depths to 3D points.

        Args:
            in_depths: Optional input depth maps (uses optimizer's if None)
            in_im_poses: Optional input camera poses (uses optimizer's if None)
            in_focals: Optional input focal lengths (uses optimizer's if None)
            in_pps: Optional input principal points (uses optimizer's if None)
            in_imshapes: Optional input image shapes (uses optimizer's if None)

        Returns:
            List of 3D point clouds [H, W, 3] for each view
        """
        focals = self.optimizer.get_focals() if in_focals is None else in_focals
        im_poses = self.optimizer.get_im_poses() if in_im_poses is None else in_im_poses
        depth = self._get_depthmaps() if in_depths is None else in_depths
        pp = self.optimizer.get_principal_points() if in_pps is None else in_pps
        imshapes = self.imshapes if in_imshapes is None else in_imshapes

        def focal_ex(i):
            return focals[i][..., None, None].expand(1, *focals[i].shape, *imshapes[i])

        dm_to_3d = [
            PointCloudUtils.depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[[i]])
            for i in range(im_poses.shape[0])
        ]

        def autoprocess(x):
            x = x[0]
            return x.transpose(-2, -1) if len(x.shape) == 4 else x

        return [
            GeometryUtils.geom_transform(GeometryUtils.inv(pose), autoprocess(pt))
            for pose, pt in zip(im_poses, dm_to_3d)
        ]

    def _TSDF_postprocess_or_not(self, pts3d, depthmaps, confs, niter=1):
        """
        Apply TSDF post-processing if threshold is set, otherwise return inputs unchanged.

        Args:
            pts3d: Input 3D point clouds
            depthmaps: Input depth maps
            confs: Input confidence maps
            niter: Number of refinement iterations

        Returns:
            Tuple of (processed pts3d, processed depthmaps)
        """
        # Get image shapes - handle both dense and sparse optimizers
        if hasattr(self.optimizer, "imgs"):
            self.imshapes = [im.shape[:2] for im in self.optimizer.imgs]
        elif hasattr(self.optimizer, "imshapes"):
            self.imshapes = self.optimizer.imshapes
        else:
            # Infer from depthmaps
            self.imshapes = [
                dm.shape[:2] if len(dm.shape) == 2 else dm.shape[1:3] for dm in depthmaps
            ]

        self.im_depthmaps = [
            dd.log().view(imshape) for dd, imshape in zip(depthmaps, self.imshapes)
        ]
        self.im_conf = confs

        if self.TSDF_thresh > 0.0:
            self._refine_depths_with_TSDF(self.TSDF_thresh, niter=niter)
            depthmaps = [dd.exp() for dd in self.TSDF_im_depthmaps]
            pts3d = self._backproj_pts3d(in_depths=depthmaps)
            depthmaps = [dd.flatten() for dd in depthmaps]
            pts3d = [pp.view(-1, 3) for pp in pts3d]
        return pts3d, depthmaps

    def get_dense_pts3d(self, clean_depth=True):
        """
        Get dense 3D points, depthmaps, and confidence maps.

        Args:
            clean_depth: If True, clean the pointcloud by reducing confidence
                        for points that are occluded or inconsistent.

        Returns:
            Tuple of (pts3d, depthmaps, confs) where:
            - pts3d: List of [H, W, 3] tensors of 3D points
            - depthmaps: List of [H, W] tensors of depth maps
            - confs: List of [H, W] tensors of confidence maps
        """

        if clean_depth:
            # Get intrinsics and camera poses - handle both dense and sparse optimizers
            if hasattr(self.optimizer, "intrinsics") and hasattr(self.optimizer, "cam2w"):
                # Sparse optimizer (SparseGA)
                K = self.optimizer.intrinsics
                cams = inv(self.optimizer.cam2w)
            elif hasattr(self.optimizer, "get_intrinsics"):
                # Dense optimizer (PointCloudOptimizer, PairViewer)
                K = self.optimizer.get_intrinsics()
                cams = inv(self.optimizer.get_im_poses())
            else:
                raise ValueError(
                    "Optimizer must have either (intrinsics, cam2w) attributes "
                    "or (get_intrinsics, get_im_poses) methods"
                )

            confs = PointCloudUtils.clean_pointcloud(
                self.confs,
                K,
                cams,
                self.depthmaps,
                self.pts3d,
            )
        else:
            confs = self.confs
        return self.pts3d, self.depthmaps, confs
