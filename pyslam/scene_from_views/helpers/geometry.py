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

Part of the code is adapted from the original code by Naver Corporation.
Original code Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""

"""
Geometric transformation and projection utilities.
"""

import torch
import numpy as np
from functools import lru_cache

# cache is only available in Python 3.9+, use lru_cache(maxsize=None) as fallback
try:
    from functools import cache
except ImportError:
    cache = lambda func: lru_cache(maxsize=None)(func)
import cv2
import roma
from scipy.spatial.distance import pdist

from pyslam.utilities.torch import to_numpy, to_device


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
