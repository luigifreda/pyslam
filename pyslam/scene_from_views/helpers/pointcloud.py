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
Point cloud processing utilities.
"""

import torch
import numpy as np

from .geometry import GeometryUtils


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
