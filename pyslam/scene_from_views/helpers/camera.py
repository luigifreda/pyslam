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
Camera and pose utilities.
"""

import torch
import numpy as np
import cv2
import roma
import tqdm
import scipy.sparse as sp

from .geometry import GeometryUtils
from pyslam.utilities.torch import to_numpy

# EdgeUtils is defined in sparse_ga.py
# Import it locally where needed to avoid circular dependency


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
            from .sparse_ga import EdgeUtils

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
            from .sparse_ga import EdgeUtils

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
        from .sparse_ga import EdgeUtils

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
                from .sparse_ga import EdgeUtils

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
                from .sparse_ga import EdgeUtils

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
                    from .sparse_ga import EdgeUtils

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
