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

Base point cloud optimizer class for scene optimization.

Part of the code is adapted from the original code by Naver Corporation.
Original code Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import roma
import tqdm
import cv2

from pyslam.utilities.math_utils import signed_log1p, signed_expm1
from pyslam.utilities.torch import to_numpy, to_cpu

from ..helpers import (
    EdgeUtils,
    ParameterUtils,
    ImageUtils,
    ConfidenceUtils,
    LossFunctions,
    GeometryUtils,
    PointCloudUtils,
)
from .learning_rate_schedules import LearningRateSchedules

# ============================================================================
# Global Alignment Loop
# ============================================================================


def global_alignment_loop(net, lr=0.01, niter=300, schedule="cosine", lr_min=1e-6):
    """
    Run global alignment optimization loop.

    Args:
        net: Optimizer network (BasePointCloudOptimizer instance)
        lr: Base learning rate
        niter: Number of iterations
        schedule: Learning rate schedule ('cosine' or 'linear')
        lr_min: Minimum learning rate

    Returns:
        Final loss value
    """
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print("Global alignement - optimizing for:")
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float("inf")
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                loss, lr = global_alignment_iter(
                    net, bar.n, niter, lr_base, lr_min, optimizer, schedule
                )
                bar.set_postfix_str(f"{lr=:g} loss={loss:g}")
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    """
    Single iteration of global alignment optimization.

    Args:
        net: Optimizer network
        cur_iter: Current iteration number
        niter: Total number of iterations
        lr_base: Base learning rate
        lr_min: Minimum learning rate
        optimizer: Optimizer instance
        schedule: Learning rate schedule ('cosine' or 'linear')

    Returns:
        Tuple of (loss, current_lr)
    """
    t = cur_iter / niter
    if schedule == "cosine":
        lr = LearningRateSchedules.cosine_schedule(t, lr_base, lr_min)
    elif schedule == "linear":
        lr = LearningRateSchedules.linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f"bad lr {schedule=}")
    LearningRateSchedules.adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()
    loss = net()
    loss.backward()
    optimizer.step()

    return float(loss.detach()), lr


# ============================================================================
# Base Point Cloud Optimizer
# ============================================================================


class BasePointCloudOptimizer(nn.Module):
    """
    Base class for optimizing a global scene from pairwise observations.

    This is an abstract base class that defines the interface and common functionality
    for point cloud optimizers. It handles:
    - Pairwise observations (graph edges)
    - Confidence maps and transformations
    - Pairwise pose parameters
    - Global alignment optimization loop

    Graph structure:
    - Nodes: images
    - Edges: pairwise observations = (pred1, pred2)
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            other = deepcopy(args[0])
            attrs = """edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
                        min_conf_thr conf_thr conf_i conf_j im_conf
                        base_scale norm_pw_scale POSE_DIM pw_poses 
                        pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose""".split()
            self.__dict__.update({k: other[k] for k in attrs})
        else:
            self._init_from_views(*args, **kwargs)

    def _init_from_views(
        self,
        view1,
        view2,
        pred1,
        pred2,
        dist="l1",
        conf="log",
        min_conf_thr=3,
        base_scale=0.5,
        allow_pw_adaptors=False,
        pw_break=20,
        rand_pose=torch.randn,
        iterationsCount=None,
        verbose=True,
    ):
        """
        Initialize optimizer from pairwise views and predictions.

        Args:
            view1: Dictionary with 'idx' (list of image indices) and optionally 'img'
            view2: Dictionary with 'idx' (list of image indices) and optionally 'img'
            pred1: Dictionary with 'pts3d' (list of point clouds) and 'conf' (confidence maps)
            pred2: Dictionary with 'pts3d_in_other_view' and 'conf'
            dist: Distance metric ('l1' or 'l2')
            conf: Confidence transformation mode ('log', 'sqrt', 'm1', 'id', 'none')
            min_conf_thr: Minimum confidence threshold
            base_scale: Base scale for pairwise poses
            allow_pw_adaptors: Whether to allow pairwise adaptors
            pw_break: Pairwise break parameter
            rand_pose: Function to generate random poses
            iterationsCount: Number of iterations (deprecated)
            verbose: Whether to print progress messages
        """
        super().__init__()
        if not isinstance(view1["idx"], list):
            view1["idx"] = view1["idx"].tolist()
        if not isinstance(view2["idx"], list):
            view2["idx"] = view2["idx"].tolist()
        self.edges = [(int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])]
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        self.dist = LossFunctions.get_all_dists()[dist]
        self.verbose = verbose

        self.n_imgs = self._check_edges()

        # input data
        pred1_pts = pred1["pts3d"]
        pred2_pts = pred2["pts3d_in_other_view"]
        self.pred_i = ParameterUtils.NoGradParamDict(
            {ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)}
        )
        self.pred_j = ParameterUtils.NoGradParamDict(
            {ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)}
        )
        self.imshapes = ImageUtils.get_imshapes(self.edges, pred1_pts, pred2_pts)

        # work in log-scale with conf
        pred1_conf = pred1["conf"]
        pred2_conf = pred2["conf"]
        self.min_conf_thr = min_conf_thr
        self.conf_trf = ConfidenceUtils.get_conf_trf(conf)

        self.conf_i = ParameterUtils.NoGradParamDict(
            {ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)}
        )
        self.conf_j = ParameterUtils.NoGradParamDict(
            {ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)}
        )
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(rand_pose((self.n_edges, 1 + self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose

        # possibly store images for show_pointcloud
        self.imgs = None
        if "img" in view1 and "img" in view2:
            imgs = [torch.zeros((3,) + hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1["idx"][v]
                imgs[idx] = view1["img"][v]
                idx = view2["idx"][v]
                imgs[idx] = view2["img"][v]
            self.imgs = ImageUtils.rgb(imgs)

    @property
    def n_edges(self):
        """Number of edges in the graph."""
        return len(self.edges)

    @property
    def str_edges(self):
        """List of edge strings in format 'i_j'."""
        return [EdgeUtils.edge_str(i, j) for i, j in self.edges]

    @property
    def imsizes(self):
        """List of image sizes as (W, H) tuples."""
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        """Device of the first parameter."""
        return next(iter(self.parameters())).device

    def state_dict(self, trainable=True):
        """Get state dict, optionally filtering to trainable parameters only."""
        all_params = super().state_dict()
        return {
            k: v
            for k, v in all_params.items()
            if k.startswith(("_", "pred_i.", "pred_j.", "conf_i.", "conf_j.")) != trainable
        }

    def load_state_dict(self, data):
        """Load state dict, merging with non-trainable parameters."""
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        """Validate that edge indices are consecutive starting from 0."""
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), "bad pair indices: missing values "
        return len(indices)

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        """Compute per-image confidence maps from pairwise confidences."""
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_adaptors(self):
        """Get pairwise adaptors for scale adaptation."""
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def _get_poses(self, poses):
        """Convert pose parameters to homogeneous transformation matrices."""
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        """
        Set pose parameters from rotation, translation, and scale.

        Args:
            poses: Pose parameter tensor
            idx: Index of pose to set
            R: Rotation matrix (3x3) or homogeneous matrix (4x4)
            T: Translation vector (3,) or None
            scale: Scale factor or None
            force: Whether to force update even if requires_grad=False
        """
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def get_pw_norm_scale_factor(self):
        """Get normalization factor for pairwise scales."""
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_scale(self):
        """Get pairwise scale factors."""
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_pw_poses(self):
        """Get pairwise poses (cam-to-world)."""
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_masks(self):
        """Get confidence masks for each image."""
        return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def depth_to_pts3d(self):
        """Convert depth maps to 3D points. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_pts3d(self, raw=False):
        """Get 3D points, optionally in raw format."""
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[: h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_focal(self, idx, focal, force=False):
        """Set focal length. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_focals(self):
        """Get focal lengths. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_known_focal_mask(self):
        """Get mask of known focal lengths. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_principal_points(self):
        """Get principal points. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_conf(self, mode=None):
        """Get confidence maps with optional transformation."""
        trf = self.conf_trf if mode is None else ConfidenceUtils.get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]

    def get_im_poses(self):
        """Get image poses. Must be implemented by subclasses."""
        raise NotImplementedError()

    def _set_depthmap(self, idx, depth, force=False):
        """Set depth map. Must be implemented by subclasses."""
        raise NotImplementedError()

    def get_depthmaps(self, raw=False):
        """Get depth maps. Must be implemented by subclasses."""
        raise NotImplementedError()

    def clean_pointcloud(self, **kw):
        """Clean pointcloud by reducing confidence for inconsistent points."""
        cams = GeometryUtils.inv(self.get_im_poses())
        K = self.get_intrinsics()
        depthmaps = self.get_depthmaps()
        all_pts3d = self.get_pts3d()

        new_im_confs = PointCloudUtils.clean_pointcloud(
            self.im_conf, K, cams, depthmaps, all_pts3d, **kw
        )

        for i, new_conf in enumerate(new_im_confs):
            self.im_conf[i].data[:] = new_conf
        return self

    def forward(self, ret_details=False):
        """
        Forward pass: compute optimization loss.

        Args:
            ret_details: If True, return loss details per edge

        Returns:
            Loss value, or (loss, details) if ret_details=True
        """
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = EdgeUtils.edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = GeometryUtils.geom_transform(
                pw_poses[e], pw_adapt[e] * self.pred_i[i_j]
            )
            aligned_pred_j = GeometryUtils.geom_transform(
                pw_poses[e], pw_adapt[e] * self.pred_j[i_j]
            )
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

    @torch.amp.autocast("cuda", enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
        """
        Compute global alignment optimization.

        Args:
            init: Initialization method ('mst', 'msp', 'known_poses', or None)
            niter_PnP: Number of PnP iterations for initialization
            **kw: Additional arguments for global_alignment_loop

        Returns:
            Final loss value
        """
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            # Import here to avoid circular dependency (helpers imports BasePointCloudOptimizer)
            from ..helpers import PoseInitialization

            PoseInitialization.init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
        elif init == "known_poses":
            # Import here to avoid circular dependency (helpers imports BasePointCloudOptimizer)
            from ..helpers import PoseInitialization

            PoseInitialization.init_from_known_poses(
                self, min_conf_thr=self.min_conf_thr, niter_PnP=niter_PnP
            )
        else:
            raise ValueError(f"bad value for {init=}")

        return global_alignment_loop(self, **kw)

    def get_intrinsics(self):
        """Get camera intrinsics. Must be implemented by subclasses."""
        raise NotImplementedError()


# ============================================================================
# Pair Viewer
# ============================================================================


class PairViewer(BasePointCloudOptimizer):
    """
    This a Dummy Optimizer.
    To use only when the goal is to visualize the results for a pair of images (with is_symmetrized)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.is_symmetrized and self.n_edges == 2
        self.has_im_poses = True

        # compute all parameters directly from raw input
        self.focals = []
        self.pp = []
        rel_poses = []
        confs = []
        for i in range(self.n_imgs):
            conf = float(
                self.conf_i[EdgeUtils.edge_str(i, 1 - i)].mean()
                * self.conf_j[EdgeUtils.edge_str(i, 1 - i)].mean()
            )
            if self.verbose:
                print(f"  - {conf=:.3} for edge {i}-{1-i}")
            confs.append(conf)

            H, W = self.imshapes[i]
            pts3d = self.pred_i[EdgeUtils.edge_str(i, 1 - i)]
            # Ensure pp is on the same device as pts3d
            device = pts3d.device if isinstance(pts3d, torch.Tensor) else self.device
            pp = torch.tensor((W / 2, H / 2), device=device)
            focal = float(
                GeometryUtils.estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld")
            )
            self.focals.append(focal)
            self.pp.append(pp)

            # estimate the pose of pts1 in image 2
            pixels = np.mgrid[:W, :H].T.astype(np.float32)
            pts3d_tensor = self.pred_j[EdgeUtils.edge_str(1 - i, i)]
            pts3d = (
                pts3d_tensor.cpu().numpy()
                if isinstance(pts3d_tensor, torch.Tensor)
                else pts3d_tensor
            )
            assert pts3d.shape[:2] == (H, W)
            msk_tensor = self.get_masks()[i]
            msk = msk_tensor.cpu().numpy() if isinstance(msk_tensor, torch.Tensor) else msk_tensor
            # Convert pp to numpy if it's a tensor
            pp_np = pp.cpu().numpy() if isinstance(pp, torch.Tensor) else np.array(pp)
            K = np.float32([(focal, 0, pp_np[0]), (0, focal, pp_np[1]), (0, 0, 1)])

            try:
                res = cv2.solvePnPRansac(
                    pts3d[msk],
                    pixels[msk],
                    K,
                    None,
                    iterationsCount=100,
                    reprojectionError=5,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                success, R, T, inliers = res
                assert success

                R = cv2.Rodrigues(R)[0]  # world to cam
                pose = GeometryUtils.inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
            except:
                pose = np.eye(4)
            rel_poses.append(torch.from_numpy(pose.astype(np.float32)))

        # Move rel_poses to the same device as pred_j tensors
        # Get device from one of the existing tensors
        device = next(iter(self.pred_j.values())).device if self.pred_j else torch.device("cpu")
        rel_poses = [pose.to(device) for pose in rel_poses]

        # let's use the pair with the most confidence
        if confs[0] > confs[1]:
            # ptcloud is expressed in camera1
            self.im_poses = [torch.eye(4, device=device), rel_poses[1]]  # I, cam2-to-cam1
            self.depth = [
                self.pred_i["0_1"][..., 2],
                GeometryUtils.geom_transform(GeometryUtils.inv(rel_poses[1]), self.pred_j["0_1"])[
                    ..., 2
                ],
            ]
        else:
            # ptcloud is expressed in camera2
            self.im_poses = [rel_poses[0], torch.eye(4, device=device)]  # I, cam1-to-cam2
            self.depth = [
                GeometryUtils.geom_transform(GeometryUtils.inv(rel_poses[0]), self.pred_j["1_0"])[
                    ..., 2
                ],
                self.pred_i["1_0"][..., 2],
            ]

        self.im_poses = nn.Parameter(torch.stack(self.im_poses, dim=0), requires_grad=False)
        self.focals = nn.Parameter(torch.tensor(self.focals), requires_grad=False)
        self.pp = nn.Parameter(torch.stack(self.pp, dim=0), requires_grad=False)
        self.depth = nn.ParameterList(self.depth)
        for p in self.parameters():
            p.requires_grad = False

    def _set_depthmap(self, idx, depth, force=False):
        if self.verbose:
            print("_set_depthmap is ignored in PairViewer")
        return

    def get_depthmaps(self, raw=False):
        depth = [d.to(self.device) for d in self.depth]
        return depth

    def _set_focal(self, idx, focal, force=False):
        self.focals[idx] = focal

    def get_focals(self):
        return self.focals

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.focals])

    def get_principal_points(self):
        return self.pp

    def get_intrinsics(self):
        focals = self.get_focals()
        pps = self.get_principal_points()
        K = torch.zeros((len(focals), 3, 3), device=self.device)
        for i in range(len(focals)):
            K[i, 0, 0] = K[i, 1, 1] = focals[i]
            K[i, :2, 2] = pps[i]
            K[i, 2, 2] = 1
        return K

    def get_im_poses(self):
        return self.im_poses

    def depth_to_pts3d(self):
        pts3d = []
        for d, intrinsics, im_pose in zip(self.depth, self.get_intrinsics(), self.get_im_poses()):
            pts, _ = PointCloudUtils.depthmap_to_absolute_camera_coordinates(
                d.cpu().numpy(), intrinsics.cpu().numpy(), im_pose.cpu().numpy()
            )
            pts3d.append(torch.from_numpy(pts).to(device=self.device))
        return pts3d

    def get_dense_pts3d(self, clean_depth=True):
        """Get dense 3D points, depthmaps, and confidence maps.

        Args:
            clean_depth: If True, clean the pointcloud by reducing confidence
                        for points that are occluded or inconsistent.

        Returns:
            Tuple of (pts3d, depthmaps, confs) where:
            - pts3d: List of [H, W, 3] tensors of 3D points
            - depthmaps: List of [H, W] tensors of depth maps
            - confs: List of [H, W] tensors of confidence maps
        """
        pts3d = self.get_pts3d(raw=False)  # Returns list of [H, W, 3] tensors
        depthmaps = self.get_depthmaps(raw=False)  # Returns list of [H, W] tensors

        # Get confidence maps (use raw im_conf, not transformed)
        confs = list(self.im_conf)  # Returns list of confidence tensors

        if clean_depth:
            # Clean pointcloud by reducing confidence for inconsistent points
            cams = GeometryUtils.inv(self.get_im_poses())
            K = self.get_intrinsics()
            confs = PointCloudUtils.clean_pointcloud(confs, K, cams, depthmaps, pts3d)

        return pts3d, depthmaps, confs

    def forward(self):
        return float("nan")


# ============================================================================
# Point Cloud Optimizer
# ============================================================================


class PointCloudOptimizer(BasePointCloudOptimizer):
    """
    Dense point cloud optimizer for global scene optimization.

    This optimizer performs dense optimization on full depth maps and point clouds.
    It optimizes camera poses, depth maps, and camera intrinsics (focal lengths,
    principal points) to align pairwise 3D predictions into a consistent global scene.

    Graph structure:
    - Nodes: images
    - Edges: pairwise observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        """
        Initialize dense point cloud optimizer.

        Args:
            *args: Arguments passed to BasePointCloudOptimizer
            optimize_pp: Whether to optimize principal points
            focal_break: Focal length break parameter
            **kwargs: Additional arguments passed to BasePointCloudOptimizer
        """
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(
            torch.randn(H, W) / 10 - 3 for H, W in self.imshapes
        )  # log(depth)
        self.im_poses = nn.ParameterList(
            self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs)
        )  # camera poses
        self.im_focals = nn.ParameterList(
            torch.FloatTensor([self.focal_break * np.log(max(H, W))]) for H, W in self.imshapes
        )  # camera intrinsics
        self.im_pp = nn.ParameterList(
            torch.zeros((2,)) for _ in range(self.n_imgs)
        )  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h * w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterUtils.ParameterStack(
            self.im_depthmaps, is_param=True, fill=self.max_area
        )
        self.im_poses = ParameterUtils.ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterUtils.ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterUtils.ParameterStack(self.im_pp, is_param=True)
        self.register_buffer("_pp", torch.tensor([(w / 2, h / 2) for h, w in self.imshapes]))
        self.register_buffer(
            "_grid",
            ParameterUtils.ParameterStack(
                [GeometryUtils.xy_grid(W, H, device=self.device) for H, W in self.imshapes],
                fill=self.max_area,
            ),
        )

        # pre-compute pixel weights
        self.register_buffer(
            "_weight_i",
            ParameterUtils.ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area
            ),
        )
        self.register_buffer(
            "_weight_j",
            ParameterUtils.ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area
            ),
        )

        # precompute aa
        self.register_buffer(
            "_stacked_pred_i",
            ParameterUtils.ParameterStack(self.pred_i, self.str_edges, fill=self.max_area),
        )
        self.register_buffer(
            "_stacked_pred_j",
            ParameterUtils.ParameterStack(self.pred_j, self.str_edges, fill=self.max_area),
        )
        self.register_buffer("_ei", torch.tensor([i for i, j in self.edges]))
        self.register_buffer("_ej", torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

    def _check_all_imgs_are_selected(self, msk):
        """Check that all images are selected in the mask."""
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), "incomplete mask!"

    def _get_msk_indices(self, msk):
        """Get indices from mask (supports various mask formats)."""
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f"bad {msk=}")

    def _set_focal(self, idx, focal, force=False):
        """Set focal length for image idx."""
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        """Get focal lengths for all images."""
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        """Get mask of known (non-trainable) focal lengths."""
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        """Set principal point for image idx."""
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W / 2, H / 2)) / 10
        return param

    def get_principal_points(self):
        """Get principal points for all images."""
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        """Get camera intrinsics matrices for all images."""
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):
        """Get camera-to-world poses for all images."""
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        """Set depth map for image idx."""
        depth = ParameterUtils.ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        """Get depth maps for all images."""
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[: h * w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        """Convert depth maps to 3D points in world coordinates."""
        # Get depths and projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = PointCloudUtils.fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return GeometryUtils.geom_transform(im_poses, rel_ptmaps)

    def get_pts3d(self, raw=False):
        """Get 3D points, optionally in raw format."""
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[: h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def forward(self):
        """Forward pass: compute optimization loss."""
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = GeometryUtils.geom_transform(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = GeometryUtils.geom_transform(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the loss
        li = (
            self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum()
            / self.total_area_i
        )
        lj = (
            self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum()
            / self.total_area_j
        )

        return li + lj

    def get_dense_pts3d(self, clean_depth=True):
        """Get dense 3D points, depthmaps, and confidence maps.

        Args:
            clean_depth: If True, clean the pointcloud by reducing confidence
                        for points that are occluded or inconsistent.

        Returns:
            Tuple of (pts3d, depthmaps, confs) where:
            - pts3d: List of [H, W, 3] tensors of 3D points
            - depthmaps: List of [H, W] tensors of depth maps
            - confs: List of [H, W] tensors of confidence maps
        """
        pts3d = self.get_pts3d(raw=False)  # Returns list of [H, W, 3] tensors
        depthmaps = self.get_depthmaps(raw=False)  # Returns list of [H, W] tensors

        # Get confidence maps (use raw im_conf, not transformed)
        confs = list(self.im_conf)  # Returns list of confidence tensors

        if clean_depth:
            # Clean pointcloud by reducing confidence for inconsistent points
            cams = GeometryUtils.inv(self.get_im_poses())
            K = self.get_intrinsics()
            confs = PointCloudUtils.clean_pointcloud(confs, K, cams, depthmaps, pts3d)

        return pts3d, depthmaps, confs
