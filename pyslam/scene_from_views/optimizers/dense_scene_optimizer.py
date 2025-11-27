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
"""

import torch
from enum import Enum

import pyslam.config as config

config.cfg.set_lib("mast3r")  # Dust3r is accessed through mast3r

from .scene_optimizer_base import SceneOptimizerBase
from .scene_optimizer_io import SceneOptimizerInput, SceneOptimizerOutput
from .point_cloud_optimizer import PairViewer, PointCloudOptimizer
from pyslam.utilities.torch import to_numpy


class GlobalAlignerMode(Enum):
    POINT_CLOUD_OPTIMIZER = "PointCloudOptimizer"
    MODULAR_POINT_CLOUD_OPTIMIZER = "ModularPointCloudOptimizer"
    PAIR_VIEWER = "PairViewer"


def global_aligner(dust3r_output, device, mode=GlobalAlignerMode.POINT_CLOUD_OPTIMIZER, **optim_kw):
    # extract all inputs
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in "view1 view2 pred1 pred2".split()]
    # build the optimizer
    if mode == GlobalAlignerMode.POINT_CLOUD_OPTIMIZER:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.MODULAR_POINT_CLOUD_OPTIMIZER:
        # Note: ModularPointCloudOptimizer is not included here to reduce complexity
        # If needed, it can be added from modular_optimizer.py
        raise NotImplementedError(
            "ModularPointCloudOptimizer not implemented in consolidated version"
        )
    elif mode == GlobalAlignerMode.PAIR_VIEWER:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f"Unknown mode {mode}")

    return net


class DenseSceneOptimizer(SceneOptimizerBase):
    """
    Wrapper class for Dust3r's global_aligner optimizer.

    This optimizer performs dense optimization on full depth maps and point clouds.
    It uses a single-stage optimization approach that aligns pairwise 3D predictions
    into a consistent global scene.

    Optimization Inputs:
        - Pairwise 3D point predictions (pred1, pred2): Dense 3D point clouds predicted
          for each image pair from different viewpoints
        - Confidence maps: Per-pixel confidence scores for each prediction, used to
          weight the optimization loss
        - Image pairs: RGB images from different viewpoints forming the pairwise observations
        - View indices: Mapping between image pairs and their corresponding views

    Optimization Targets:
        - Camera poses (im_poses): Camera-to-world transformation matrices for each view,
          optimized to align all pairwise observations
        - Depth maps (im_depthmaps): Dense per-pixel depth estimates (in log space) for
          each image, representing the 3D structure of the scene
        - Camera focal lengths (im_focals): Intrinsic camera parameters optimized to
          ensure consistent 3D reconstruction across views
        - Principal points (im_pp): Optional camera intrinsic parameters for principal
          point offsets

    The optimization minimizes a geometric consistency loss that measures the distance
    between aligned pairwise 3D predictions and the projected 3D points from the
    optimized global scene. This ensures all pairwise observations are consistent with
    a single unified 3D reconstruction.

    This class consolidates all code from dust3r.cloud_opt module.
    """

    def __init__(self, device=None, **kwargs):
        """
        Initialize the global aligner optimizer.

        Args:
            device: Device to run optimization on (e.g., 'cuda', 'cpu')
            **kwargs: Additional optimizer parameters
        """
        super().__init__(device=device, **kwargs)

    def optimize(
        self,
        optimizer_input: "SceneOptimizerInput",
        verbose: bool = True,
        **optimizer_kwargs,
    ) -> "SceneOptimizerOutput":
        """
        Run global alignment optimization.

        Args:
            optimizer_input: SceneOptimizerInput containing view1, view2, pred1, pred2, pairs, and images
            verbose: If True, print progress messages
            **optimizer_kwargs: Additional optimizer parameters:
                - niter: Number of iterations (default: 300)
                - schedule: Learning rate schedule ('linear', 'cosine', etc.) (default: 'linear')
                - lr: Learning rate (default: 0.01)

        Returns:
            SceneOptimizerOutput: Unified output containing optimized scene
        """

        # Validate required fields
        if not optimizer_input.pair_predictions:
            raise ValueError("DenseSceneOptimizer requires pair_predictions in optimizer_input")

        # Get optimizer parameters
        niter = optimizer_kwargs.get("niter", self.kwargs.get("niter", 300))
        schedule = optimizer_kwargs.get("schedule", self.kwargs.get("schedule", "linear"))
        lr = optimizer_kwargs.get("lr", self.kwargs.get("lr", 0.01))

        # Determine mode based on number of images
        mode = (
            GlobalAlignerMode.POINT_CLOUD_OPTIMIZER
            if len(optimizer_input.images) > 2
            else GlobalAlignerMode.PAIR_VIEWER
        )

        if verbose:
            print(f"[Global Aligner] Initializing (mode: {mode.name})...")

        # Convert unified format to Dust3r format for global_aligner
        # Extract data from pair_predictions
        view1_idx = []
        view2_idx = []
        pred1_pts3d = []
        pred2_pts3d = []
        pred1_conf = []
        pred2_conf = []
        view1_imgs = []
        view2_imgs = []

        # Ensure device is a torch.device object
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

        for pred in optimizer_input.pair_predictions:
            view1_idx.append(pred.image_idx_i)
            view2_idx.append(pred.image_idx_j)
            # Ensure all tensors are on the correct device
            pred1_pts3d.append(
                pred.pts3d_i.to(device) if isinstance(pred.pts3d_i, torch.Tensor) else pred.pts3d_i
            )
            pred2_pts3d.append(
                pred.pts3d_j.to(device) if isinstance(pred.pts3d_j, torch.Tensor) else pred.pts3d_j
            )
            pred1_conf.append(
                pred.conf_i.to(device) if isinstance(pred.conf_i, torch.Tensor) else pred.conf_i
            )
            pred2_conf.append(
                pred.conf_j.to(device) if isinstance(pred.conf_j, torch.Tensor) else pred.conf_j
            )
            if pred.image_i is not None:
                img_i = (
                    pred.image_i.to(device)
                    if isinstance(pred.image_i, torch.Tensor)
                    else pred.image_i
                )
                view1_imgs.append(img_i)
            if pred.image_j is not None:
                img_j = (
                    pred.image_j.to(device)
                    if isinstance(pred.image_j, torch.Tensor)
                    else pred.image_j
                )
                view2_imgs.append(img_j)

        # Build view1 and view2 dicts
        view1 = {"idx": view1_idx}
        view2 = {"idx": view2_idx}
        if view1_imgs:
            view1["img"] = view1_imgs
        if view2_imgs:
            view2["img"] = view2_imgs

        # Build pred1 and pred2 dicts
        pred1 = {"pts3d": pred1_pts3d, "conf": pred1_conf}
        pred2 = {"pts3d_in_other_view": pred2_pts3d, "conf": pred2_conf}

        # Create dust3r output dict
        dust3r_output = {
            "view1": view1,
            "view2": view2,
            "pred1": pred1,
            "pred2": pred2,
        }

        # Create global aligner scene
        scene = global_aligner(dust3r_output, device=device, mode=mode, verbose=verbose)

        # Run global alignment optimization
        if mode == GlobalAlignerMode.POINT_CLOUD_OPTIMIZER:
            if verbose:
                print(
                    f"[Global Aligner] Running optimization ({niter} iterations, lr={lr}, schedule={schedule})..."
                )
            scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        else:
            if verbose:
                print("[Global Aligner] Skipping optimization (pair viewer mode)...")

        return SceneOptimizerOutput(
            scene=scene,
            optimizer_type=self.optimizer_type,
        )

    @property
    def optimizer_type(self) -> str:
        """Return the optimizer type identifier."""
        return "dense_scene_optimizer"

    def extract_results(
        self,
        optimizer_output: "SceneOptimizerOutput",
        processed_images: list = None,
        **kwargs,
    ) -> "SceneOptimizerOutput":
        """
        Extract results from the optimized scene.

        Args:
            optimizer_output: SceneOptimizerOutput from optimize()
            processed_images: Preprocessed images (not used, kept for compatibility)
            **kwargs: Additional parameters

        Returns:
            SceneOptimizerOutput: Updated output with extracted results
        """
        from .scene_optimizer_io import SceneOptimizerOutput

        scene = optimizer_output.scene

        # Extract results from scene
        pts3d = scene.get_pts3d()  # List of [h, w, 3] point clouds
        rgb_imgs = scene.imgs  # List of [h, w, 3] RGB images
        cams2world = scene.get_im_poses()  # [n, 4, 4] camera poses
        focals = scene.get_focals()  # [n] focal lengths
        confs = scene.im_conf  # List of [h, w] confidence maps

        # Convert to numpy
        rgb_imgs = to_numpy(rgb_imgs)
        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)
        pts3d = to_numpy(pts3d)
        confs = [to_numpy(x) for x in confs]

        return SceneOptimizerOutput(
            scene=scene,
            optimizer_type=optimizer_output.optimizer_type,
            rgb_imgs=rgb_imgs,
            focals=focals,
            cams2world=cams2world,
            pts3d=pts3d,
            confs=confs,
            additional_data=optimizer_output.additional_data,
        )
