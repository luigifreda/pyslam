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
"""

import torch
import numpy as np
import scipy.cluster.hierarchy as sch

from .scene_optimizer_base import SceneOptimizerBase
from .scene_optimizer_config import adapt_config_for_subsample

# Import all helper functions and classes from helpers module
from ..helpers import (
    SparseGA,
    CanonicalViewUtils,
    GraphUtils,
    sparse_scene_optimizer,
)
from .tsdf_postprocess import TSDFPostProcess
from pyslam.utilities.torch import to_numpy


class SparseSceneOptimizer(SceneOptimizerBase):
    """
    Wrapper class for MASt3r's sparse_scene_optimizer.

    This optimizer performs sparse optimization on feature correspondences.
    It uses a two-stage optimization approach:
    - 1. Coarse: 3D point matching loss (loss_3d) - aligns 3D correspondences
        It optimizes camera poses (rotations and translations) and scale parameters.
        Intrinsics (focals, principal points) and depths are frozen.
    - 2. Fine: 2D reprojection loss (loss_2d) - refines with pixel reprojection
        It optimizes camera poses, intrinsics (focals, principal points),
        scale parameters, and depths (if opt_depth=True).
    """

    def __init__(self, device=None, **kwargs):
        """
        Initialize the sparse scene optimizer.

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
        Run sparse scene optimization.

        Args:
            optimizer_input: SceneOptimizerInput containing pairs_output, pairs, images, filelist, and cache_dir
            verbose: If True, print progress messages
            **optimizer_kwargs: Additional optimizer parameters:
                - subsample: Subsampling factor (default: 8)
                - lr1: Coarse learning rate (default: 0.07)
                - niter1: Number of coarse iterations (default: 500)
                - lr2: Fine learning rate (default: 0.014)
                - niter2: Number of fine iterations (default: 200)
                - matching_conf_thr: Matching confidence threshold (default: 5.0)
                - shared_intrinsics: Whether to use shared intrinsics (default: False)
                - optim_level: Optimization level ('coarse', 'refine', 'refine+depth') (default: 'refine+depth')
                - kinematic_mode: Kinematic chain mode ('mst', 'hclust-ward', etc.) (default: 'hclust-ward')

        Returns:
            SceneOptimizerOutput: Unified output containing optimized scene and results
        """
        from .scene_optimizer_io import SceneOptimizerInput, SceneOptimizerOutput

        # Validate required fields
        if optimizer_input.filelist is None:
            raise ValueError("filelist is required in optimizer_input for SparseSceneOptimizer")
        if optimizer_input.cache_dir is None:
            raise ValueError("cache_dir is required in optimizer_input for SparseSceneOptimizer")

        # Sparse optimizer needs pairs_output for canonical view processing
        # pairs_output contains paths to cached pair data needed for canonical view computation
        pairs_output = optimizer_input.pairs_output
        if pairs_output is None:
            raise ValueError(
                "SparseSceneOptimizer requires pairs_output in optimizer_input "
                "for canonical view processing. pairs_output should contain paths to cached pair data."
            )

        # Get optimizer parameters
        subsample = optimizer_kwargs.get("subsample", self.kwargs.get("subsample", 8))
        
        # Check if adaptive scaling is enabled (default: True)
        adaptive_scaling = optimizer_kwargs.get(
            "adaptive_scaling", self.kwargs.get("adaptive_scaling", True)
        )
        
        # Get base parameters from config or kwargs
        base_config = {
            "lr1": optimizer_kwargs.get("lr1", self.kwargs.get("lr1", 0.08)),
            "niter1": optimizer_kwargs.get("niter1", self.kwargs.get("niter1", 600)),
            "lr2": optimizer_kwargs.get("lr2", self.kwargs.get("lr2", 0.02)),
            "niter2": optimizer_kwargs.get("niter2", self.kwargs.get("niter2", 300)),
        }
        
        # Apply adaptive scaling if enabled and subsample != 8
        if adaptive_scaling and subsample != 8:
            adapted_config = adapt_config_for_subsample(base_config, subsample)
            if verbose:
                print(f"[Sparse Scene Optimizer] Adaptive scaling enabled for subsample={subsample}")
                print(f"  Original: lr1={base_config['lr1']:.4f}, niter1={base_config['niter1']}, "
                      f"lr2={base_config['lr2']:.4f}, niter2={base_config['niter2']}")
                print(f"  Adapted:  lr1={adapted_config['lr1']:.4f}, niter1={adapted_config['niter1']}, "
                      f"lr2={adapted_config['lr2']:.4f}, niter2={adapted_config['niter2']}")
            base_config = adapted_config
        
        lr1 = base_config["lr1"]
        niter1 = base_config["niter1"]
        lr2 = base_config["lr2"]
        niter2 = base_config["niter2"]
        
        matching_conf_thr = optimizer_kwargs.get(
            "matching_conf_thr", self.kwargs.get("matching_conf_thr", 5.0)
        )
        shared_intrinsics = optimizer_kwargs.get(
            "shared_intrinsics", self.kwargs.get("shared_intrinsics", False)
        )
        optim_level = optimizer_kwargs.get(
            "optim_level", self.kwargs.get("optim_level", "refine+depth")
        )
        kinematic_mode = optimizer_kwargs.get(
            "kinematic_mode", self.kwargs.get("kinematic_mode", "hclust-ward")
        )
        skip_fine = optimizer_kwargs.get(
            "skip_fine", False
        )  # Debug option to skip fine optimization

        if optim_level == "coarse":
            niter2 = 0

        # Extract fields from optimizer_input
        filelist = optimizer_input.filelist
        pairs = optimizer_input.pairs
        cache_dir = optimizer_input.cache_dir

        # Convert pair naming convention from dust3r to mast3r
        pairs_in = CanonicalViewUtils.convert_dust3r_pairs_naming(filelist, pairs)

        # Extract canonical pointmaps per image
        tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = (
            CanonicalViewUtils.prepare_canonical_data(
                filelist,
                pairs_output,
                subsample=subsample,
                cache_path=cache_dir,
                mode="avg-angle",
                device=self.device,
                verbose=verbose,  # Pass verbose flag to prepare_canonical_data
            )
        )

        # Smartly combine all useful data
        imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = (
            CanonicalViewUtils.condense_data(
                filelist, tmp_pairs, canonical_views, preds_21, dtype=torch.float32
            )
        )

        # Build kinematic chain (minimum spanning tree or hierarchical clustering)
        if kinematic_mode == "mst":
            mst = GraphUtils.compute_min_spanning_tree(pairwise_scores)
        elif kinematic_mode.startswith("hclust"):
            mode, linkage = kinematic_mode.split("-")
            n_patches = (imsizes // subsample).prod(dim=1)
            max_n_corres = 3 * torch.minimum(n_patches[:, None], n_patches[None, :])
            pws = (pairwise_scores.clone() / max_n_corres).clip(max=1)
            pws.fill_diagonal_(1)
            pws = to_numpy(pws)
            distance_matrix = np.where(pws, 1 - pws, 2)
            condensed_distance_matrix = sch.distance.squareform(distance_matrix)
            Z = sch.linkage(condensed_distance_matrix, method=linkage)

            tree = np.eye(len(filelist))
            new_to_old_nodes = {i: i for i in range(len(filelist))}
            for i, (a, b) in enumerate(Z[:, :2].astype(int)):
                a = new_to_old_nodes[a]
                b = new_to_old_nodes[b]
                tree[a, b] = tree[b, a] = 1
                best = a if pws[a].sum() > pws[b].sum() else b
                new_to_old_nodes[len(filelist) + i] = best
                pws[best] = np.maximum(pws[a], pws[b])

            pairwise_scores = torch.from_numpy(tree)
            mst = GraphUtils.compute_min_spanning_tree(pairwise_scores)
        else:
            raise ValueError(f"bad {kinematic_mode=}")

        if verbose:
            print("[Sparse Scene Optimizer] Running two-stage optimization...")
            print(f"  Coarse: {niter1} iterations, lr={lr1}")
            if skip_fine:
                print(f"  Fine: SKIPPED (debug mode)")
            else:
                print(f"  Fine: {niter2} iterations, lr={lr2}")

        # Sparse scene optimizer
        # Two-stage optimization:
        # - Coarse: 3D point matching loss (loss_3d) — aligns 3D correspondences
        # - Fine: 2D reprojection loss (loss_2d) — refines with pixel reprojection
        imgs, res_coarse, res_fine = sparse_scene_optimizer(
            filelist,
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
            shared_intrinsics=shared_intrinsics,
            cache_path=cache_dir,
            device=self.device,
            dtype=torch.float32,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=0 if skip_fine else niter2,  # Skip fine optimization if skip_fine=True
            opt_depth="depth" in optim_level,
            matching_conf_thr=matching_conf_thr,
            pairs_output=pairs_output,  # Pass pairs_output for initial pose extraction
        )

        # Create scene object
        scene = SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths)

        from .scene_optimizer_io import SceneOptimizerOutput

        return SceneOptimizerOutput(
            scene=scene,
            optimizer_type=self.optimizer_type,
            additional_data={
                "pairs_in": pairs_in,
                "anchors": anchors,
                "canonical_paths": canonical_paths,
            },
        )

    @property
    def optimizer_type(self) -> str:
        """Return the optimizer type identifier."""
        return "sparse_scene_optimizer"

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
            **kwargs: Additional parameters:
                - clean_depth: Whether to clean depth maps (default: True)
                - use_tsdf: Whether to use TSDF post-processing (default: False)
                - TSDF_thresh: TSDF threshold if use_tsdf=True (default: 0.0)

        Returns:
            SceneOptimizerOutput: Updated output with extracted results
        """
        from .scene_optimizer_io import SceneOptimizerOutput
        from pyslam.utilities.torch import to_numpy

        scene = optimizer_output.scene

        clean_depth = kwargs.get("clean_depth", True)
        use_tsdf = kwargs.get("use_tsdf", False)
        TSDF_thresh = kwargs.get("TSDF_thresh", 0.0)

        rgb_imgs = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()

        # Get dense point clouds
        # TSDFPostProcess is optional (now included in this file)
        if use_tsdf and TSDF_thresh > 0:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
        else:
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

        # Convert to numpy
        rgb_imgs = to_numpy(rgb_imgs)
        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)
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
