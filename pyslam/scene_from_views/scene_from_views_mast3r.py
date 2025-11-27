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

import cv2
import numpy as np
import torch
from typing import List, Optional
import trimesh
import copy
import os

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from pyslam.utilities.dust3r import (
    Dust3rImagePreprocessor,
    convert_mv_output_to_geometry,
    estimate_focal_knowing_depth,
    calibrate_camera_pnpransac,
)
from pyslam.utilities.torch import to_numpy

import pyslam.config as config

config.cfg.set_lib("mast3r")


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SceneFromViewsMast3r(SceneFromViewsBase):
    """
    Scene reconstruction using MASt3R (Grounding Image Matching in 3D with MASt3R).

    This model performs multi-view reconstruction with pose estimation.
    """

    def __init__(
        self,
        device=None,
        model_path=None,
        inference_size=512,
        min_conf_thr=1.5,
        scenegraph_type="complete",
        winsize=1,
        win_cyclic=False,
        refid=0,
        TSDF_thresh=0.0,
        use_tsdf=None,
        clean_depth=True,
        optimizer_config=None,
        **kwargs,
    ):
        """
        Initialize MASt3R model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_path: Path to model checkpoint
            inference_size: Image size for inference (224 or 512)
            min_conf_thr: Minimum confidence threshold for filtering points
            scenegraph_type: Scene graph type ('complete', 'swin', 'logwin', 'oneref')
            winsize: Window size for scene graph
            win_cyclic: Whether to use cyclic windows
            refid: Reference image ID for 'oneref' scenegraph
            TSDF_thresh: TSDF threshold for post-processing (only used if use_tsdf=True)
            use_tsdf: Whether to apply TSDFPostProcess (None = auto-detect from TSDF_thresh > 0)
            clean_depth: Whether to clean up depth maps
            optimizer_config: Dictionary specifying optimizer configuration:
                - type: 'dense_scene_optimizer' or 'sparse_scene_optimizer'
                - Additional parameters specific to the optimizer type
                If None, defaults to 'sparse_scene_optimizer' with default parameters.
                If type='dense_scene_optimizer', uses DEFAULT_DENSE_OPTIMIZER_CONFIG as base.
                Note: dense_scene_optimizer requires format conversion (not yet implemented).
                Optimizer parameters can be provided via kwargs.
            **kwargs: Additional parameters including optimizer parameters:
                - type: Optimizer type ('sparse_scene_optimizer' or 'dense_scene_optimizer')
                For sparse_scene_optimizer:
                    - optim_level: Optimization level ('coarse', 'refine', 'refine+depth')
                    - lr1: Coarse learning rate
                    - niter1: Number of iterations for coarse alignment
                    - lr2: Fine learning rate
                    - niter2: Number of iterations for refinement
                    - matching_conf_thr: Matching confidence threshold
                    - shared_intrinsics: Whether to use shared intrinsics
                    - subsample: Subsampling factor for feature extraction
                    - kinematic_mode: Kinematic chain mode ('mst', 'hclust-ward', etc.)
                For dense_scene_optimizer:
                    - niter: Number of iterations
                    - schedule: Learning rate schedule ('linear', 'cosine', etc.)
                    - lr: Learning rate
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from mast3r.model import AsymmetricMASt3R

        if model_path is None:
            model_path = (
                kRootFolder
                + "/thirdparty/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            )

        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(self.device)
        self.inference_size = inference_size
        self.min_conf_thr = min_conf_thr
        self.scenegraph_type = scenegraph_type
        self.winsize = winsize
        self.win_cyclic = win_cyclic
        self.refid = refid
        self.TSDF_thresh = TSDF_thresh
        # Auto-detect use_tsdf from TSDF_thresh if not explicitly set
        if use_tsdf is None:
            use_tsdf = TSDF_thresh > 0
        self.use_tsdf = use_tsdf
        self.clean_depth = clean_depth

        # Setup optimizer configuration
        # Extract optimizer parameters from kwargs (including type)
        optimizer_params = {}
        for param in [
            "type",  # Allow type to be specified in kwargs
            "optim_level",
            "lr1",
            "niter1",
            "lr2",
            "niter2",
            "matching_conf_thr",
            "shared_intrinsics",
            "subsample",
            "kinematic_mode",
            # Dense optimizer parameters
            "niter",
            "schedule",
            "lr",
        ]:
            if param in kwargs:
                optimizer_params[param] = kwargs.pop(param)

        # Determine optimizer type (from optimizer_config or kwargs)
        optimizer_type = None
        if optimizer_config is not None:
            optimizer_type = optimizer_config.get("type")
        if optimizer_type is None:
            optimizer_type = optimizer_params.get("type", "sparse_scene_optimizer")

        if optimizer_config is None:
            # Use appropriate default config based on optimizer type
            from pyslam.scene_from_views.optimizers import (
                DEFAULT_SPARSE_OPTIMIZER_CONFIG,
                DEFAULT_DENSE_OPTIMIZER_CONFIG,
            )

            if optimizer_type == "dense_scene_optimizer":
                optimizer_config = DEFAULT_DENSE_OPTIMIZER_CONFIG.copy()
            else:
                optimizer_config = DEFAULT_SPARSE_OPTIMIZER_CONFIG.copy()

            # Override with user-provided parameters from kwargs
            optimizer_config.update(optimizer_params)
        else:
            # Merge optimizer_params into provided config
            optimizer_config = optimizer_config.copy()
            optimizer_config.update(optimizer_params)
            # Ensure type is set
            if "type" not in optimizer_config:
                optimizer_config["type"] = optimizer_type

        self.optimizer_config = optimizer_config

        # Initialize optimizer based on config
        self._init_optimizer()

        self.dust3r_preprocessor = Dust3rImagePreprocessor(
            inference_size=inference_size, verbose=True
        )

    def _init_optimizer(self):
        """Initialize the optimizer based on optimizer_config."""
        from pyslam.scene_from_views.optimizers import (
            scene_optimizer_factory,
            SceneOptimizerType,
        )

        # Determine default type based on optimizer_config
        optimizer_type_str = self.optimizer_config.get("type", "sparse_scene_optimizer")
        if optimizer_type_str == "dense_scene_optimizer":
            default_type = SceneOptimizerType.DENSE
        else:
            default_type = SceneOptimizerType.SPARSE

        # Both dense and sparse optimizers are now supported with unified format
        self.optimizer = scene_optimizer_factory(
            optimizer_config=self.optimizer_config,
            device=self.device,
            default_type=default_type,
        )

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List:
        """
        Preprocess images for MASt3R.

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (not used for this model)
        """
        return self.dust3r_preprocessor.preprocess_images(images)

    def infer(self, processed_images: List, **kwargs):
        """
        Run inference on preprocessed images using MASt3R.

        Args:
            processed_images: Preprocessed images from preprocess_images()
            **kwargs: Additional inference parameters (can override init parameters)

        Returns:
            Dictionary containing scene object and parameters
        """
        # Override parameters with kwargs if provided
        min_conf_thr = kwargs.get("min_conf_thr", self.min_conf_thr)
        scenegraph_type = kwargs.get("scenegraph_type", self.scenegraph_type)
        winsize = kwargs.get("winsize", self.winsize)
        win_cyclic = kwargs.get("win_cyclic", self.win_cyclic)
        refid = kwargs.get("refid", self.refid)
        TSDF_thresh = kwargs.get("TSDF_thresh", self.TSDF_thresh)
        use_tsdf = kwargs.get("use_tsdf", self.use_tsdf)
        clean_depth = kwargs.get("clean_depth", self.clean_depth)
        verbose = kwargs.get("verbose", True)

        # Extract optimizer parameters from kwargs or use defaults from optimizer_config
        optim_level = kwargs.pop(
            "optim_level", self.optimizer_config.get("optim_level", "refine+depth")
        )
        lr1 = kwargs.pop("lr1", self.optimizer_config.get("lr1", 0.07))
        niter1 = kwargs.pop("niter1", self.optimizer_config.get("niter1", 500))
        lr2 = kwargs.pop("lr2", self.optimizer_config.get("lr2", 0.014))
        niter2 = kwargs.pop("niter2", self.optimizer_config.get("niter2", 200))
        matching_conf_thr = kwargs.pop(
            "matching_conf_thr", self.optimizer_config.get("matching_conf_thr", 5.0)
        )
        shared_intrinsics = kwargs.pop(
            "shared_intrinsics", self.optimizer_config.get("shared_intrinsics", False)
        )
        subsample = kwargs.pop("subsample", self.optimizer_config.get("subsample", 8))
        kinematic_mode = kwargs.pop(
            "kinematic_mode", self.optimizer_config.get("kinematic_mode", "hclust-ward")
        )

        imgs_preproc = processed_images

        # Prepare filelist (dummy names)
        filelist = [f"image_{i}" for i in range(len(imgs_preproc))]

        # Handle single image case
        if len(imgs_preproc) == 1:
            imgs_preproc = [imgs_preproc[0], copy.deepcopy(imgs_preproc[0])]
            imgs_preproc[1]["idx"] = 1
            filelist = [filelist[0], filelist[0] + "_2"]

        # Build scene graph
        import mast3r.utils.path_to_dust3r  # Ensure path is set
        from dust3r.image_pairs import make_pairs

        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(refid))
        if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
            scene_graph_params.append("noncyclic")
        scene_graph = "-".join(scene_graph_params)

        pairs = make_pairs(imgs_preproc, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        if optim_level == "coarse":
            niter2 = 0

        # Create cache directory
        import tempfile
        import os

        cache_dir = tempfile.mkdtemp(suffix="_mast3r_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # ====================================================================
        # INFERENCE PART: Forward pass to extract features and correspondences
        # ====================================================================
        from mast3r.cloud_opt.sparse_ga import (
            convert_dust3r_pairs_naming,
            forward_mast3r,
            prepare_canonical_data,
            condense_data,
            compute_min_spanning_tree,
        )

        # Convert pair naming convention from dust3r to mast3r
        pairs_in = convert_dust3r_pairs_naming(filelist, pairs)

        # Forward pass: runs MASt3R model on all pairs
        # This extracts 3D points, confidence maps, and descriptors for each pair
        pairs_output, cache_path = forward_mast3r(
            pairs_in,
            self.model,
            cache_path=cache_dir,
            subsample=subsample,
            desc_conf="desc_conf",
            device=self.device,
        )

        # Extract canonical pointmaps per image
        tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = (
            prepare_canonical_data(
                filelist,
                pairs_output,
                subsample=subsample,
                cache_path=cache_dir,
                mode="avg-angle",
                device=self.device,
            )
        )

        # Smartly combine all useful data
        imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = condense_data(
            filelist, tmp_pairs, canonical_views, preds_21, dtype=torch.float32
        )

        # ====================================================================
        # OPTIMIZER: Optimize camera poses, intrinsics, depth maps
        # ====================================================================
        optimizer_type = self.optimizer_config.get("type", "sparse_scene_optimizer")

        if verbose:
            print(f"[Optimizer] Using {optimizer_type}...")

        # Create unified optimizer input
        optimizer_input = self.create_optimizer_input(
            raw_output=pairs_output,  # pairs_output for canonical view processing
            pairs=pairs,
            processed_images=imgs_preproc,
            filelist=filelist,
            cache_dir=cache_dir,
        )

        # Optimize using unified interface
        optimizer_output = self.optimizer.optimize(
            optimizer_input=optimizer_input,
            verbose=verbose,
            subsample=subsample,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            matching_conf_thr=matching_conf_thr,
            shared_intrinsics=shared_intrinsics,
            optim_level=optim_level,
            kinematic_mode=kinematic_mode,
        )

        scene = optimizer_output.scene

        # Extract additional data if available
        additional_data = optimizer_output.additional_data or {}
        pairs_in = additional_data.get("pairs_in", pairs_in)
        anchors = additional_data.get("anchors", anchors)
        canonical_paths = additional_data.get("canonical_paths", canonical_paths)

        # Extract dense 3D map
        rgb_imgs = scene.imgs
        if rgb_imgs is None:
            # If images are not available, create dummy images from image shapes
            rgb_imgs = [torch.zeros((h, w, 3), device=scene.device) for h, w in scene.imshapes]

        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()

        # Get dense point clouds
        # TSDFPostProcess is optional and can be disabled via use_tsdf parameter
        if use_tsdf and TSDF_thresh > 0:
            from pyslam.scene_from_views.optimizers.tsdf_postprocess import TSDFPostProcess

            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
            pts3d, _, confs = tsdf.get_dense_pts3d(clean_depth=clean_depth)
        else:
            pts3d, _, confs = scene.get_dense_pts3d(clean_depth=clean_depth)

        # Convert to numpy (convert each element separately)
        pts3d = to_numpy(pts3d)
        confs = [to_numpy(x) for x in confs]
        rgb_imgs = to_numpy(rgb_imgs)

        return {
            "scene": scene,
            "rgb_imgs": rgb_imgs,
            "focals": focals,
            "cams2world": cams2world,
            "pts3d": pts3d,
            "confs": confs,
            "min_conf_thr": min_conf_thr,
        }

    def postprocess_results(
        self,
        raw_output,
        images: List[np.ndarray],
        processed_images: List,
        as_pointcloud: bool = True,
        **kwargs,
    ) -> SceneFromViewsResult:
        """
        Postprocess raw MASt3R output into SceneFromViewsResult.

        Args:
            raw_output: Dictionary containing scene data and parameters from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        rgb_imgs = raw_output["rgb_imgs"]
        focals = raw_output["focals"]
        cams2world = raw_output["cams2world"]
        pts3d = raw_output["pts3d"]
        confs = raw_output["confs"]
        min_conf_thr = raw_output["min_conf_thr"]

        # Ensure rgb_imgs is numpy (it should already be converted in infer())
        if not isinstance(rgb_imgs, (list, tuple)) or (
            rgb_imgs and isinstance(rgb_imgs[0], torch.Tensor)
        ):
            rgb_imgs = to_numpy(rgb_imgs)

        # Ensure pts3d and confs are numpy (they should already be converted in infer())
        if not isinstance(pts3d, (list, tuple)) or (pts3d and isinstance(pts3d[0], torch.Tensor)):
            pts3d = to_numpy(pts3d)
        if not isinstance(confs, (list, tuple)) or (confs and isinstance(confs[0], torch.Tensor)):
            confs = [to_numpy(x) for x in confs]

        # Ensure all arrays have correct shapes: [H, W, 3] for pts3d and rgb_imgs, [H, W] for confs
        # Remove any extra dimensions (e.g., if shape is [1, H, W, 3], reshape to [H, W, 3])
        rgb_imgs = [np.squeeze(img) for img in rgb_imgs]
        pts3d = [np.squeeze(pts) for pts in pts3d]
        confs = [np.squeeze(conf) for conf in confs]

        # Ensure rgb_imgs and pts3d have 3 dimensions [H, W, 3]
        rgb_imgs = [img if img.ndim == 3 else img.reshape(*img.shape[:2], -1) for img in rgb_imgs]
        pts3d = [pts if pts.ndim == 3 else pts.reshape(*pts.shape[:2], -1) for pts in pts3d]

        # Ensure confs have 2 dimensions [H, W]
        confs = [conf if conf.ndim == 2 else conf.reshape(*conf.shape[:2]) for conf in confs]

        # Create mask with correct shape matching confs
        mask = [c > min_conf_thr for c in confs]

        # Convert to geometry
        global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, mask, as_pointcloud)
        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)
        confs = [to_numpy(x) for x in confs]

        # Extract depth maps from point clouds
        depth_predictions = []
        for i, (pts, msk) in enumerate(zip(pts3d, mask)):
            h, w = rgb_imgs[i].shape[:2]
            depth = np.zeros((h, w), dtype=pts.dtype)
            pts_reshaped = pts.reshape(h, w, 3)
            depth[msk] = pts_reshaped[msk, 2]  # z-component
            depth_predictions.append(depth)

        # Process images to match original format
        processed_images_list = []
        for img in rgb_imgs:
            # Convert from normalized [-1, 1] to [0, 255]
            img_processed = (img * 255.0).astype(np.uint8)  # ((img + 1) / 2 * 255).astype(np.uint8)
            # Convert RGB to BGR for consistency
            # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
            processed_images_list.append(img_processed)

        # Create per-view point clouds (in world frame)
        point_clouds = []
        for i, (pts, msk, img) in enumerate(zip(pts3d, mask, rgb_imgs)):
            h, w = rgb_imgs[i].shape[:2]
            pts_reshaped = pts.reshape(h, w, 3)
            pts_valid = pts_reshaped[msk].reshape(-1, 3)
            img_valid = img[msk].reshape(-1, 3)
            if len(pts_valid) > 0:
                pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                point_clouds.append(pc)
            else:
                point_clouds.append(None)

        # Build intrinsics list
        intrinsics_list = []
        for focal in focals:
            K = np.eye(3)
            K[0, 0] = focal
            K[1, 1] = focal
            h, w = rgb_imgs[0].shape[:2]
            K[0, 2] = w / 2
            K[1, 2] = h / 2
            intrinsics_list.append(K)

        return SceneFromViewsResult(
            global_point_cloud=global_pc,
            global_mesh=global_mesh,
            camera_poses=[cams2world[i] for i in range(len(cams2world))],
            processed_images=processed_images_list,
            depth_predictions=depth_predictions,
            point_clouds=point_clouds,
            intrinsics=intrinsics_list,
            confidences=confs,
        )

    def create_optimizer_input(
        self,
        raw_output,
        pairs: List,
        processed_images: List,
        **kwargs,
    ):
        """
        Create SceneOptimizerInput from MASt3r model output.

        Args:
            raw_output: Output from forward_mast3r() (pairs_output)
            pairs: List of image pairs
            processed_images: Preprocessed images
            **kwargs: Additional parameters:
                - filelist: List of image file names (required)
                - cache_dir: Cache directory (required)

        Returns:
            SceneOptimizerInput: Unified input representation
        """
        from pyslam.scene_from_views.optimizers import SceneOptimizerInput, PairPrediction
        import torch

        # Extract required parameters
        filelist = kwargs.get("filelist")
        cache_dir = kwargs.get("cache_dir")

        if filelist is None:
            raise ValueError(
                "filelist is required for create_optimizer_input() in SceneFromViewsMast3r"
            )
        if cache_dir is None:
            raise ValueError(
                "cache_dir is required for create_optimizer_input() in SceneFromViewsMast3r"
            )

        # Extract images from processed_images
        images = []
        for img_data in processed_images:
            if isinstance(img_data, dict) and "img" in img_data:
                images.append(img_data["img"])
            elif isinstance(img_data, torch.Tensor):
                images.append(img_data)

        # Convert pairs_output to pair_predictions format
        # pairs_output is a dict: {(img1, img2): ((path1, path2), path_corres)}
        # We need to load the cached data to extract pts3d and conf
        pair_predictions = []

        # Create mapping from image names to indices
        img_to_idx = {img: idx for idx, img in enumerate(filelist)}

        # Extract pair predictions from pairs_output
        if raw_output:  # pairs_output from forward_mast3r
            for (img1, img2), ((path1, path2), path_corres) in raw_output.items():
                try:
                    # Load cached data: (X, C, X2, C2) where:
                    # X: pts3d for view1, C: conf for view1
                    # X2: pts3d for view2 (in view1's frame), C2: conf for view2
                    # Convert device to torch.device if it's a string
                    device = (
                        self.device
                        if isinstance(self.device, torch.device)
                        else torch.device(self.device)
                    )
                    X1, C1, X2, C2 = torch.load(path1, map_location=device)

                    # Ensure all tensors are on the correct device
                    X1 = X1.to(device) if isinstance(X1, torch.Tensor) else X1
                    C1 = C1.to(device) if isinstance(C1, torch.Tensor) else C1
                    X2 = X2.to(device) if isinstance(X2, torch.Tensor) else X2
                    C2 = C2.to(device) if isinstance(C2, torch.Tensor) else C2

                    # Get image indices
                    idx_i = img_to_idx.get(img1)
                    idx_j = img_to_idx.get(img2)

                    if idx_i is not None and idx_j is not None:
                        # Get images and ensure they're on the correct device
                        img_i = images[idx_i] if idx_i < len(images) else None
                        img_j = images[idx_j] if idx_j < len(images) else None
                        if img_i is not None and isinstance(img_i, torch.Tensor):
                            img_i = img_i.to(device)
                        if img_j is not None and isinstance(img_j, torch.Tensor):
                            img_j = img_j.to(device)

                        # Create PairPrediction
                        pair_pred = PairPrediction(
                            image_idx_i=idx_i,
                            image_idx_j=idx_j,
                            pts3d_i=X1,  # [H, W, 3] 3D points for image i
                            pts3d_j=X2,  # [H, W, 3] 3D points for image j (in image i's frame)
                            conf_i=C1,  # [H, W] confidence for image i
                            conf_j=C2,  # [H, W] confidence for image j
                            image_i=img_i,
                            image_j=img_j,
                        )
                        pair_predictions.append(pair_pred)
                except Exception as e:
                    # Skip pairs that can't be loaded (shouldn't happen, but be safe)
                    verbose = kwargs.get("verbose", False)
                    if verbose:
                        print(f"Warning: Could not load pair ({img1}, {img2}): {e}")
                    continue

        return SceneOptimizerInput(
            images=images,
            pairs=pairs,
            filelist=filelist,
            cache_dir=cache_dir,
            pair_predictions=pair_predictions,
            pairs_output=raw_output,  # pairs_output for canonical view processing
        )
