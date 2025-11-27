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

config.cfg.set_lib("mast3r")  # Dust3r is accessed through mast3r

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SceneFromViewsDust3r(SceneFromViewsBase):
    """
    Scene reconstruction using DUSt3R (Geometric 3D Vision Made Easy).

    This model performs multi-view reconstruction with pose estimation.
    Similar to MASt3r but uses the original Dust3r model.
    """

    def __init__(
        self,
        device=None,
        model_path=None,
        inference_size=512,
        min_conf_thr=20.0,
        scenegraph_type="complete",
        winsize=1,
        win_cyclic=False,
        refid=0,
        batch_size=1,
        verbose=True,
        optimizer_config=None,
        **kwargs,
    ):
        """
        Initialize Dust3r model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_path: Path to model checkpoint (None to use default)
            inference_size: Image size for inference (224 or 512)
            min_conf_thr: Minimum confidence threshold (percentage of max confidence)
            scenegraph_type: Scene graph type ('complete', 'swin', 'logwin', 'oneref')
            winsize: Window size for scene graph
            win_cyclic: Whether to use cyclic windows
            refid: Reference image ID for 'oneref' scenegraph
            batch_size: Batch size for inference
            verbose: If True, print progress messages
            optimizer_config: Dictionary specifying optimizer configuration:
                - type: 'dense_scene_optimizer' or 'sparse_scene_optimizer'
                - Additional parameters specific to the optimizer type
                If None, defaults to 'dense_scene_optimizer' with default parameters.
                Optimizer parameters (niter, schedule, lr) can be provided via kwargs.
            **kwargs: Additional parameters including optimizer parameters:
                - niter: Number of iterations for global alignment
                - schedule: Learning rate schedule ('linear', 'cosine', etc.)
                - lr: Learning rate for global alignment
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        import mast3r.utils.path_to_dust3r  # Setup path
        from dust3r.model import AsymmetricCroCo3DStereo

        if model_path is None:
            # Try to use default model
            try:
                self.model = AsymmetricCroCo3DStereo.from_pretrained(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
                ).to(self.device)
            except:
                model_path = (
                    kRootFolder
                    + "/thirdparty/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
                )
                self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(self.device)
        else:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(self.device)

        self.inference_size = inference_size
        self.min_conf_thr = min_conf_thr
        self.scenegraph_type = scenegraph_type
        self.winsize = winsize
        self.win_cyclic = win_cyclic
        self.refid = refid
        self.batch_size = batch_size
        self.verbose = verbose

        # Setup optimizer configuration
        # Extract optimizer parameters from kwargs
        optimizer_params = {}
        # Dense optimizer parameters
        for param in ["niter", "schedule", "lr"]:
            if param in kwargs:
                optimizer_params[param] = kwargs.pop(param)
        # Sparse optimizer parameters
        for param in [
            "type",
            "subsample",
            "lr1",
            "niter1",
            "lr2",
            "niter2",
            "matching_conf_thr",
            "shared_intrinsics",
            "optim_level",
            "kinematic_mode",
        ]:
            if param in kwargs:
                optimizer_params[param] = kwargs.pop(param)

        if optimizer_config is None:
            # Determine default optimizer type
            optimizer_type = optimizer_params.get("type", "dense_scene_optimizer")

            from pyslam.scene_from_views.optimizers import (
                DEFAULT_DENSE_OPTIMIZER_CONFIG,
                DEFAULT_SPARSE_OPTIMIZER_CONFIG,
            )

            if optimizer_type == "sparse_scene_optimizer":
                optimizer_config = DEFAULT_SPARSE_OPTIMIZER_CONFIG.copy()
            else:
                optimizer_config = DEFAULT_DENSE_OPTIMIZER_CONFIG.copy()

            # Override with user-provided parameters from kwargs
            optimizer_config.update(optimizer_params)
        else:
            # Merge optimizer_params into provided config
            optimizer_config = optimizer_config.copy()
            optimizer_config.update(optimizer_params)
            # Ensure type is set
            if "type" not in optimizer_config:
                optimizer_config["type"] = optimizer_params.get("type", "dense_scene_optimizer")
        self.optimizer_config = optimizer_config

        # Initialize optimizer based on config
        self._init_optimizer()

        self.dust3r_preprocessor = Dust3rImagePreprocessor(
            inference_size=inference_size, verbose=verbose
        )

    def _init_optimizer(self):
        """Initialize the optimizer based on optimizer_config."""
        from pyslam.scene_from_views.optimizers import (
            scene_optimizer_factory,
            SceneOptimizerType,
        )

        # Determine default type based on optimizer_config
        optimizer_type_str = self.optimizer_config.get("type", "dense_scene_optimizer")
        if optimizer_type_str == "sparse_scene_optimizer":
            default_type = SceneOptimizerType.SPARSE
        else:
            default_type = SceneOptimizerType.DENSE

        self.optimizer = scene_optimizer_factory(
            optimizer_config=self.optimizer_config,
            device=self.device,
            default_type=default_type,
        )

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List:
        """
        Preprocess images for Dust3r.

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (not used for this model)
        """
        return self.dust3r_preprocessor.preprocess_images(images)

    def infer(self, processed_images: List, **kwargs):
        """
        Run inference on preprocessed images using Dust3r.

        Args:
            processed_images: Preprocessed images from preprocess_images()
            **kwargs: Additional inference parameters (can override init parameters)

        Returns:
            Dictionary containing raw inference output and scene object
        """
        # Override parameters with kwargs if provided
        min_conf_thr = kwargs.get("min_conf_thr", self.min_conf_thr)
        scenegraph_type = kwargs.get("scenegraph_type", self.scenegraph_type)
        winsize = kwargs.get("winsize", self.winsize)
        win_cyclic = kwargs.get("win_cyclic", self.win_cyclic)
        refid = kwargs.get("refid", self.refid)
        batch_size = kwargs.get("batch_size", self.batch_size)
        verbose = kwargs.get("verbose", self.verbose)

        # Extract optimizer parameters from kwargs or use defaults from optimizer_config
        optimizer_type = self.optimizer_config.get("type", "dense_scene_optimizer")

        # Dense optimizer parameters
        niter = kwargs.pop("niter", self.optimizer_config.get("niter", 300))
        schedule = kwargs.pop("schedule", self.optimizer_config.get("schedule", "linear"))
        lr = kwargs.pop("lr", self.optimizer_config.get("lr", 0.01))

        # Sparse optimizer parameters (extract but only use if sparse optimizer)
        subsample = kwargs.pop("subsample", self.optimizer_config.get("subsample", 8))
        lr1 = kwargs.pop("lr1", self.optimizer_config.get("lr1", 0.08))
        niter1 = kwargs.pop("niter1", self.optimizer_config.get("niter1", 600))
        lr2 = kwargs.pop("lr2", self.optimizer_config.get("lr2", 0.02))
        niter2 = kwargs.pop("niter2", self.optimizer_config.get("niter2", 300))
        matching_conf_thr = kwargs.pop(
            "matching_conf_thr", self.optimizer_config.get("matching_conf_thr", 3.0)
        )
        shared_intrinsics = kwargs.pop(
            "shared_intrinsics", self.optimizer_config.get("shared_intrinsics", False)
        )
        optim_level = kwargs.pop(
            "optim_level", self.optimizer_config.get("optim_level", "refine+depth")
        )
        kinematic_mode = kwargs.pop(
            "kinematic_mode", self.optimizer_config.get("kinematic_mode", "hclust-ward")
        )

        if verbose:
            print("=" * 60)
            print("Dust3r Scene Reconstruction")
            print("=" * 60)
            print(f"Processing {len(processed_images)} image(s)...")

        # Handle single image case
        imgs_preproc = processed_images
        if len(imgs_preproc) == 1:
            if verbose:
                print("   Single image detected, duplicating for pair processing...")
            imgs_preproc = [imgs_preproc[0], copy.deepcopy(imgs_preproc[0])]
            imgs_preproc[1]["idx"] = 1

        # Build scene graph
        import mast3r.utils.path_to_dust3r  # Ensure path is set
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference

        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(refid))
        if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
            scene_graph_params.append("noncyclic")
        scene_graph = "-".join(scene_graph_params)

        # Create pairs from all images
        if verbose:
            print(f"[2/6] Creating image pairs (scene graph: {scene_graph})...")
        pairs = make_pairs(imgs_preproc, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        if verbose:
            print(f"   Created {len(pairs)} image pair(s)")

        # Run inference on all pairs
        # This runs the forward pass of the model on all pairs.
        # The output is a dictionary containing the 3D points and confidence maps for each pair.
        if verbose:
            print(f"[3/6] Running inference on {len(pairs)} pair(s) (batch_size={batch_size})...")
        output = inference(pairs, self.model, self.device, batch_size=batch_size, verbose=verbose)

        # Check if we should skip optimization and use initial poses
        skip_optimizer = kwargs.pop("skip_optimizer", False)

        # Import torch here to avoid linter warnings (it's already imported at top)
        import torch

        if skip_optimizer:
            # Skip optimization and use initial poses from Dust3r inference
            if verbose:
                print("[4/6] Skipping optimizer - using initial poses from Dust3r inference...")

            # Use global_aligner to create scene with initial poses (MST initialization)
            from .optimizers.dense_scene_optimizer import global_aligner, GlobalAlignerMode
            from .optimizers.scene_optimizer_io import SceneOptimizerInput, SceneOptimizerOutput

            # Create filelist for optimizer input
            filelist = [f"image_{i}" for i in range(len(imgs_preproc))]

            # Create unified optimizer input (needed for global_aligner)
            optimizer_input = self.create_optimizer_input(
                raw_output=output,
                pairs=pairs,
                processed_images=imgs_preproc,
                filelist=filelist,
                cache_dir=None,  # Not needed for initial poses
                subsample=subsample,
                verbose=verbose,
            )

            # Create global aligner scene with initial poses (MST initialization)
            device = (
                self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            )
            scene = global_aligner(
                {
                    "view1": output["view1"],
                    "view2": output["view2"],
                    "pred1": output["pred1"],
                    "pred2": output["pred2"],
                },
                device=device,
                mode=GlobalAlignerMode.POINT_CLOUD_OPTIMIZER,
                verbose=verbose,
            )

            # Initialize with MST but don't optimize (niter=0)
            if verbose:
                print("   Initializing poses with MST (no optimization)...")
            scene.compute_global_alignment(init="mst", niter=0, schedule=schedule, lr=lr)

            # Create optimizer_output-like object for compatibility
            optimizer_output = SceneOptimizerOutput(
                scene=scene,
                optimizer_type=optimizer_type,
            )
        else:
            # Use optimizer to align all views (using unified interface)
            if verbose:
                print(f"[4/6] Using {optimizer_type}...")

            # Create cache directory if using sparse optimizer
            cache_dir = None
            if optimizer_type == "sparse_scene_optimizer":
                import tempfile
                import os

                cache_dir = tempfile.mkdtemp(suffix="_dust3r_cache")
                os.makedirs(cache_dir, exist_ok=True)
                if verbose:
                    print(f"   Created cache directory: {cache_dir}")

            # Create filelist for optimizer input
            filelist = [f"image_{i}" for i in range(len(imgs_preproc))]

            # Create unified optimizer input
            optimizer_input = self.create_optimizer_input(
                raw_output=output,
                pairs=pairs,
                processed_images=imgs_preproc,
                filelist=filelist,
                cache_dir=cache_dir,
                subsample=subsample,
                verbose=verbose,
            )

            # Optimize using unified interface
            if optimizer_type == "sparse_scene_optimizer":
                # Debug option: skip fine optimization to test if coarse stage works correctly
                skip_fine = kwargs.pop("skip_fine", False)
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
                    skip_fine=skip_fine,  # Debug option to skip fine optimization
                )
            else:
                optimizer_output = self.optimizer.optimize(
                    optimizer_input=optimizer_input,
                    verbose=verbose,
                    niter=niter,
                    schedule=schedule,
                    lr=lr,
                )

            scene = optimizer_output.scene

        return {
            "scene": scene,
            "min_conf_thr": min_conf_thr,
            "verbose": verbose,
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
        Postprocess raw Dust3r output into SceneFromViewsResult.

        Args:
            raw_output: Dictionary containing scene object and parameters from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        scene = raw_output["scene"]
        min_conf_thr = raw_output["min_conf_thr"]
        verbose = raw_output["verbose"]

        # Extract results from scene
        if verbose:
            print("[6/6] Extracting results and post-processing...")

        # Handle different scene types (DenseGA vs SparseGA)
        # SparseGA has canonical_paths attribute, DenseGA does not
        # This is the most reliable way to distinguish them
        is_sparse = hasattr(scene, "canonical_paths") and scene.canonical_paths is not None

        if is_sparse:
            # SparseGA: use get_dense_pts3d() to get dense point clouds
            pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
            rgb_imgs = scene.imgs  # List of [h, w, 3] RGB images (numpy arrays)
            # Ensure rgb_imgs are numpy arrays
            if rgb_imgs and isinstance(rgb_imgs[0], torch.Tensor):
                rgb_imgs = [to_numpy(img) for img in rgb_imgs]

            # Debug: check camera poses and point clouds
            if verbose:
                from pyslam.utilities.system import Printer

                # Check camera poses
                cam2w = scene.cam2w
                Printer.blue("\n[Debug] Camera poses after optimization:")
                for i in range(len(cam2w)):
                    trans = cam2w[i][:3, 3].cpu().numpy()
                    rot_first_row = cam2w[i][0, :3].cpu().numpy()
                    Printer.blue(f"  Camera {i}: translation={trans}, rotation[0]={rot_first_row}")

                # Check if camera poses are different
                if len(cam2w) > 1:
                    for i in range(len(cam2w)):
                        for j in range(i + 1, len(cam2w)):
                            trans_diff = np.linalg.norm(
                                cam2w[i][:3, 3].cpu().numpy() - cam2w[j][:3, 3].cpu().numpy()
                            )
                            rot_diff = np.linalg.norm(
                                cam2w[i][:3, :3].cpu().numpy() - cam2w[j][:3, :3].cpu().numpy()
                            )
                            if trans_diff < 1e-6 and rot_diff < 1e-6:
                                Printer.red(f"  WARNING: Cameras {i} and {j} have identical poses!")
                            else:
                                Printer.blue(
                                    f"  Cameras {i} and {j}: trans_diff={trans_diff:.6f}, rot_diff={rot_diff:.6f}"
                                )

                # Check point clouds
                Printer.blue("\n[Debug] Point cloud statistics:")
                for i in range(len(pts3d)):
                    if isinstance(pts3d[i], torch.Tensor):
                        pts3d_np = (
                            pts3d[i].detach().cpu().numpy()
                            if hasattr(pts3d[i], "detach")
                            else pts3d[i].cpu().numpy()
                        )
                    else:
                        pts3d_np = pts3d[i]
                    pts3d_flat = pts3d_np.reshape(-1, 3)
                    mean_pt = pts3d_flat.mean(axis=0)
                    std_pt = pts3d_flat.std(axis=0)
                    min_pt = pts3d_flat.min(axis=0)
                    max_pt = pts3d_flat.max(axis=0)
                    Printer.blue(
                        f"  Camera {i}: mean={mean_pt}, std={std_pt}, min={min_pt}, max={max_pt}, shape={pts3d_np.shape}"
                    )

                # Check camera-to-point-cloud distances (should be reasonable)
                Printer.blue("\n[Debug] Camera-to-point-cloud distances:")
                for i in range(len(cam2w)):
                    cam_pos = cam2w[i][:3, 3].cpu().numpy()
                    if isinstance(pts3d[i], torch.Tensor):
                        pts3d_np = (
                            pts3d[i].detach().cpu().numpy()
                            if hasattr(pts3d[i], "detach")
                            else pts3d[i].cpu().numpy()
                        )
                    else:
                        pts3d_np = pts3d[i]
                    pts3d_flat = pts3d_np.reshape(-1, 3)
                    # Compute distance from camera to point cloud center
                    pc_center = pts3d_flat.mean(axis=0)
                    dist = np.linalg.norm(cam_pos - pc_center)
                    Printer.blue(
                        f"  Camera {i}: pos={cam_pos}, pc_center={pc_center}, dist={dist:.6f}"
                    )

                # Check if point clouds are different
                if len(pts3d) > 1:
                    for i in range(len(pts3d)):
                        for j in range(i + 1, len(pts3d)):
                            if isinstance(pts3d[i], torch.Tensor):
                                pts3d_i = (
                                    pts3d[i].detach().cpu().numpy()
                                    if hasattr(pts3d[i], "detach")
                                    else pts3d[i].cpu().numpy()
                                )
                            else:
                                pts3d_i = pts3d[i]
                            if isinstance(pts3d[j], torch.Tensor):
                                pts3d_j = (
                                    pts3d[j].detach().cpu().numpy()
                                    if hasattr(pts3d[j], "detach")
                                    else pts3d[j].cpu().numpy()
                                )
                            else:
                                pts3d_j = pts3d[j]

                            if pts3d_i.shape == pts3d_j.shape:
                                diff = np.abs(pts3d_i - pts3d_j).mean()
                                if diff < 1e-6:
                                    Printer.red(
                                        f"  WARNING: Point clouds {i} and {j} are identical (diff={diff:.2e})!"
                                    )
                                else:
                                    Printer.blue(f"  Point clouds {i} and {j}: diff={diff:.6f}")

            # Debug: verify images are different
            if verbose and len(rgb_imgs) > 1:
                from pyslam.utilities.system import Printer

                for i in range(len(rgb_imgs)):
                    for j in range(i + 1, len(rgb_imgs)):
                        if rgb_imgs[i].shape == rgb_imgs[j].shape:
                            diff = np.abs(rgb_imgs[i] - rgb_imgs[j]).mean()
                            if diff < 1e-6:
                                Printer.red(
                                    f"WARNING: Images {i} and {j} appear to be identical (diff={diff:.2e})"
                                )
                            else:
                                Printer.blue(f"Images {i} and {j} are different (diff={diff:.4f})")
        else:
            # DenseGA: use get_pts3d() method (original behavior)
            pts3d = scene.get_pts3d()  # List of [h, w, 3] point clouds
            rgb_imgs = scene.imgs  # List of [h, w, 3] RGB images
            confs = scene.im_conf  # List of [h, w] confidence maps

        cams2world = scene.get_im_poses()  # [n, 4, 4] camera poses
        focals = scene.get_focals()  # [n] focal lengths

        if verbose:
            print(f"   Extracted {len(pts3d)} view(s)")
            # Debug: print camera poses
            print(f"   Camera poses shape: {cams2world.shape}")
            for i, pose in enumerate(cams2world):
                print(f"   Camera {i} pose translation: {pose[:3, 3]}")
                print(f"   Camera {i} pose rotation (first row): {pose[0, :3]}")
            # Debug: print confidence maps info
            if confs:
                print(f"   Confidence maps: {len(confs)} maps")
                for i, conf in enumerate(confs):
                    print(
                        f"   Conf map {i} shape: {conf.shape}, mean: {conf.mean():.4f}, std: {conf.std():.4f}"
                    )

        # Create confidence masks (work with torch tensors first)
        if verbose:
            print(f"   Creating confidence masks (threshold: {min_conf_thr}% of max confidence)...")
        conf_vec = torch.stack([x.reshape(-1) for x in confs])
        conf_sorted = conf_vec.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        mask = [x >= conf_thres for x in confs]

        # Convert to numpy
        rgb_imgs = to_numpy(rgb_imgs)
        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)
        pts3d = to_numpy(pts3d)
        confs = [to_numpy(x) for x in confs]
        mask = [to_numpy(x) for x in mask]

        # Convert to geometry
        if verbose:
            print(f"   Converting to {'point cloud' if as_pointcloud else 'mesh'}...")
        global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, mask, as_pointcloud)

        # Extract depth maps from point clouds (use z-component)
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
            img_processed = (img * 255.0).astype(np.uint8)
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
        for i, focal in enumerate(focals):
            K = np.eye(3)
            K[0, 0] = focal
            K[1, 1] = focal
            h, w = rgb_imgs[i].shape[:2]
            K[0, 2] = w / 2
            K[1, 2] = h / 2
            intrinsics_list.append(K)

        if verbose:
            print("   Reconstruction complete!")
            print("=" * 60)

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
        subsample: int = 8,
        **kwargs,
    ):
        """
        Create SceneOptimizerInput from Dust3r model output.

        Args:
            raw_output: Output from dust3r.inference() containing view1, view2, pred1, pred2
            pairs: List of image pairs
            processed_images: Preprocessed images
            **kwargs: Additional parameters:
                - filelist: Optional list of image file names
                - cache_dir: Optional cache directory (required for sparse optimizer)

        Returns:
            SceneOptimizerInput: Unified input representation with extracted fields
        """
        from pyslam.scene_from_views.optimizers import SceneOptimizerInput, PairPrediction
        import torch
        import os

        # Extract images from processed_images
        images = []
        if processed_images:
            for img_data in processed_images:
                if isinstance(img_data, dict) and "img" in img_data:
                    images.append(img_data["img"])
                elif isinstance(img_data, torch.Tensor):
                    images.append(img_data)

        # Create filelist if not provided
        filelist = kwargs.get("filelist", None)
        if filelist is None:
            filelist = [f"image_{i}" for i in range(len(images))]

        cache_dir = kwargs.get("cache_dir", None)
        verbose = kwargs.get("verbose", False)

        # Extract Dust3r-specific fields from output
        view1 = raw_output.get("view1")
        view2 = raw_output.get("view2")
        pred1 = raw_output.get("pred1")
        pred2 = raw_output.get("pred2")

        # Convert to unified format: create PairPrediction for each pair
        pair_predictions = []
        pairs_output = None  # For sparse optimizer

        if view1 and view2 and pred1 and pred2:
            view1_idx = view1.get("idx", [])
            view2_idx = view2.get("idx", [])
            view1_imgs = view1.get("img", [])
            view2_imgs = view2.get("img", [])
            pred1_pts3d = pred1.get("pts3d", [])
            pred2_pts3d = pred2.get("pts3d_in_other_view", [])
            pred1_conf = pred1.get("conf", [])
            pred2_conf = pred2.get("conf", [])
            pred1_desc = pred1.get("desc", [])
            pred2_desc = pred2.get("desc", [])

            # If cache_dir is provided (for sparse optimizer), cache the pair data
            if cache_dir is not None:
                pairs_output = {}
                # Create forward subdirectory structure like Mast3r
                forward_dir = os.path.join(cache_dir, "forward")
                corres_dir = os.path.join(cache_dir, "corres")
                os.makedirs(forward_dir, exist_ok=True)
                os.makedirs(corres_dir, exist_ok=True)

                # Create a mapping from (img1_idx, img2_idx) to pair index to find reverse pairs
                pair_map = {}
                for i in range(len(view1_idx)):
                    img1_idx = int(view1_idx[i])
                    img2_idx = int(view2_idx[i])
                    pair_map[(img1_idx, img2_idx)] = i

                # Debug: check if reverse pairs exist
                if verbose:
                    print(f"  [Debug] Total pairs: {len(view1_idx)}")
                    reverse_count = 0
                    for i in range(len(view1_idx)):
                        img1_idx = int(view1_idx[i])
                        img2_idx = int(view2_idx[i])
                        if (img2_idx, img1_idx) in pair_map:
                            reverse_count += 1
                    print(f"  [Debug] Reverse pairs found: {reverse_count}/{len(view1_idx)}")

                for i in range(len(view1_idx)):
                    img1_idx = int(view1_idx[i])
                    img2_idx = int(view2_idx[i])
                    img1_name = filelist[img1_idx]
                    img2_name = filelist[img2_idx]

                    # Cache the pair data in the format expected by sparse optimizer
                    # Format: (X1, C1, X2, C2) where:
                    # X1: pts3d for view1, C1: conf for view1
                    # X2: pts3d for view2 (in view1's frame), C2: conf for view2
                    pair_key = (img1_name, img2_name)

                    # Create separate cache files for (img1, img2) and (img2, img1) directions
                    # This matches Mast3r's format: path1 = forward/{idx1}/{idx2}.pth, path2 = forward/{idx2}/{idx1}.pth
                    import hashlib

                    idx1 = hashlib.md5(img1_name.encode()).hexdigest()
                    idx2 = hashlib.md5(img2_name.encode()).hexdigest()

                    # Path for (img1, img2): stores (X11, C11, X21, C21)
                    path1_dir = os.path.join(forward_dir, idx1)
                    os.makedirs(path1_dir, exist_ok=True)
                    path1 = os.path.join(path1_dir, f"{idx2}.pth")

                    # Path for (img2, img1): stores (X22, C22, X12, C12)
                    path2_dir = os.path.join(forward_dir, idx2)
                    os.makedirs(path2_dir, exist_ok=True)
                    path2 = os.path.join(path2_dir, f"{idx1}.pth")

                    # Ensure tensors are on CPU for caching
                    X1 = (
                        pred1_pts3d[i].cpu()
                        if isinstance(pred1_pts3d[i], torch.Tensor)
                        else pred1_pts3d[i]
                    )
                    C1 = (
                        pred1_conf[i].cpu()
                        if isinstance(pred1_conf[i], torch.Tensor)
                        else pred1_conf[i]
                    )
                    X2 = (
                        pred2_pts3d[i].cpu()
                        if isinstance(pred2_pts3d[i], torch.Tensor)
                        else pred2_pts3d[i]
                    )
                    C2 = (
                        pred2_conf[i].cpu()
                        if isinstance(pred2_conf[i], torch.Tensor)
                        else pred2_conf[i]
                    )

                    # Save (img1, img2) direction: (X11, C11, X21, C21)
                    # X11 = X1 (pts3d for img1 in img1's frame), C11 = C1 (conf for img1)
                    # X21 = X2 (pts3d for img2 in img1's frame), C21 = C2 (conf for img2)
                    torch.save((X1, C1, X2, C2), path1)

                    # For (img2, img1) direction, check if we have the reverse pair in the output
                    # If symmetrize=True, pairs should include both directions
                    reverse_pair_idx = pair_map.get((img2_idx, img1_idx))
                    if reverse_pair_idx is not None:
                        # We have the reverse pair - use its predictions
                        # For reverse pair (img2, img1):
                        # - pred1_pts3d[reverse_pair_idx] = pts3d for img2 in img2's frame (X22)
                        # - pred2_pts3d[reverse_pair_idx] = pts3d for img1 in img2's frame (X12)
                        X22 = (
                            pred1_pts3d[reverse_pair_idx].cpu()
                            if isinstance(pred1_pts3d[reverse_pair_idx], torch.Tensor)
                            else pred1_pts3d[reverse_pair_idx]
                        )
                        C22 = (
                            pred1_conf[reverse_pair_idx].cpu()
                            if isinstance(pred1_conf[reverse_pair_idx], torch.Tensor)
                            else pred1_conf[reverse_pair_idx]
                        )
                        X12 = (
                            pred2_pts3d[reverse_pair_idx].cpu()
                            if isinstance(pred2_pts3d[reverse_pair_idx], torch.Tensor)
                            else pred2_pts3d[reverse_pair_idx]
                        )
                        C12 = (
                            pred2_conf[reverse_pair_idx].cpu()
                            if isinstance(pred2_conf[reverse_pair_idx], torch.Tensor)
                            else pred2_conf[reverse_pair_idx]
                        )
                        # Save (img2, img1) direction: (X22, C22, X12, C12)
                        torch.save((X22, C22, X12, C12), path2)
                    else:
                        # No reverse pair found - this shouldn't happen with symmetrize=True
                        # But if it does, we can't create proper reverse pair data because coordinate frames differ
                        # For now, set path2 to None - the optimizer should handle missing pairs gracefully
                        # TODO: Investigate why symmetrize=True doesn't always create reverse pairs
                        if verbose:
                            print(
                                f"  Warning: No reverse pair found for ({img2_idx}, {img1_idx}), "
                                f"using None for path2. This may cause issues in sparse optimizer."
                            )
                        path2 = None

                    # Create correspondence file from predictions.
                    # Priority: 1) descriptor matching (if available), 2) 3D NN matching, 3) dense grid fallback.
                    corres_path = os.path.join(corres_dir, f"{idx1}-{idx2}.pth")
                    # Prepare confidence maps for correspondence scoring
                    conf1_corr = C1.squeeze(0) if hasattr(C1, "dim") and C1.dim() == 3 else C1
                    conf2_corr = C2.squeeze(0) if hasattr(C2, "dim") and C2.dim() == 3 else C2

                    try:
                        from mast3r.fast_nn import (
                            extract_correspondences_nonsym,
                            fast_reciprocal_NNs,
                        )

                        desc1 = pred1_desc[i] if i < len(pred1_desc) else None
                        desc2 = pred2_desc[i] if i < len(pred2_desc) else None

                        if desc1 is not None and desc2 is not None:
                            # Remove batch dimension if present
                            if isinstance(desc1, torch.Tensor) and desc1.dim() == 4:
                                desc1 = desc1.squeeze(0)
                            if isinstance(desc2, torch.Tensor) and desc2.dim() == 4:
                                desc2 = desc2.squeeze(0)

                            xy1, xy2, confs_flat = extract_correspondences_nonsym(
                                desc1,
                                desc2,
                                conf1_corr,
                                conf2_corr,
                                subsample=subsample,
                                device="cpu",
                            )
                        else:
                            raise ValueError(
                                "Descriptors not available for correspondence extraction"
                            )
                    except Exception as e:
                        # Second attempt: use 3D nearest-neighbor correspondences (works even without descriptors)
                        try:
                            xy1_idx, xy2_idx = fast_reciprocal_NNs(
                                X1, X2, subsample_or_initxy1=subsample, ret_xy=True, device="cpu"
                            )
                            xy1 = torch.stack(
                                [
                                    torch.from_numpy(xy1_idx[:, 0]).float(),
                                    torch.from_numpy(xy1_idx[:, 1]).float(),
                                ],
                                dim=1,
                            )
                            xy2 = torch.stack(
                                [
                                    torch.from_numpy(xy2_idx[:, 0]).float(),
                                    torch.from_numpy(xy2_idx[:, 1]).float(),
                                ],
                                dim=1,
                            )
                            # Use minimum confidence as match weight
                            confs_flat = torch.minimum(
                                C1[xy1_idx[:, 1], xy1_idx[:, 0]].flatten(),
                                C2[xy2_idx[:, 1], xy2_idx[:, 0]].flatten(),
                            )
                        except Exception as e2:
                            if verbose:
                                print(
                                    f"  Warning: Falling back to dense-grid correspondences for pair "
                                    f"({img1_idx}, {img2_idx}): descriptor/3D matching failed ({e}); "
                                    f"3D NN failed ({e2})"
                                )
                            # Fallback: dense grid correspondences weighted by confidence
                            h1, w1 = X1.shape[:2]
                            h2, w2 = X2.shape[:2]
                            y1_coords, x1_coords = torch.meshgrid(
                                torch.arange(h1, dtype=torch.long),
                                torch.arange(w1, dtype=torch.long),
                                indexing="ij",
                            )
                            y2_coords, x2_coords = torch.meshgrid(
                                torch.arange(h2, dtype=torch.long),
                                torch.arange(w2, dtype=torch.long),
                                indexing="ij",
                            )
                            xy1 = torch.stack(
                                [x1_coords.flatten().float(), y1_coords.flatten().float()], dim=1
                            )
                            xy2 = torch.stack(
                                [x2_coords.flatten().float(), y2_coords.flatten().float()], dim=1
                            )
                            confs_flat = torch.minimum(C1.flatten(), C2.flatten())

                    # Score format: (conf_score, corres_sum, len_corres) as expected by sparse optimizer
                    confs_flat_tensor = (
                        confs_flat
                        if isinstance(confs_flat, torch.Tensor)
                        else torch.tensor(confs_flat)
                    )
                    conf1_corr_tensor = (
                        conf1_corr if torch.is_tensor(conf1_corr) else torch.tensor(conf1_corr)
                    )
                    conf2_corr_tensor = (
                        conf2_corr if torch.is_tensor(conf2_corr) else torch.tensor(conf2_corr)
                    )
                    conf_score = (
                        torch.sqrt(
                            conf1_corr_tensor.float().mean() * conf2_corr_tensor.float().mean()
                        ).item()
                        if confs_flat_tensor.numel() > 0
                        else 0.0
                    )
                    corres_sum = (
                        float(confs_flat_tensor.sum().item())
                        if confs_flat_tensor.numel() > 0
                        else 0.0
                    )
                    len_corres = int(confs_flat_tensor.numel())
                    score = (conf_score, corres_sum, len_corres)

                    # Save correspondences: (score, (xy1, xy2, confs))
                    torch.save((score, (xy1, xy2, confs_flat)), corres_path)

                    # Also create reverse correspondence file for (img2, img1)
                    corres_path2 = os.path.join(corres_dir, f"{idx2}-{idx1}.pth")
                    torch.save((score, (xy2, xy1, confs_flat)), corres_path2)

                    # pairs_output format: {(img1, img2): ((path1, path2), path_corres)}
                    # path1: (X11, C11, X21, C21) for (img1, img2)
                    # path2: (X22, C22, X12, C12) for (img2, img1) - may be None if reverse pair not available
                    # path_corres: correspondence file for (img1, img2) direction
                    # Note: prepare_canonical_data expects path_corres to work for both directions
                    # When img == img2, it uses the same path_corres but swaps xy1/xy2
                    if reverse_pair_idx is not None:
                        pairs_output[pair_key] = ((path1, path2), corres_path)
                    else:
                        # Only forward direction available - optimizer should handle this
                        # Use corres_path for forward direction, but prepare_canonical_data will need
                        # to handle the case where path2 is None
                        pairs_output[pair_key] = ((path1, None), corres_path)

            # Create PairPrediction objects
            for i in range(len(view1_idx)):
                pair_pred = PairPrediction(
                    image_idx_i=int(view1_idx[i]),
                    image_idx_j=int(view2_idx[i]),
                    pts3d_i=pred1_pts3d[i],
                    pts3d_j=pred2_pts3d[i],
                    conf_i=pred1_conf[i],
                    conf_j=pred2_conf[i],
                    image_i=view1_imgs[i] if i < len(view1_imgs) else None,
                    image_j=view2_imgs[i] if i < len(view2_imgs) else None,
                )
                pair_predictions.append(pair_pred)

        return SceneOptimizerInput(
            images=images,
            pairs=pairs,
            filelist=filelist,
            cache_dir=cache_dir,
            pair_predictions=pair_predictions,
            pairs_output=pairs_output,  # Required for sparse optimizer
        )
