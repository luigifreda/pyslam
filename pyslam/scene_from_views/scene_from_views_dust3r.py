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
        niter=300,
        schedule="linear",
        lr=0.01,
        batch_size=1,
        verbose=True,
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
            niter: Number of iterations for global alignment
            schedule: Learning rate schedule ('linear', 'cosine', etc.)
            lr: Learning rate for global alignment
            batch_size: Batch size for inference
            verbose: If True, print progress messages
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
        self.niter = niter
        self.schedule = schedule
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose

        self.dust3r_preprocessor = Dust3rImagePreprocessor(
            inference_size=inference_size, verbose=verbose
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
        niter = kwargs.get("niter", self.niter)
        schedule = kwargs.get("schedule", self.schedule)
        lr = kwargs.get("lr", self.lr)
        batch_size = kwargs.get("batch_size", self.batch_size)
        verbose = kwargs.get("verbose", self.verbose)

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
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

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
        if verbose:
            print(f"[3/6] Running inference on {len(pairs)} pair(s) (batch_size={batch_size})...")
        output = inference(pairs, self.model, self.device, batch_size=batch_size, verbose=verbose)

        # Use global aligner to align all views
        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(imgs_preproc) > 2
            else GlobalAlignerMode.PairViewer
        )
        if verbose:
            print(f"[4/6] Initializing global aligner (mode: {mode.name})...")
        scene = global_aligner(output, device=self.device, mode=mode, verbose=verbose)

        # Run global alignment optimization
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            if verbose:
                print(
                    f"[5/6] Running global alignment optimization ({niter} iterations, lr={lr}, schedule={schedule})..."
                )
            scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        else:
            if verbose:
                print("[5/6] Skipping global alignment (pair viewer mode)...")

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
        **kwargs
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
        pts3d = scene.get_pts3d()  # List of [h, w, 3] point clouds
        rgb_imgs = scene.imgs  # List of [h, w, 3] RGB images
        cams2world = scene.get_im_poses()  # [n, 4, 4] camera poses
        focals = scene.get_focals()  # [n] focal lengths
        confs = scene.im_conf  # List of [h, w] confidence maps

        if verbose:
            print(f"   Extracted {len(pts3d)} view(s)")

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
