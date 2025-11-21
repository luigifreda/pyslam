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
        optim_level="refine+depth",
        lr1=0.07,
        niter1=500,
        lr2=0.014,
        niter2=200,
        matching_conf_thr=5.0,
        shared_intrinsics=False,
        scenegraph_type="complete",
        winsize=1,
        win_cyclic=False,
        refid=0,
        TSDF_thresh=0.0,
        clean_depth=True,
        **kwargs,
    ):
        """
        Initialize MASt3R model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_path: Path to model checkpoint
            inference_size: Image size for inference (224 or 512)
            min_conf_thr: Minimum confidence threshold for filtering points
            optim_level: Optimization level ('coarse', 'refine', 'refine+depth')
            lr1: Coarse learning rate
            niter1: Number of iterations for coarse alignment
            lr2: Fine learning rate
            niter2: Number of iterations for refinement
            matching_conf_thr: Matching confidence threshold
            shared_intrinsics: Whether to use shared intrinsics
            scenegraph_type: Scene graph type ('complete', 'swin', 'logwin', 'oneref')
            winsize: Window size for scene graph
            win_cyclic: Whether to use cyclic windows
            refid: Reference image ID for 'oneref' scenegraph
            TSDF_thresh: TSDF threshold (0 to disable)
            clean_depth: Whether to clean up depth maps
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
        self.optim_level = optim_level
        self.lr1 = lr1
        self.niter1 = niter1
        self.lr2 = lr2
        self.niter2 = niter2
        self.matching_conf_thr = matching_conf_thr
        self.shared_intrinsics = shared_intrinsics
        self.scenegraph_type = scenegraph_type
        self.winsize = winsize
        self.win_cyclic = win_cyclic
        self.refid = refid
        self.TSDF_thresh = TSDF_thresh
        self.clean_depth = clean_depth

        self.dust3r_preprocessor = Dust3rImagePreprocessor(
            inference_size=inference_size, verbose=True
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
        optim_level = kwargs.get("optim_level", self.optim_level)
        lr1 = kwargs.get("lr1", self.lr1)
        niter1 = kwargs.get("niter1", self.niter1)
        lr2 = kwargs.get("lr2", self.lr2)
        niter2 = kwargs.get("niter2", self.niter2)
        matching_conf_thr = kwargs.get("matching_conf_thr", self.matching_conf_thr)
        shared_intrinsics = kwargs.get("shared_intrinsics", self.shared_intrinsics)
        scenegraph_type = kwargs.get("scenegraph_type", self.scenegraph_type)
        winsize = kwargs.get("winsize", self.winsize)
        win_cyclic = kwargs.get("win_cyclic", self.win_cyclic)
        refid = kwargs.get("refid", self.refid)
        TSDF_thresh = kwargs.get("TSDF_thresh", self.TSDF_thresh)
        clean_depth = kwargs.get("clean_depth", self.clean_depth)

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

        # Run sparse global alignment
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        import tempfile
        import os

        cache_dir = tempfile.mkdtemp(suffix="_mast3r_cache")
        os.makedirs(cache_dir, exist_ok=True)

        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            self.model,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            device=self.device,
            opt_depth="depth" in optim_level,
            shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr,
        )

        # Extract dense 3D map
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

        rgb_imgs = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()

        # Get dense point clouds
        if TSDF_thresh > 0:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
        else:
            pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

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

        mask = to_numpy([c > min_conf_thr for c in confs])

        # Convert to geometry
        global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, mask, as_pointcloud)

        # Convert to numpy
        rgb_imgs = to_numpy(rgb_imgs)
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
