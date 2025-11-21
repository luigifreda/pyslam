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

config.cfg.set_lib("mvdust3r")

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SceneFromViewsMvdust3r(SceneFromViewsBase):
    """
    Scene reconstruction using MVDust3r (Multi-view Dust3r).

    This model performs multi-view reconstruction with pose estimation.
    """

    def __init__(
        self,
        device=None,
        model_path=None,
        model_name="MVD",
        inference_size=224,
        min_conf_thr=5.0,
        **kwargs,
    ):
        """
        Initialize MVDust3r model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_path: Path to model checkpoint
            model_name: Model name ('MVD' or 'MVDp')
            inference_size: Image size for inference (224)
            min_conf_thr: Minimum confidence threshold for filtering points
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from mvdust3r.dust3r.model import AsymmetricCroCo3DStereoMultiView

        if model_path is None:
            model_path = kRootFolder + "/thirdparty/mvdust3r/checkpoints/MVD.pth"

        # Determine model configuration based on model_name
        inf = np.inf
        if model_name == "MVD":
            model = AsymmetricCroCo3DStereoMultiView(
                pos_embed="RoPE100",
                img_size=(224, 224),
                head_type="linear",
                output_mode="pts3d",
                depth_mode=("exp", -inf, inf),
                conf_mode=("exp", 1, 1e9),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12,
                GS=True,
                sh_degree=0,
                pts_head_config={"skip": True},
            )
        elif model_name == "MVDp":
            model = AsymmetricCroCo3DStereoMultiView(
                pos_embed="RoPE100",
                img_size=(224, 224),
                head_type="linear",
                output_mode="pts3d",
                depth_mode=("exp", -inf, inf),
                conf_mode=("exp", 1, 1e9),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12,
                GS=True,
                sh_degree=0,
                pts_head_config={"skip": True},
                m_ref_flag=True,
                n_ref=4,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        model.to(self.device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(model_path).to(self.device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)

        self.model = model
        self.model_name = model_name
        self.inference_size = inference_size
        self.min_conf_thr = min_conf_thr

        self.dust3r_preprocessor = Dust3rImagePreprocessor(
            inference_size=inference_size, verbose=True
        )

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List:
        """
        Preprocess images for MVDust3r.

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (not used for this model)
        """
        return self.dust3r_preprocessor.preprocess_images(images)

    def infer(self, processed_images: List, **kwargs):
        """
        Run inference on preprocessed images using MVDust3r.

        Args:
            processed_images: Preprocessed images from preprocess_images()
            **kwargs: Additional inference parameters (can override init parameters)

        Returns:
            Dictionary containing raw inference output and parameters
        """
        min_conf_thr = kwargs.get("min_conf_thr", self.min_conf_thr)

        imgs_preproc = processed_images

        # Handle single image case
        if len(imgs_preproc) == 1:
            imgs_preproc = [imgs_preproc[0], copy.deepcopy(imgs_preproc[0])]
            imgs_preproc[1]["idx"] = 1

        # Rearrange images for better matching (from test file)
        if len(imgs_preproc) < 12:
            if len(imgs_preproc) > 3:
                imgs_preproc[1], imgs_preproc[3] = copy.deepcopy(imgs_preproc[3]), copy.deepcopy(
                    imgs_preproc[1]
                )
            if len(imgs_preproc) > 6:
                imgs_preproc[2], imgs_preproc[6] = copy.deepcopy(imgs_preproc[6]), copy.deepcopy(
                    imgs_preproc[2]
                )
        else:
            change_id = len(imgs_preproc) // 4 + 1
            imgs_preproc[1], imgs_preproc[change_id] = copy.deepcopy(
                imgs_preproc[change_id]
            ), copy.deepcopy(imgs_preproc[1])
            change_id = (len(imgs_preproc) * 2) // 4 + 1
            imgs_preproc[2], imgs_preproc[change_id] = copy.deepcopy(
                imgs_preproc[change_id]
            ), copy.deepcopy(imgs_preproc[2])
            change_id = (len(imgs_preproc) * 3) // 4 + 1
            imgs_preproc[3], imgs_preproc[change_id] = copy.deepcopy(
                imgs_preproc[change_id]
            ), copy.deepcopy(imgs_preproc[3])

        # Run inference
        from mvdust3r.dust3r.inference import inference_mv

        for img in imgs_preproc:
            img["true_shape"] = torch.from_numpy(img["true_shape"]).long()

        output = inference_mv(imgs_preproc, self.model, self.device, verbose=True)

        # Replace RGB with original images
        output["pred1"]["rgb"] = imgs_preproc[0]["img"].permute(0, 2, 3, 1)
        for x, img in zip(output["pred2s"], imgs_preproc[1:]):
            x["rgb"] = img["img"].permute(0, 2, 3, 1)

        return {
            "output": output,
            "imgs_preproc": imgs_preproc,
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
        Postprocess raw MVDust3r output into SceneFromViewsResult.

        Args:
            raw_output: Dictionary containing inference output and parameters from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        output = raw_output["output"]
        imgs_preproc = raw_output["imgs_preproc"]
        min_conf_thr = raw_output["min_conf_thr"]

        # Extract 3D dense map
        _, h, w = output["pred1"]["rgb"].shape[0:3]
        rgb_imgs = [output["pred1"]["rgb"][0]] + [x["rgb"][0] for x in output["pred2s"]]
        for i in range(len(rgb_imgs)):
            rgb_imgs[i] = (rgb_imgs[i] + 1) / 2

        pts3d = [output["pred1"]["pts3d"][0]] + [
            x["pts3d_in_other_view"][0] for x in output["pred2s"]
        ]

        conf = torch.stack(
            [output["pred1"]["conf"][0]] + [x["conf"][0] for x in output["pred2s"]], 0
        )

        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres

        # Estimate focal length
        conf_first = conf[0].reshape(-1)
        conf_first_sorted = conf_first.sort()[0]
        conf_first_thres = conf_first_sorted[int(conf_first_sorted.shape[0] * 0.03)]
        valid_first = conf_first >= conf_first_thres
        valid_first = valid_first.reshape(h, w)

        focals = (
            estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda())
            .cpu()
            .item()
        )

        intrinsics = torch.eye(3)
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()

        # Estimate camera poses
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()

        c2ws = []
        for pr_pt, valid in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(
                pr_pt.cuda().flatten(0, 1)[None],
                pixel_coords.flatten(0, 1)[None],
                valid.cuda().flatten(0, 1)[None],
                intrinsics[None],
            )
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu()

        # Convert to numpy
        cams2world = to_numpy(cams2world)
        focals = to_numpy(focals)
        msk = to_numpy(msk)
        confs = [to_numpy(x[0]) for x in conf.split(1, dim=0)]
        rgb_imgs = [to_numpy(x) for x in rgb_imgs]
        pts3d = to_numpy(pts3d)

        # Convert to geometry
        global_pc, global_mesh = convert_mv_output_to_geometry(rgb_imgs, pts3d, msk, as_pointcloud)

        # Extract depth maps
        depth_predictions = []
        for i, (pts, msk_i) in enumerate(zip(pts3d, msk)):
            h_i, w_i = rgb_imgs[i].shape[:2]
            depth = np.zeros((h_i, w_i), dtype=pts.dtype)
            pts_reshaped = pts.reshape(h_i, w_i, 3)
            depth[msk_i] = pts_reshaped[msk_i, 2]
            depth_predictions.append(depth)

        # Process images
        processed_images_list = []
        for img in rgb_imgs:
            img_processed = (img * 255).astype(np.uint8)
            # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
            processed_images_list.append(img_processed)

        # Create per-view point clouds
        point_clouds = []
        for i, (pts, msk_i, img) in enumerate(zip(pts3d, msk, rgb_imgs)):
            pts_valid = pts[msk_i].reshape(-1, 3)
            img_valid = img[msk_i].reshape(-1, 3)
            if len(pts_valid) > 0:
                pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                point_clouds.append(pc)
            else:
                point_clouds.append(None)

        # Build intrinsics list
        intrinsics_list = []
        for i in range(len(rgb_imgs)):
            K = np.eye(3)
            K[0, 0] = focals
            K[1, 1] = focals
            h_i, w_i = rgb_imgs[i].shape[:2]
            K[0, 2] = w_i / 2
            K[1, 2] = h_i / 2
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
