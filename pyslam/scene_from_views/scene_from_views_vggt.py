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

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from pyslam.utilities.dust3r import convert_mv_output_to_geometry
from pyslam.utilities.torch import to_numpy
from pyslam.utilities.geometry import inv_poseRt

import pyslam.config as config

config.cfg.set_lib("vggt")


class SceneFromViewsVggt(SceneFromViewsBase):
    """
    Scene reconstruction using VGGT (Visual Geometry Grounded Transformer).

    This model performs multi-view reconstruction with pose estimation.
    NOTE: This model requires a lot of GPU memory!
    """

    def __init__(
        self,
        device=None,
        model_path=None,
        conf_thres=3.0,
        prediction_mode="Pointmap Regression",
        mask_black_bg=False,
        mask_white_bg=False,
        mask_sky=False,
        **kwargs,
    ):
        """
        Initialize VGGT model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_path: Path to model checkpoint (None to use default)
            conf_thres: Confidence threshold for filtering points
            prediction_mode: Prediction mode ('Pointmap Regression' or 'Depthmap Regression')
            mask_black_bg: Whether to mask out black background
            mask_white_bg: Whether to mask out white background
            mask_sky: Whether to apply sky segmentation masks
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from vggt.models.vggt import VGGT

        self.model = VGGT()
        if model_path is None:
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        else:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        self.model = self.model.to(self.device)

        self.conf_thres = conf_thres
        self.prediction_mode = prediction_mode
        self.mask_black_bg = mask_black_bg
        self.mask_white_bg = mask_white_bg
        self.mask_sky = mask_sky

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> torch.Tensor:
        """
        Preprocess images for VGGT.

        Returns batched tensor of shape (N, 3, H, W), float32 in [0, 1].

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (target_size can be overridden)
        """
        processed_images = []
        shapes = []
        target_size = kwargs.get("target_size", 518)

        for img in images:
            # Convert BGR to RGB
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Handle alpha channel
            if img.shape[-1] == 4:
                b, g, r, a = cv2.split(img)
                alpha = a.astype(np.float32) / 255.0
                rgb = cv2.merge((r, g, b)).astype(np.float32) / 255.0
                white = np.ones_like(rgb)
                img = alpha[..., None] * rgb + (1 - alpha[..., None]) * white
            else:
                img = img.astype(np.float32) / 255.0

            h, w = img.shape[:2]

            # Resize (pad mode)
            if w >= h:
                new_w = target_size
                new_h = round(h * (new_w / w) / 14) * 14
            else:
                new_h = target_size
                new_w = round(w * (new_h / h) / 14) * 14

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Pad to square
            pad_h = target_size - img.shape[0]
            pad_w = target_size - img.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img = cv2.copyMakeBorder(
                img,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(1.0, 1.0, 1.0),  # white
            )

            shapes.append(img.shape[:2])
            processed_images.append(img)

        # Ensure consistent shapes
        if len(set(shapes)) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded_images = []
            for img in processed_images:
                h, w = img.shape[:2]
                pad_top = (max_h - h) // 2
                pad_bottom = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                img = cv2.copyMakeBorder(
                    img,
                    top=pad_top,
                    bottom=pad_bottom,
                    left=pad_left,
                    right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(1.0, 1.0, 1.0),
                )
                padded_images.append(img)
            processed_images = padded_images

        # Stack: (N, 3, H, W)
        images_tensor = np.stack([img.transpose(2, 0, 1) for img in processed_images])
        return torch.from_numpy(images_tensor).float()

    def infer(self, processed_images: torch.Tensor, **kwargs):
        """
        Run inference on preprocessed images using VGGT.

        Args:
            processed_images: Preprocessed images tensor from preprocess_images()
            **kwargs: Additional inference parameters (can override init parameters)

        Returns:
            Dictionary containing predictions and parameters
        """
        conf_thres = kwargs.get("conf_thres", self.conf_thres)
        prediction_mode = kwargs.get("prediction_mode", self.prediction_mode)
        mask_black_bg = kwargs.get("mask_black_bg", self.mask_black_bg)
        mask_white_bg = kwargs.get("mask_white_bg", self.mask_white_bg)
        mask_sky = kwargs.get("mask_sky", self.mask_sky)

        images_tensor = processed_images.to(self.device)

        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images_tensor)

        # Convert pose encoding to extrinsic and intrinsic matrices
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )

        # Remove singleton batch dimension if present
        if extrinsic.dim() == 4 and extrinsic.shape[0] == 1:
            extrinsic = extrinsic.squeeze(0)
        if intrinsic is not None and intrinsic.dim() == 4 and intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Generate world points from depth map
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        depth_map = predictions["depth"]
        if depth_map.dim() == 5 and depth_map.shape[0] == 1:
            depth_map = depth_map.squeeze(0)
            predictions["depth"] = depth_map

        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # Convert to numpy
        def _to_numpy_cpu(data):
            """Recursively move torch tensors to CPU and convert them to numpy arrays."""
            if isinstance(data, torch.Tensor):
                tensor = data.detach().cpu()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                if tensor.ndim > 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                return tensor.numpy()
            if isinstance(data, dict):
                return {key: _to_numpy_cpu(value) for key, value in data.items()}
            if isinstance(data, list):
                return [_to_numpy_cpu(item) for item in data]
            if isinstance(data, tuple):
                return tuple(_to_numpy_cpu(item) for item in data)
            return data

        predictions = {key: _to_numpy_cpu(value) for key, value in predictions.items()}

        return {
            "predictions": predictions,
            "conf_thres": conf_thres,
            "prediction_mode": prediction_mode,
            "mask_black_bg": mask_black_bg,
            "mask_white_bg": mask_white_bg,
            "mask_sky": mask_sky,
        }

    def postprocess_results(
        self,
        raw_output,
        images: List[np.ndarray],
        processed_images: torch.Tensor,
        as_pointcloud: bool = True,
        **kwargs,
    ) -> SceneFromViewsResult:
        """
        Postprocess raw VGGT output into SceneFromViewsResult.

        Args:
            raw_output: Dictionary containing predictions and parameters from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images tensor used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        predictions = raw_output["predictions"]
        conf_thres = raw_output["conf_thres"]
        prediction_mode = raw_output["prediction_mode"]
        mask_black_bg = raw_output["mask_black_bg"]
        mask_white_bg = raw_output["mask_white_bg"]
        mask_sky = raw_output["mask_sky"]

        # Extract 3D data
        if "Pointmap" in prediction_mode:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )

        images_pred = predictions["images"]  # (S, H, W, 3)
        if images_pred.ndim == 4 and images_pred.shape[1] == 3:
            images_pred = np.transpose(images_pred, (0, 2, 3, 1))

        # Convert BGR to RGB
        images_pred = images_pred[..., ::-1]

        camera_matrices = predictions["extrinsic"]  # (S, 3, 4)
        S, H, W = pred_world_points_conf.shape

        # Optional sky masking
        if mask_sky:
            # Sky masking would require additional dependencies
            # For now, skip it
            pass

        # Per-frame confidence maps and masks
        confs = [pred_world_points_conf[i] for i in range(S)]
        masks = [c > np.percentile(c, conf_thres) for c in confs]

        # Optional background masking
        if mask_black_bg:
            black_mask = [img.sum(axis=-1) >= 16 for img in images_pred]
            masks = [m & b for m, b in zip(masks, black_mask)]

        if mask_white_bg:
            white_mask = [
                ~((img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] > 240))
                for img in images_pred
            ]
            masks = [m & w for m, w in zip(masks, white_mask)]

        # Normalize mask shapes
        normalized_masks = []
        for i in range(S):
            img_h, img_w = images_pred[i].shape[:2]
            m = masks[i]
            if m.ndim == 1:
                if m.size != img_h * img_w:
                    raise ValueError(
                        f"Cannot reshape flat mask of size {m.size} to match image {img_h}x{img_w}"
                    )
                m = m.reshape((img_h, img_w))
            elif m.shape != (img_h, img_w):
                raise ValueError(
                    f"Mask shape {m.shape} does not match image shape {(img_h, img_w)}"
                )
            normalized_masks.append(m)

        # Convert to point cloud or mesh
        global_pc, global_mesh = convert_mv_output_to_geometry(
            imgs=images_pred,
            pts3d=pred_world_points,
            mask=normalized_masks,
            as_pointcloud=as_pointcloud,
        )

        # Camera extrinsics
        # VGGT extrinsics are in w2c (world-to-camera) format [R|t] where:
        #   X_cam = R * X_world + t
        # SceneFromViewsBase expects c2w (camera-to-world) format, so we need to invert
        cams2world = np.zeros((S, 4, 4))
        for i in range(S):
            # Extract rotation and translation from w2c matrix
            Rcw = camera_matrices[i][:3, :3]  # world-to-camera rotation
            tcw = camera_matrices[i][:3, 3]  # world-to-camera translation
            # Convert to camera-to-world transformation
            cams2world[i] = inv_poseRt(Rcw, tcw)

        # Extract depth maps
        depth_predictions = []
        depth_maps = predictions.get("depth", None)
        if depth_maps is not None:
            if depth_maps.ndim == 4:
                # (S, 1, H, W) or (S, H, W)
                if depth_maps.shape[1] == 1:
                    depth_maps = depth_maps[:, 0]
            for i in range(S):
                depth_predictions.append(depth_maps[i])
        else:
            # Extract depth from world points (z-component)
            for i in range(S):
                depth = pred_world_points[i][..., 2]
                depth_predictions.append(depth)

        # Process images
        processed_images_list = []
        for img in images_pred:
            img_processed = (img * 255).astype(np.uint8)
            # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
            processed_images_list.append(img_processed)

        # Create per-view point clouds
        point_clouds = []
        for i, (pts, msk) in enumerate(zip(pred_world_points, normalized_masks)):
            pts_valid = pts[msk].reshape(-1, 3)
            img_valid = images_pred[i][msk].reshape(-1, 3)
            if len(pts_valid) > 0:
                pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                point_clouds.append(pc)
            else:
                point_clouds.append(None)

        # Build intrinsics list
        intrinsics_list = []
        focals = predictions.get("focals", None)
        if focals is not None:
            for i in range(S):
                K = np.eye(3)
                if focals.ndim == 2:
                    K[0, 0] = focals[i, 0]
                    K[1, 1] = focals[i, 1]
                else:
                    K[0, 0] = focals[i]
                    K[1, 1] = focals[i]
                h, w = images_pred[i].shape[:2]
                K[0, 2] = w / 2
                K[1, 2] = h / 2
                intrinsics_list.append(K)
        else:
            # Use intrinsic from predictions if available
            intrinsic_pred = predictions.get("intrinsic", None)
            if intrinsic_pred is not None:
                for i in range(S):
                    if intrinsic_pred.ndim == 3:
                        intrinsics_list.append(intrinsic_pred[i])
                    else:
                        intrinsics_list.append(intrinsic_pred)
            else:
                intrinsics_list = None

        return SceneFromViewsResult(
            global_point_cloud=global_pc,
            global_mesh=global_mesh,
            camera_poses=[cams2world[i] for i in range(S)],
            processed_images=processed_images_list,
            depth_predictions=depth_predictions,
            point_clouds=point_clouds,
            intrinsics=intrinsics_list,
            confidences=confs,
        )
