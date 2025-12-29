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
import tempfile
import os

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from pyslam.utilities.dust3r import convert_mv_output_to_geometry
from pyslam.utilities.torch import to_numpy
from pyslam.utilities.geometry import inv_poseRt

import pyslam.config as config

config.cfg.set_lib("fast3r")


class SceneFromViewsFast3r(SceneFromViewsBase):
    """
    Scene reconstruction using Fast3R (Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass).

    This model performs multi-view reconstruction with pose estimation in a single forward pass.
    NOTE: This model requires a lot of GPU memory!
    """

    def __init__(
        self,
        device=None,
        checkpoint_dir=None,
        is_lightning_checkpoint=False,
        image_size=512,
        min_conf_thr_percentile=10,
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head",
        **kwargs,
    ):
        """
        Initialize Fast3R model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            checkpoint_dir: Path to model checkpoint or HF model name (None to use default)
            is_lightning_checkpoint: Whether the checkpoint is from Lightning (default: False)
            image_size: Image size for inference (224 or 512)
            min_conf_thr_percentile: Minimum confidence percentile threshold for alignment
            niter_PnP: Number of iterations for PnP pose estimation
            focal_length_estimation_method: Method for focal length estimation
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from fast3r.utils.checkpoint_utils import load_model

        if checkpoint_dir is None:
            checkpoint_dir = "jedyang97/Fast3R_ViT_Large_512"

        self.model, self.lit_module = load_model(
            checkpoint_dir, device=self.device, is_lightning_checkpoint=is_lightning_checkpoint
        )

        self.image_size = image_size
        self.min_conf_thr_percentile = min_conf_thr_percentile
        self.niter_PnP = niter_PnP
        self.focal_length_estimation_method = focal_length_estimation_method

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List:
        """
        Preprocess images for Fast3R.

        Args:
            images: List of input images (BGR or RGB format, numpy arrays)
            **kwargs: Additional preprocessing parameters (target_size can be overridden)

        Returns:
            List of dictionaries with preprocessed image data
        """
        from fast3r.dust3r.utils.image import load_images

        # Save images temporarily to disk for Fast3R's load_images function
        # Fast3R expects file paths, not numpy arrays
        temp_dir = tempfile.mkdtemp(suffix="_fast3r_images")
        filelist = []

        for i, img in enumerate(images):
            # Convert BGR to RGB if needed
            if img.ndim == 3 and img.shape[2] == 3:
                # Check if it's BGR (OpenCV format) by checking if it looks like BGR
                # For safety, we'll assume BGR and convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            # Save image
            img_path = os.path.join(temp_dir, f"image_{i:04d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            filelist.append(img_path)

        # Use Fast3R's load_images function
        image_size = kwargs.get("target_size", self.image_size)
        processed_images = load_images(
            filelist,
            size=image_size,
            verbose=kwargs.get("verbose", False),
            rotate_clockwise_90=kwargs.get("rotate_clockwise_90", False),
            crop_to_landscape=kwargs.get("crop_to_landscape", False),
        )

        # Store temp_dir for cleanup in postprocess
        self._temp_dir = temp_dir

        return processed_images

    def infer(self, processed_images: List, **kwargs):
        """
        Run inference on preprocessed images using Fast3R.

        Args:
            processed_images: Preprocessed images from preprocess_images()
            **kwargs: Additional inference parameters (can override init parameters)

        Returns:
            Dictionary containing predictions and parameters
        """
        from fast3r.dust3r.inference_multiview import inference

        dtype = kwargs.get("dtype", torch.float32)
        verbose = kwargs.get("verbose", True)
        profiling = kwargs.get("profiling", False)

        # Run inference
        # inference() returns (result, profiling_info) when profiling=True, otherwise just result
        inference_result = inference(
            processed_images,
            self.model,
            self.device,
            dtype=dtype,
            verbose=verbose,
            profiling=profiling,
        )

        # Handle both return cases
        if profiling and isinstance(inference_result, tuple):
            output_dict, profiling_info = inference_result
        else:
            output_dict = inference_result
            profiling_info = None

        # Process predictions and move tensors to CPU
        try:
            for pred in output_dict["preds"]:
                for k, v in pred.items():
                    if isinstance(v, torch.Tensor):
                        pred[k] = v.cpu()
            for view in output_dict["views"]:
                for k, v in view.items():
                    if isinstance(v, torch.Tensor):
                        view[k] = v.cpu()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            if verbose:
                print(f"Warning during tensor conversion: {e}")

        # Align local points to global coordinate frame
        min_conf_thr_percentile = kwargs.get(
            "min_conf_thr_percentile", self.min_conf_thr_percentile
        )
        self.lit_module.align_local_pts3d_to_global(
            preds=output_dict["preds"],
            views=output_dict["views"],
            min_conf_thr_percentile=min_conf_thr_percentile,
        )

        return {
            "output_dict": output_dict,
            "profiling_info": profiling_info if profiling else None,
            "niter_PnP": kwargs.get("niter_PnP", self.niter_PnP),
            "focal_length_estimation_method": kwargs.get(
                "focal_length_estimation_method", self.focal_length_estimation_method
            ),
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
        Postprocess raw Fast3R output into SceneFromViewsResult.

        Args:
            raw_output: Dictionary containing output_dict and parameters from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        # Import to_numpy from fast3r for consistency with Fast3R demo
        from fast3r.dust3r.utils.device import to_numpy as fast3r_to_numpy

        output_dict = raw_output["output_dict"]
        niter_PnP = raw_output["niter_PnP"]
        focal_length_estimation_method = raw_output["focal_length_estimation_method"]

        # Estimate camera poses using PnP
        # Note: estimate_camera_poses expects preds with batch dimension and as torch.Tensor
        # We need to convert numpy arrays back to tensors and add batch dimension if not present
        preds_with_batch = []
        for pred in output_dict["preds"]:
            pred_batched = {}
            for k, v in pred.items():
                # Convert numpy arrays back to torch tensors if needed
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                elif not isinstance(v, torch.Tensor):
                    pred_batched[k] = v
                    continue

                # Add batch dimension if needed
                if v.ndim == 3:  # (H, W, ...)
                    pred_batched[k] = v.unsqueeze(0)  # Add batch dimension -> (1, H, W, ...)
                elif v.ndim == 2:  # (H, W)
                    pred_batched[k] = v.unsqueeze(0)  # Add batch dimension -> (1, H, W)
                elif v.ndim == 4:  # Already has batch dimension (1, H, W, ...)
                    pred_batched[k] = v
                else:
                    pred_batched[k] = v
            preds_with_batch.append(pred_batched)

        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

        poses_c2w_batch, estimated_focals_batch = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds_with_batch,
            views=output_dict.get("views"),
            niter_PnP=niter_PnP,
            focal_length_estimation_method=focal_length_estimation_method,
        )

        # poses_c2w_batch is a list; the first element contains the estimated poses
        camera_poses = poses_c2w_batch[0]  # List of (4, 4) numpy arrays
        estimated_focals = estimated_focals_batch[0]  # List of focal lengths

        # Extract 3D points and confidence maps
        preds = output_dict["preds"]
        views = output_dict["views"]

        num_views = len(preds)
        images_pred = []
        depth_predictions = []
        point_clouds = []
        confidences = []

        # Convert images from views
        from fast3r.dust3r.utils.image import rgb

        for i, view in enumerate(views):
            # Get image from view
            img_tensor = view.get("img", None)
            if img_tensor is not None:
                true_shape = view.get("true_shape", None)
                if true_shape is not None:
                    if isinstance(true_shape, torch.Tensor):
                        true_shape = true_shape.cpu().numpy()
                    # true_shape is stored as (1, 2) array [H, W], extract the actual shape
                    if isinstance(true_shape, np.ndarray):
                        if true_shape.ndim == 2 and true_shape.shape[0] == 1:
                            true_shape = tuple(true_shape[0])  # Extract (H, W) from (1, 2) as tuple
                        elif true_shape.ndim == 1 and true_shape.shape[0] == 2:
                            true_shape = tuple(true_shape)  # Convert to tuple (H, W)
                        else:
                            true_shape = None  # Invalid shape, ignore it
                img_rgb = rgb(img_tensor, true_shape=true_shape)
                # Convert to uint8 and ensure RGB format (rgb() returns RGB in [0,1])
                img_uint8 = (img_rgb * 255).astype(np.uint8)
                # Ensure contiguous array and RGB format
                img_uint8 = np.ascontiguousarray(img_uint8, dtype=np.uint8)
                images_pred.append(img_uint8)
            else:
                # Fallback: use original image
                if i < len(images):
                    img = images[i].copy()
                    if img.ndim == 3 and img.shape[2] == 3:
                        # Convert BGR to RGB (OpenCV loads as BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Ensure contiguous array
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                    images_pred.append(img)
                else:
                    # Create dummy image
                    h, w = 512, 512
                    images_pred.append(np.zeros((h, w, 3), dtype=np.uint8))

        # Extract point clouds, depth maps, and confidences
        for i, pred in enumerate(preds):
            # Get aligned points (in global coordinate frame)
            if "pts3d_local_aligned_to_global" in pred:
                pts3d = pred["pts3d_local_aligned_to_global"]
            elif "pts3d_in_other_view" in pred:
                pts3d = pred["pts3d_in_other_view"]
            else:
                # Fallback to local points
                pts3d = pred.get("pts3d_local", None)
                if pts3d is None:
                    raise ValueError(f"No valid 3D points found in prediction {i}")

            # Convert to numpy if needed
            if isinstance(pts3d, torch.Tensor):
                pts3d = to_numpy(pts3d)
            if pts3d.ndim == 4 and pts3d.shape[0] == 1:
                pts3d = pts3d[0]  # Remove batch dimension

            # Get confidence map
            conf = pred.get("conf", None)
            if conf is None:
                conf = pred.get("conf_local", None)
            if conf is not None:
                if isinstance(conf, torch.Tensor):
                    conf = to_numpy(conf)
                # Remove batch dimension if present
                while conf.ndim > 2 and conf.shape[0] == 1:
                    conf = conf[0]
                # Remove channel dimension if present
                if conf.ndim == 3 and conf.shape[2] == 1:
                    conf = conf[:, :, 0]
                # Ensure conf is 2D (H, W)
                if conf.ndim == 1:
                    # If conf is 1D, we need to reshape it to match pts3d
                    h, w = pts3d.shape[:2]
                    if conf.shape[0] == h * w:
                        conf = conf.reshape(h, w)
                    else:
                        # Create dummy confidence map if shape doesn't match
                        conf = np.ones((h, w), dtype=np.float32)
            else:
                # Create dummy confidence map
                h, w = pts3d.shape[:2]
                conf = np.ones((h, w), dtype=np.float32)

            # Ensure conf shape matches pts3d spatial dimensions
            h, w = pts3d.shape[:2]
            if conf.shape != (h, w):
                # Reshape or recreate if shape doesn't match
                if conf.size == h * w:
                    conf = conf.reshape(h, w)
                else:
                    conf = np.ones((h, w), dtype=np.float32)

            # Extract depth from points (z-component)
            depth = pts3d[:, :, 2]
            depth_predictions.append(depth)

            # Store confidence
            confidences.append(conf)

            # Create per-view point cloud
            # Apply confidence threshold
            conf_threshold = np.percentile(conf, self.min_conf_thr_percentile)
            mask = conf > conf_threshold

            # Ensure mask has correct shape
            if mask.shape != pts3d.shape[:2]:
                if mask.size == pts3d.shape[0] * pts3d.shape[1]:
                    mask = mask.reshape(pts3d.shape[:2])
                else:
                    # Fallback: use all points
                    mask = np.ones(pts3d.shape[:2], dtype=bool)

            pts_valid = pts3d[mask].reshape(-1, 3)

            # Extract colors directly from the view's img tensor (in [-1, 1] range)
            # This matches exactly how Fast3R demo extracts colors (see viser_visualizer.py line 354-372)
            view = views[i]
            img_tensor = view.get("img", None)

            if img_tensor is not None:
                # Use the same method as Fast3R demo: to_numpy, squeeze, permute
                # Convert tensor to numpy and permute: (1, 3, H, W) or (3, H, W) -> (H, W, 3)
                if isinstance(img_tensor, torch.Tensor):
                    img_rgb_norm = fast3r_to_numpy(img_tensor.cpu().squeeze().permute(1, 2, 0))
                else:
                    # Already numpy, but might need permute
                    img_np = np.array(img_tensor)
                    if img_np.ndim == 4 and img_np.shape[0] == 1:
                        img_np = img_np[0]  # Remove batch
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = img_np.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                    img_rgb_norm = img_np

                # Get true_shape to crop if needed (before flattening)
                true_shape = view.get("true_shape", None)
                if true_shape is not None:
                    if isinstance(true_shape, torch.Tensor):
                        true_shape = true_shape.cpu().numpy()
                    if isinstance(true_shape, np.ndarray):
                        if true_shape.ndim == 2 and true_shape.shape[0] == 1:
                            true_shape = tuple(true_shape[0])
                        elif true_shape.ndim == 1 and true_shape.shape[0] == 2:
                            true_shape = tuple(true_shape)
                        if isinstance(true_shape, tuple) and len(true_shape) == 2:
                            H, W = true_shape
                            img_rgb_norm = img_rgb_norm[:H, :W]

                # CRITICAL: Resize image to match point cloud resolution before flattening
                # The point cloud pts3d has shape (H_pts, W_pts, 3), and we need the image to match
                h_pts, w_pts = pts3d.shape[:2]
                h_img, w_img = img_rgb_norm.shape[:2]
                if h_img != h_pts or w_img != w_pts:
                    # Convert from [-1, 1] to [0, 1] for cv2.resize (it expects [0, 1] or [0, 255])
                    img_rgb_01 = (img_rgb_norm + 1.0) / 2.0
                    # Resize to match point cloud resolution
                    img_rgb_01 = cv2.resize(
                        img_rgb_01, (w_pts, h_pts), interpolation=cv2.INTER_LINEAR
                    )
                    # Convert back to [-1, 1] range
                    img_rgb_norm = img_rgb_01 * 2.0 - 1.0

                # Flatten image to match point cloud: (H, W, 3) -> (H*W, 3)
                # This matches Fast3R demo line 355: img_rgb_flat = img_rgb.reshape(-1, 3)
                img_rgb_flat = img_rgb_norm.reshape(-1, 3)

                # Apply mask to get colors for valid points
                # This matches Fast3R demo: sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
                img_valid_norm = img_rgb_flat[mask.flatten()]

                # Convert from [-1, 1] range to [0, 255] uint8
                # This matches Fast3R demo line 371: ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8)
                # But we keep as uint8 [0, 255] for trimesh (not dividing by 255.0)
                img_valid = ((img_valid_norm + 1) * 127.5).astype(np.uint8).clip(0, 255)
            else:
                # Fallback: use original input image
                h, w = pts3d.shape[:2]
                if i < len(images):
                    img_orig = images[i].copy()
                    if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                    img_h, img_w = img_orig.shape[:2]
                    if img_h != h or img_w != w:
                        img_orig = cv2.resize(img_orig, (w, h), interpolation=cv2.INTER_LINEAR)
                    img_rgb = img_orig
                    img_valid = img_rgb[mask].reshape(-1, 3)
                else:
                    # Last resort: use processed image
                    img_rgb = images_pred[i]
                    img_h, img_w = img_rgb.shape[:2]
                    if img_h != h or img_w != w:
                        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                    img_valid = img_rgb[mask].reshape(-1, 3)

            # Ensure colors are contiguous and in RGB format (uint8 [0, 255])
            img_valid = np.ascontiguousarray(img_valid, dtype=np.uint8)

            if len(pts_valid) > 0:
                pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                point_clouds.append(pc)
            else:
                point_clouds.append(None)

        # Create global point cloud or mesh
        # Use all points with confidence filtering
        all_pts = []
        all_colors = []
        all_masks = []

        for i, (pred, img) in enumerate(zip(preds, images_pred)):
            if "pts3d_local_aligned_to_global" in pred:
                pts3d = pred["pts3d_local_aligned_to_global"]
            elif "pts3d_in_other_view" in pred:
                pts3d = pred["pts3d_in_other_view"]
            else:
                pts3d = pred.get("pts3d_local", None)
                if pts3d is None:
                    continue

            if isinstance(pts3d, torch.Tensor):
                pts3d = to_numpy(pts3d)
            if pts3d.ndim == 4 and pts3d.shape[0] == 1:
                pts3d = pts3d[0]

            conf = pred.get("conf", None)
            if conf is None:
                conf = pred.get("conf_local", None)
            if conf is not None:
                if isinstance(conf, torch.Tensor):
                    conf = to_numpy(conf)
                # Remove batch dimension if present
                while conf.ndim > 2 and conf.shape[0] == 1:
                    conf = conf[0]
                # Remove channel dimension if present
                if conf.ndim == 3 and conf.shape[2] == 1:
                    conf = conf[:, :, 0]
                # Ensure conf is 2D (H, W)
                if conf.ndim == 1:
                    h, w = pts3d.shape[:2]
                    if conf.shape[0] == h * w:
                        conf = conf.reshape(h, w)
                    else:
                        conf = np.ones((h, w), dtype=np.float32)
            else:
                h, w = pts3d.shape[:2]
                conf = np.ones((h, w), dtype=np.float32)

            # Ensure conf shape matches pts3d spatial dimensions
            h, w = pts3d.shape[:2]
            if conf.shape != (h, w):
                if conf.size == h * w:
                    conf = conf.reshape(h, w)
                else:
                    conf = np.ones((h, w), dtype=np.float32)

            conf_threshold = np.percentile(conf, self.min_conf_thr_percentile)
            mask = conf > conf_threshold

            # Ensure mask has correct shape
            if mask.shape != pts3d.shape[:2]:
                if mask.size == pts3d.shape[0] * pts3d.shape[1]:
                    mask = mask.reshape(pts3d.shape[:2])
                else:
                    mask = np.ones(pts3d.shape[:2], dtype=bool)

            pts_valid = pts3d[mask].reshape(-1, 3)

            # Extract colors directly from the view's img tensor (in [-1, 1] range)
            # This matches exactly how Fast3R demo extracts colors (see viser_visualizer.py line 354-372)
            view = views[i]
            img_tensor = view.get("img", None)

            if img_tensor is not None:
                # Use the same method as Fast3R demo: to_numpy, squeeze, permute
                # Convert tensor to numpy and permute: (1, 3, H, W) or (3, H, W) -> (H, W, 3)
                if isinstance(img_tensor, torch.Tensor):
                    img_rgb_norm = fast3r_to_numpy(img_tensor.cpu().squeeze().permute(1, 2, 0))
                else:
                    # Already numpy, but might need permute
                    img_np = np.array(img_tensor)
                    if img_np.ndim == 4 and img_np.shape[0] == 1:
                        img_np = img_np[0]  # Remove batch
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = img_np.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
                    img_rgb_norm = img_np

                # Get true_shape to crop if needed (before flattening)
                true_shape = view.get("true_shape", None)
                if true_shape is not None:
                    if isinstance(true_shape, torch.Tensor):
                        true_shape = true_shape.cpu().numpy()
                    if isinstance(true_shape, np.ndarray):
                        if true_shape.ndim == 2 and true_shape.shape[0] == 1:
                            true_shape = tuple(true_shape[0])
                        elif true_shape.ndim == 1 and true_shape.shape[0] == 2:
                            true_shape = tuple(true_shape)
                        if isinstance(true_shape, tuple) and len(true_shape) == 2:
                            H, W = true_shape
                            img_rgb_norm = img_rgb_norm[:H, :W]

                # CRITICAL: Resize image to match point cloud resolution before flattening
                # The point cloud pts3d has shape (H_pts, W_pts, 3), and we need the image to match
                h_pts, w_pts = pts3d.shape[:2]
                h_img, w_img = img_rgb_norm.shape[:2]
                if h_img != h_pts or w_img != w_pts:
                    # Convert from [-1, 1] to [0, 1] for cv2.resize (it expects [0, 1] or [0, 255])
                    img_rgb_01 = (img_rgb_norm + 1.0) / 2.0
                    # Resize to match point cloud resolution
                    img_rgb_01 = cv2.resize(
                        img_rgb_01, (w_pts, h_pts), interpolation=cv2.INTER_LINEAR
                    )
                    # Convert back to [-1, 1] range
                    img_rgb_norm = img_rgb_01 * 2.0 - 1.0

                # Flatten image to match point cloud: (H, W, 3) -> (H*W, 3)
                # This matches Fast3R demo line 355: img_rgb_flat = img_rgb.reshape(-1, 3)
                img_rgb_flat = img_rgb_norm.reshape(-1, 3)

                # Apply mask to get colors for valid points
                # This matches Fast3R demo: sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
                img_valid_norm = img_rgb_flat[mask.flatten()]

                # Convert from [-1, 1] range to [0, 255] uint8
                # This matches Fast3R demo line 371: ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8)
                # But we keep as uint8 [0, 255] for trimesh (not dividing by 255.0)
                img_valid = ((img_valid_norm + 1) * 127.5).astype(np.uint8).clip(0, 255)
            else:
                # Fallback: use original input image
                h, w = pts3d.shape[:2]
                if i < len(images):
                    img_orig = images[i].copy()
                    if img_orig.ndim == 3 and img_orig.shape[2] == 3:
                        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                    img_h, img_w = img_orig.shape[:2]
                    if img_h != h or img_w != w:
                        img_orig = cv2.resize(img_orig, (w, h), interpolation=cv2.INTER_LINEAR)
                    img_rgb = img_orig
                    img_valid = img_rgb[mask].reshape(-1, 3)
                else:
                    # Last resort: use processed image
                    img_rgb = img  # img is from images_pred in the loop
                    img_h, img_w = img_rgb.shape[:2]
                    if img_h != h or img_w != w:
                        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                    img_valid = img_rgb[mask].reshape(-1, 3)

            # Ensure colors are contiguous and in RGB format (uint8 [0, 255])
            img_valid = np.ascontiguousarray(img_valid, dtype=np.uint8)

            if len(pts_valid) > 0:
                all_pts.append(pts_valid)
                all_colors.append(img_valid)
                all_masks.append(mask)

        if len(all_pts) > 0:
            global_pts = np.concatenate(all_pts, axis=0)
            global_colors = np.concatenate(all_colors, axis=0)
            # Convert RGB to BGR by swapping first and third channels (most efficient: fancy indexing)
            global_colors = global_colors[:, [2, 1, 0]]  # RGB -> BGR
            # Ensure colors are contiguous
            global_colors = np.ascontiguousarray(global_colors, dtype=np.uint8)
            global_pc = trimesh.PointCloud(vertices=global_pts, colors=global_colors)

            if as_pointcloud:
                global_mesh = None
            else:
                # Create mesh from point cloud
                # Prepare pts3d list for convert_mv_output_to_geometry
                pts3d_list = []
                for pred in preds:
                    if "pts3d_local_aligned_to_global" in pred:
                        pts3d = pred["pts3d_local_aligned_to_global"]
                    elif "pts3d_in_other_view" in pred:
                        pts3d = pred["pts3d_in_other_view"]
                    else:
                        pts3d = pred.get("pts3d_local", None)

                    if pts3d is not None:
                        if isinstance(pts3d, torch.Tensor):
                            pts3d = to_numpy(pts3d)
                        if pts3d.ndim == 4 and pts3d.shape[0] == 1:
                            pts3d = pts3d[0]
                        pts3d_list.append(pts3d)
                    else:
                        pts3d_list.append(None)

                global_mesh, global_pc = convert_mv_output_to_geometry(
                    imgs=images_pred,
                    pts3d=pts3d_list,
                    mask=all_masks,
                    as_pointcloud=False,
                )
        else:
            global_pc = None
            global_mesh = None

        # Build intrinsics from estimated focals
        intrinsics_list = []
        for i in range(len(camera_poses)):
            focal = estimated_focals[i] if i < len(estimated_focals) else None
            K = np.eye(3)
            if focal is not None:
                K[0, 0] = focal
                K[1, 1] = focal
            h, w = images_pred[i].shape[:2]
            K[0, 2] = w / 2
            K[1, 2] = h / 2
            intrinsics_list.append(K)

        # Sanitize processed_images so downstream viewers always get HxWx3 uint8 (BGR for OpenCV)
        sanitized_images = []
        for i, img in enumerate(images_pred):
            if img is None:
                sanitized_images.append(np.zeros((1, 1, 3), dtype=np.uint8))
                continue

            img_np = np.asarray(img)
            # Drop leading batch dim if present
            if img_np.ndim == 4 and img_np.shape[0] == 1:
                img_np = img_np[0]
            # Channel-first -> channel-last
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                if img_np.shape[0] == 1 and img_np.shape[2] == 1:
                    img_np = np.squeeze(img_np, axis=0)
                elif img_np.shape[0] in (1, 3) and img_np.shape[2] not in (1, 3):
                    img_np = img_np.transpose(1, 2, 0)
            # If grayscale, replicate to 3 channels
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            # If still not 3D, fallback to dummy
            if img_np.ndim != 3 or img_np.shape[2] not in (1, 3):
                sanitized_images.append(np.zeros((1, 1, 3), dtype=np.uint8))
                continue
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            # Convert to uint8 [0,255]
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255)
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            # Convert RGB -> BGR for OpenCV viewers
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            sanitized_images.append(np.ascontiguousarray(img_np))

        # Align list length with camera poses (avoid zip truncation issues)
        if len(sanitized_images) < len(camera_poses):
            pad_count = len(camera_poses) - len(sanitized_images)
            sanitized_images.extend(
                [
                    np.zeros_like(
                        sanitized_images[0]
                        if sanitized_images
                        else np.zeros((1, 1, 3), dtype=np.uint8)
                    )
                ]
                * pad_count
            )
        elif len(sanitized_images) > len(camera_poses):
            sanitized_images = sanitized_images[: len(camera_poses)]

        # Clean up temporary directory
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp directory {self._temp_dir}: {e}")

        return SceneFromViewsResult(
            global_point_cloud=global_pc,
            global_mesh=global_mesh,
            camera_poses=[pose for pose in camera_poses],
            processed_images=sanitized_images,
            depth_predictions=depth_predictions,
            point_clouds=point_clouds,
            intrinsics=intrinsics_list,
            confidences=confidences,
        )
