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

from __future__ import annotations

import gc
import math
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from pyslam.utilities.dust3r import convert_mv_output_to_geometry
from pyslam.utilities.geometry import inv_poseRt
from pyslam.utilities.torch import safe_empty_cache, invert_se3

import pyslam.config as config

config.cfg.set_lib("vggt_robust")


def extrinsics_to_matrix(extrinsics: torch.Tensor) -> torch.Tensor:
    """Convert [N, 3, 4] extrinsics to homogenous [N, 4, 4] matrices."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics of shape [N,3,4], got {tuple(extrinsics.shape)}")
    n = extrinsics.shape[0]
    device = extrinsics.device
    dtype = extrinsics.dtype
    mats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(n, 4, 4).clone()
    mats[:, :3, :3] = extrinsics[:, :3, :3]
    mats[:, :3, 3] = extrinsics[:, :3, 3]
    return mats


def convert_world_to_cam_to_cam_to_world(extrinsics: torch.Tensor) -> torch.Tensor:
    """Invert world-to-camera extrinsics to obtain camera-to-world transforms."""
    T_w2c = extrinsics_to_matrix(extrinsics)
    return invert_se3(T_w2c)


class SceneFromViewsVggtRobust(SceneFromViewsBase):
    """
    Scene reconstruction using VGGT with robust outlier rejection.

    This variant mirrors the reference robust_vggt demo while conforming to
    the SceneFromViewsBase interface.

    The rejection logic is anchor-based: by default it scores every image against a single reference
    (anchor) frame and drops any non-reference whose combined score falls below rej_thresh. The anchor
    can be the first frame or the most "central" frame (highest average similarity to all others).
    """

    def __init__(
        self,
        device=None,
        model_id: str = "facebook/VGGT-1B",
        attn_a: float = 0.5,
        cos_a: float = 0.5,
        rej_thresh: float = 0.3,  # -1: disabled, 0.1: default,
        conf_thres: float = 3.0,  # confidence threshold for filtering points
        prediction_mode: str = "Depthmap Regression",
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        mask_sky: bool = False,
        target_size: int = 518,
        use_most_central_as_reference: bool = True,  # True: use the most central frame as the reference, False: use the first frame as the reference,
        use_anchor_relative_score: bool = True,  # True: normalize scores relative to anchor instead of global min-max
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from vggt.models.vggt import VGGT

        self.model = VGGT.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability()
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.amp_dtype = torch.float32

        self.attn_a = attn_a
        self.cos_a = cos_a
        self.rej_thresh = rej_thresh
        self.conf_thres = conf_thres
        self.prediction_mode = prediction_mode
        self.mask_black_bg = mask_black_bg
        self.mask_white_bg = mask_white_bg
        self.mask_sky = mask_sky
        self.target_size = target_size
        self.use_most_central_as_reference = use_most_central_as_reference
        self.use_anchor_relative_score = use_anchor_relative_score

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> torch.Tensor:
        """
        Preprocess images for VGGT.

        Returns batched tensor of shape (N, 3, H, W), float32 in [0, 1].
        """
        processed_images = []
        shapes = []
        target_size = kwargs.get("target_size", self.target_size)

        for img in images:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[-1] == 4:
                b, g, r, a = cv2.split(img)
                alpha = a.astype(np.float32) / 255.0
                rgb = cv2.merge((r, g, b)).astype(np.float32) / 255.0
                white = np.ones_like(rgb)
                img = alpha[..., None] * rgb + (1 - alpha[..., None]) * white
            else:
                img = img.astype(np.float32) / 255.0

            h, w = img.shape[:2]

            if w >= h:
                new_w = target_size
                new_h = round(h * (new_w / w) / 14) * 14
            else:
                new_h = target_size
                new_w = round(w * (new_h / h) / 14) * 14

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

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
                value=(1.0, 1.0, 1.0),
            )

            shapes.append(img.shape[:2])
            processed_images.append(img)

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

        images_tensor = np.stack([img.transpose(2, 0, 1) for img in processed_images])
        return torch.from_numpy(images_tensor).float()

    def infer(self, processed_images: torch.Tensor, **kwargs):
        """
        Run inference with robust outlier rejection.
        """
        attn_a = kwargs.get("attn_a", self.attn_a)
        cos_a = kwargs.get("cos_a", self.cos_a)
        rej_thresh = kwargs.get("rej_thresh", self.rej_thresh)
        conf_thres = kwargs.get("conf_thres", self.conf_thres)
        prediction_mode = kwargs.get("prediction_mode", self.prediction_mode)
        mask_black_bg = kwargs.get("mask_black_bg", self.mask_black_bg)
        mask_white_bg = kwargs.get("mask_white_bg", self.mask_white_bg)
        mask_sky = kwargs.get("mask_sky", self.mask_sky)

        if processed_images.ndim == 3:
            processed_images = processed_images.unsqueeze(0)
        if processed_images.ndim == 4:
            processed_images = processed_images.unsqueeze(0)  # (1, N, 3, H, W)

        images = processed_images
        device = self.device
        non_blocking = device.type == "cuda"
        images_device = images.to(device=device, non_blocking=non_blocking)
        if device.type == "cuda":
            images_device = images_device.to(
                device=device, dtype=self.amp_dtype, non_blocking=non_blocking
            )

        # We follow the robust_vggt demo and probe the last global attention block
        # (layer 23) to score frames against the first (anchor) frame.
        attn_layers = [23]
        q_out = {}
        k_out = {}
        handles = []

        def _make_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()

            return _hook

        for i in attn_layers:
            blk = self.model.aggregator.global_blocks[i].attn
            handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, i)))
            handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, i)))

        def _run_model(images_dev: torch.Tensor, need_tokens: bool = False):
            if need_tokens:
                agg_tokens, patch_start_idx_local = self.model.aggregator(images_dev)
                preds = {}
                with torch.cuda.amp.autocast(enabled=False):
                    if self.model.camera_head is not None:
                        pose_enc_list = self.model.camera_head(agg_tokens)
                        preds["pose_enc"] = pose_enc_list[-1]
                        preds["pose_enc_list"] = pose_enc_list
                    if self.model.depth_head is not None:
                        depth, depth_conf = self.model.depth_head(
                            agg_tokens, images=images_dev, patch_start_idx=patch_start_idx_local
                        )
                        preds["depth"] = depth
                        preds["depth_conf"] = depth_conf
                    if self.model.point_head is not None:
                        pts3d, pts3d_conf = self.model.point_head(
                            agg_tokens, images=images_dev, patch_start_idx=patch_start_idx_local
                        )
                        preds["world_points"] = pts3d
                        preds["world_points_conf"] = pts3d_conf
                if self.model.track_head is not None:
                    # Tracking head not used in this pipeline
                    pass
                if not self.model.training:
                    preds["images"] = images_dev
                return preds, agg_tokens, patch_start_idx_local

            preds = self.model(images_dev)
            return preds, None, None

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    predictions_full, aggregated_tokens_list, patch_start_idx = _run_model(
                        images_device, need_tokens=True
                    )
            else:
                predictions_full, aggregated_tokens_list, patch_start_idx = _run_model(
                    images_device, need_tokens=True
                )

        H, W = tuple(int(dim) for dim in images.shape[-2:])
        aggregator = self.model.aggregator
        patch_size = aggregator.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens

        target_layers = attn_layers
        aggregated_tokens_selected: List[torch.Tensor] = [
            aggregated_tokens_list[idx]
            for idx in target_layers
            if aggregated_tokens_list is not None and idx < len(aggregated_tokens_list)
        ]
        global_tokens: List[torch.Tensor] = [
            tokens[..., 1024:] for tokens in aggregated_tokens_selected
        ]

        cosine_similarities: List[torch.Tensor] = []
        anchor_idx: int = 0  # index in the flattened (B*S) sequence used as reference
        anchor_chosen = False
        for feature in global_tokens:
            if feature.ndim != 4:
                continue
            if feature.shape[-2] == 0 or feature.shape[-1] == 0:
                continue

            # Compare every frame's global tokens to a single anchor frame (first by default).
            feature = feature[:, :, patch_start_idx:, :]
            layer_feat = feature.detach().to(dtype=torch.float32)
            num_samples = layer_feat.shape[0] * layer_feat.shape[1]
            layer_feat = layer_feat.reshape(num_samples, layer_feat.shape[2], layer_feat.shape[3])

            layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)

            if (
                self.use_most_central_as_reference
                and not anchor_chosen
                and layer_feat_norm.shape[0] > 0
            ):
                # Pick the frame whose tokens are, on average, most similar to all others (medoid).
                # pair_sim: (N, N) average cosine over all patch tokens.
                pair_sim = torch.einsum("ntc,mtc->nmt", layer_feat_norm, layer_feat_norm).mean(
                    dim=-1
                )
                centrality = pair_sim.mean(dim=1)
                anchor_idx = int(torch.argmax(centrality).item())
                anchor_chosen = True

            # First sample in the batch/seq is treated as the reference frame unless overridden above.
            anchor_idx_safe = min(max(anchor_idx, 0), layer_feat_norm.shape[0] - 1)
            ref_feat_norm = layer_feat_norm[anchor_idx_safe : anchor_idx_safe + 1, :, :]
            cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)
            cos_sim_mean = cos_sim.mean(-1).mean(-1)
            cosine_similarities.append(cos_sim_mean)

        global_mean_vals: List[float] = []
        num_images_total = (
            images.shape[1] if images.ndim == 5 else (images.shape[0] if images.ndim == 4 else 1)
        )
        # Map flattened anchor_idx back to sequence frame index (assumes contiguous flattening).
        anchor_frame_idx = min(max(anchor_idx % max(num_images_total, 1), 0), num_images_total - 1)
        for i in attn_layers:
            if i not in q_out or i not in k_out:
                continue
            Q = q_out[i]
            K = k_out[i]

            T = int(K.shape[-2])
            num_images_in_seq = T // tokens_per_image
            if num_images_in_seq <= 0:
                continue

            # Reference (anchor) queries: tokens from the selected anchor frame.
            q_start = anchor_frame_idx * tokens_per_image + patch_start_idx
            q_end = q_start + num_patch_tokens
            q_first_image = Q[:, :, q_start:q_end, :]
            Tk = int(min(num_images_in_seq, num_images_total) * tokens_per_image)
            K_slice = K[:, :, :Tk, :]
            scale = 1.0 / math.sqrt(float(q_first_image.shape[-1]))
            logits = torch.einsum("bhqd,bhtd->bhqt", q_first_image, K_slice) * scale
            probs = torch.softmax(logits, dim=-1)
            attn_first_image = probs.mean(dim=1).mean(dim=1)[0]

            attn_maps: List[torch.Tensor] = []
            for img_idx in range(num_images_in_seq):
                start = img_idx * tokens_per_image + patch_start_idx
                end = start + num_patch_tokens
                if end > attn_first_image.shape[-1]:
                    break
                patch_attn = attn_first_image[start:end]
                if patch_attn.numel() != num_patch_tokens:
                    continue
                attn_maps.append(patch_attn.view(h_patches, w_patches))

            if not attn_maps:
                continue

            stacked = torch.stack(attn_maps)
            gmin = stacked.min()
            gmax = stacked.max()
            denom = gmax - gmin + 1e-12
            for attn2d in attn_maps:
                global_norm = (attn2d - gmin) / denom
                global_mean_vals.append(float(global_norm.mean().item()))

        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()
        q_out.clear()
        k_out.clear()

        reject_flags = [False] * num_images_total
        score_summary = []
        if cosine_similarities and global_mean_vals:
            cos_sim = cosine_similarities[0]
            attn_val = torch.tensor(global_mean_vals, device=cos_sim.device, dtype=cos_sim.dtype)
            min_len = min(cos_sim.shape[0], attn_val.shape[0])
            cos_sim = cos_sim[:min_len]
            attn_val = attn_val[:min_len]
            if self.use_anchor_relative_score:
                # Anchor-relative: scale by anchor value (keeps anchor at 1.0, others in [0, 1]).
                anchor_pos = min(max(anchor_frame_idx, 0), min_len - 1)
                eps = 1e-6
                cos_anchor = torch.clamp(cos_sim[anchor_pos], min=eps)
                attn_anchor = torch.clamp(attn_val[anchor_pos], min=eps)
                cos_sim = torch.clamp(cos_sim / cos_anchor, 0.0, 1.0)
                attn_val = torch.clamp(attn_val / attn_anchor, 0.0, 1.0)
            else:
                # Global min-max normalization (previous behavior).
                cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-6)
                attn_val = (attn_val - attn_val.min()) / (attn_val.max() - attn_val.min() + 1e-6)
            combined_score = attn_a * attn_val + cos_a * cos_sim  # anchor-vs-frame score
            score_summary = [
                (int(idx), float(combined_score[idx].item())) for idx in range(min_len)
            ]
            for idx in range(min_len):
                if idx == anchor_frame_idx:
                    continue
                if combined_score[idx].item() < rej_thresh:
                    reject_flags[idx] = True

        rejected_indices = [idx for idx, flag in enumerate(reject_flags) if flag]
        survivors = [i for i in range(num_images_total) if i not in rejected_indices]

        # Informative prints for debugging / transparency
        if score_summary:
            print(
                f"[VGGT_ROBUST] attn_a={attn_a:.3f}, cos_a={cos_a:.3f}, "
                f"rej_thresh={rej_thresh:.3f}, anchor={anchor_frame_idx}, scores={score_summary}, "
                f"rejected={rejected_indices}, survivors={survivors}"
            )
        else:
            print(
                f"[VGGT_ROBUST] no valid scores collected; skipping rejection. "
                f"survivors={survivors}"
            )

        if survivors and len(survivors) < num_images_total:
            images_subset = (
                images[:, survivors, ...] if images.ndim == 5 else images[survivors, ...]
            )
            images_subset_device = images_subset.to(device=device, non_blocking=non_blocking)
            if device.type == "cuda":
                images_subset_device = images_subset_device.to(
                    device=device, dtype=self.amp_dtype, non_blocking=non_blocking
                )
            with torch.inference_mode():
                predictions, _, _ = _run_model(images_subset_device, need_tokens=False)
        else:
            survivors = list(range(num_images_total))
            predictions = predictions_full

        safe_empty_cache()

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        image_hw = (H, W)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
        if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
            extrinsics = extrinsics.squeeze(0)
        if intrinsics is not None and intrinsics.ndim == 4 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics.squeeze(0)

        predictions["extrinsic"] = extrinsics
        predictions["intrinsic"] = intrinsics
        predictions["cams2world"] = convert_world_to_cam_to_cam_to_world(extrinsics)

        depth_map = predictions["depth"]
        if depth_map.dim() == 5 and depth_map.shape[0] == 1:
            depth_map = depth_map.squeeze(0)
            predictions["depth"] = depth_map

        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )

        predictions_numpy = {key: self._to_numpy_cpu(value) for key, value in predictions.items()}

        return {
            "predictions": predictions_numpy,
            "conf_thres": conf_thres,
            "prediction_mode": prediction_mode,
            "mask_black_bg": mask_black_bg,
            "mask_white_bg": mask_white_bg,
            "mask_sky": mask_sky,
            "rejected_indices": rejected_indices,
            "survivor_indices": survivors,
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
        """
        predictions = raw_output["predictions"]
        conf_thres = raw_output["conf_thres"]
        prediction_mode = raw_output["prediction_mode"]
        mask_black_bg = raw_output["mask_black_bg"]
        mask_white_bg = raw_output["mask_white_bg"]
        mask_sky = raw_output["mask_sky"]
        survivor_indices: Sequence[int] = raw_output.get(
            "survivor_indices", list(range(len(images)))
        )

        if "Pointmap" in prediction_mode and "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            pred_world_points = predictions.get(
                "world_points_from_depth", predictions.get("world_points")
            )
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )

        if pred_world_points.ndim == 5 and pred_world_points.shape[0] == 1:
            pred_world_points = pred_world_points[0]
            if pred_world_points_conf is not None and pred_world_points_conf.shape[0] == 1:
                pred_world_points_conf = pred_world_points_conf[0]

        images_pred = predictions.get("images", None)
        if images_pred is None:
            if isinstance(processed_images, torch.Tensor):
                images_pred = processed_images.detach().cpu().permute(0, 2, 3, 1).numpy()
            else:
                images_pred = np.asarray(images, dtype=np.float32) / 255.0
        if images_pred.ndim == 5 and images_pred.shape[0] == 1:
            images_pred = images_pred[0]
        if images_pred.ndim == 4 and images_pred.shape[1] == 3:
            images_pred = np.transpose(images_pred, (0, 2, 3, 1))

        images_pred = images_pred[..., ::-1]

        camera_matrices = predictions["extrinsic"]
        camera_matrices = np.asarray(camera_matrices)
        if camera_matrices.ndim == 2 and camera_matrices.shape == (3, 4):
            camera_matrices = camera_matrices[None, ...]
        elif camera_matrices.ndim == 1 and camera_matrices.size % 12 == 0:
            camera_matrices = camera_matrices.reshape((-1, 3, 4))
        if camera_matrices.ndim != 3 or camera_matrices.shape[-2:] != (3, 4):
            raise ValueError(f"Unexpected extrinsic shape: {camera_matrices.shape}")

        S = camera_matrices.shape[0]

        if mask_sky:
            pass

        confs = [pred_world_points_conf[i] for i in range(S)]
        masks = [c > np.percentile(c, conf_thres) for c in confs]

        if mask_black_bg:
            black_mask = [img.sum(axis=-1) >= 16 for img in images_pred]
            masks = [m & b for m, b in zip(masks, black_mask)]

        if mask_white_bg:
            white_mask = [
                ~((img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] > 240))
                for img in images_pred
            ]
            masks = [m & w for m, w in zip(masks, white_mask)]

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

        global_pc, global_mesh = convert_mv_output_to_geometry(
            imgs=images_pred,
            pts3d=pred_world_points,
            mask=normalized_masks,
            as_pointcloud=as_pointcloud,
        )

        cams2world = np.zeros((S, 4, 4))
        for i in range(S):
            Rcw = camera_matrices[i][:3, :3]
            tcw = camera_matrices[i][:3, 3]
            cams2world[i] = inv_poseRt(Rcw, tcw)

        depth_predictions = []
        depth_maps = predictions.get("depth", None)
        if depth_maps is not None:
            if depth_maps.ndim == 4 and depth_maps.shape[1] == 1:
                depth_maps = depth_maps[:, 0]
            for i in range(S):
                depth_predictions.append(depth_maps[i])
        else:
            for i in range(S):
                depth = pred_world_points[i][..., 2]
                depth_predictions.append(depth)

        processed_images_list = []
        for img in images_pred:
            img_processed = (img * 255).astype(np.uint8)
            processed_images_list.append(img_processed)

        point_clouds = []
        for i, (pts, msk) in enumerate(zip(pred_world_points, normalized_masks)):
            pts_valid = pts[msk].reshape(-1, 3)
            img_valid = images_pred[i][msk].reshape(-1, 3)
            if len(pts_valid) > 0:
                pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                point_clouds.append(pc)
            else:
                point_clouds.append(None)

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
            intrinsic_pred = predictions.get("intrinsic", None)
            if intrinsic_pred is not None:
                for i in range(S):
                    if intrinsic_pred.ndim == 3:
                        intrinsics_list.append(intrinsic_pred[i])
                    else:
                        intrinsics_list.append(intrinsic_pred)
            else:
                intrinsics_list = None

        result = SceneFromViewsResult(
            global_point_cloud=global_pc,
            global_mesh=global_mesh,
            camera_poses=[cams2world[i] for i in range(S)],
            processed_images=processed_images_list,
            depth_predictions=depth_predictions,
            point_clouds=point_clouds,
            intrinsics=intrinsics_list,
            confidences=confs,
        )

        result.rejected_indices = raw_output.get("rejected_indices", [])
        result.survivor_indices = list(survivor_indices)
        return result

    def _to_numpy_cpu(self, data):
        """Recursively move torch tensors to CPU and convert to numpy."""
        if isinstance(data, torch.Tensor):
            tensor = data.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            if tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            return tensor.numpy()
        if isinstance(data, dict):
            return {key: self._to_numpy_cpu(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._to_numpy_cpu(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self._to_numpy_cpu(item) for item in data)
        return data
