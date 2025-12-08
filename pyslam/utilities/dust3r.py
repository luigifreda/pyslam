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
import torchvision.transforms as tvf
import trimesh
import torch

from .torch import to_numpy


# NOTE: some of the following functions have been adapted from https://github.com/naver/dust3r


# =========================================================
# preprocessing
# =========================================================


# define the standard image transforms
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# resize input image so that its long side size is resized to "long_edge_size"
def _resize_cv_image(img, long_edge_size):
    H, W = img.shape[:2]
    S = max(H, W)
    if S > long_edge_size:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_CUBIC
    new_size = (int(round(W * long_edge_size / S)), int(round(H * long_edge_size / S)))
    return cv2.resize(img, new_size, interpolation=interp)


# preprocess a list of images:
# size can be either 224 or 512
# resize and center crop to 224x224 or 512x512
def dust3r_preprocess_images(imgs_raw, size, square_ok=False, verbose=True):
    imgs = []
    for j, img in enumerate(imgs_raw):
        H1, W1, _ = img.shape
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_cv_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            # print(f'resizing {H1}x{W1} to {size}x{size}')
            img = _resize_cv_image(img, size)
        H, W, _ = img.shape
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img[cy - half : cy + half, cx - half : cx + half]
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw // 4
            img = img[cy - halfh : cy + halfh, cx - halfw : cx + halfw]

        H2, W2, _ = img.shape
        if verbose:
            print(f"preprocessing image {j} - resolution {W1}x{H1} --> {W2}x{H2}")
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.shape[:2]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )
        # print('adding image', imgs[-1]['img'].shape, imgs[-1]['img'].max(), imgs[-1]['img'].min(), imgs[-1]['true_shape'], imgs[-1]['idx'], imgs[-1]['instance'])
        # adding image torch.Size([1, 3, 384, 512]) tensor(1.) tensor(-0.9608) [[384 512]] 3 3
    return imgs


# Image processor for dust3r
class Dust3rImagePreprocessor:
    def __init__(self, inference_size=224, square_ok=False, verbose=True):
        self.inference_size = inference_size
        self.square_ok = square_ok
        self.verbose = verbose
        self.metadata = {}

    # Resize input image so that its long side size is resized to "long_edge_size"
    def _resize_cv_image(self, img, long_edge_size):
        H, W = img.shape[:2]
        S = max(H, W)
        interp = cv2.INTER_LANCZOS4 if S > long_edge_size else cv2.INTER_CUBIC
        new_size = (int(round(W * long_edge_size / S)), int(round(H * long_edge_size / S)))
        return cv2.resize(img, new_size, interpolation=interp)

    # Preprocess an input image:
    # Size can be either 224 or 512
    # 1. Resize
    # 2. center crop to 224x224 or 512x512
    # Output: a list of preprocessed torch images as required by dust3r
    def preprocess_image(self, img_raw, image_idx=0):
        """Preprocess images: resize, center crop, and normalize."""

        H1, W1 = img_raw.shape[0:2]  # Original shape
        original_shape = (H1, W1)

        if self.inference_size == 224:
            # Resize short side to 224 (then crop)
            img_resized = self._resize_cv_image(
                img_raw, round(self.inference_size * max(W1 / H1, H1 / W1))
            )
        else:
            # Resize long side to 512
            img_resized = self._resize_cv_image(img_raw, self.inference_size)

        H, W = img_resized.shape[0:2]
        cx, cy = W // 2, H // 2

        # Cropping
        if self.inference_size == 224:
            half = min(cx, cy)
            img_cropped = img_resized[cy - half : cy + half, cx - half : cx + half]
            crop_offset = (cx - half, cy - half)
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not self.square_ok and W == H:
                halfh = (3 * halfw) // 4
            img_cropped = img_resized[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
            crop_offset = (cx - halfw, cy - halfh)

        processed_image = img_cropped
        processed_shape = img_cropped.shape[:2]

        if self.verbose:
            print(
                f"Preprocessing image {image_idx}: {W1}x{H1} -> {processed_shape[1]}x{processed_shape[0]}"
            )

        # Store metadata for inversion
        self.metadata[image_idx] = {
            "original_shape": original_shape[0:2],
            "crop_offset": crop_offset,
            "cropped_shape": img_cropped.shape[0:2],
            "resized_shape": img_resized.shape[0:2],
        }

        img_processed = dict(
            img=ImgNorm(processed_image)[None],
            true_shape=np.int32([processed_image.shape[:2]]),
            idx=image_idx,
            instance=str(image_idx),
        )
        return img_processed

    # Preprocess a list of input images:
    def preprocess_images(self, imgs_raw, reset_metadata=True):
        """Preprocess images: resize, center crop, and normalize."""
        imgs_processed = []
        if reset_metadata:
            self.metadata = {}  # Reset metadata
        for idx, img_raw in enumerate(imgs_raw):
            img_processed = self.preprocess_image(img_raw, image_idx=idx)
            imgs_processed.append(img_processed)
        return imgs_processed

    # Get original-resolution images from inference-size images.
    # This is like inverting the preprocessing above.
    def rescale_image(self, inference_size_img, image_idx=0, original_shape=None):
        if original_shape is None:
            assert image_idx in self.metadata
            original_shape = self.metadata[image_idx]["original_shape"]

        assert image_idx in self.metadata
        resized_shape = self.metadata[image_idx][
            "resized_shape"
        ]  # Get stored shape before cropping

        # we assume the input image has been correctly extracted and transformed back to numpy image
        # img = img_dict['img'].squeeze(0).permute(1, 2, 0).numpy()  # Convert back to HWC format
        # img = (img * 0.5 + 0.5) * 255  # Denormalize
        # img = img.astype(np.uint8)

        # Get the current dimensions
        H, W = inference_size_img.shape[:2]

        # Undo cropping by padding
        if resized_shape[0] == 224:
            half = min(W, H)  # Cropped square size
            pad_h = (resized_shape[0] - half) // 2
            pad_w = (resized_shape[1] - half) // 2
            pad_h = max(pad_h, 0)
            pad_w = max(pad_w, 0)
            if pad_h == 0 and pad_w == 0:
                img_padded = inference_size_img
            else:
                img_padded = cv2.copyMakeBorder(
                    inference_size_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0
                )
        else:
            halfw, halfh = ((2 * W) // 16) * 8, ((2 * H) // 16) * 8
            if W == H:
                halfh = 3 * halfw // 4
            pad_h = (resized_shape[0] - 2 * halfh) // 2
            pad_w = (resized_shape[1] - 2 * halfw) // 2
            pad_h = max(pad_h, 0)
            pad_w = max(pad_w, 0)
            if pad_h == 0 and pad_w == 0:
                img_padded = inference_size_img
            else:
                img_padded = cv2.copyMakeBorder(
                    inference_size_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0
                )

        # Resize back to original dimensions
        img_raw = cv2.resize(
            img_padded, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC
        )
        return img_raw

    # Get original-resolution images from inference-size images.
    # This is like inverting the preprocessing above.
    def rescale_images(self, inference_size_imgs, original_shapes=None):
        if original_shapes is None:
            original_shapes = [
                self.metadata[idx]["original_shape"] for idx in range(len(inference_size_imgs))
            ]
        imgs_raw = []
        for idx, inference_size_img in enumerate(inference_size_imgs):
            img_raw = self.rescale_image(
                inference_size_img, image_idx=idx, original_shape=original_shapes[idx]
            )
            imgs_raw.append(img_raw)
        return imgs_raw

    # Get original-resolution depth from inference-size depth
    def rescale_depth(self, inference_size_depth, image_idx=0, original_shape=None):
        if original_shape is None:
            assert image_idx in self.metadata
            original_shape = self.metadata[image_idx]["original_shape"]

        resized_shape = self.metadata[image_idx][
            "resized_shape"
        ]  # Get stored shape before cropping

        # Get the current dimensions
        H, W = inference_size_depth.shape[:2]

        # Undo cropping by padding
        if resized_shape[0] == 224:
            half = min(W, H)  # Cropped square size
            pad_h = (resized_shape[0] - half) // 2
            pad_w = (resized_shape[1] - half) // 2
            pad_h = max(pad_h, 0)
            pad_w = max(pad_w, 0)
            if pad_h == 0 and pad_w == 0:
                depth_padded = inference_size_depth
            else:
                depth_padded = cv2.copyMakeBorder(
                    inference_size_depth, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, 0
                )
        else:
            halfw, halfh = ((2 * W) // 16) * 8, ((2 * H) // 16) * 8
            if W == H:
                halfh = 3 * halfw // 4
            pad_h = (resized_shape[0] - 2 * halfh) // 2
            pad_w = (resized_shape[1] - 2 * halfw) // 2
            pad_h = max(pad_h, 0)
            pad_w = max(pad_w, 0)
            if pad_h == 0 and pad_w == 0:
                depth_padded = inference_size_depth
            else:
                depth_padded = cv2.copyMakeBorder(
                    inference_size_depth, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, 0
                )

        # Resize back to original dimensions
        depth_raw = cv2.resize(
            depth_padded, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return depth_raw

    # Rescale inference-size keypoints to original-size keypoints.
    def rescale_keypoints(self, inference_size_kps, image_idx=0, original_shape=None):
        if original_shape is None:
            original_shape = self.metadata[image_idx]["original_shape"]

        resized_shape = self.metadata[image_idx][
            "resized_shape"
        ]  # Get stored shape before cropping

        # First, invert cropping:
        # Assume keypoints are given relative to the cropped image.
        x_offset, y_offset = self.metadata[image_idx]["crop_offset"]
        keypoints_in_resized = [(x + x_offset, y + y_offset) for (x, y) in inference_size_kps]

        # Next, compute scale factor used during resizing.
        H_raw, W_raw = original_shape
        # Long side of raw image
        S_raw = max(H_raw, W_raw)
        # The resized image was computed with scale factor s: long_edge_size / S_raw
        # For instance, if we know long_edge_size was used, then:
        r = resized_shape[0] / H_raw if H_raw > W_raw else resized_shape[1] / W_raw

        # Now invert the resize transformation:
        keypoints_raw = [(x / r, y / r) for (x, y) in keypoints_in_resized]

        return keypoints_raw

    # Transform keypoints from original image coordinates to processed image coordinates.
    # Input:
    # - keypoints_list: List of (x, y) keypoints in the original image.
    # - image_index: Index of the image in the batch (to access metadata).
    # Output:
    # - Transformed keypoints as a NumPy array of shape (N, 2).
    def scale_down_keypoints(self, kps, image_idx=0):
        if image_idx >= len(self.metadata):
            raise IndexError("Image index out of range for metadata.")

        meta = self.metadata[image_idx]
        orig_H, orig_W = meta["original_shape"]
        resized_H, resized_W = meta["resized_shape"]
        crop_x, crop_y = meta["crop_offset"]

        # Compute scaling factors between original and resized dimensions
        scale_x = resized_W / orig_W
        scale_y = resized_H / orig_H

        # First, rescale keypoints to resized image coordinates.
        keypoints_rescaled = np.array([(kp[0] * scale_x, kp[1] * scale_y) for kp in kps])

        # Next, adjust keypoints by subtracting the crop offset.
        keypoints_transformed = keypoints_rescaled - np.array([crop_x, crop_y])

        return keypoints_transformed


# =========================================================
# cv
# =========================================================


def calibrate_camera_pnpransac(pointclouds, img_points, masks, intrinsics):
    """
    Input:
        pointclouds: (bs, N, 3)
        img_points: (bs, N, 2)
    Return:
        rotations: (bs, 3, 3)
        translations: (bs, 3, 1)
        c2ws: (bs, 4, 4)
    """
    bs = pointclouds.shape[0]

    camera_matrix = intrinsics.cpu().numpy()  # (bs, 3, 3)

    dist_coeffs = np.zeros((5, 1))

    rotations = []
    translations = []

    for i in range(bs):
        obj_points = pointclouds[i][masks[i]].cpu().numpy()
        img_pts = img_points[i][[masks[i]]].cpu().numpy()

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_pts, camera_matrix[i], dist_coeffs
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotations.append(torch.tensor(rotation_matrix, dtype=torch.float32))
            translations.append(torch.tensor(tvec, dtype=torch.float32))
        else:
            rotations.append(torch.eye(3))
            translations.append(torch.ones(3, 1))

    rotations = torch.stack(rotations).to(pointclouds.device)
    translations = torch.stack(translations).to(pointclouds.device)
    w2cs = torch.eye(4).repeat(bs, 1, 1).to(pointclouds.device)
    w2cs[:, :3, :3] = rotations
    w2cs[:, :3, 3:] = translations
    return torch.linalg.inv(w2cs)


def xy_grid(
    W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal_knowing_depth(pts3d, valid_mask, min_focal=0.0, max_focal=np.inf):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape  # valid_mask: [1, H, W], bs = 1
    assert THREE == 3

    # centered pixel grid
    pp = torch.tensor([[W / 2, H / 2]], dtype=torch.float32, device=pts3d.device)
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)
    valid_mask = valid_mask.flatten(1, 2)  # (B, HW, 1)
    pixels = pixels[valid_mask].unsqueeze(0)  # (1, N, 2)
    pts3d = pts3d[valid_mask].unsqueeze(0)  # (1, N, 3)

    # init focal with l2 closed form
    # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
    xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
        posinf=0, neginf=0
    )  # homogeneous (x,y,1)

    dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy = xy_over_z.square().sum(dim=-1)

    focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

    # iterative re-weighted least-squares
    for iter in range(10):
        # re-weighting by inverse of distance
        dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
        # print(dis.nanmean(-1))
        w = dis.clip(min=1e-8).reciprocal()
        # update the scaling with the new weights
        focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
    return focal


# =========================================================
# viz
# =========================================================


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate(
        (
            np.c_[idx1, idx2, idx3],
            np.c_[
                idx3, idx2, idx1
            ],  # same triangle, but backward (cheap solution to cancel face culling)
            np.c_[idx2, idx3, idx4],
            np.c_[
                idx4, idx3, idx2
            ],  # same triangle, but backward (cheap solution to cancel face culling)
        ),
        axis=0,
    )

    # prepare triangle colors
    face_colors = np.concatenate(
        (
            img[:-1, :-1].reshape(-1, 3),
            img[:-1, :-1].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
        ),
        axis=0,
    )

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m["vertices"], m["faces"], m["face_colors"]) for m in meshes])
    n_vertices = np.cumsum([0] + [len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def convert_mv_output_to_geometry(imgs, pts3d, mask, as_pointcloud):
    assert len(pts3d) == len(mask) <= len(imgs)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)

    global_pc = None
    global_mesh = None

    for i, p in enumerate(pts3d):
        if pts3d[i].shape[0:2] != mask[i].shape[0:2]:
            pts3d[i] = pts3d[i].reshape(mask[i].shape[0], mask[i].shape[1], 3)
        # print(f'p.shape: {pts3d[i].shape}, m.shape: {mask[i].shape}')

    # if as_pointcloud:
    #     # full pointcloud
    #     pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    #     col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    #     global_pc = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    # else:
    #     meshes = []
    #     for i in range(len(imgs)):
    #         meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
    #     global_mesh = trimesh.Trimesh(**cat_meshes(meshes))

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        global_pc = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        global_mesh = trimesh.Trimesh(**cat_meshes(meshes))

    return global_pc, global_mesh
