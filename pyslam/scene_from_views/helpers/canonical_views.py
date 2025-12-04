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

Part of the code is adapted from the original code by Naver Corporation.
Original code Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""

"""
Canonical view processing and correspondence utilities.
"""

import torch
import numpy as np
import os
import tqdm
import torch.nn.functional as F

from .camera import CameraUtils
from pyslam.utilities.file_management import mkdir_for, hash_md5
from pyslam.utilities.torch import to_cpu

# PairOfSlices will be imported locally where needed to avoid circular dependency


class CanonicalViewUtils:
    """Canonical view processing and correspondence utilities."""

    @staticmethod
    def convert_dust3r_pairs_naming(imgs, pairs_in):
        """Convert Dust3r pair naming convention to use image instances.

        This function updates the "instance" field in pairs_in to use the actual
        image paths from the imgs list, converting from index-based to path-based naming.

        Args:
            imgs: List of image paths
            pairs_in: List of image pairs, each containing dicts with "idx" and "instance" fields

        Returns:
            Modified pairs_in list with updated "instance" fields
        """
        for pair_id in range(len(pairs_in)):
            for i in range(2):
                pairs_in[pair_id][i]["instance"] = imgs[pairs_in[pair_id][i]["idx"]]
        return pairs_in

    @staticmethod
    def load_corres(path_corres, device, min_conf_thr):
        """Load correspondences from a cached file and filter by confidence.

        Args:
            path_corres: Path to correspondence cache file
            device: Device for loaded tensors
            min_conf_thr: Minimum confidence threshold for filtering

        Returns:
            Tuple of (score, (xy1, xy2, confs)) where:
                score: Pairwise matching score
                xy1: Corresponding points in first image (N, 2)
                xy2: Corresponding points in second image (N, 2)
                confs: Confidence values (N,)
        """
        score, (xy1, xy2, confs) = torch.load(path_corres, map_location=device)
        valid = confs > min_conf_thr if min_conf_thr else slice(None)
        return score, (xy1[valid], xy2[valid], confs[valid])

    @staticmethod
    def canonical_view(ptmaps11, confs11, subsample, mode="avg-angle"):
        """Compute canonical view from multiple point maps and confidence maps.

        This function aggregates multiple point maps into a canonical representation
        using weighted averaging. It supports two modes: "avg-angle" (angle-based)
        and "avg-reldepth" (relative depth-based).

        Args:
            ptmaps11: List of point maps (N, H, W, 3)
            confs11: List of confidence maps (N, H, W)
            subsample: Subsampling factor for anchor points
            mode: Aggregation mode ("avg-angle" or "avg-reldepth") (default: "avg-angle")

        Returns:
            Tuple of (canon, canon2, confs) where:
                canon: Canonical point map (H, W, 3)
                canon2: Canonical depth offset map (H, W)
                confs: Aggregated confidence map (H, W)
        """
        assert len(ptmaps11) == len(confs11) > 0, "not a single view1 for img={i}"

        # canonical pointmap is just a weighted average
        confs11 = confs11.unsqueeze(-1) - 0.999
        canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

        canon_depth = ptmaps11[..., 2].unsqueeze(1)
        S = slice(subsample // 2, None, subsample)
        center_depth = canon_depth[:, :, S, S]
        center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

        stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
        stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

        if mode == "avg-reldepth":
            rel_depth = stacked_depth / center_depth
            stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
            canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

        elif mode == "avg-angle":
            xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
            stacked_xy = F.pixel_unshuffle(xy, subsample)
            B, _, H, W = stacked_xy.shape
            stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
            stacked_radius.clip_(min=1e-8)

            stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
            avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

            # back to depth
            stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

            canon2 = F.pixel_shuffle(
                (1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample
            ).squeeze()
        else:
            raise ValueError(f"bad {mode=}")

        confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
        return canon, canon2, confs

    @staticmethod
    def anchor_depth_offsets(canon_depth, pixels, subsample=8):
        """Compute depth offsets for pixels relative to anchor points.

        This function creates a grid of anchor points and computes relative depth
        offsets for given pixel coordinates. The anchors are placed on a subsampled grid.

        Args:
            canon_depth: Canonical depth map (H, W)
            pixels: Dict mapping image index to (pixel_coords, confs) tuples
            subsample: Subsampling factor for anchor grid (default: 8)

        Returns:
            Tuple of (core_idxs, core_offs) where:
                core_idxs: Dict mapping image index to anchor indices
                core_offs: Dict mapping image index to depth offset ratios
        """
        device = canon_depth.device

        # create a 2D grid of anchor 3D points
        H1, W1 = canon_depth.shape
        yx = np.mgrid[subsample // 2 : H1 : subsample, subsample // 2 : W1 : subsample]
        H2, W2 = yx.shape[1:]
        cy, cx = yx.reshape(2, -1)
        core_depth = canon_depth[cy, cx]
        assert (core_depth > 0).all()

        # slave 3d points (attached to core 3d points)
        core_idxs = {}
        core_offs = {}

        for img2, (xy1, _confs) in pixels.items():
            px, py = xy1.long().T

            # find nearest anchor == block quantization
            core_idx = (py // subsample) * W2 + (px // subsample)
            core_idxs[img2] = core_idx.to(device)

            # compute relative depth offsets w.r.t. anchors
            ref_z = core_depth[core_idx]
            pts_z = canon_depth[py, px]
            offset = pts_z / ref_z
            core_offs[img2] = offset.detach().to(device)

        return core_idxs, core_offs

    @staticmethod
    @torch.no_grad()
    def prepare_canonical_data(
        imgs,
        tmp_pairs,
        subsample,
        order_imgs=False,
        min_conf_thr=0,
        cache_path=None,
        device="cuda",
        **kw,
    ):
        """Prepare canonical views and correspondence data for sparse optimization.

        This function processes pairwise predictions to create canonical views for
        each image, computes focal lengths, and extracts depth offsets. It can
        cache results to disk for faster subsequent runs.

        Args:
            imgs: List of image paths
            tmp_pairs: Dict mapping (img1, img2) to ((path1, path2), path_corres)
            subsample: Subsampling factor for canonical views
            order_imgs: Whether to order images (default: False)
            min_conf_thr: Minimum confidence threshold (default: 0)
            cache_path: Optional path for caching canonical views
            device: Device for tensors (default: "cuda")
            **kw: Additional keyword arguments passed to canonical_view

        Returns:
            Tuple of (tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21)
        """
        canonical_views = {}
        pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
        canonical_paths = []
        preds_21 = {}

        for img in tqdm.tqdm(imgs):
            if cache_path:
                cache = os.path.join(
                    cache_path, "canon_views", hash_md5(img) + f"_{subsample=}_{kw=}.pth"
                )
                canonical_paths.append(cache)
            try:
                (canon, canon2, cconf), focal = torch.load(cache, map_location=device)
            except IOError:
                canon = focal = None

            # collect all pred1
            n_pairs = sum((img in pair) for pair in tmp_pairs)

            ptmaps11 = None
            pixels = {}
            n = 0
            for (img1, img2), ((path1, path2), path_corres) in tmp_pairs.items():
                score = None
                if img == img1:
                    X, C, X2, C2 = torch.load(path1, map_location=device)
                    score, (xy1, xy2, confs) = CanonicalViewUtils.load_corres(
                        path_corres, device, min_conf_thr
                    )
                    pixels[img2] = xy1, confs
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img2] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].ravel(),
                    )

                if img == img2:
                    if path2 is None:
                        # No reverse pair available - try to use path1 with swapped data
                        # This can happen with Dust3r when symmetrize=True doesn't create reverse pairs
                        # Load path1 and swap X1/X2 and C1/C2 to get reverse direction
                        X1_loaded, C1_loaded, X2_loaded, C2_loaded = torch.load(
                            path1, map_location=device
                        )
                        # For reverse: X = X2 (pts3d for img2), C = C2, X2 = X1 (pts3d for img1), C2 = C1
                        X, C, X2, C2 = X2_loaded, C2_loaded, X1_loaded, C1_loaded
                        # Try to find reverse correspondence file by swapping indices in filename
                        # Extract base directory and filename
                        corres_dir = os.path.dirname(path_corres)
                        corres_filename = os.path.basename(path_corres)
                        # Try to extract indices from filename (format: "idx1-idx2.pth")
                        # If filename contains "-", try swapping
                        if "-" in corres_filename:
                            parts = corres_filename.rsplit(".", 1)[0].split("-", 1)
                            if len(parts) == 2:
                                idx1_str, idx2_str = parts
                                corres_path_rev = os.path.join(
                                    corres_dir, f"{idx2_str}-{idx1_str}.pth"
                                )
                                if os.path.exists(corres_path_rev):
                                    score, (xy1_rev, xy2_rev, confs_rev) = (
                                        CanonicalViewUtils.load_corres(
                                            corres_path_rev, device, min_conf_thr
                                        )
                                    )
                                    pixels[img1] = (
                                        xy2_rev,
                                        confs_rev,
                                    )  # For img2, pixels[img1] uses xy2_rev
                                else:
                                    # Use forward correspondences but swap xy1 and xy2
                                    score, (xy1_fwd, xy2_fwd, confs_fwd) = (
                                        CanonicalViewUtils.load_corres(
                                            path_corres, device, min_conf_thr
                                        )
                                    )
                                    pixels[img1] = (
                                        xy2_fwd,
                                        confs_fwd,
                                    )  # Swap: use xy2_fwd for img1 when viewing from img2
                            else:
                                # Fallback: use forward correspondences with swapped xy
                                score, (xy1_fwd, xy2_fwd, confs_fwd) = (
                                    CanonicalViewUtils.load_corres(
                                        path_corres, device, min_conf_thr
                                    )
                                )
                                pixels[img1] = xy2_fwd, confs_fwd
                        else:
                            # Fallback: use forward correspondences with swapped xy
                            score, (xy1_fwd, xy2_fwd, confs_fwd) = CanonicalViewUtils.load_corres(
                                path_corres, device, min_conf_thr
                            )
                            pixels[img1] = xy2_fwd, confs_fwd
                    else:
                        X, C, X2, C2 = torch.load(path2, map_location=device)
                        score, (xy1, xy2, confs) = CanonicalViewUtils.load_corres(
                            path_corres, device, min_conf_thr
                        )
                        pixels[img1] = xy2, confs
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img1] = (
                        X2[::subsample, ::subsample].reshape(-1, 3),
                        C2[::subsample, ::subsample].ravel(),
                    )

                if score is not None:
                    i, j = imgs.index(img1), imgs.index(img2)
                    score = score[2]
                    pairwise_scores[i, j] = score
                    pairwise_scores[j, i] = score

                    if canon is not None:
                        continue
                    if ptmaps11 is None:
                        H, W = C.shape
                        ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                        confs11 = torch.empty((n_pairs, H, W), device=device)

                    ptmaps11[n] = X
                    confs11[n] = C
                    n += 1

            if canon is None:
                # Filter out verbose from kw as canonical_view doesn't accept it
                kw_filtered = {k: v for k, v in kw.items() if k != "verbose"}
                canon, canon2, cconf = CanonicalViewUtils.canonical_view(
                    ptmaps11, confs11, subsample, **kw_filtered
                )
                del ptmaps11
                del confs11

            # compute focals
            H, W = canon.shape[:2]
            pp = torch.tensor([W / 2, H / 2], device=device)
            if focal is None:
                focal = CameraUtils.estimate_focal_knowing_depth(
                    canon[None], pp, focal_mode="weiszfeld", min_focal=0.5, max_focal=3.5
                )
                if cache:
                    torch.save(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))

            # extract depth offsets with correspondences
            core_depth = canon[subsample // 2 :: subsample, subsample // 2 :: subsample, 2]
            idxs, offsets = CanonicalViewUtils.anchor_depth_offsets(
                canon2, pixels, subsample=subsample
            )

            # Debug: check depth initialization
            if kw.get("verbose", False):
                print(
                    f"[prepare_canonical_data] Image {img}: core_depth shape={core_depth.shape}, median={core_depth.median().item():.6f}, min={core_depth.min().item():.6f}, max={core_depth.max().item():.6f}"
                )
                print(
                    f"  canon shape={canon.shape}, canon2 shape={canon2.shape if canon2 is not None else None}"
                )

            canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

        return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21

    @staticmethod
    def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32):
        """Condense and aggregate canonical data into optimization-ready format.

        This function aggregates principal points, shapes, focals, core depths,
        anchors, and correspondences from canonical views into structured data
        suitable for sparse scene optimization.

        Args:
            imgs: List of image paths
            tmp_paths: List of (img1, img2) tuples representing pairs
            canonical_views: Dict mapping image paths to canonical view data
            preds_21: Dict mapping image paths to predictions
            dtype: Data type for tensors (default: torch.float32)

        Returns:
            Tuple of (imsizes, principal_points, focals, core_depth, img_anchors,
                      corres, corres2d, subsamp_preds_21)
        """
        # aggregate all data properly
        set_imgs = set(imgs)

        principal_points = []
        shapes = []
        focals = []
        core_depth = []
        img_anchors = {}
        tmp_pixels = {}

        for idx1, img1 in enumerate(imgs):
            # load stuff
            pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views[img1]

            principal_points.append(pp)
            shapes.append(shape)
            focals.append(focal)
            core_depth.append(anchors)

            img_uv1 = []
            img_idxs = []
            img_offs = []
            cur_n = [0]

            for img2, (pixels, match_confs) in pixels_confs.items():
                if img2 not in set_imgs:
                    continue
                assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
                img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
                img_idxs.append(idxs[img2])
                img_offs.append(offsets[img2])
                cur_n.append(cur_n[-1] + len(pixels))
                tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
            img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))

        all_confs = []
        imgs_slices = []
        corres2d = {img: [] for img in range(len(imgs))}

        for img1, img2 in tmp_paths:
            try:
                pix1, confs1, slice1 = tmp_pixels[img1, img2]
                pix2, confs2, slice2 = tmp_pixels[img2, img1]
            except KeyError:
                continue
            img1 = imgs.index(img1)
            img2 = imgs.index(img2)
            confs = (confs1 * confs2).sqrt()

            # prepare for loss_3d
            all_confs.append(confs)
            anchor_idxs1 = canonical_views[imgs[img1]][5][imgs[img2]]
            anchor_idxs2 = canonical_views[imgs[img2]][5][imgs[img1]]
            from .sparse_ga import PairOfSlices

            imgs_slices.append(
                PairOfSlices(
                    img1,
                    slice1,
                    pix1,
                    anchor_idxs1,
                    img2,
                    slice2,
                    pix2,
                    anchor_idxs2,
                    confs,
                    float(confs.sum()),
                )
            )

            # prepare for loss_2d
            corres2d[img1].append((pix1, confs, img2, slice2))
            corres2d[img2].append((pix2, confs, img1, slice1))

        all_confs = torch.cat(all_confs)
        corres = (all_confs, float(all_confs.sum()), imgs_slices)

        def aggreg_matches(img1, list_matches):
            pix1, confs, img2, slice2 = zip(*list_matches)
            all_pix1 = torch.cat(pix1).to(dtype)
            all_confs = torch.cat(confs).to(dtype)
            return (
                img1,
                all_pix1,
                all_confs,
                float(all_confs.sum()),
                [(j, sl2) for j, sl2 in zip(img2, slice2)],
            )

        corres2d = [aggreg_matches(img, m) for img, m in corres2d.items()]

        imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)
        principal_points = torch.stack(principal_points)
        focals = torch.cat(focals)

        # Subsample preds_21
        subsamp_preds_21 = {}
        for imk, imv in preds_21.items():
            subsamp_preds_21[imk] = {}
            for im2k, (pred, conf) in preds_21[imk].items():
                idxs = img_anchors[imgs.index(im2k)][1]
                subsamp_preds_21[imk][im2k] = (pred[idxs], conf[idxs])

        return (
            imsizes,
            principal_points,
            focals,
            core_depth,
            img_anchors,
            corres,
            corres2d,
            subsamp_preds_21,
        )
