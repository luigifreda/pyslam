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
Sparse Global Alignment and sparse scene optimization utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from copy import deepcopy
import copy
import scipy.sparse as sp
import tqdm
import roma
import torch.nn.functional as F

from .geometry import GeometryUtils, ProjectionUtils
from .camera import CameraUtils, PoseInitialization
from .pointcloud import PointCloudUtils
from .loss import LossFunctions
from .spectral import SpectralUtils

# CanonicalViewUtils imported locally where needed to avoid circular dependency
from ..optimizers.learning_rate_schedules import LearningRateSchedules
from pyslam.utilities.torch import to_numpy, to_cpu, to_device


# Named tuple for image pairs
PairOfSlices = namedtuple(
    "ImgPair",
    "img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum",
)


class EdgeUtils:
    """Edge and graph edge utility functions."""

    @staticmethod
    def edge_str(i, j):
        """Convert edge indices to string representation.

        Args:
            i: First image index
            j: Second image index

        Returns:
            String representation of edge "i_j"
        """
        return f"{i}_{j}"

    @staticmethod
    def i_j_ij(ij):
        """Convert edge tuple to string and return both.

        Args:
            ij: Edge tuple (i, j)

        Returns:
            Tuple of (edge_string, edge_tuple)
        """
        return EdgeUtils.edge_str(*ij), ij

    @staticmethod
    def edge_conf(conf_i, conf_j, edge):
        """Compute edge confidence from two confidence maps.

        Args:
            conf_i: Confidence map for view i
            conf_j: Confidence map for view j
            edge: Edge string identifier

        Returns:
            Combined confidence score (float)
        """
        return float(conf_i[edge].mean() * conf_j[edge].mean())

    @staticmethod
    def compute_edge_scores(edges, conf_i, conf_j):
        """Compute confidence scores for all edges.

        Args:
            edges: Iterable of (edge_string, (i, j)) tuples
            conf_i: Dictionary of confidence maps for view i
            conf_j: Dictionary of confidence maps for view j

        Returns:
            Dictionary mapping (i, j) tuples to confidence scores
        """
        return {(i, j): EdgeUtils.edge_conf(conf_i, conf_j, e) for e, (i, j) in edges}


class GraphUtils:
    """Graph algorithms and utilities."""

    @staticmethod
    def bfs(tree, start_node):
        """Perform breadth-first search on a graph.

        Args:
            tree: Sparse graph matrix
            start_node: Starting node index

        Returns:
            Tuple of (ranks, predecessors) where:
                ranks: Node ranks in BFS order
                predecessors: Predecessor array for each node
        """
        order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
        ranks = np.arange(len(order))
        ranks[order] = ranks.copy()
        return ranks, predecessors

    @staticmethod
    def compute_min_spanning_tree(pws):
        """Compute minimum spanning tree from pairwise scores and find root node.

        This function builds a minimum spanning tree from pairwise scores, then
        finds the central node (farthest from any leaf) to use as the root.
        Returns the root index and ordered list of MST edges.

        Args:
            pws: Pairwise scores matrix (N, N)

        Returns:
            Tuple of (root_idx, edges) where:
                root_idx: Index of the root node
                edges: List of (parent, child) tuples in BFS order
        """
        sparse_graph = sp.dok_array(pws.shape)
        for i, j in pws.nonzero().cpu().tolist():
            sparse_graph[i, j] = -float(pws[i, j])
        msp = sp.csgraph.minimum_spanning_tree(sparse_graph)

        # now reorder the oriented edges, starting from the central point
        ranks1, _ = GraphUtils.bfs(msp, 0)
        ranks2, _ = GraphUtils.bfs(msp, ranks1.argmax())
        ranks1, _ = GraphUtils.bfs(msp, ranks2.argmax())
        # this is the point farther from any leaf
        root = np.minimum(ranks1, ranks2).argmax()

        # find the ordered list of edges that describe the tree
        order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
        order = order[1:]
        edges = [(predecessors[i], i) for i in order]

        return root, edges


class SparseGA:
    """Sparse Global Alignment class for managing sparse scene optimization results.

    This class stores and provides access to optimized camera poses, depthmaps,
    and 3D points from sparse scene optimization. It can also generate dense
    point clouds from cached canonical views.

    Args:
        img_paths: List of image file paths
        pairs_in: List of image pairs with their data
        res_fine: Dictionary with optimization results (intrinsics, cam2w, depthmaps, pts3d)
        anchors: List of anchor points for each image
        canonical_paths: Optional list of paths to cached canonical views
    """

    def __init__(self, img_paths, pairs_in, res_fine, anchors, canonical_paths=None):
        def torgb(x):
            return (x[0].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(min=0.0, max=1.0)

        # Build a mapping from image instance to image tensor
        # This ensures each image is fetched correctly and uniquely
        img_dict = {}
        for im1, im2 in pairs_in:
            if im1["instance"] not in img_dict:
                img_dict[im1["instance"]] = torgb(im1["img"])
            if im2["instance"] not in img_dict:
                img_dict[im2["instance"]] = torgb(im2["img"])

        self.canonical_paths = canonical_paths
        self.img_paths = img_paths
        # Fetch images in order, ensuring each img_path gets its corresponding image
        self.imgs = []
        missing = []
        for img_path in img_paths:
            img = img_dict.get(img_path, None)
            if img is None:
                missing.append(img_path)
            self.imgs.append(img)

        # Verify all images were found
        if missing:
            raise ValueError(f"Could not find images for: {missing} in pairs_in")
        self.intrinsics = res_fine["intrinsics"]
        self.cam2w = res_fine["cam2w"]
        self.depthmaps = res_fine["depthmaps"]
        self.pts3d = res_fine["pts3d"]
        self.pts3d_colors = []
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            # Convert to integers for array indexing
            x = x.astype(np.int64)
            y = y.astype(np.int64)
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        """Get focal lengths for all cameras.

        Returns:
            Tensor of focal lengths (N,)
        """
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        """Get principal points for all cameras.

        Returns:
            Tensor of principal points (N, 2)
        """
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        """Get camera-to-world poses for all cameras.

        Returns:
            Tensor of camera poses (N, 4, 4)
        """
        return self.cam2w

    def get_sparse_pts3d(self):
        """Get sparse 3D points for all images.

        Returns:
            List of 3D point clouds, one per image
        """
        return self.pts3d

    def get_dense_pts3d(self, clean_depth=True, subsample=8):
        """Generate dense 3D points from cached canonical views.

        This function loads canonical views from cache and generates dense
        point clouds by densifying the sparse depthmaps.

        Args:
            clean_depth: Whether to clean the point cloud (default: True)
            subsample: Subsampling factor for anchor points (default: 8)

        Returns:
            Tuple of (pts3d, depthmaps, confs) where:
                pts3d: List of dense 3D point clouds
                depthmaps: List of dense depth maps
                confs: List of confidence maps
        """
        assert self.canonical_paths, "cache_path is required for dense 3d points"
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = torch.load(canon_path, map_location=device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            from .canonical_views import CanonicalViewUtils

            idxs, offsets = CanonicalViewUtils.anchor_depth_offsets(
                canon2, {i: (pixels, None)}, subsample=subsample
            )
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = ProjectionUtils.make_pts3d(
            anchors,
            self.intrinsics,
            self.cam2w,
            [d.ravel() for d in self.depthmaps],
            base_focals=base_focals,
            ret_depth=True,
        )

        if clean_depth:
            confs = PointCloudUtils.clean_pointcloud(
                confs, self.intrinsics, GeometryUtils.inv(self.cam2w), depthmaps, pts3d
            )

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        """Get colors for sparse 3D points.

        Returns:
            List of color arrays, one per image
        """
        return self.pts3d_colors

    def get_depthmaps(self):
        """Get sparse depthmaps for all images.

        Returns:
            List of depth maps, one per image
        """
        return self.depthmaps

    def get_masks(self):
        """Get masks for all images (currently returns full masks).

        Returns:
            List of slice objects, one per image
        """
        return [slice(None, None) for _ in range(len(self.imgs))]


def _extract_initial_poses_from_pairwise(
    filelist,
    pairs_output,
    imsizes,
    base_focals,
    pps,
    mst,
    subsample,
    cache_dir,
    device,
    dtype,
    verbose=True,
):
    """
    Extract initial camera poses from pairwise predictions using MST.

    Args:
        filelist: List of image file names
        pairs_output: Dictionary mapping (img1, img2) -> ((path1, path2), path_corres) with cached pair data
        imsizes: Tensor of image sizes (W, H) for each image
        base_focals: Base focal lengths
        pps: Principal points
        mst: Minimum spanning tree (root_idx, [(parent, child), ...])
        subsample: Subsampling factor (not used here, but kept for compatibility)
        cache_dir: Cache directory containing pair data files
        device: Device for tensors
        dtype: Data type for tensors
        verbose: Whether to print debug messages

    Returns:
        rel_quats: List of relative quaternions for each camera (along MST)
        rel_trans: List of relative translations for each camera (along MST)
        abs_cam2w: Absolute camera-to-world poses (for reference)
    """
    n_imgs = len(filelist)

    # Build edges from pairs_output - load full-resolution point clouds from cache
    edges = []
    pred_i_dict = {}
    pred_j_dict = {}
    conf_i_dict = {}
    conf_j_dict = {}

    if pairs_output is None:
        if verbose:
            print("Warning: pairs_output is None, cannot extract initial poses")
        return None, None, None

    for pair_key, (pair_paths, corres_path) in pairs_output.items():
        img1_name, img2_name = pair_key
        try:
            img1_idx = filelist.index(img1_name)
            img2_idx = filelist.index(img2_name)
        except ValueError:
            # Image not in filelist, skip
            continue

        # Load cached pair data: (X1, C1, X2, C2)
        # X1: pts3d for view1 (full resolution), C1: conf for view1
        # X2: pts3d for view2 (in view1's frame), C2: conf for view2
        try:
            cache_path1, cache_path2 = pair_paths
            X1, C1, X2, C2 = torch.load(cache_path1, map_location=device)

            # Ensure tensors are on correct device and dtype
            if isinstance(X1, torch.Tensor):
                X1 = X1.to(device=device, dtype=dtype)
            else:
                X1 = torch.tensor(X1, device=device, dtype=dtype)

            if isinstance(C1, torch.Tensor):
                C1 = C1.to(device=device, dtype=dtype)
            else:
                C1 = torch.tensor(C1, device=device, dtype=dtype)

            if isinstance(X2, torch.Tensor):
                X2 = X2.to(device=device, dtype=dtype)
            else:
                X2 = torch.tensor(X2, device=device, dtype=dtype)

            if isinstance(C2, torch.Tensor):
                C2 = C2.to(device=device, dtype=dtype)
            else:
                C2 = torch.tensor(C2, device=device, dtype=dtype)

            edge_key = (img1_idx, img2_idx)
            if edge_key not in edges:
                edges.append(edge_key)

            # X1 is the point cloud for view1 (img1), X2 is for view2 (img2) in view1's frame
            pred_i_dict[edge_key] = X1
            pred_j_dict[edge_key] = X2
            conf_i_dict[edge_key] = C1
            conf_j_dict[edge_key] = C2

        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load pair data for {img1_name}-{img2_name}: {e}")
            continue

    if len(edges) == 0:
        if verbose:
            print("Warning: No pairwise predictions found, using identity poses")
        return None, None, None

    # Use minimum_spanning_tree to extract initial poses
    # Convert to format expected by minimum_spanning_tree
    # Note: imsizes is (W, H), but imshapes expects (H, W)
    imshapes_full = [(int(h), int(w)) for w, h in imsizes]

    # Build pred_i and pred_j dictionaries in the format expected by minimum_spanning_tree
    # The function expects EdgeUtils.edge_str(i, j) format for keys and (H, W, 3) point clouds
    pred_i = {}
    pred_j = {}
    conf_i = {}
    conf_j = {}
    imshapes_used = []

    for i, j in edges:
        # Use EdgeUtils.edge_str function format: "i_j" string
        key = EdgeUtils.edge_str(i, j)

        # Point clouds from cache are full resolution (H, W, 3) tensors
        pts3d_i_tensor = pred_i_dict[(i, j)]
        pts3d_j_tensor = pred_j_dict.get((i, j), pred_i_dict[(i, j)])
        conf_i_tensor = conf_i_dict[(i, j)]
        conf_j_tensor = conf_j_dict.get((i, j), conf_i_dict[(i, j)])

        # Determine actual dimensions from point cloud shape
        if len(pts3d_i_tensor.shape) == 3:
            # Already in (H, W, 3) format - use directly
            H_used, W_used = pts3d_i_tensor.shape[:2]
            pred_i[key] = pts3d_i_tensor
            pred_j[key] = pts3d_j_tensor

            # Reshape confidence maps if needed
            if len(conf_i_tensor.shape) == 2:
                conf_i_reshaped = conf_i_tensor
                conf_j_reshaped = conf_j_tensor
            else:
                conf_i_reshaped = conf_i_tensor.reshape(H_used, W_used)
                conf_j_reshaped = conf_j_tensor.reshape(H_used, W_used)

            conf_i[key] = conf_i_reshaped
            conf_j[key] = conf_j_reshaped

            # Store dimensions for this image
            if i < len(imshapes_used):
                if H_used * W_used > imshapes_used[i][0] * imshapes_used[i][1]:
                    imshapes_used[i] = (H_used, W_used)
            else:
                while len(imshapes_used) <= i:
                    H_full, W_full = imshapes_full[len(imshapes_used)]
                    imshapes_used.append((H_full, W_full))
                imshapes_used[i] = (H_used, W_used)

            continue  # Skip to next edge since we've already processed this one
        else:
            # Flattened point cloud - need to reshape
            pts3d_i_flat = pts3d_i_tensor
            pts3d_j_flat = pts3d_j_tensor
            conf_i_flat = conf_i_tensor
            conf_j_flat = conf_j_tensor

            H_full, W_full = imshapes_full[i]
            num_points = pts3d_i_flat.shape[0]

            # Check if it matches full resolution
            if num_points == H_full * W_full:
                H_used, W_used = H_full, W_full
            # Check if it matches subsampled resolution
            elif num_points == (H_full // subsample) * (W_full // subsample):
                H_used, W_used = H_full // subsample, W_full // subsample
            else:
                # Try to infer dimensions from the point cloud size
                # Find factors that could represent H and W
                # Common aspect ratios: 4:3, 16:9, 1:1
                if verbose:
                    print(
                        f"Warning: Point cloud size {num_points} doesn't match expected dimensions for edge {i}-{j}"
                    )
                    print(f"  Full resolution: {H_full}x{W_full} = {H_full * W_full}")
                    print(
                        f"  Subsampled ({subsample}x): {H_full // subsample}x{W_full // subsample} = {(H_full // subsample) * (W_full // subsample)}"
                    )

                # Try to find reasonable H, W that multiply to num_points
                # Use the full resolution aspect ratio as a guide
                aspect_ratio = W_full / H_full if H_full > 0 else 1.0

                # Find H such that H * W = num_points and W/H ≈ aspect_ratio
                # Start with a guess based on aspect ratio
                H_guess = int((num_points / aspect_ratio) ** 0.5)
                W_guess = int(num_points / H_guess) if H_guess > 0 else int(num_points**0.5)

                # Adjust to make sure H * W == num_points exactly
                # Try nearby values to find exact factors
                best_H, best_W = None, None
                min_aspect_diff = float("inf")

                # Try values around the guess
                for H_test in range(
                    max(1, H_guess - 10), min(H_guess + 10, int(num_points**0.5) + 1)
                ):
                    if num_points % H_test == 0:
                        W_test = num_points // H_test
                        aspect_diff = abs(W_test / H_test - aspect_ratio)
                        if aspect_diff < min_aspect_diff:
                            min_aspect_diff = aspect_diff
                            best_H, best_W = H_test, W_test

                if best_H is not None and best_W is not None:
                    H_used, W_used = best_H, best_W
                    if verbose:
                        print(
                            f"  Inferred dimensions: {H_used}x{W_used} (aspect ratio: {W_used/H_used:.2f}, target: {aspect_ratio:.2f})"
                        )
                else:
                    # Last resort: try to find any factors
                    # Find the largest factor close to sqrt(num_points)
                    sqrt_n = int(num_points**0.5)
                    for H_test in range(sqrt_n, 0, -1):
                        if num_points % H_test == 0:
                            W_test = num_points // H_test
                            H_used, W_used = H_test, W_test
                            if verbose:
                                print(f"  Inferred dimensions (fallback): {H_used}x{W_used}")
                            break
                    else:
                        # Skip this edge if we can't determine dimensions
                        if verbose:
                            print(f"  Skipping edge {i}-{j} - cannot factor {num_points}")
                        continue

                # Reshape to (H, W, 3) format expected by minimum_spanning_tree
                try:
                    pred_i[key] = pts3d_i_flat.reshape(H_used, W_used, 3)
                    pred_j[key] = pts3d_j_flat.reshape(H_used, W_used, 3)
                    conf_i[key] = conf_i_flat.reshape(H_used, W_used)
                    conf_j[key] = conf_j_flat.reshape(H_used, W_used)
                except RuntimeError as e:
                    if verbose:
                        print(f"Warning: Failed to reshape point cloud for edge {i}-{j}: {e}")
                    continue

            # Store the dimensions used for this image
            if i < len(imshapes_used):
                # Update if we have a better match
                if H_used * W_used > imshapes_used[i][0] * imshapes_used[i][1]:
                    imshapes_used[i] = (H_used, W_used)
            else:
                # Extend list if needed
                while len(imshapes_used) <= i:
                    H_full, W_full = imshapes_full[len(imshapes_used)]
                    imshapes_used.append((H_full, W_full))
                imshapes_used[i] = (H_used, W_used)

    if len(pred_i) == 0:
        if verbose:
            print("Warning: No valid edges after reshaping, using identity poses")
        return None, None, None

    # Use the determined dimensions for im_conf
    # If we didn't determine dimensions for some images, use full resolution
    imshapes_final = []
    for idx in range(n_imgs):
        if idx < len(imshapes_used):
            imshapes_final.append(imshapes_used[idx])
        else:
            H_full, W_full = imshapes_full[idx]
            imshapes_final.append((H_full, W_full))

    # Create dummy im_conf (not used for pose extraction)
    im_conf = [torch.ones((h, w), device=device) for h, w in imshapes_final]

    # Extract initial poses using minimum_spanning_tree
    try:
        pts3d, msp_edges, im_focals, im_poses = PoseInitialization.minimum_spanning_tree(
            imshapes=imshapes_final,
            edges=edges,
            pred_i=pred_i,
            pred_j=pred_j,
            conf_i=conf_i,
            conf_j=conf_j,
            im_conf=im_conf,
            min_conf_thr=0.0,
            device=device,
            has_im_poses=True,
            niter_PnP=10,
            verbose=verbose,
        )

        if im_poses is None:
            return None, None, None

        # Convert absolute poses to relative poses along MST
        # mst is (root_idx, [(parent, child), ...])
        root_idx = mst[0]
        mst_edges = mst[1]

        # Initialize relative poses
        rel_quats = []
        rel_trans = []

        # Root camera has identity relative pose
        for i in range(n_imgs):
            if i == root_idx:
                rel_quats.append(torch.tensor([0, 0, 0, 1], dtype=dtype, device=device))
                rel_trans.append(torch.zeros(3, dtype=dtype, device=device))
            else:
                # Will be set based on MST
                rel_quats.append(torch.tensor([0, 0, 0, 1], dtype=dtype, device=device))
                rel_trans.append(torch.zeros(3, dtype=dtype, device=device))

        # Build parent map for MST
        parent_map = {root_idx: None}
        for parent, child in mst_edges:
            parent_map[child] = parent

        # Convert absolute poses to relative poses along MST
        abs_cam2w = im_poses.clone()

        if verbose:
            print("\n[Initial Pose Extraction] Absolute camera poses from MST:")
            for idx in range(n_imgs):
                pose = abs_cam2w[idx]
                trans = pose[:3, 3]
                rot_first_row = pose[0, :3]
                print(
                    f"  Camera {idx}: translation={trans.cpu().numpy()}, rotation[0]={rot_first_row.cpu().numpy()}"
                )
            print(f"  MST root: {root_idx}, edges: {mst_edges}")

        # For each camera, compute relative pose w.r.t. its parent
        for child_idx in range(n_imgs):
            if child_idx == root_idx:
                continue

            parent_idx = parent_map.get(child_idx)
            if parent_idx is None:
                # No parent found, use identity
                if verbose:
                    print(
                        f"  Warning: Camera {child_idx} has no parent in MST, using identity relative pose"
                    )
                continue

            # Compute relative transform: child_pose = parent_pose @ rel_pose
            # So: rel_pose = GeometryUtils.inv(parent_pose) @ child_pose
            parent_pose = abs_cam2w[parent_idx]
            child_pose = abs_cam2w[child_idx]

            rel_pose = torch.inverse(parent_pose) @ child_pose

            # Extract rotation and translation
            rel_rot = rel_pose[:3, :3]
            rel_t = rel_pose[:3, 3]

            # Convert to quaternion
            rel_quat = roma.rotmat_to_unitquat(rel_rot)

            rel_quats[child_idx] = rel_quat
            rel_trans[child_idx] = rel_t

            if verbose:
                print(
                    f"  Camera {child_idx} (parent={parent_idx}): rel_trans={rel_t.cpu().numpy()}, rel_quat={rel_quat.cpu().numpy()}"
                )

        if verbose:
            print("[Initial Pose Extraction] Relative poses summary:")
            for idx in range(n_imgs):
                if idx == root_idx:
                    print(f"  Camera {idx} (root): identity relative pose")
                else:
                    print(
                        f"  Camera {idx}: rel_trans={rel_trans[idx].cpu().numpy()}, rel_quat={rel_quats[idx].cpu().numpy()}"
                    )

        return rel_quats, rel_trans, abs_cam2w

    except Exception as e:
        if verbose:
            print(f"Warning: Failed to extract initial poses: {e}")
        return None, None, None


def sparse_scene_optimizer(
    imgs,
    subsample,
    imsizes,
    pps,
    base_focals,
    core_depth,
    anchors,
    corres,
    corres2d,
    preds_21,
    canonical_paths,
    mst,
    cache_path,
    lr1=0.07,
    niter1=300,
    loss1=None,
    lr2=0.01,
    niter2=300,
    loss2=None,
    lossd=None,
    opt_pp=True,
    opt_depth=True,
    schedule=LearningRateSchedules.cosine_schedule,
    depth_mode="add",
    exp_depth=False,
    lora_depth=False,
    shared_intrinsics=False,
    init={},
    device="cuda",
    dtype=torch.float32,
    matching_conf_thr=5.0,
    loss_dust3r_w=0.01,
    verbose=False,
    dbg=(),
    pairs_output=None,  # Optional: full-resolution pair data for initial pose extraction
):
    """Optimize sparse scene reconstruction using two-stage optimization.

    This function performs sparse scene optimization in two stages:
    1. Coarse optimization: Optimizes 3D point correspondences
    2. Fine optimization: Refines with 2D reprojection loss

    The optimization uses a kinematic chain representation where cameras are
    parameterized relative to their parent in the minimum spanning tree.

    Args:
        imgs: List of image paths
        subsample: Subsampling factor
        imsizes: Tensor of image sizes (N, 2) as (W, H)
        pps: List of principal points (N, 2)
        base_focals: List of base focal lengths (N,)
        core_depth: List of core depth maps
        anchors: Dict mapping image index to anchor data
        corres: Tuple of (all_confs, confs_sum, imgs_slices)
        corres2d: List of 2D correspondences per image
        preds_21: Dict mapping image paths to predictions
        canonical_paths: List of paths to cached canonical views
        mst: Tuple of (root_idx, edges) for minimum spanning tree
        cache_path: Path for caching intermediate results
        lr1: Learning rate for coarse stage (default: 0.07)
        niter1: Number of iterations for coarse stage (default: 300)
        loss1: Loss function for coarse stage (default: gamma_loss(1.5))
        lr2: Learning rate for fine stage (default: 0.01)
        niter2: Number of iterations for fine stage (default: 300)
        loss2: Loss function for fine stage (default: gamma_loss(0.5))
        lossd: Loss function for Dust3r predictions (default: gamma_loss(1.1))
        opt_pp: Whether to optimize principal points (default: True)
        opt_depth: Whether to optimize depthmaps (default: True)
        schedule: Learning rate schedule function (default: LearningRateSchedules.cosine_schedule)
        depth_mode: Depth parameterization mode ("add" or "mul") (default: "add")
        exp_depth: Whether to use exponential depth parameterization (default: False)
        lora_depth: Whether to use LoRA depth parameterization (default: False)
        shared_intrinsics: Whether to share intrinsics across images (default: False)
        init: Dict of initialization values per image (default: {})
        device: Device for computation (default: "cuda")
        dtype: Data type for tensors (default: torch.float32)
        matching_conf_thr: Confidence threshold for matching (default: 5.0)
        loss_dust3r_w: Weight for Dust3r loss term (default: 0.01)
        verbose: Whether to print debug information (default: False)
        dbg: Debug tuple (default: ())
        pairs_output: Optional full-resolution pair data for initial pose extraction

    Returns:
        Tuple of (imgs, res_coarse, res_fine) where:
            imgs: List of image paths
            res_coarse: Dict with coarse optimization results
            res_fine: Dict with fine optimization results or None
    """
    # Set default loss functions if not provided
    if loss1 is None:
        loss1 = LossFunctions.gamma_loss(1.5)
    if loss2 is None:
        loss2 = LossFunctions.gamma_loss(0.5)
    if lossd is None:
        lossd = LossFunctions.gamma_loss(1.1)

    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # Try to extract initial poses from pairwise predictions if not provided in init
    rel_quats_init = None
    rel_trans_init = None
    # Check if init already contains pose information
    has_init_poses = False
    if init:
        for img_key, init_val in init.items():
            if isinstance(init_val, dict):
                if "cam2w" in init_val or "rel_quat" in init_val:
                    has_init_poses = True
                    break

    if not has_init_poses:
        # Extract initial poses from pairwise predictions
        # Use pairs_output if available (full-resolution point clouds from cache)
        rel_quats_init = None
        rel_trans_init = None

        if pairs_output is not None:
            rel_quats_init, rel_trans_init, abs_cam2w = _extract_initial_poses_from_pairwise(
                filelist=imgs,
                pairs_output=pairs_output,
                imsizes=imsizes,
                base_focals=base_focals,
                pps=pps,
                mst=mst,
                subsample=subsample,
                cache_dir=cache_path,
                device=device,
                dtype=dtype,
                verbose=verbose,
            )

        if rel_quats_init is not None and verbose:
            print(
                f" >> Extracted initial poses from pairwise predictions for {len(rel_quats_init)} cameras"
            )

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f" >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}")

        K = init_values.get("intrinsics")
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]

        depth = init_values.get("depthmap")
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        if verbose:
            print(
                f"  Camera {idx}: core_depth shape={core_depth[idx].shape}, median={med_depth.item():.6f}, min={core_depth[idx].min().item():.6f}, max={core_depth[idx].max().item():.6f}"
            )

        # Check for relative pose initialization (preferred for kinematic chain)
        rel_quat = init_values.get("rel_quat")
        rel_t = init_values.get("rel_trans")
        if rel_quat is not None and rel_t is not None:
            quats[idx].data[:] = rel_quat.detach().to(dtype).to(device)
            trans[idx].data[:] = rel_t.detach().to(dtype).to(device)
        elif (
            rel_quats_init is not None and rel_trans_init is not None and idx < len(rel_quats_init)
        ):
            # Use extracted initial relative poses
            quats[idx].data[:] = rel_quats_init[idx].to(dtype).to(device)
            trans[idx].data[:] = rel_trans_init[idx].to(dtype).to(device)
            if verbose:
                print(f"  Camera {idx}: Initialized from extracted relative pose")
                print(f"    rel_trans={rel_trans_init[idx].cpu().numpy()}")
                print(f"    rel_quat={rel_quats_init[idx].cpu().numpy()}")
        else:
            # Fall back to absolute pose initialization (if provided)
            cam2w = init_values.get("cam2w")
            if cam2w is not None:
                rot = cam2w[:3, :3].detach()
                cam_center = cam2w[:3, 3].detach()
                quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
                trans_offset = med_depth * torch.cat(
                    (imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0])
                )
                trans[idx].data[:] = cam_center + rot @ trans_offset
                del rot
                if verbose:
                    print(
                        f"Warning: Initializing from absolute pose for camera {idx}, but kinematic chain conversion not fully implemented"
                    )
                # Note: This will initialize the root camera correctly, but child cameras
                # will need to be converted to relative poses along the MST

    # Debug: print initial quats and trans after initialization
    if verbose:
        print("\n[Sparse Optimizer] Initial quats and trans after initialization:")
        for idx in range(len(imgs)):
            print(
                f"  Camera {idx}: quat={quats[idx].data.detach().cpu().numpy()}, trans={trans[idx].data.detach().cpu().numpy()}"
            )

    # intrinsics parameters
    if shared_intrinsics:
        confs = torch.stack([torch.load(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags
    max_focals = 10 * diags

    assert len(mst[1]) == len(pps) - 1

    # Track if this is the first call to make_K_cam_depth for debugging
    make_K_cam_depth_call_count = [0]  # Use list to allow modification in nested function

    # Store initial relative translations for scale regularization
    initial_rel_trans = None
    if rel_trans_init is not None:
        # rel_trans_init is a list of tensors, convert to stacked tensor
        if isinstance(rel_trans_init, list):
            initial_rel_trans = torch.stack(rel_trans_init).clone().detach()
        else:
            initial_rel_trans = rel_trans_init.clone().detach()

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        nonlocal make_K_cam_depth_call_count
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        # Root camera's absolute pose is its relative transform (it's the reference)
        # For child cameras, absolute pose = parent's absolute pose @ child's relative transform
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # Debug: print initial absolute poses computed from kinematic chain (only first time)
        if (
            verbose
            and trans is not None
            and quats is not None
            and make_K_cam_depth_call_count[0] == 0
        ):
            make_K_cam_depth_call_count[0] += 1
            print("\n[Sparse Optimizer] Initial absolute poses from kinematic chain:")
            for idx in range(len(K)):
                pose = tmp_cam2w[idx]
                trans_val = pose[:3, 3]
                rot_first_row = pose[0, :3]
                print(
                    f"  Camera {idx}: translation={trans_val.detach().cpu().numpy()}, rotation[0]={rot_first_row.detach().cpu().numpy()}"
                )
            print(f"  Root camera: {mst[0]}, MST edges: {mst[1]}")

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat(
            (imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1
        )
        new_trans = global_scaling * (
            tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1)
        )

        # Enforce root camera at origin with identity rotation
        # Root camera has identity relative pose, so its absolute pose should be identity
        root_idx = mst[0]
        if trans is not None and quats is not None:
            # Root camera should be at origin with identity rotation
            # Use clone() to avoid inplace modification that breaks gradients
            new_trans = new_trans.clone()
            new_trans[root_idx] = torch.zeros(3, 1, dtype=new_trans.dtype, device=new_trans.device)
            # Create a new rotation matrix tensor for root camera (avoid inplace modification)
            root_rot = torch.eye(3, dtype=tmp_cam2w.dtype, device=tmp_cam2w.device)
            # Clone tmp_cam2w and modify the clone to avoid breaking gradients
            tmp_cam2w_rot = tmp_cam2w[:, :3, :3].clone()
            tmp_cam2w_rot[root_idx] = root_rot
        else:
            tmp_cam2w_rot = tmp_cam2w[:, :3, :3]

        cam2w = torch.cat(
            (
                torch.cat((tmp_cam2w_rot, new_trans), dim=2),
                vec0001.view(1, 1, 4).expand(len(K), 1, 4),
            ),
            dim=1,
        )

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == "add":
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == "mul":
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f"Bad {depth_mode=}")
            depthmaps.append(global_scaling * core_depth_img)

        return K, (GeometryUtils.inv(cam2w), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics:
        print("init focal (shared) = ", to_numpy(K[0, 0, 0]).round(2))
    else:
        print("init focals =", to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = SpectralUtils.spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth
        )
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    core_depth = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres

    # Define which pairs are fine to use with matching
    def matching_check(x):
        return x.max() > matching_conf_thr

    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]

    if verbose:
        print(f"\n[Optimization Setup] Correspondence slices:")
        print(f"  Total slices: {len(imgs_slices)}")
        print(f"  loss3d_slices (matching OK): {len(loss3d_slices)}")
        print(f"  dust3r_slices (matching not OK): {len(dust3r_slices)}")
        if len(loss3d_slices) > 0:
            print(f"  loss3d_slices pairs: {[(s.img1, s.img2) for s in loss3d_slices]}")
        if len(loss3d_slices) == 0:
            print("  WARNING: No loss3d_slices! This may cause optimization issues.")
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    def loss_dust3r(cam2w, pts3d, pix_loss):
        loss = 0.0
        cf_sum = 0.0
        for s in dust3r_slices:
            if init[imgs[s.img1]].get("freeze") and init[imgs[s.img2]].get("freeze"):
                continue
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = GeometryUtils.geom_transform(GeometryUtils.inv(cam2w[s.img2]), tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0.0 else 0.0

    # Track loss_3d calls for debugging
    loss_3d_call_count = [0]

    def loss_3d(K, w2cam, pts3d, pix_loss):
        nonlocal loss_3d_call_count
        if any(v.get("freeze") for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get("freeze") and init[imgs[s.img2]].get("freeze"):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.0
            cf_sum = 1.0

        # Debug: check if loss is zero or very small (might indicate no correspondences)
        loss_3d_call_count[0] += 1
        if verbose and loss_3d_call_count[0] == 1:
            print(
                f"\n[loss_3d] First call: loss={loss.item() if isinstance(loss, torch.Tensor) else loss:.6f}, cf_sum={cf_sum.item() if isinstance(cf_sum, torch.Tensor) else cf_sum:.6f}, num_slices={len(loss3d_slices)}"
            )
            if len(pts3d_1) == 0:
                print("  WARNING: No point correspondences in loss_3d!")
            else:
                # Check if correspondences have reasonable distances
                pts3d_1_cat = torch.cat(pts3d_1) if isinstance(pts3d_1, list) else pts3d_1
                pts3d_2_cat = torch.cat(pts3d_2) if isinstance(pts3d_2, list) else pts3d_2
                if len(pts3d_1_cat) > 0 and len(pts3d_2_cat) > 0:
                    dists = torch.norm(pts3d_1_cat - pts3d_2_cat, dim=-1)
                    print(
                        f"  Correspondence distances: mean={dists.mean().item():.6f}, std={dists.std().item():.6f}, min={dists.min().item():.6f}, max={dists.max().item():.6f}"
                    )
                    print(
                        f"  pts3d_1 shape: {pts3d_1_cat.shape}, pts3d_2 shape: {pts3d_2_cat.shape}"
                    )

        return loss / cf_sum

    def loss_2d(K, w2cam, pts3d, pix_loss):
        proj_matrix = K @ w2cam[:, :3]
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get("freeze", 0) >= 1:
                continue
            pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(
                    pix1_filtered, ProjectionUtils.reproj2d(proj_matrix[img1], pts3d_in_img1)
                )
                npix += confs_filtered.sum()

        return loss / npix if npix != 0 else 0.0

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        # Exclude root camera's quat, trans, log_sizes, and log_focals from optimization
        root_idx = mst[0]
        params = (
            pps
            + [f for i, f in enumerate(log_focals) if i != root_idx]
            + [q for i, q in enumerate(quats) if i != root_idx]
            + [t for i, t in enumerate(trans) if i != root_idx]
            + [s for i, s in enumerate(log_sizes) if i != root_idx]
            + core_depth
        )
        if verbose:
            print(f"\n[Optimize Loop] Root camera {root_idx} excluded from optimization")
            print(
                f"  Total params: {len(params)} (excluding root camera quat/trans/log_sizes/log_focals)"
            )
        optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if "meta" in repr(pix_loss) else (lambda a: pix_loss)

        with tqdm.tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(
                    log_focals, pps, trans, quats, log_sizes, core_depth
                )
                pts3d = ProjectionUtils.make_pts3d(
                    anchors, K, cam2w, depthmaps, base_focals=base_focals
                )

                # Debug: check depth maps and 3D points at key iterations
                if verbose and (
                    iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0)
                ):
                    print(f"\n[Optimization iter {iter}/{niter}] Depth maps and 3D points:")
                    for idx in range(len(imgs)):
                        depth_val = depthmaps[idx]
                        if isinstance(depth_val, torch.Tensor):
                            depth_mean = depth_val.mean().item()
                            depth_std = depth_val.std().item()
                            depth_min = depth_val.min().item()
                            depth_max = depth_val.max().item()
                            print(
                                f"  Camera {idx}: depth mean={depth_mean:.6f}, std={depth_std:.6f}, min={depth_min:.6f}, max={depth_max:.6f}"
                            )
                        if idx < len(pts3d):
                            pts3d_val = pts3d[idx]
                            if isinstance(pts3d_val, torch.Tensor):
                                pts3d_mean = pts3d_val.mean(dim=0).detach().cpu().numpy()
                                pts3d_std = pts3d_val.std(dim=0).detach().cpu().numpy()
                                print(f"  Camera {idx}: pts3d mean={pts3d_mean}, std={pts3d_std}")

                if niter == 0:
                    break

                alpha = iter / niter
                lr = schedule(alpha, lr_base, lr_end)
                LearningRateSchedules.adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(
                    cam2w, pts3d, lossd
                )

                # Add scale regularization to prevent camera collapse
                # Penalize cameras getting too close to each other
                # Use initial relative translations as reference to maintain scale
                root_idx = mst[0]
                scale_reg = 0.0
                if len(imgs) > 1:
                    # Compute pairwise distances between cameras
                    cam_positions = cam2w[:, :3, 3]  # [n_cameras, 3]

                    # Get initial distances from initial relative translations
                    initial_distances = {}
                    if initial_rel_trans is not None:
                        for i in range(len(imgs)):
                            if i == root_idx:
                                continue
                            if i < len(initial_rel_trans):
                                initial_dist = torch.norm(initial_rel_trans[i])
                                initial_distances[i] = initial_dist

                    for i in range(len(imgs)):
                        if i == root_idx:
                            continue
                        # Distance from root camera
                        dist_to_root = torch.norm(cam_positions[i] - cam_positions[root_idx])

                        # Use initial distance as reference if available
                        if i in initial_distances:
                            initial_dist = initial_distances[i]
                            # Penalize if current distance is much smaller than initial (collapse)
                            # Use a threshold of 70% of initial distance to allow some refinement
                            if dist_to_root < initial_dist * 0.7:  # More than 30% collapse
                                collapse_ratio = dist_to_root / (initial_dist + 1e-8)
                                # Strong penalty that increases as collapse gets worse
                                scale_reg += (
                                    (initial_dist * 0.7 - dist_to_root) ** 2
                                    * 200.0
                                    * (1.0 - collapse_ratio)
                                )
                        else:
                            # Fallback: penalize if distance is too small (< 0.05)
                            if dist_to_root < 0.05:
                                scale_reg += (0.05 - dist_to_root) ** 2 * 100.0  # Increased penalty

                    # Also penalize cameras getting too close to each other
                    for i in range(len(imgs)):
                        for j in range(i + 1, len(imgs)):
                            if i == root_idx or j == root_idx:
                                continue
                            dist = torch.norm(cam_positions[i] - cam_positions[j])
                            if dist < 0.05:
                                scale_reg += (0.05 - dist) ** 2 * 50.0  # Increased penalty

                loss = loss + scale_reg

                # Debug: print scale regularization at key iterations
                if (
                    verbose
                    and scale_reg > 0
                    and (iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0))
                ):
                    print(f"  [Scale Reg] iter {iter}: scale_reg={scale_reg.item():.6f}")

                loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                # Also enforce root camera identity relative pose
                root_idx = mst[0]
                for i in range(len(imgs)):
                    if i == root_idx:
                        # Root camera: enforce identity relative pose (don't normalize first)
                        quats[i].data[:] = torch.tensor(
                            [0, 0, 0, 1], dtype=quats[i].dtype, device=quats[i].device
                        )
                        trans[i].data[:] = torch.zeros(
                            3, dtype=trans[i].dtype, device=trans[i].device
                        )
                    else:
                        # Normalize quaternions for non-root cameras
                        quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                if loss != loss:
                    break  # NaN loss

                # Debug: print poses at key iterations
                if verbose and (
                    iter == 0 or iter == niter - 1 or (iter % max(1, niter // 10) == 0)
                ):
                    print(f"\n[Optimization iter {iter}/{niter}] Camera poses:")
                    for idx in range(len(imgs)):
                        pose = cam2w[idx]
                        trans_val = pose[:3, 3]
                        rot_first_row = pose[0, :3]
                        print(
                            f"  Camera {idx}: translation={trans_val.detach().cpu().numpy()}, rotation[0]={rot_first_row.detach().cpu().numpy()}"
                        )
                    print(f"  Loss: {loss:.6f}, LR: {lr:.6f}")

                bar.set_postfix_str(f"{lr=:.4f}, {loss=:.3f}")
                bar.update(1)

        if niter:
            print(f">> final loss = {loss}")
        return dict(
            intrinsics=K.detach(),
            cam2w=cam2w.detach(),
            depthmaps=[d.detach() for d in depthmaps],
            pts3d=[p.detach() for p in pts3d],
        )

    # at start, don't optimize 3d points
    # Freeze root camera's relative pose (it should remain identity)
    root_idx = mst[0]
    for i, img in enumerate(imgs):
        trainable = not (init[img].get("freeze"))
        # Root camera should have identity relative pose, so freeze its quat, trans, and log_sizes
        # (log_sizes affects absolute pose computation through z_cameras)
        is_root = i == root_idx
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable and not is_root)  # Freeze root camera's relative pose
        trans[i].requires_grad_(trainable and not is_root)  # Freeze root camera's relative pose
        log_sizes[i].requires_grad_(trainable and not is_root)  # Freeze root camera's log_sizes too
        core_depth[i].requires_grad_(False)

        if verbose and is_root:
            print(
                f"\n[Optimization Setup] Freezing root camera {i} relative pose (identity) and log_sizes"
            )

    res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

    res_fine = None
    if niter2:
        # now we can optimize 3d points
        # Keep root camera frozen
        root_idx = mst[0]
        if verbose:
            print(f"\n[Fine Optimization Setup] Root camera {root_idx} will remain frozen")
        for i, img in enumerate(imgs):
            if init[img].get("freeze", 0) >= 1:
                continue
            is_root = i == root_idx
            pps[i].requires_grad_(bool(opt_pp))
            # Keep root camera's log_focals frozen too (affects absolute pose through z_cameras)
            log_focals[i].requires_grad_(True and not is_root)
            # Keep root camera's relative pose and log_sizes frozen
            if not is_root:
                quats[i].requires_grad_(True)
                trans[i].requires_grad_(True)
                log_sizes[i].requires_grad_(True)
            else:
                # Explicitly freeze root camera (quat, trans, log_sizes, and log_focals)
                # log_sizes and log_focals affect absolute pose through z_cameras calculation
                quats[i].requires_grad_(False)
                trans[i].requires_grad_(False)
                log_sizes[i].requires_grad_(False)
                log_focals[i].requires_grad_(False)
                if verbose:
                    print(f"  Camera {i} (root): quat, trans, log_sizes, and log_focals frozen")
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)
    if shared_intrinsics:
        print("Final focal (shared) = ", to_numpy(K[0, 0, 0]).round(2))
    else:
        print("Final focals =", to_numpy(K[:, 0, 0]))

    return imgs, res_coarse, res_fine
