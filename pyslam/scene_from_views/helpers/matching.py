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
Nearest neighbor and correspondence matching utilities.
"""

import torch
import numpy as np
import math
from scipy.spatial import KDTree

from pyslam.utilities.torch import to_numpy


class MatchingUtils:
    """Nearest neighbor and correspondence matching utilities."""

    @staticmethod
    @torch.no_grad()
    def bruteforce_reciprocal_nns(A, B, device="cuda", block_size=None, dist="l2"):
        """Compute reciprocal nearest neighbors between two point sets.

        For each point in A, find its nearest neighbor in B, and vice versa.
        Only pairs that are mutual nearest neighbors are returned. Supports
        block-wise processing for large point sets.

        Args:
            A: First point set (N, D)
            B: Second point set (M, D)
            device: Device for computation (default: "cuda")
            block_size: Block size for memory-efficient processing (default: None)
            dist: Distance metric ("l2" or "dot") (default: "l2")

        Returns:
            Tuple of (nn_A, nn_B) where:
                nn_A: Indices of nearest neighbors in B for each point in A (N,)
                nn_B: Indices of nearest neighbors in A for each point in B (M,)
        """
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).to(device)
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).to(device)

        A = A.to(device)
        B = B.to(device)

        if dist == "l2":
            dist_func = torch.cdist
            argmin = torch.min
        elif dist == "dot":

            def dist_func(A, B):
                return A @ B.T

            def argmin(X, dim):
                sim, nn = torch.max(X, dim=dim)
                return sim.neg_(), nn

        else:
            raise ValueError(f"Unknown {dist=}")

        if block_size is None or len(A) * len(B) <= block_size**2:
            dists = dist_func(A, B)
            _, nn_A = argmin(dists, dim=1)
            _, nn_B = argmin(dists, dim=0)
        else:
            dis_A = torch.full((A.shape[0],), float("inf"), device=device, dtype=A.dtype)
            dis_B = torch.full((B.shape[0],), float("inf"), device=device, dtype=B.dtype)
            nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
            nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
            number_of_iteration_A = math.ceil(A.shape[0] / block_size)
            number_of_iteration_B = math.ceil(B.shape[0] / block_size)

            for i in range(number_of_iteration_A):
                A_i = A[i * block_size : (i + 1) * block_size]
                for j in range(number_of_iteration_B):
                    B_j = B[j * block_size : (j + 1) * block_size]
                    dists_blk = dist_func(A_i, B_j)
                    min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                    min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                    col_mask = min_A_i < dis_A[i * block_size : (i + 1) * block_size]
                    line_mask = min_B_j < dis_B[j * block_size : (j + 1) * block_size]

                    dis_A[i * block_size : (i + 1) * block_size][col_mask] = min_A_i[col_mask]
                    dis_B[j * block_size : (j + 1) * block_size][line_mask] = min_B_j[line_mask]

                    nn_A[i * block_size : (i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (
                        j * block_size
                    )
                    nn_B[j * block_size : (j + 1) * block_size][line_mask] = argmin_B_j[
                        line_mask
                    ] + (i * block_size)
        nn_A = nn_A.cpu().numpy()
        nn_B = nn_B.cpu().numpy()
        return nn_A, nn_B

    @staticmethod
    def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
        """Merge correspondences and convert from linear indices to pixel coordinates.

        This function takes two arrays of linear indices and merges them into
        unique correspondences, optionally converting them to (x, y) pixel coordinates.

        Args:
            idx1: Linear indices for first image (N,)
            idx2: Linear indices for second image (N,)
            shape1: Shape (H, W) of first image (required if ret_xy=True)
            shape2: Shape (H, W) of second image (required if ret_xy=True)
            ret_xy: Whether to return pixel coordinates (default: True)
            ret_index: Whether to return indices of unique correspondences (default: False)

        Returns:
            If ret_index=False: Tuple of (xy1, xy2) pixel coordinates
            If ret_index=True: Tuple of (xy1, xy2, indices)
        """
        assert idx1.dtype == idx2.dtype == np.int32

        # unique and sort along idx1
        corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
        if ret_index:
            corres, indices = corres
        xy2, xy1 = corres[:, None].view(np.int32).T

        if ret_xy:
            assert shape1 and shape2
            xy1 = np.unravel_index(xy1, shape1)
            xy2 = np.unravel_index(xy2, shape2)
            if ret_xy != "y_x":
                xy1 = xy1[0].base[:, ::-1]
                xy2 = xy2[0].base[:, ::-1]

        if ret_index:
            return xy1, xy2, indices
        return xy1, xy2

    @staticmethod
    def fast_reciprocal_NNs(
        pts1,
        pts2,
        subsample_or_initxy1=8,
        ret_xy=True,
        pixel_tol=0,
        ret_basin=False,
        device="cuda",
        **matcher_kw,
    ):
        """Find reciprocal nearest neighbors between two point sets iteratively.

        This function performs iterative reciprocal nearest neighbor search,
        starting from a subsampled grid or initial correspondences. It converges
        to stable correspondences through multiple iterations.

        Args:
            pts1: First point set (H1, W1, D)
            pts2: Second point set (H2, W2, D)
            subsample_or_initxy1: Subsampling factor (int) or initial (x, y) coordinates
            ret_xy: Whether to return pixel coordinates (default: True)
            pixel_tol: Pixel tolerance for convergence (default: 0)
            ret_basin: Whether to return basin of attraction map (default: False)
            device: Device for computation (default: "cuda")
            **matcher_kw: Additional keyword arguments for matcher

        Returns:
            Tuple of (xy1, xy2) correspondences, optionally with basin map
        """
        from scipy.spatial import KDTree

        H1, W1, DIM1 = pts1.shape
        H2, W2, DIM2 = pts2.shape
        assert DIM1 == DIM2

        pts1 = pts1.reshape(-1, DIM1)
        pts2 = pts2.reshape(-1, DIM2)

        if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
            S = subsample_or_initxy1
            y1, x1 = np.mgrid[S // 2 : H1 : S, S // 2 : W1 : S].reshape(2, -1)
            max_iter = 10
        else:
            x1, y1 = subsample_or_initxy1
            if isinstance(x1, torch.Tensor):
                x1 = x1.cpu().numpy()
            if isinstance(y1, torch.Tensor):
                y1 = y1.cpu().numpy()
            max_iter = 1

        xy1 = np.int32(np.unique(x1 + W1 * y1))
        xy2 = np.full_like(xy1, -1)
        old_xy1 = xy1.copy()
        old_xy2 = xy2.copy()

        if (
            "dist" in matcher_kw
            or "block_size" in matcher_kw
            or (isinstance(device, str) and device.startswith("cuda"))
            or (isinstance(device, torch.device) and device.type.startswith("cuda"))
        ):
            pts1 = pts1.to(device)
            pts2 = pts2.to(device)
            tree1 = NearestNeighborMatcher(pts1, device=device)
            tree2 = NearestNeighborMatcher(pts2, device=device)
        else:
            pts1, pts2 = to_numpy((pts1, pts2))
            tree1 = KDTree(pts1)
            tree2 = KDTree(pts2)

        notyet = np.ones(len(xy1), dtype=bool)
        basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

        niter = 0
        while notyet.any():
            _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
            if not ret_basin:
                notyet &= old_xy2 != xy2

            _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
            if ret_basin:
                basin[old_xy1[notyet]] = xy1[notyet]
            notyet &= old_xy1 != xy1

            niter += 1
            if niter >= max_iter:
                break

            old_xy2[:] = xy2
            old_xy1[:] = xy1

        if pixel_tol > 0:
            old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
            new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
            dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
            converged = dis < pixel_tol
            if not isinstance(subsample_or_initxy1, int):
                xy1 = old_xy1
        else:
            converged = ~notyet

        xy1, xy2 = MatchingUtils.merge_corres(
            xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy
        )
        if ret_basin:
            return xy1, xy2, basin
        return xy1, xy2


class NearestNeighborMatcher:
    """Matcher class for finding nearest neighbors using brute-force distance computation.

    This class provides a simple interface for nearest neighbor search using
    GPU-accelerated distance computation. It wraps bruteforce_reciprocal_nns
    for query operations.

    Args:
        db_pts: Database points (N, D) tensor
        device: Device for computation (default: "cuda")
    """

    def __init__(self, db_pts, device="cuda"):
        self.db_pts = db_pts.to(device)
        self.device = device

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = MatchingUtils.bruteforce_reciprocal_nns(
            queries, self.db_pts, device=self.device, **kw
        )
        dis = None
        return dis, nnA
