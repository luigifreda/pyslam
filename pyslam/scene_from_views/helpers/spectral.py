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
Spectral clustering and projection utilities.
"""

import torch
import numpy as np
import os
import tqdm

from .geometry import GeometryUtils, ProjectionUtils
from pyslam.utilities.file_management import mkdir_for, hash_md5


class SpectralUtils:
    """Spectral clustering and projection utilities."""

    @staticmethod
    def spectral_clustering(graph, k=None, normalized_cuts=False):
        """Perform spectral clustering on a graph.

        This function computes the graph Laplacian and returns its k smallest
        eigenvalues and corresponding eigenvectors, which can be used for
        clustering or dimensionality reduction.

        Args:
            graph: Adjacency matrix (N, N)
            k: Number of eigenvectors to return (default: None, returns all)
            normalized_cuts: Whether to use normalized Laplacian (default: False)

        Returns:
            Tuple of (eigenvalues, eigenvectors) where:
                eigenvalues: k smallest eigenvalues (k,)
                eigenvectors: Corresponding eigenvectors (N, k)
        """
        graph.fill_diagonal_(0)
        degrees = graph.sum(dim=-1)
        laplacian = torch.diag(degrees) - graph
        if normalized_cuts:
            i_inv = torch.diag(degrees.sqrt().reciprocal())
            laplacian = i_inv @ laplacian @ i_inv
        eigval, eigvec = torch.linalg.eigh(laplacian)
        return eigval[:k], eigvec[:, :k]

    @staticmethod
    def spectral_projection_of_depthmaps(
        imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw
    ):
        """Apply spectral projection to depthmaps for low-rank parameterization.

        This function applies spectral clustering to depthmaps to create a low-rank
        representation (LoRA) that can be used for efficient optimization.

        Args:
            imgs: List of image paths
            intrinsics: Camera intrinsics (N, 3, 3)
            depthmaps: List of depth maps (N,)
            subsample: Subsampling factor
            cache_path: Optional path for caching projections
            **kw: Additional keyword arguments for spectral_projection_depth

        Returns:
            Tuple of (core_depth, lora_proj) where:
                core_depth: List of depth coefficients
                lora_proj: List of projection matrices
        """
        core_depth = []
        lora_proj = []
        for i, img in enumerate(tqdm.tqdm(imgs)):
            cache = os.path.join(cache_path, "lora_depth", hash_md5(img)) if cache_path else None
            depth, proj = SpectralUtils.spectral_projection_depth(
                intrinsics[i], depthmaps[i], subsample, cache_path=cache, **kw
            )
            core_depth.append(depth)
            lora_proj.append(proj)
        return core_depth, lora_proj

    @staticmethod
    def spectral_projection_depth(
        K, depthmap, subsample, k=64, cache_path="", normalized_cuts=True, gamma=7, min_norm=5
    ):
        """Compute spectral projection of a single depthmap for low-rank representation.

        This function backprojects the depthmap to 3D, builds a similarity graph,
        performs spectral clustering, and encodes the depthmap using the resulting
        low-rank basis.

        Args:
            K: Camera intrinsics (3, 3)
            depthmap: Depth map (H, W)
            subsample: Subsampling factor
            k: Number of basis vectors (default: 64)
            cache_path: Optional path for caching projection matrix
            normalized_cuts: Whether to use normalized Laplacian (default: True)
            gamma: Similarity function parameter (default: 7)
            min_norm: Minimum norm for coefficient normalization (default: 5)

        Returns:
            Tuple of (coeffs, lora_proj) where:
                coeffs: Depth coefficients (k,)
                lora_proj: Projection matrix (H*W, k)
        """
        try:
            if cache_path:
                cache_path = cache_path + f"_{k=}_norm={normalized_cuts}_{gamma=}.pth"
            lora_proj = torch.load(cache_path, map_location=K.device)
        except IOError:
            xyz = ProjectionUtils.backproj(K, depthmap, subsample)
            xyz = xyz.reshape(-1, 3)
            graph = SpectralUtils.sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
            _, lora_proj = SpectralUtils.spectral_clustering(
                graph, k, normalized_cuts=normalized_cuts
            )
            if cache_path:
                torch.save(lora_proj.cpu(), mkdir_for(cache_path))
        lora_proj, coeffs = SpectralUtils.lora_encode_normed(
            lora_proj, depthmap.ravel(), min_norm=min_norm
        )
        return coeffs, lora_proj

    @staticmethod
    def sim_func(p1, p2, gamma):
        """Compute similarity between two point sets using exponential distance.

        The similarity is computed as exp(-gamma * (relative_distance)^2), where
        relative_distance is the Euclidean distance normalized by average depth.

        Args:
            p1: First point set (N1, 1, 3) or (N1, 3)
            p2: Second point set (1, N2, 3) or (N2, 3)
            gamma: Scaling parameter for exponential decay

        Returns:
            Similarity matrix (N1, N2)
        """
        diff = (p1 - p2).norm(dim=-1)
        avg_depth = p1[:, :, 2] + p2[:, :, 2]
        rel_distance = diff / avg_depth
        sim = torch.exp(-gamma * rel_distance.square())
        return sim

    @staticmethod
    def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
        """Encode a signal using a low-rank projection with normalized coefficients.

        This function projects a signal onto a low-rank basis and normalizes the
        projection matrix to ensure minimum norm constraints on coefficients.

        Args:
            lora_proj: Projection matrix (M, k)
            x: Signal to encode (M,)
            min_norm: Minimum norm for coefficient normalization
            global_norm: Whether to use global normalization (default: False)

        Returns:
            Tuple of (lora_proj_normalized, coeffs) where:
                lora_proj_normalized: Normalized projection matrix (M, k)
                coeffs: Encoded coefficients (k,)
        """
        coeffs = torch.linalg.pinv(lora_proj) @ x
        if coeffs.ndim == 1:
            coeffs = coeffs[:, None]
        if global_norm:
            lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
        elif min_norm:
            lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
        coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()
        return lora_proj.detach(), coeffs.detach()
