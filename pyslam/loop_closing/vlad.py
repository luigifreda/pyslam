"""
* This file is part of PYSLAM
* Adpated from adapted from https://github.com/lzx551402/contextdesc/blob/master/utils/tf.py, see the license therein.
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

import numpy as np
import cv2
import os
import torch

import einops as ein
import fast_pytorch_kmeans as fpk
from torch.nn import functional as F

from typing import Union, List, Tuple, Literal


# VLAD global descriptor implementation. VLAD needs a "Vocabulary" in order to work.
# from https://github.com/AnyLoc/AnyLoc
# See demos here
#     https://github.com/AnyLoc/AnyLoc/tree/main/demo
#     In particular, https://github.com/AnyLoc/AnyLoc/blob/main/demo/images_vlad_colab.ipynb
class VLAD:
    """
    An implementation of VLAD algorithm given database and query
    descriptors.

    Constructor arguments:
    - num_clusters:     Number of cluster centers for VLAD
    - desc_dim:         Descriptor dimension. If None, then it is
                        inferred when running `fit` method.
    - intra_norm:       If True, intra normalization is applied
                        when constructing VLAD
    - norm_descs:       If True, the given descriptors are
                        normalized before training and predicting
                        VLAD descriptors. Different from the
                        `intra_norm` argument.
    - dist_mode:        Distance mode for KMeans clustering for
                        vocabulary (not residuals). Must be in
                        {'euclidean', 'cosine'}.
    - vlad_mode:        Mode for descriptor assignment (to cluster
                        centers) in VLAD generation. Must be in
                        {'soft', 'hard'}
    - soft_temp:        Temperature for softmax (if 'vald_mode' is
                        'soft') for assignment
    - cache_dir:        Directory to cache the VLAD vectors. If
                        None, then no caching is done. If a str,
                        then it is assumed as the folder path. Use
                        absolute paths.

    Notes:
    - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
        Proceedings of the IEEE conference on Computer Vision and
        Pattern Recognition. 2013.
    """

    def __init__(
        self,
        desc_dim: Union[int, None] = None,
        num_clusters: int = 8,
        intra_norm: bool = True,
        norm_descs: bool = True,
        dist_mode: str = "cosine",
        vlad_mode: str = "hard",
        soft_temp: float = 1.0,
        cache_dir: Union[str, None] = None,
    ) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ["soft", "hard"]
        self.soft_temp = soft_temp
        # Set in the training phase
        self.c_centers = None
        self.kmeans = None
        # Set the caching
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            else:
                print("Warning: Cache directory already exists: " f"{self.cache_dir}")
        else:
            print("VLAD caching is disabled.")

    def can_use_cache_vlad(self):
        """
        Checks if the cache directory is a valid cache directory.
        For it to be valid, it must exist and should at least
        include the cluster centers file.

        Returns:
        - True if the cache directory is valid
        - False if
            - the cache directory doesn't exist
            - exists but doesn't contain the cluster centers
            - no caching is set in constructor
        """
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/c_centers.pt"):
            return True
        else:
            return False

    def can_use_cache_ids(
        self, cache_ids: Union[List[str], str, None], only_residuals: bool = False
    ) -> bool:
        """
        Checks if the given cache IDs exist in the cache directory
        and returns True if all of them exist.
        The cache is stored in the following files:
        - c_centers.pt:     Cluster centers
        - `cache_id`_r.pt:  Residuals for VLAD
        - `cache_id`_l.pt:  Labels for VLAD (hard assignment)
        - `cache_id`_s.pt:  Soft assignment for VLAD

        The function returns False if cache cannot be used or if
        any of the cache IDs are not found. If all cache IDs are
        found, then True is returned.

        This function is mainly for use outside the VLAD class.
        """
        if not self.can_use_cache_vlad():
            return False
        if cache_ids is None:
            return False
        if isinstance(cache_ids, str):
            cache_ids = [cache_ids]
        for cache_id in cache_ids:
            if not os.path.exists(f"{self.cache_dir}/{cache_id}_r.pt"):
                return False
            if (
                self.vlad_mode == "hard"
                and not os.path.exists(f"{self.cache_dir}/{cache_id}_l.pt")
                and not only_residuals
            ):
                return False
            if (
                self.vlad_mode == "soft"
                and not os.path.exists(f"{self.cache_dir}/{cache_id}_s.pt")
                and not only_residuals
            ):
                return False
        return True

    # Generate cluster centers
    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None]):
        """
        Using the training descriptors, generate the cluster
        centers (vocabulary). Function expects all descriptors in
        a single list (see `fit_and_generate` for a batch of
        images).
        If the cache directory is valid, then retrieves cluster
        centers from there (the `train_descs` are ignored).
        Otherwise, stores the cluster centers in the cache
        directory (if using caching).

        Parameters:
        - train_descs:  Training descriptors of shape
                        [num_train_desc, desc_dim]. If None, then
                        caching should be valid (else ValueError).
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        # Check if cache exists
        if self.can_use_cache_vlad():
            print("Using cached cluster centers")
            self.c_centers = torch.load(f"{self.cache_dir}/c_centers.pt")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            if train_descs is None:
                raise ValueError("No training descriptors given")
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("Caching cluster centers")
                torch.save(self.c_centers, f"{self.cache_dir}/c_centers.pt")

    def fit_and_generate(self, train_descs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Given a batch of descriptors over images, `fit` the VLAD
        and generate the global descriptors for the training
        images. Use only when there are a fixed number of
        descriptors in each image.

        Parameters:
        - train_descs:  Training image descriptors of shape
                        [num_imgs, num_descs, desc_dim]. There are
                        'num_imgs' images, each image has
                        'num_descs' descriptors and each
                        descriptor is 'desc_dim' dimensional.

        Returns:
        - train_vlads:  The VLAD vectors of all training images.
                        Shape: [num_imgs, num_clusters*desc_dim]
        """
        # Generate vocabulary
        all_descs = ein.rearrange(train_descs, "n k d -> (n k) d")
        self.fit(all_descs)
        # For each image, stack VLAD
        return torch.stack([self.generate(tr) for tr in train_descs])

    def load(self, path: str):
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        print("VLAD: Loading cluster centers...")
        self.c_centers = torch.load(path)
        self.kmeans.centroids = self.c_centers
        if self.desc_dim is None:
            self.desc_dim = self.c_centers.shape[1]
            print(f"Desc dim set to {self.desc_dim}")

    def save(self, path: str):
        print("VLAD: Saving cluster centers...")
        torch.save(self.c_centers, path)

    def generate(
        self, query_descs: Union[np.ndarray, torch.Tensor], cache_id: Union[str, None] = None
    ) -> torch.Tensor:
        """
        Given the query descriptors, generate a VLAD vector. Call
        `fit` before using this method. Use this for only single
        images and with descriptors stacked. Use function
        `generate_multi` for multiple images.

        Parameters:
        - query_descs:  Query descriptors of shape [n_q, desc_dim]
                        where 'n_q' is number of 'desc_dim'
                        dimensional descriptors in a query image.
        - cache_id:     If not None, then the VLAD vector is
                        constructed using the residual and labels
                        from this file.

        Returns:
        - n_vlas:   Normalized VLAD: [num_clusters*desc_dim]
        """
        if type(query_descs) == np.ndarray:
            query_descs = torch.from_numpy(query_descs).to(torch.float32)
        residuals = self.generate_res_vec(query_descs, cache_id)
        # Un-normalized VLAD vector: [c*d,]
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == "hard":
            # Get labels for assignment of descriptors
            if (
                cache_id is not None
                and self.can_use_cache_vlad()
                and os.path.isfile(f"{self.cache_dir}/{cache_id}_l.pt")
            ):
                labels = torch.load(f"{self.cache_dir}/{cache_id}_l.pt")
            else:
                labels = self.kmeans.predict(query_descs)  # [q]
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(labels, f"{self.cache_dir}/{cache_id}_l.pt")
            # Create VLAD from residuals and labels
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                #  Shape:[q, c, d]  ->  [q', d] -> [d]
                cd_sum = residuals[labels == k, k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum
        else:  # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            if (
                cache_id is not None
                and self.can_use_cache_vlad()
                and os.path.isfile(f"{self.cache_dir}/{cache_id}_s.pt")
            ):
                soft_assign = torch.load(f"{self.cache_dir}/{cache_id}_s.pt")
            else:
                cos_sims = F.cosine_similarity(  # [q, c]
                    ein.rearrange(query_descs, "q d -> q 1 d"),
                    ein.rearrange(self.c_centers, "c d -> 1 c d"),
                    dim=2,
                )
                soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(soft_assign, f"{self.cache_dir}/{cache_id}_s.pt")
            # Soft assignment scores (as probabilities): [q, c]
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Sum of residuals for all descriptors (for cluster k)
                cd_sum = ein.rearrange(w * residuals, "q c d -> (q c) d").sum(dim=0)  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum
        # Normalize the VLAD vector
        n_vlad = F.normalize(un_vlad, dim=0)
        return n_vlad

    def generate_multi(
        self,
        multi_query: Union[np.ndarray, torch.Tensor, list],
        cache_ids: Union[List[str], None] = None,
    ) -> Union[torch.Tensor, list]:
        """
        Given query descriptors from multiple images, generate
        the VLAD for them.

        Parameters:
        - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                        There are 'n_imgs' and each image has
                        'n_kpts' keypoints, with 'd' dimensional
                        descriptor each. If a List (can then have
                        different number of keypoints in each
                        image), then the result is also a list.
        - cache_ids:    Cache IDs for the VLAD vectors. If None,
                        then no caching is done (stored or
                        retrieved). If a list, then the length
                        should be 'n_imgs' (one per image).

        Returns:
        - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate(q, c) for (q, c) in zip(multi_query, cache_ids)]
        try:  # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:  # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass  # Let it remain as a list
        return res

    def generate_res_vec(
        self, query_descs: Union[np.ndarray, torch.Tensor], cache_id: Union[str, None] = None
    ) -> torch.Tensor:
        """
        Given the query descriptors, generate a VLAD vector. Call
        `fit` before using this method. Use this for only single
        images and with descriptors stacked. Use function
        `generate_multi` for multiple images.

        Parameters:
        - query_descs:  Query descriptors of shape [n_q, desc_dim]
                        where 'n_q' is number of 'desc_dim'
                        dimensional descriptors in a query image.
        - cache_id:     If not None, then the VLAD vector is
                        constructed using the residual and labels
                        from this file.

        Returns:
        - residuals:    Residual vector: shape [n_q, n_c, d]
        """
        assert self.kmeans is not None
        assert self.c_centers is not None
        # Compute residuals (all query to cluster): [q, c, d]
        if (
            cache_id is not None
            and self.can_use_cache_vlad()
            and os.path.isfile(f"{self.cache_dir}/{cache_id}_r.pt")
        ):
            residuals = torch.load(f"{self.cache_dir}/{cache_id}_r.pt")
        else:
            if type(query_descs) == np.ndarray:
                query_descs = torch.from_numpy(query_descs).to(torch.float32)
            if self.norm_descs:
                query_descs = F.normalize(query_descs)
            residuals = ein.rearrange(query_descs, "q d -> q 1 d") - ein.rearrange(
                self.c_centers, "c d -> 1 c d"
            )
            if cache_id is not None and self.can_use_cache_vlad():
                cid_dir = f"{self.cache_dir}/" f"{os.path.split(cache_id)[0]}"
                if not os.path.isdir(cid_dir):
                    os.makedirs(cid_dir)
                    print(f"Created directory: {cid_dir}")
                torch.save(residuals, f"{self.cache_dir}/{cache_id}_r.pt")
        # print("residuals",residuals.shape)
        return residuals

    def generate_multi_res_vec(
        self,
        multi_query: Union[np.ndarray, torch.Tensor, list],
        cache_ids: Union[List[str], None] = None,
    ) -> Union[torch.Tensor, list]:
        """
        Given query descriptors from multiple images, generate
        the VLAD for them.

        Parameters:
        - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                        There are 'n_imgs' and each image has
                        'n_kpts' keypoints, with 'd' dimensional
                        descriptor each. If a List (can then have
                        different number of keypoints in each
                        image), then the result is also a list.
        - cache_ids:    Cache IDs for the VLAD vectors. If None,
                        then no caching is done (stored or
                        retrieved). If a list, then the length
                        should be 'n_imgs' (one per image).

        Returns:
        - multi_res:    VLAD descriptors for the queries
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate_res_vec(q, c) for (q, c) in zip(multi_query, cache_ids)]
        try:  # Most likely pytorch
            res = torch.stack(res)
        except TypeError:
            try:  # Otherwise numpy
                res = np.stack(res)
            except TypeError:
                pass  # Let it remain as a list
        return res

    # Place recognition function (returns indices of most similar images)
    # input:
    # - database_descs: descriptors of database images
    # - query_descs: descriptors of query image
    # output: indices of most similar images
    def place_recognition_query(self, database_descs, query_descs):
        # Generate VLAD descriptors for the database images
        database_vlads = self.generate_multi(database_descs)

        # Generate VLAD descriptor for the query image
        query_vlad = self.generate(query_descs)

        # Compute similarities between query VLAD and each database VLAD
        similarities = [
            torch.cosine_similarity(query_vlad, db_vlad, dim=0).item() for db_vlad in database_vlads
        ]

        # Sort indices by similarity to get the most similar places
        most_similar_idx = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )

        return most_similar_idx  # Indices of the most similar images in the database

    # g_des1 is [1, D], g_des2 is [M, D]
    @staticmethod
    def score(desc_a, desc_b):
        similarities = [torch.cosine_similarity(desc_a, d, dim=0).item() for d in desc_b]
        return similarities
