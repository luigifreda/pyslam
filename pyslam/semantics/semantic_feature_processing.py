"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
import torch

from pyslam.semantics.semantic_types import SemanticFeatureType
from pyslam.utilities.logging import Printer


class SemanticFeatureProcessing:

    @staticmethod
    def features_to_sims(semantics, text_embs, device="cpu"):
        """Compute similarities between semantics and text embeddings

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        if device == "cpu":
            return SemanticFeatureProcessing._features_to_sims_cpu(semantics, text_embs)
        elif device == "gpu":
            return SemanticFeatureProcessing._features_to_sims_gpu(semantics, text_embs)
        else:
            raise ValueError(f"Invalid device: {device}")

    @staticmethod
    def features_to_labels(semantics, text_embs, device="cpu"):
        """Compute labels from semantics and text embeddings

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        if device == "cpu":
            return SemanticFeatureProcessing._features_to_labels_cpu(semantics, text_embs)
        elif device == "gpu":
            return SemanticFeatureProcessing._features_to_labels_gpu(semantics, text_embs)
        else:
            raise ValueError(f"Invalid device: {device}")

    # =================================================================

    @staticmethod
    def _features_to_sims_cpu(semantics, text_embs):
        """Compute similarities between semantics and text embeddings

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        # check if we need to convert text_embs to a numpy array
        if isinstance(text_embs, torch.Tensor):
            Printer.warning("WARNING: text_embs is a tensor, converting to numpy array")
            text_embs = text_embs.cpu().detach().numpy()
        # check if we need to convert semantics to a numpy array
        if isinstance(semantics, torch.Tensor):
            Printer.warning("WARNING: semantics is a tensor, converting to numpy array")
            semantics = semantics.cpu().detach().numpy()
        # Compute similarity using numpy
        similarity = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        return similarity

    @staticmethod
    def _features_to_sims_gpu(semantics, text_embs):
        """Compute similarities between semantics and text embeddings

        Args:
            semantics (torch.Tensor): Semantic features of generic shape ([1] or [H, W], D)
        """
        if not isinstance(semantics, torch.Tensor):
            Printer.warning("WARNING: semantics is not a tensor, converting to tensor")
            semantics = torch.from_numpy(semantics)
        if not isinstance(text_embs, torch.Tensor):
            Printer.warning("WARNING: text_embs is not a tensor, converting to tensor")
            text_embs = torch.from_numpy(text_embs)
        # Move tensors to GPU
        if not semantics.is_cuda:
            semantics = semantics.cuda()
        if not text_embs.is_cuda:
            text_embs = text_embs.cuda()
        similarity = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        return similarity.cpu().detach().numpy()

    @staticmethod
    def _features_to_labels_cpu(semantics, text_embs):
        """Compute labels from semantics and text embeddings

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        # check if we need to convert text_embs to a numpy array
        if isinstance(text_embs, torch.Tensor):
            Printer.warning("WARNING: text_embs is a tensor, converting to numpy array")
            text_embs = text_embs.cpu().detach().numpy()
        # check if we need to convert semantics to a numpy array
        if isinstance(semantics, torch.Tensor):
            Printer.warning("WARNING: semantics is a tensor, converting to numpy array")
            semantics = semantics.cpu().detach().numpy()
        # Compute similarity using numpy
        similarity = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        pred = np.argmax(similarity, axis=-1)
        return pred

    @staticmethod
    def _features_to_labels_gpu(semantics, text_embs):
        """Compute labels from semantics and text embeddings

        Args:
            semantics (torch.Tensor): Semantic features of generic shape ([1] or [H, W], D)
        """
        if not isinstance(semantics, torch.Tensor):
            Printer.warning("WARNING: semantics is not a tensor, converting to tensor")
            semantics = torch.from_numpy(semantics)
        if not isinstance(text_embs, torch.Tensor):
            Printer.warning("WARNING: text_embs is not a tensor, converting to tensor")
            text_embs = torch.from_numpy(text_embs)
        # Move tensors to GPU
        if not semantics.is_cuda:
            semantics = semantics.cuda()
        if not text_embs.is_cuda:
            text_embs = text_embs.cuda()
        similarity = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        pred = similarity.argmax(dim=-1)
        return pred.cpu().detach().numpy()
