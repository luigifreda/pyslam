"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
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

import cv2
from einops import rearrange
import numpy as np
import os
import sys
import platform

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose
from f3rm.features.clip import clip as f3rm_clip
from f3rm.features.clip import tokenize

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_utils import (
    similarity_heatmap_image,
    labels_color_map_factory,
    labels_to_image,
)
from .semantic_labels_factory import semantic_labels_factory

from pyslam.utilities.logging import Printer


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SemanticSegmentationCLIP(SemanticSegmentationBase):
    # CLIP available models: https://github.com/f3rm/f3rm/tree/main/f3rm/features/clip
    available_configs = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ]

    supported_feature_types = [
        SemanticFeatureType.LABEL,
        SemanticFeatureType.PROBABILITY_VECTOR,
        SemanticFeatureType.FEATURE_VECTOR,
    ]

    # TODO(dvdmc): take the text query parameter out
    def __init__(
        self,
        device=None,
        encoder_name="ViT-L/14@336px",
        model_path="",
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 1024),
        semantic_feature_type=SemanticFeatureType.LABEL,
        custom_set_labels=None,
        sim_text_query="clock",
        skip_center_crop=True,
        **kwargs,
    ):

        device = self.init_device(device)

        self.encoder_name = encoder_name  # Keep the encoder name to infer if it's ViT or RN

        # NOTE: transform is called preprocess in the original code
        model, transform = self.init_model(
            device, encoder_name, semantic_dataset_type, image_size, model_path
        )

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        # Config the dataset type
        if semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
            if custom_set_labels is None:
                raise ValueError(
                    "custom_set_labels must be provided if semantic_dataset_type is CUSTOM_SET"
                )
            self.semantic_color_map = labels_color_map_factory(
                semantic_dataset_type, num_classes=len(custom_set_labels)
            )
        elif semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            if semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
                raise ValueError(
                    "semantic_feature_type must be FEATURE_VECTOR if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            if sim_text_query == "":
                raise ValueError(
                    "sim_text_query must be provided if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            self.semantic_color_map = None
            self.sim_scale = 3.0  # NOTE: This is for visualization
        else:
            self.semantic_color_map = labels_color_map_factory(semantic_dataset_type)

        self.semantic_dataset_type = semantic_dataset_type

        # Config the text encodings
        if semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
            self.label_names = custom_set_labels
            self.tokens = [tokenize(text_query).to(device) for text_query in custom_set_labels]
        elif semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            # We already checked that semantic_feature_type is SemanticFeatureType.FEATURE_VECTOR
            self.label_names = [sim_text_query]
            self.tokens = [
                tokenize(sim_text_query).to(device)
            ]  # We will only work with a single text query
        else:
            self.label_names = semantic_labels_factory(semantic_dataset_type)
            self.tokens = torch.stack(
                [tokenize(text_query).to(device) for text_query in self.label_names]
            )  # Shape: (N, token_dim)

        self.text_embs = torch.stack([model.encode_text(token).squeeze(0) for token in self.tokens])

        self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)

        # Patch the preprocess if we want to skip center crop
        if skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in transform.transforms]
            assert sum(is_center_crop) == 1, "There should be exactly one CenterCrop transform"
            # Create new transform without center crop
            transform = Compose([t for t in transform.transforms if not isinstance(t, CenterCrop)])
            print("Skipping center crop")

        super().__init__(model, transform, device, semantic_feature_type)

    def init_model(self, device, encoder_name, semantic_dataset_type, image_size, model_path):

        # Check if selected config is available
        if encoder_name not in self.available_configs:
            raise ValueError(
                f"Segformer does not support {encoder_name} model with size {image_size} and dataset {semantic_dataset_type}"
            )

        if model_path == "":  # Load pre-trained models
            model, preprocess = f3rm_clip.load(encoder_name, device)
        else:
            raise NotImplementedError(
                "Segformer only supports pre-trained model for now"
            )  # TODO(dvdmc): allow to load a custom model
        model = model.eval()

        return model, preprocess

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            # Convert string to torch.device object
            device = torch.device(device)
        # At this point, device should be a torch.device object
        if device.type == "cuda":
            print("SemanticSegmentationCLIP: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():  # Should return True for MPS availability
                raise Exception("SemanticSegmentationCLIP: MPS is not available")
            print("SemanticSegmentationCLIP: Using MPS")
        else:
            print("SemanticSegmentationCLIP: Using CPU")
        return device

    def set_query_word(self, query_word):
        if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.label_names = [query_word]
            self.tokens = [tokenize(query_word).to(self.device)]
            self.text_embs = torch.stack(
                [self.model.encode_text(token).squeeze(0) for token in self.tokens]
            )
            self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)
        else:
            Printer.red(
                "Setting the query word will have no effect since semantic_dataset_type is not FEATURE_SIMILARITY"
            )

    def get_output_dims(self, h_in, w_in):
        """Compute output dimensions."""
        # from https://github.com/f3rm/f3rm/blob/main/f3rm/features/clip_extract.py
        if self.encoder_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
            return h_out, w_out

        if self.encoder_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            return int(h_out), int(w_out)

        raise ValueError(f"unknown clip model: {self.encoder_name}")

    def num_classes(self):
        return len(self.label_names)

    @torch.no_grad()
    def infer(self, image) -> SemanticSegmentationOutput:
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize(
            (prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST
        )
        image_pil = Image.fromarray(image)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)

        # We use a batch inference approach for using the implementation in f3rm
        embeddings = []
        embeddings.append(self.model.get_patch_encodings(img_t))

        embeddings = torch.cat(embeddings, dim=0)

        h_out, w_out = self.get_output_dims(img_t.shape[-2], img_t.shape[-1])

        embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)  # (H, W, D)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.squeeze(0)

        if self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Nothing to do except recover the size
            # We need to permute channels to do the recover size correctly TODO(dvdmc): can we improve this?
            self.semantics = (
                recover_size(embeddings.permute(2, 0, 1)).permute(1, 2, 0).cpu().numpy()
            )
            return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

        # Compute similarities
        sims = embeddings @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)

        # NOTE: Careful with channel ordering here. It differs from other sem seg modules!

        # Normalize to get "probabilities"
        probs = (sims / sims.norm(dim=-1, keepdim=True)).permute(2, 0, 1)
        probs = recover_size(probs)

        if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

        # Get the label
        pred = probs.argmax(dim=0)
        # if self.semantic_feature_type == SemanticFeatureType.LABEL: # NOT NECESSARY FOR NOW
        self.semantics = pred.cpu().numpy()
        return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        return self.sem_img_to_rgb(semantics, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantic_img, self.semantic_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(
                np.argmax(semantic_img, axis=-1), self.semantic_color_map, bgr=bgr
            )
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Transform semantic to tensor
            # TODO(dvdmc): check if doing these operations (and functions below) in CPU is more efficient (it probably is)
            semantics = torch.from_numpy(semantic_img).to(self.device)
            # Compute similarity
            sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                return similarity_heatmap_image(
                    sims.cpu().detach().numpy(),
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                pred = sims.argmax(dim=-1)
                return labels_to_image(
                    pred.cpu().detach().numpy(), self.semantic_color_map, bgr=bgr
                )

    def features_to_sims(self, semantics):
        """Public interface to compute similarity

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """

        if self.semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_sims()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        return sims.cpu().detach().numpy()

    def features_to_labels(self, semantics):
        """Public interface to compute labels

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        if self.semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_labels()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        pred = sims.argmax(dim=-1)
        return pred.cpu().detach().numpy()
