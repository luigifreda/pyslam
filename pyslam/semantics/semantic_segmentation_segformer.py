"""
* This file is part of PYSLAM 
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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
import os
import sys
import platform

import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from torchvision import transforms

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_types import SemanticFeatureType
from .semantic_utils import SemanticDatasetType, labels_color_map_factory, labels_to_image

from pyslam.utilities.utils_sys import Printer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'

class SemanticSegmentationSegformer(SemanticSegmentationBase):
    # Segformer available models: https://huggingface.co/models?search=nvidia/segformer
    # They can be configured by:
    # - Model variant: b0, b1, b2, b3, b4, b5
    # - Sizes: (512,512), (512,1024), (768,768), (1024,1024), (640, 1280)
    # - Dataset: cityscapes, ade
    # Check the specific available configurations
    available_configs = [
        ('b0', (1024, 1024), SemanticDatasetType.CITYSCAPES),
        ('b0', (512, 512), SemanticDatasetType.ADE20K),
        ('b0', (512, 1024), SemanticDatasetType.CITYSCAPES),
        ('b0', (640, 1280), SemanticDatasetType.CITYSCAPES),
        ('b0', (768, 768), SemanticDatasetType.CITYSCAPES),
        ('b1', (1024, 1024), SemanticDatasetType.CITYSCAPES),
        ('b1', (512, 512), SemanticDatasetType.ADE20K),
        ('b2', (1024, 1024), SemanticDatasetType.CITYSCAPES),
        ('b2', (512, 512), SemanticDatasetType.ADE20K),
        ('b3', (1024, 1024), SemanticDatasetType.CITYSCAPES),
        ('b3', (512, 512), SemanticDatasetType.ADE20K),
        ('b4', (1024, 1024), SemanticDatasetType.CITYSCAPES),
        ('b4', (512, 512), SemanticDatasetType.ADE20K),
        ('b5', (1024, 1024), SemanticDatasetType.CITYSCAPES),
    ]
    # TODO(dvdmc): this can be used to make mappings more generic NOTE: not currently used
    available_mappings = [
        {'in': SemanticDatasetType.ADE20K, 'out': SemanticDatasetType.NYU40, 'map': get_ade20k_to_scannet40_map()},
    ]
    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]
    def __init__(self, device=None, encoder_name='b0', semantic_dataset_type=SemanticDatasetType.CITYSCAPES, image_size=(512, 1024), model_path='', semantic_feature_type=SemanticFeatureType.LABEL):
        
        self.label_mapping = None

        device = self.init_device(device)
        
        model, transform = self.init_model(device, encoder_name, semantic_dataset_type, image_size, model_path)
        
        self.semantics_color_map = labels_color_map_factory(semantic_dataset_type)

        self.semantic_dataset_type = semantic_dataset_type

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}")
        
        super().__init__(model, transform, device, semantic_feature_type)

    def init_model(self, device, encoder_name, semantic_dataset_type, image_size, model_path):

        # Convert image_size to appropiate form
        if semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
            image_size = (512, 1024)
        else:
            image_size = (512, 512)

        friendly_dataset_type = semantic_dataset_type
        # We allow to use this model on NYU40 by mapping labels from ADE20K
        if semantic_dataset_type == SemanticDatasetType.NYU40:
            self.label_mapping = get_ade20k_to_scannet40_map()
            friendly_dataset_type = SemanticDatasetType.ADE20K # From here

        # Check if selected config is available
        if (encoder_name, image_size, friendly_dataset_type) not in self.available_configs:
            raise ValueError(f"Segformer does not support {encoder_name} model with size {image_size} and dataset {semantic_dataset_type}")

        # Convert dataset type to appropiate form
        dataset = friendly_dataset_type.name.lower()
        if dataset == 'ade20k':
            dataset = 'ade'
        
        if model_path == '': # Load pre-trained models
            model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-{encoder_name}-finetuned-{dataset}-{image_size[0]}-{image_size[1]}")
        else:
            raise NotImplementedError("Segformer only supports pre-trained model for now") #TODO(dvdmc): allow to load a custom model
        model = model.to(device).eval()
        transform = AutoImageProcessor.from_pretrained(f"nvidia/segformer-{encoder_name}-finetuned-{dataset}-{image_size[0]}-{image_size[1]}")
        return model,transform

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != 'cuda':
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == 'cuda':
            print('SemanticSegmentationSegformer: Using CUDA')
        elif device.type == 'mps':
            if not torch.backends.mps.is_available():  # Should return True for MPS availability        
                raise Exception('SemanticSegmentationSegformer: MPS is not available')
            print('SemanticSegmentationSegformer: Using MPS')
        else:
            print('SemanticSegmentationSegformer: Using CPU')
        return device

    @torch.no_grad()
    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        image_pil = Image.fromarray(image)
        batch = self.transform(images=image_pil, return_tensors="pt").to(self.device)
        prediction = self.model(**batch).logits
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])

        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            self.semantics = probs.argmax(dim=0).cpu().numpy()
            if self.label_mapping is not None:
                self.semantics = self.label_mapping[self.semantics]
        
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
        
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            if self.label_mapping is not None:
                self.semantics = self.aggregate_probabilities(self.semantics, self.label_mapping)

        return self.semantics
        
    def aggregate_probabilities(self, semantics: np.ndarray, label_mapping: np.ndarray) -> np.ndarray:
        """
        Aggregates original probabilities into output probabilities using a label mapping.

        Args:
            semantics: np.ndarray of shape [H, W, original num classes] - softmaxed class probabilities.
            label_mapping: np.ndarray of shape [original num classes] - maps original class indices to output classes.

        Returns:
            np.ndarray of shape [H, W, num output classes] - aggregated probabilities.
        """
        H, W, num_original_classes = semantics.shape
        num_output_classes = label_mapping.max() + 1

        aggregated = np.zeros((H, W, num_output_classes), dtype=semantics.dtype)

        for in_idx, out_idx in enumerate(label_mapping):
            aggregated[..., out_idx] += semantics[..., in_idx]

        return aggregated
        
    def to_rgb(self, semantics, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantics, self.semantics_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(np.argmax(semantics, axis=-1), self.semantics_color_map, bgr=bgr)
    