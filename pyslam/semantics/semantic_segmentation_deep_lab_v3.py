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
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision import transforms

from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_types import SemanticFeatureType
from .semantic_utils import SemanticDatasetType, labels_color_map_factory, labels_to_image

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'

class SemanticSegmentationDeepLabV3(SemanticSegmentationBase):
    model_configs = {
        'resnet50': {'encoder': 'resnet50', 'model': deeplabv3_resnet50, 'weights': DeepLabV3_ResNet50_Weights.DEFAULT, 'dataset': SemanticDatasetType.VOC, 'image_size': (512, 512)},
        'resnet101': {'encoder': 'resnet101', 'model': deeplabv3_resnet101, 'weights': DeepLabV3_ResNet101_Weights.DEFAULT, 'dataset': SemanticDatasetType.VOC, 'image_size': (512, 512)},
        'mobilenetv3': {'encoder': 'mobilenetv3', 'model': deeplabv3_mobilenet_v3_large, 'weights': DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, 'dataset': SemanticDatasetType.VOC, 'image_size': (512, 512)},
    }
    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]
    def __init__(self, device=None, encoder_name='resnet50', model_path='', semantic_dataset_type=SemanticDatasetType.VOC, image_size=(512, 512), semantic_feature_type=SemanticFeatureType.LABEL):
        
        device = self.init_device(device)

        model, transform = self.init_model(device, encoder_name, model_path, semantic_dataset_type)
        
        self.semantics_color_map = labels_color_map_factory(semantic_dataset_type)

        self.semantic_dataset_type = semantic_dataset_type
        
        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}")

        super().__init__(model, transform, device, semantic_feature_type)

    def init_model(self, device, encoder_name, model_path, semantic_dataset_type):
        if encoder_name not in self.model_configs:
            raise ValueError(f"Encoder name {encoder_name} is not supported for {self.__class__.__name__}")
        model = self.model_configs[encoder_name]['model'](self.model_configs[encoder_name]['weights'])
        if model_path != '': # Load pre-trained models
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device).eval()
        transform = self.model_configs[encoder_name]['weights'].transforms()
        return model,transform

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != 'cuda':
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == 'cuda':
            print('SemanticSegmentationDeepLabV3: Using CUDA')
        elif device.type == 'mps':
            if not torch.backends.mps.is_available():  # Should return True for MPS availability        
                raise Exception('SemanticSegmentationDeepLabV3: MPS is not available')
            print('SemanticSegmentationDeepLabV3: Using MPS')
        else:
            print('SemanticSegmentationDeepLabV3: Using CPU')
        return device

    @torch.no_grad()
    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        image_torch = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        batch = self.transform(image_torch).unsqueeze(0)
        prediction = self.model(batch)["out"]
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])
        
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            self.semantics = probs.argmax(dim=0).cpu().numpy()
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()

        return self.semantics
    
    def to_rgb(self, semantics, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantics, self.semantics_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(np.argmax(semantics, axis=-1), self.semantics_color_map, bgr=bgr)