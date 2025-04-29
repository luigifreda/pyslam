"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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

from semantic_estimator_base import SemanticEstimator
from semantic_feature_types import SemanticFeatureTypes
from semantic_fusion_methods import bayesian_fusion, count_labels
from utils_semantics import information_weights_factory, labels_map_factory, labels_to_image, single_label_to_color

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

class SemanticEstimatorDeepLabV3(SemanticEstimator):
    model_configs = {
        'resnet50': {'encoder': 'resnet50', 'model': deeplabv3_resnet50, 'weights': DeepLabV3_ResNet50_Weights.DEFAULT},
        'resnet101': {'encoder': 'resnet101', 'model': deeplabv3_resnet101, 'weights': DeepLabV3_ResNet101_Weights.DEFAULT},
        'mobilenetv3': {'encoder': 'mobilenetv3', 'model': deeplabv3_mobilenet_v3_large, 'weights': DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT},
    }
    feature_type_configs = {
        'label': {'type':SemanticFeatureTypes.LABEL, 'fusion':count_labels}, 
        'probability_vector': {'type':SemanticFeatureTypes.PROBABILITY_VECTOR, 'fusion':bayesian_fusion}
    }

    def __init__(self, device=None, encoder_name='resnet50', model_path='', dataset_name='voc', semantic_feature_type='label'):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != 'cuda':
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == 'cuda':
            print('SemanticEstimatorDeepLabV3: Using CUDA')
        elif device.type == 'mps':
            if not torch.backends.mps.is_available():  # Should return True for MPS availability        
                raise Exception('SemanticEstimatorDeepLabV3: MPS is not available')
            print('SemanticEstimatorDeepLabV3: Using MPS')
        else:
            print('SemanticEstimatorDeepLabV3: Using CPU')

        model = self.model_configs[encoder_name]['model'](self.model_configs[encoder_name]['weights'])
        if model_path != '': # Load pre-trained models
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device).eval()
        transform = self.model_configs[encoder_name]['weights'].transforms()
        
        self.semantics_rgb_map = labels_map_factory(dataset_name)
        self.semantic_sigma2_factor = information_weights_factory(dataset_name)

        if semantic_feature_type not in self.feature_type_configs:
            raise ValueError(f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}")

        semantic_type_config = self.feature_type_configs[semantic_feature_type]
        super().__init__(model, transform, device, semantic_type_config['type'], semantic_type_config['fusion'])

    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        image_torch = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        batch = self.transform(image_torch).unsqueeze(0)
        prediction = self.model(batch)["out"]
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])
        
        if self.semantic_feature_type == SemanticFeatureTypes.LABEL:
            self.semantics = probs.argmax(dim=0).cpu().numpy()
        elif self.semantic_feature_type == SemanticFeatureTypes.PROBABILITY_VECTOR:
            self.semantics = probs.cpu().numpy()

        return self.semantics
    
    def to_rgb(self, semantics, bgr=False):
        if self.semantic_feature_type == SemanticFeatureTypes.LABEL:
            return labels_to_image(semantics, self.semantics_rgb_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureTypes.PROBABILITY_VECTOR:
            return labels_to_image(np.argmax(semantics, axis=-1), self.semantics_rgb_map, bgr=bgr)
        
    def single_to_rgb(self, semantic_des, bgr=False):
        if self.semantic_feature_type == SemanticFeatureTypes.LABEL:
            return single_label_to_color(semantic_des, self.semantics_rgb_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureTypes.PROBABILITY_VECTOR:
            return single_label_to_color(np.argmax(semantic_des, axis=-1), self.semantics_rgb_map, bgr=bgr)
    
    def get_semantic_weight(self, semantic_des):
        if self.semantic_feature_type == SemanticFeatureTypes.LABEL:
            return self.semantic_sigma2_factor[semantic_des]
        elif self.semantic_feature_type == SemanticFeatureTypes.PROBABILITY_VECTOR:
            return self.semantic_sigma2_factor[np.argmax(semantic_des, axis=-1)]