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
from utils_semantics import labels_map_factory

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

class SemanticEstimatorDeepLabV3(SemanticEstimator):
    model_configs = {
        'resnet50': {'encoder': 'resnet50', 'model': deeplabv3_resnet50, 'weights': DeepLabV3_ResNet50_Weights.DEFAULT},
        'resnet101': {'encoder': 'resnet101', 'model': deeplabv3_resnet101, 'weights': DeepLabV3_ResNet101_Weights.DEFAULT},
        'mobilenetv3': {'encoder': 'mobilenetv3', 'model': deeplabv3_mobilenet_v3_large, 'weights': DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT},
    }
    def __init__(self, device=None, encoder_name='resnet50', model_path='', dataset_name='voc'):
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
        semantics_rgb_map = labels_map_factory(dataset_name)
        super().__init__(model, transform, device, semantics_rgb_map)

    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        image_torch = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        batch = self.transform(image_torch).unsqueeze(0)
        prediction = self.model(batch)["out"]
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])
        self.semantics = probs.argmax(dim=0).cpu().numpy() # Labels in this case
        return self.semantics