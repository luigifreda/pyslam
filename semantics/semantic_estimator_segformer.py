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
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from torchvision import transforms

from semantic_estimator_base import SemanticEstimator
from utils_semantics import labels_map_factory

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

class SemanticEstimatorSegformer(SemanticEstimator):
    # Segformer available models: https://huggingface.co/models?search=nvidia/segformer
    # They can be configured by:
    # - Model variant: b0, b1, b2, b3, b4, b5
    # - Sizes: (512,512), (512,1024), (768,768), (1024,1024), (640, 1280)
    # - Dataset: cityscapes, ade
    # Check the specific available configurations
    available_configs = [
        ('b0', (1024, 1024), 'cityscapes'),
        ('b0', (512, 512), 'ade'),
        ('b0', (512, 1024), 'cityscapes'),
        ('b0', (640, 1280), 'cityscapes'),
        ('b0', (768, 768), 'cityscapes'),
        ('b1', (1024, 1024), 'cityscapes'),
        ('b1', (512, 512), 'ade'),
        ('b2', (1024, 1024), 'cityscapes'),
        ('b2', (512, 512), 'ade'),
        ('b3', (1024, 1024), 'cityscapes'),
        ('b3', (512, 512), 'ade'),
        ('b4', (1024, 1024), 'cityscapes'),
        ('b4', (512, 512), 'ade'),
        ('b5', (1024, 1024), 'cityscapes'),
    ]
    def __init__(self, device=None, encoder_name='b0', dataset_name='cityscapes', image_size=(512, 1024), model_path=''):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != 'cuda':
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == 'cuda':
            print('SemanticEstimatorSegformer: Using CUDA')
        elif device.type == 'mps':
            if not torch.backends.mps.is_available():  # Should return True for MPS availability        
                raise Exception('SemanticEstimatorSegformer: MPS is not available')
            print('SemanticEstimatorSegformer: Using MPS')
        else:
            print('SemanticEstimatorSegformer: Using CPU')

        # Check if selected config is available
        if (encoder_name, image_size, dataset_name) not in self.available_configs:
            raise ValueError(f"Segformer does not support {encoder_name} model with size {image_size} and dataset {dataset_name}")
        
        if model_path == '': # Load pre-trained models
            model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-{encoder_name}-finetuned-{dataset_name}-{image_size[0]}-{image_size[1]}")
        else:
            raise NotImplementedError("Segformer only supports pre-trained model for now") #TODO(@dvdmc): allow to load a custom model
        
        transform = AutoImageProcessor.from_pretrained(f"nvidia/segformer-{encoder_name}-finetuned-{dataset_name}-{image_size[0]}-{image_size[1]}")
        model = model.to(device).eval()
        semantics_map = labels_map_factory(dataset_name)
        super().__init__(model, transform, device, semantics_map)

    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        image_pil = Image.fromarray(image)
        batch = self.transform(images=image_pil, return_tensors="pt").to(self.device)
        prediction = self.model(**batch).logits
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])
        self.semantics = probs.argmax(dim=0).cpu().numpy() # Labels in this case
        return self.semantics