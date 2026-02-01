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

import numpy as np
import os
import sys
import platform

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from torchvision import transforms

from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_utils import labels_color_map_factory, labels_to_image

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SemanticSegmentationDeepLabV3(SemanticSegmentationBase):
    model_configs = {
        "resnet50": {
            "encoder": "resnet50",
            "model": deeplabv3_resnet50,
            "weights": DeepLabV3_ResNet50_Weights.DEFAULT,
            "dataset": SemanticDatasetType.VOC,
            "image_size": (512, 512),
        },
        "resnet101": {
            "encoder": "resnet101",
            "model": deeplabv3_resnet101,
            "weights": DeepLabV3_ResNet101_Weights.DEFAULT,
            "dataset": SemanticDatasetType.VOC,
            "image_size": (512, 512),
        },
        "mobilenetv3": {
            "encoder": "mobilenetv3",
            "model": deeplabv3_mobilenet_v3_large,
            "weights": DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            "dataset": SemanticDatasetType.VOC,
            "image_size": (512, 512),
        },
    }
    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]

    def __init__(
        self,
        device=None,
        encoder_name="resnet50",
        model_path="",
        semantic_dataset_type=SemanticDatasetType.VOC,
        image_size=(512, 512),
        semantic_feature_type=SemanticFeatureType.LABEL,
        **kwargs,
    ):

        device = self.init_device(device)

        model, transform = self.init_model(device, encoder_name, model_path, semantic_dataset_type)

        self.semantic_color_map = labels_color_map_factory(semantic_dataset_type)

        self.semantic_dataset_type = semantic_dataset_type

        self.encoder_name = encoder_name

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        super().__init__(model, transform, device, semantic_feature_type)

    def init_model(self, device, encoder_name, model_path, semantic_dataset_type):
        if encoder_name not in self.model_configs:
            raise ValueError(
                f"Encoder name {encoder_name} is not supported for {self.__class__.__name__}"
            )
        model = self.model_configs[encoder_name]["model"](
            self.model_configs[encoder_name]["weights"]
        )
        if model_path != "":  # Load pre-trained models
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device).eval()
        transform = self.model_configs[encoder_name]["weights"].transforms()
        return model, transform

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
            print("SemanticSegmentationDeepLabV3: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():  # Should return True for MPS availability
                raise Exception("SemanticSegmentationDeepLabV3: MPS is not available")
            print("SemanticSegmentationDeepLabV3: Using MPS")
        else:
            print("SemanticSegmentationDeepLabV3: Using CPU")
        return device

    def num_classes(self):
        # 1) Prefer the weight metadata (works for torchvision pretrained weights)
        try:
            weights = self.model_configs[self.encoder_name]["weights"]
            meta = getattr(weights, "meta", None)
            if meta and "categories" in meta and meta["categories"]:
                return len(meta["categories"])
        except Exception:
            pass

        # 2) Introspect the classifier head: grab the last Conv2d's out_channels
        try:
            head = getattr(self.model, "classifier", None)
            if isinstance(head, torch.nn.Module):
                for m in reversed(list(head.modules())):
                    if isinstance(m, torch.nn.Conv2d):
                        return m.out_channels
        except Exception:
            pass

        # 3) Last resort: run a tiny forward and read the channel count
        try:
            with torch.no_grad():
                H, W = self.model_configs[self.encoder_name]["image_size"]
                dummy = torch.zeros(1, 3, H, W, device=self.device)
                out = self.model(dummy)["out"]
                return out.shape[1]
        except Exception:
            raise Exception("SemanticSegmentationDeepLabV3: Failed to get number of classes")

    @torch.no_grad()
    def infer(self, image) -> SemanticSegmentationOutput:
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize(
            (prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST
        )
        image_torch = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        batch = self.transform(image_torch).unsqueeze(0)
        prediction = self.model(batch)["out"]
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])

        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            # Ensure int32 labels for downstream consumers (e.g., volumetric integration)
            self.semantics = probs.argmax(dim=0).cpu().numpy().astype(np.int32, copy=False)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()

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
