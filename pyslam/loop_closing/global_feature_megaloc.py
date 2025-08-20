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
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.multiprocessing as mp

from typing import List
import numpy as np
from tqdm.auto import tqdm


class ImageDataset(data.Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.mytransform = self.input_transform()
        self.images = imgs

    def __getitem__(self, index):
        img = self.images[index]
        img = self.mytransform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform():
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(480),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class GlobalFeatureMegaloc(torch.nn.Module):
    def __init__(self, device=None, share_memory=False):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                print("GlobalFeatureMegaloc - Using GPU")
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("GlobalFeatureMegaloc - Using MPS")
                self.device = torch.device("mps")
            else:
                print("GlobalFeatureMegaloc - Using CPU")
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")

        if share_memory:
            print("GlobalFeatureMegaloc - Share memory")
            self.model.share_memory()  # Share the model parameters among processes

        self.dim = 2048
        self.model = self.model.to(self.device)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        # use_cuda = torch.cuda.is_available()
        use_cuda = self.device.type == "cuda"
        img_set = ImageDataset(imgs)
        test_data_loader = DataLoader(
            dataset=img_set, num_workers=4, batch_size=4, shuffle=False, pin_memory=use_cuda
        )
        self.model.eval()
        with torch.no_grad():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            test_data_ = tqdm(test_data_loader) if len(imgs) > 1 else test_data_loader
            for input_data, indices in test_data_:
                indices_np = indices.numpy()
                input_data = input_data.to(self.device)
                image_encoding = self.model(input_data)
                global_feats[indices_np, :] = image_encoding.cpu().numpy()
        return global_feats

    def compute_features_step(self, img: np.ndarray) -> np.ndarray:
        # use_cuda = torch.cuda.is_available()
        use_cuda = self.device.type == "cuda"
        self.model.eval()
        with torch.no_grad():

            # # Ensure image has 3 channels (RGB)
            # if img.ndim == 2:  # If grayscale
            #     img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            # elif img.shape[2] == 1:  # If single channel
            #     img = np.repeat(img, 3, axis=2)

            # Apply transformations to the image
            img = ImageDataset.input_transform()(img)
            # Add a batch dimension
            input_data = img.unsqueeze(0).to(self.device)
            # Compute the image encoding
            image_encoding = self.model(input_data)
        return image_encoding.cpu().numpy()
