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
*
* Device arguments and single-image/batch equivalence without model downloads.
"""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "thirdparty"))


class _Model(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(1, 2, 3))[:, None].expand(-1, 2048)


class VprDeviceTests(unittest.TestCase):
    def test_device_arguments_and_feature_equivalence(self):
        for module_name, class_name in (
            ("feature_extractor_cosplace", "CosPlaceFeatureExtractor"),
            ("feature_extractor_eigenplaces", "EigenPlacesFeatureExtractor"),
        ):
            module = importlib.import_module("vpr.feature_extraction." + module_name)
            for device in ("cpu", torch.device("cpu")):
                with self.subTest(extractor=class_name, device=device):
                    def loader(**kwargs):
                        kwargs["num_workers"] = 0
                        return torch.utils.data.DataLoader(**kwargs)

                    with patch("torch.hub.load", return_value=_Model()), patch.object(
                        module, "DataLoader", side_effect=loader
                    ):
                        extractor = getattr(module, class_name)(device=device)
                        self.assertEqual(extractor.device, torch.device("cpu"))
                        image = np.full((24, 32, 3), 127, dtype=np.uint8)
                        batch = extractor.compute_features([image])
                        single = extractor.compute_features_step(image)
                        self.assertEqual(single.shape, (1, 2048))
                        np.testing.assert_allclose(single, batch)


if __name__ == "__main__":
    unittest.main()
