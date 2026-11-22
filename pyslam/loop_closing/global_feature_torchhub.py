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

Global (place-recognition) descriptors published through torch.hub, pinned to a commit.

One extractor class serves every such model; a model is a TorchHubGlobalFeatureSpec.
The interface matches GlobalFeatureMegaloc (compute_features, compute_features_step,
dim, device), so LoopDetectorVprBase uses them interchangeably. Models are created in
the loop-detecting process (LoopDetectorVprBase.init), never at import.

torch.hub caches the pinned code under $TORCH_HOME/hub and the checkpoints under
$TORCH_HOME/hub/checkpoints; with both cached no network access is needed.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


@dataclass(frozen=True)
class TorchHubGlobalFeatureSpec:
    name: str
    repo: str  # "owner/repo:commit"
    entrypoint: str
    dim: int
    image_size: tuple  # (height, width) used by the upstream evaluation
    kwargs: dict = field(default_factory=dict)
    license: str = ""
    reference: str = ""


SALAD = TorchHubGlobalFeatureSpec(
    name="SALAD",
    repo="serizba/salad:6aede13a3f6c25750bf7fde10209c06cb73060bb",
    entrypoint="dinov2_salad",
    dim=8448,  # 64 clusters x 128 + 256 global token
    image_size=(322, 322),
    license="GPL-3.0 (code and weights); DINOv2 backbone Apache-2.0",
    reference="Izquierdo and Civera, 'Optimal Transport Aggregation for Visual Place Recognition', CVPR 2024",
)

BOQ = TorchHubGlobalFeatureSpec(
    name="BoQ",
    repo="amaralibey/Bag-of-Queries:1a4965ea7dfd9bd0dd846adf7a0e430f68101d12",
    entrypoint="get_trained_boq",
    kwargs={"backbone_name": "dinov2", "output_dim": 12288},
    dim=12288,
    image_size=(322, 322),
    license="MIT; DINOv2 backbone Apache-2.0",
    reference="Ali-bey et al., 'BoQ: A Place is Worth a Bag of Learnable Queries', CVPR 2024",
)


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


class GlobalFeatureTorchHub(torch.nn.Module):
    def __init__(self, spec: TorchHubGlobalFeatureSpec, device=None, share_memory=False, batch_size=8):
        super().__init__()
        self.spec = spec
        self.device = torch.device(device) if device is not None else _default_device()
        print(f"GlobalFeature{spec.name} - device: {self.device}, repo: {spec.repo}")
        self.model = self._build_model().eval().to(self.device)
        if share_memory:
            self.model.share_memory()
        self.dim = spec.dim
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(spec.image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self):
        return torch.hub.load(self.spec.repo, self.spec.entrypoint, trust_repo=True, **self.spec.kwargs)

    def _to_rgb(self, img):
        if img.ndim == 2:
            return np.repeat(img[:, :, None], 3, axis=2)
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        return img

    @torch.inference_mode()
    def _encode(self, batch):
        out = self.model(batch.to(self.device, non_blocking=True))
        if isinstance(out, (tuple, list)):  # BoQ returns (descriptor, attentions)
            out = out[0]
        out = F.normalize(out.flatten(1).float(), p=2, dim=-1)
        if out.shape[1] != self.dim:
            raise RuntimeError(f"{self.spec.name}: descriptor dim {out.shape[1]} != {self.dim}")
        return out.cpu().numpy()

    def compute_features_step(self, img: np.ndarray) -> np.ndarray:
        return self._encode(self.transform(self._to_rgb(img)).unsqueeze(0))

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        feats = np.empty((len(imgs), self.dim), dtype=np.float32)
        for start in range(0, len(imgs), self.batch_size):
            chunk = imgs[start : start + self.batch_size]
            batch = torch.stack([self.transform(self._to_rgb(img)) for img in chunk])
            feats[start : start + len(chunk)] = self._encode(batch)
        return feats


def github_snapshot(repo):
    """Pinned source of "owner/repo:commit" in torch.hub's cache folder (same layout as
    torch.hub.load), downloaded once as a GitHub archive. Nothing is imported from it."""
    import os
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    owner_repo, sha = repo.split(":")
    owner, name = owner_repo.split("/")
    folder = os.path.join(torch.hub.get_dir(), f"{owner}_{name}_{sha}")
    if os.path.isdir(folder):
        return folder
    from pyslam.visual_models.weights import offline_requested

    if offline_requested():
        raise RuntimeError(f"{repo} is not cached in {torch.hub.get_dir()} (offline mode)")
    os.makedirs(torch.hub.get_dir(), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=torch.hub.get_dir()) as tmp:
        archive = os.path.join(tmp, "src.zip")
        urllib.request.urlretrieve(f"https://github.com/{owner}/{name}/archive/{sha}.zip", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
            top = os.path.join(tmp, zf.namelist()[0].split("/")[0])
        shutil.move(top, folder)
    return folder


def load_module_from_file(module_name, path):
    """Import one vendor file under a private name: no sys.path change and no clash with
    other repos' generic top-level packages (e.g. `models`)."""
    import importlib.util
    import sys

    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _BackboneAggregator(torch.nn.Module):
    # Same parameter names as SALAD's VPRModel ("backbone.", "aggregator."): the released
    # checkpoint loads with strict=True.
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        return self.aggregator(self.backbone(x))


class GlobalFeatureSalad(GlobalFeatureTorchHub):
    """SALAD built from its pinned source without the Lightning training wrapper
    (vpr_model.py imports pytorch_lightning, hence wandb). Configuration and checkpoint
    are exactly those of the upstream dinov2_salad() hub entry point."""

    kCheckpointUrl = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"

    def __init__(self, device=None, share_memory=False):
        super().__init__(SALAD, device=device, share_memory=share_memory)

    def _build_model(self):
        import os

        root = github_snapshot(self.spec.repo)
        dinov2 = load_module_from_file("pyslam_vendor_salad_dinov2", os.path.join(root, "models/backbones/dinov2.py"))
        salad = load_module_from_file("pyslam_vendor_salad_salad", os.path.join(root, "models/aggregators/salad.py"))
        backbone = dinov2.DINOv2(model_name="dinov2_vitb14", num_trainable_blocks=4, return_token=True, norm_layer=True)
        aggregator = salad.SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
        model = _BackboneAggregator(backbone, aggregator)
        state = torch.hub.load_state_dict_from_url(self.kCheckpointUrl, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return model


class GlobalFeatureBoQ(GlobalFeatureTorchHub):
    def __init__(self, device=None, share_memory=False):
        super().__init__(BOQ, device=device, share_memory=share_memory)
