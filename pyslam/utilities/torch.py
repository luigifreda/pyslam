"""
* This file is part of PYSLAM
* Adpated from adapted from https://github.com/lzx551402/contextdesc/blob/master/utils/tf.py, see the license therein.
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

import os
import torch
import numpy as np


# Convert to numpy
def to_np(x, ret_type=float) -> np.ndarray:
    x_np: np.ndarray = None
    if type(x) == torch.Tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)
    x_np = x_np.astype(ret_type)
    return x_np


def to_device(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x):
    return to_device(x, "numpy")


def to_cpu(x):
    return to_device(x, "cpu")


def to_cuda(x):
    return to_device(x, "cuda")


def safe_empty_cache() -> None:
    """Aggressively free CUDA/CPU memory."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert batched SE3 matrices."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    Tin = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    Tin = Tin.clone()
    Tin[..., :3, :3] = Rt
    Tin[..., :3, 3] = t_inv
    return Tin
