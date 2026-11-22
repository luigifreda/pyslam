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

Opt-in batched LightGlue (Parameters.kLightGlueBatchedMatching, PYSLAM_LIGHTGLUE_BATCHED=1).

Matches one frame against several frames (compute_frame_matches) in one forward pass:
the transformer layers (the cost) run on a zero-padded batch with per-item attention
masks shaped (B, 1, M, 1), so padding never attends or is attended to; the final
assignment then runs per pair on the unpadded descriptors, so padding never enters the
softmax. Differences from single-pair LightGlue, accepted for this opt-in: no adaptive
early stopping / point pruning (both are per-pair decisions), and padded-row numerics.
The keypoint normalization (image_size convention) is the same as LightGlueMatcher's.
"""

import numpy as np
import torch


def _pad(tensors, length):
    batch = tensors[0].new_zeros((len(tensors), length, tensors[0].shape[-1]))
    mask = torch.zeros((len(tensors), 1, length, 1), dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        batch[i, : t.shape[0]] = t
        mask[i, 0, : t.shape[0]] = True
    return batch, mask


@torch.inference_mode()
def lightglue_match_batch(model, items, device):
    """items: list of dicts with numpy 'kpts0', 'desc0', 'size0', 'kpts1', 'desc1', 'size1'
    (+ 'scales0'/'oris0'/'scales1'/'oris1' when the model uses scale/orientation).
    Returns a list of (idxs0, idxs1) int32 arrays, one per item."""
    from lightglue.lightglue import filter_matches, normalize_keypoints

    conf = model.conf
    tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=torch.float32)  # noqa: E731
    sides = {}
    for side in ("0", "1"):
        kpts, descs, counts = [], [], []
        for item in items:
            k = normalize_keypoints(tensor(item["kpts" + side])[None], tensor(item["size" + side])[None])[0]
            if conf.add_scale_ori:
                k = torch.cat([k, tensor(item["scales" + side])[:, None], tensor(item["oris" + side])[:, None]], -1)
            kpts.append(k)
            descs.append(tensor(item["desc" + side]))
            counts.append(k.shape[0])
        length = max(max(counts), 1)
        kpts, mask = _pad(kpts, length)
        descs, _ = _pad(descs, length)
        sides[side] = dict(desc=model.input_proj(descs), enc=model.posenc(kpts), mask=mask, counts=counts)
    desc0, desc1 = sides["0"]["desc"], sides["1"]["desc"]
    for layer in model.transformers[: conf.n_layers]:
        desc0, desc1 = layer(desc0, desc1, sides["0"]["enc"], sides["1"]["enc"],
                             mask0=sides["0"]["mask"], mask1=sides["1"]["mask"])
    results = []
    assignment = model.log_assignment[conf.n_layers - 1]
    for b, (m, n) in enumerate(zip(sides["0"]["counts"], sides["1"]["counts"])):
        if m == 0 or n == 0:
            results.append((np.array([], np.int32), np.array([], np.int32)))
            continue
        scores, _ = assignment(desc0[b:b + 1, :m], desc1[b:b + 1, :n])
        m0, _, _, _ = filter_matches(scores, conf.filter_threshold)
        valid = m0[0] > -1
        idx0 = torch.nonzero(valid)[:, 0]
        results.append((idx0.cpu().numpy().astype(np.int32), m0[0][valid].cpu().numpy().astype(np.int32)))
    return results
