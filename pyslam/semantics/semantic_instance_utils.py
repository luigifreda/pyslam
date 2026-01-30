"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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

"""
Utility helpers for instance segmentation outputs.
"""

from typing import Optional

import cv2
import numpy as np


def ensure_unique_instance_ids(
    instance_ids: Optional[np.ndarray],
    background_id: int = 0,
    min_pixels: int = 1,
    start_id: int = 1,
) -> Optional[np.ndarray]:
    """
    Relabel instance IDs so each connected component gets a unique ID.

    Args:
        instance_ids: 2D array of instance IDs (H, W) or None.
        background_id: Value treated as background (and any value <= background_id).
        min_pixels: Minimum component size to keep; smaller components become background.
        start_id: First ID to assign to the new components.

    Returns:
        2D array of relabeled instance IDs (int32), or None if input is None.
    """
    if instance_ids is None:
        return None

    instance_ids = np.asarray(instance_ids)
    if instance_ids.size == 0:
        return instance_ids.astype(np.int32, copy=False)
    if instance_ids.ndim != 2:
        raise ValueError("ensure_unique_instance_ids expects a 2D instance-id map")

    # Initialize output with background and preserve any negative labels.
    output = np.where(instance_ids <= background_id, instance_ids, background_id).astype(
        np.int32, copy=False
    )
    next_id = int(start_id)

    for instance_id in np.unique(instance_ids):
        if instance_id <= background_id:
            continue
        mask = instance_ids == instance_id
        if np.count_nonzero(mask) < min_pixels:
            continue
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for comp_id in range(1, num_labels):
            comp_mask = labels == comp_id
            if np.count_nonzero(comp_mask) < min_pixels:
                continue
            output[comp_mask] = next_id
            next_id += 1

    return output
