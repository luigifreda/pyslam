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

from .semantic_labels import (
    get_voc_labels,
    get_cityscapes_labels,
    get_ade20k_labels,
    get_nyu40_labels,
)
from .semantic_types import SemanticDatasetType


def semantic_labels_factory(semantic_dataset_type=SemanticDatasetType.CITYSCAPES):
    if semantic_dataset_type == SemanticDatasetType.VOC:
        return get_voc_labels()
    elif semantic_dataset_type == SemanticDatasetType.CITYSCAPES:
        return get_cityscapes_labels()
    elif semantic_dataset_type == SemanticDatasetType.ADE20K:
        return get_ade20k_labels()
    elif semantic_dataset_type == SemanticDatasetType.NYU40:
        return get_nyu40_labels()
    elif semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
        raise ValueError("CUSTOM_SET does not have predefined labels")
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))
