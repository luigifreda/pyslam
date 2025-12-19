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

import numpy as np
import ujson as json
from .semantic_types import (
    SemanticFeatureType,
)


def serialize_semantic_des(semantic_des, semantic_type: SemanticFeatureType):
    if semantic_des is None or semantic_type is None:
        return {"type": None, "value": None}
    return {
        "type": semantic_type.name,  # store name, e.g., "LABEL"
        "value": (
            int(semantic_des) if semantic_type == SemanticFeatureType.LABEL else list(semantic_des)
        ),
    }


def deserialize_semantic_des(data):
    """
    Deserialize semantic descriptor data.

    Args:
        data: Can be None, dict, or string (JSON-encoded dict)

    Returns:
        tuple: (semantic_des, semantic_type) where both can be None
    """
    # Handle None values
    if data is None:
        return None, None

    # Handle stringified JSON (Python serialization)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return None, None

    # Handle dict (C++ serialization or parsed Python serialization)
    if isinstance(data, dict):
        data_type = data.get("type")
        semantic_type = SemanticFeatureType[data_type] if data_type is not None else None
        value = data.get("value")
        if semantic_type is None or value is None:
            return None, None
        if semantic_type == SemanticFeatureType.LABEL:
            return int(value), semantic_type
        else:
            return np.array(value), semantic_type

    # Unknown format
    return None, None
