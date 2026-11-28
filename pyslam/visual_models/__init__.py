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
* Visual-model metadata and explicit construction, without importing optional models.
*
* The serialized feature enums and FeatureInfo remain the public compatibility API.
"""

from .model_registry import ModelSpec, get_spec, list_models, spec_for_enum, validate_frontend
from .factory import get_local_feature, get_pair_matcher, get_global_descriptor, load_constructor, ModelUnavailable

__all__ = ["ModelSpec", "get_spec", "list_models", "spec_for_enum", "validate_frontend",
           "get_local_feature", "get_pair_matcher", "get_global_descriptor", "load_constructor",
           "ModelUnavailable"]
