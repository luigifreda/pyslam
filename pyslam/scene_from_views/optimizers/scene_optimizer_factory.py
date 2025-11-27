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

from typing import Dict, Any, Optional
import torch

from .scene_optimizer_base import SceneOptimizerBase
from .dense_scene_optimizer import DenseSceneOptimizer
from .sparse_scene_optimizer import SparseSceneOptimizer
from .scene_optimizer_config import SceneOptimizerType, get_default_config


def scene_optimizer_factory(
    optimizer_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    default_type: SceneOptimizerType = SceneOptimizerType.DENSE,
) -> SceneOptimizerBase:
    """
    Factory function to create a scene optimizer instance.

    Args:
        optimizer_config: Dictionary specifying optimizer configuration.
            Must contain "type" key with value from SceneOptimizerType.
            If None, uses default configuration for default_type.
        device: Device to run optimization on (e.g., 'cuda', 'cpu').
            If None, will be determined by the optimizer.
        default_type: Default optimizer type to use if optimizer_config is None.

    Returns:
        SceneOptimizerBase: Instance of the requested optimizer (supports unified interface)

    Raises:
        ValueError: If optimizer type is not recognized

    Examples:
        >>> # Create dense optimizer with defaults
        >>> optimizer = scene_optimizer_factory()
        >>>
        >>> # Create sparse optimizer with custom config
        >>> config = {
        ...     "type": SceneOptimizerType.SPARSE,
        ...     "subsample": 4,
        ...     "lr1": 0.1,
        ... }
        >>> optimizer = scene_optimizer_factory(config, device="cuda")
        >>>
        >>> # Create optimizer from config dict with string type
        >>> config = {"type": "dense_scene_optimizer", "niter": 500}
        >>> optimizer = scene_optimizer_factory(config)
        >>>
        >>> # Use unified interface (built-in)
        >>> # Create optimizer input using the model's create_optimizer_input() method
        >>> optimizer_input = scene_from_views.create_optimizer_input(
        ...     raw_output=model_output,
        ...     pairs=pairs,
        ...     processed_images=processed_images,
        ... )
        >>> result = optimizer.optimize(optimizer_input)
    """
    if optimizer_config is None:
        optimizer_config = get_default_config(default_type)

    # Extract optimizer type
    optimizer_type_str = optimizer_config.get("type", default_type.value)

    # Handle both enum and string inputs
    if isinstance(optimizer_type_str, SceneOptimizerType):
        optimizer_type = optimizer_type_str
    elif isinstance(optimizer_type_str, str):
        try:
            optimizer_type = SceneOptimizerType(optimizer_type_str)
        except ValueError:
            raise ValueError(
                f"Unknown optimizer type: {optimizer_type_str}. "
                f"Must be one of: {[e.value for e in SceneOptimizerType]}"
            )
    else:
        raise ValueError(
            f"Invalid optimizer type: {optimizer_type_str}. "
            f"Must be a string or SceneOptimizerType enum."
        )

    # Extract optimizer-specific parameters (excluding 'type')
    optimizer_kwargs = {k: v for k, v in optimizer_config.items() if k != "type"}

    # Create optimizer instance
    if optimizer_type == SceneOptimizerType.DENSE:
        return DenseSceneOptimizer(device=device, **optimizer_kwargs)
    elif optimizer_type == SceneOptimizerType.SPARSE:
        return SparseSceneOptimizer(device=device, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
