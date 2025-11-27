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

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch

from .scene_optimizer_io import SceneOptimizerInput, SceneOptimizerOutput


class SceneOptimizerBase(ABC):
    """
    Base class for scene optimization methods.

    This abstract base class defines the unified interface that all scene optimizers
    must implement to be used interchangeably with SceneFromViews classes.

    All methods operate on SceneOptimizerInput and SceneOptimizerOutput for a
    consistent, unified interface across different optimizer implementations.
    """

    def __init__(self, device=None, **kwargs):
        """
        Initialize the scene optimizer.

        Args:
            device: Device to run optimization on (e.g., 'cuda', 'cpu')
            **kwargs: Additional optimizer-specific parameters
        """
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def optimize(
        self,
        optimizer_input: "SceneOptimizerInput",
        verbose: bool = True,
        **optimizer_kwargs,
    ) -> "SceneOptimizerOutput":
        """
        Run scene optimization to align multiple views.

        Args:
            optimizer_input: SceneOptimizerInput containing model output and metadata
            verbose: If True, print progress messages
            **optimizer_kwargs: Additional optimizer-specific parameters

        Returns:
            SceneOptimizerOutput: Unified output containing optimized scene
        """
        pass

    @abstractmethod
    def extract_results(
        self,
        optimizer_output: "SceneOptimizerOutput",
        processed_images: List = None,
        **kwargs,
    ) -> "SceneOptimizerOutput":
        """
        Extract results from the optimized scene.

        Args:
            optimizer_output: SceneOptimizerOutput from optimize()
            processed_images: Preprocessed images
            **kwargs: Additional parameters (e.g., clean_depth, use_tsdf, TSDF_thresh)

        Returns:
            SceneOptimizerOutput: Updated output with extracted results
        """
        pass

    @property
    @abstractmethod
    def optimizer_type(self) -> str:
        """
        Return the optimizer type identifier.

        Returns:
            String identifier (e.g., 'dense_scene_optimizer', 'sparse_scene_optimizer')
        """
        pass

    def can_optimize_from_result(self, result: Any) -> bool:
        """
        Check if this optimizer can work with a given SceneFromViewsResult.

        This method allows optimizers to be used as post-processing steps
        on results from other SceneFromViews classes.

        Args:
            result: SceneFromViewsResult object

        Returns:
            True if this optimizer can process the result, False otherwise
        """
        # Default implementation: check if result has required attributes
        required_attrs = ["camera_poses", "depth_predictions", "intrinsics"]
        return all(hasattr(result, attr) for attr in required_attrs)

    def optimize_from_result(
        self,
        result: Any,
        processed_images: List = None,
        verbose: bool = True,
        **optimizer_kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize scene from a SceneFromViewsResult (post-processing mode).

        This allows using optimizers as post-processing steps on results
        from other SceneFromViews classes.

        Args:
            result: SceneFromViewsResult object from another SceneFromViews class
            processed_images: Preprocessed images
            verbose: If True, print progress messages
            **optimizer_kwargs: Additional optimizer-specific parameters

        Returns:
            Dictionary containing optimized scene data (same format as optimize())
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support optimization from SceneFromViewsResult. "
            "This optimizer requires model-specific output format."
        )
