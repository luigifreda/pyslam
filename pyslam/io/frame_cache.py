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
from typing import Optional, Tuple


class FrameCache:
    """
    Cache for storing decoded frames to enable fast random access.

    Stores frames as: frame_id -> (color_img, depth_img, right_color_img, timestamp)
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize the frame cache.

        Args:
            enabled: Whether caching is enabled by default
        """
        self.enabled = enabled
        self._cache: dict = {}  # frame_id -> (color_img, depth_img, right_color_img, timestamp)

    def enable(self):
        """Enable caching."""
        self.enabled = True

    def disable(self):
        """Disable caching and clear existing cache."""
        self.enabled = False
        self.clear()

    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()

    def store(
        self,
        frame_id: int,
        color_img: Optional[np.ndarray] = None,
        depth_img: Optional[np.ndarray] = None,
        right_color_img: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Store a frame in the cache.

        Args:
            frame_id: Frame identifier
            color_img: Color image (left image for stereo)
            depth_img: Depth image
            right_color_img: Right color image (for stereo)
            timestamp: Timestamp of the frame
        """
        if not self.enabled:
            return

        # Make copies to avoid reference issues
        self._cache[frame_id] = (
            color_img.copy() if color_img is not None else None,
            depth_img.copy() if depth_img is not None else None,
            right_color_img.copy() if right_color_img is not None else None,
            timestamp,
        )

    def get(
        self, frame_id: int
    ) -> Optional[
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]
    ]:
        """
        Retrieve a cached frame.

        Args:
            frame_id: Frame identifier

        Returns:
            Tuple of (color_img, depth_img, right_color_img, timestamp) or None if not cached
        """
        if not self.enabled:
            return None
        return self._cache.get(frame_id)

    def has(self, frame_id: int) -> bool:
        """
        Check if a frame is cached.

        Args:
            frame_id: Frame identifier

        Returns:
            True if frame is cached, False otherwise
        """
        if not self.enabled:
            return False
        return frame_id in self._cache

    def get_color(self, frame_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get cached color image and timestamp.

        Args:
            frame_id: Frame identifier

        Returns:
            Tuple of (color_img, timestamp) or None if not cached
        """
        cached = self.get(frame_id)
        if cached is None:
            return None
        color_img, _, _, timestamp = cached
        if color_img is None:
            return None
        return (color_img.copy(), timestamp)

    def get_depth(self, frame_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get cached depth image and timestamp.

        Args:
            frame_id: Frame identifier

        Returns:
            Tuple of (depth_img, timestamp) or None if not cached
        """
        cached = self.get(frame_id)
        if cached is None:
            return None
        _, depth_img, _, timestamp = cached
        if depth_img is None:
            return None
        return (depth_img.copy(), timestamp)

    def get_right(self, frame_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get cached right color image and timestamp.

        Args:
            frame_id: Frame identifier

        Returns:
            Tuple of (right_color_img, timestamp) or None if not cached
        """
        cached = self.get(frame_id)
        if cached is None:
            return None
        _, _, right_img, timestamp = cached
        if right_img is None:
            return None
        return (right_img.copy(), timestamp)

    def size(self) -> int:
        """Get the number of cached frames."""
        return len(self._cache)

    def memory_usage_mb(self) -> float:
        """
        Estimate memory usage in MB (rough estimate).

        Returns:
            Estimated memory usage in megabytes
        """
        total_bytes = 0
        for color_img, depth_img, right_img, _ in self._cache.values():
            if color_img is not None:
                total_bytes += color_img.nbytes
            if depth_img is not None:
                total_bytes += depth_img.nbytes
            if right_img is not None:
                total_bytes += right_img.nbytes
        return total_bytes / (1024 * 1024)
