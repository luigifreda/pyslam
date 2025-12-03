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
from scipy.spatial import cKDTree
from typing import Tuple, List, Union, Optional
import scipy.sparse


class CKDTreeWrapper:
    """
    Python wrapper around scipy.spatial.cKDTree that matches the C++ CKDTree interface exactly.

    This class provides the same interface as the C++ implementation for any dimension D:
    - Constructor: CKDTree(points) where points.shape[1] = D
    - Properties: n (number of points), d (dimensions)
    - Methods: query(x, k, return_distance=True), query_ball_point(x, r),
              query_pairs(r), sparse_distance_matrix(other, max_distance)
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize KDTree with D-dimensional points.

        Args:
            points: numpy array of shape (N, D) containing D-dimensional points
        """
        if points.ndim != 2:
            raise ValueError("points must be 2D (N, D)")
        if points.shape[0] == 0:
            raise ValueError("points cannot be empty")

        self._points = points.copy()  # Store points for reference
        self._tree = cKDTree(points)
        self._n = points.shape[0]
        self._d = points.shape[1]  # Dynamic dimension

    @property
    def n(self) -> int:
        """Number of points in the tree"""
        return self._n

    @property
    def d(self) -> int:
        """Number of dimensions"""
        return self._d

    def query(
        self, x: np.ndarray, k: int = 1, return_distance: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Query k nearest neighbors.

        Args:
            x: Query point of shape (D,)
            k: Number of nearest neighbors to return
            return_distance: Whether to return distances

        Returns:
            Tuple of (distances, indices) as numpy arrays
            If return_distance=False, distances will be None
        """
        if x.ndim != 1 or x.shape[0] != self._d:
            raise ValueError(f"x must be 1D array of length {self._d}")

        # scipy query always returns both distances and indices
        distances, indices = self._tree.query(x, k=k)

        # Handle scipy's scalar return when k=1
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])
        else:
            distances = np.array(distances)
            indices = np.array(indices)

        # Convert indices to int64 to match C++ interface
        indices = indices.astype(np.int64)

        if return_distance:
            return distances, indices
        else:
            # Return None for distances when return_distance=False (C++ behavior)
            return None, indices

    def query_ball_point(self, x: np.ndarray, r: float) -> np.ndarray:
        """
        Query all points within radius r of query point x.

        Args:
            x: Query point of shape (D,)
            r: Radius

        Returns:
            Array of indices of points within radius r
        """
        if x.ndim != 1 or x.shape[0] != self._d:
            raise ValueError(f"x must be 1D array of length {self._d}")

        indices = self._tree.query_ball_point(x, r)

        # Convert to numpy array and ensure int64 type
        indices = np.array(indices, dtype=np.int64)

        return indices

    def query_pairs(self, r: float) -> List[Tuple[int, int]]:
        """
        Find all pairs of points within distance r.

        Args:
            r: Maximum distance between pairs

        Returns:
            List of tuples (i, j) where i < j and distance between points i and j <= r
        """
        pairs = self._tree.query_pairs(r)

        # Convert to list of tuples with int64 indices
        return [(int(i), int(j)) for i, j in pairs]

    def sparse_distance_matrix(
        self, other: "CKDTreeWrapper", max_distance: float
    ) -> List[Tuple[int, int, float]]:
        """
        Compute sparse distance matrix between this tree and another tree.

        Args:
            other: Another CKDTreeWrapper instance
            max_distance: Maximum distance to include in the matrix

        Returns:
            List of tuples (i, j, distance) for pairs within max_distance
        """
        if not isinstance(other, CKDTreeWrapper):
            raise ValueError("other must be a CKDTreeWrapper instance")

        if self._d != other._d:
            raise ValueError(f"Dimension mismatch: {self._d}D vs {other._d}D")

        # Use scipy's sparse_distance_matrix
        sparse_matrix = self._tree.sparse_distance_matrix(other._tree, max_distance)

        # Convert sparse matrix to list of triplets
        coo = sparse_matrix.tocoo()
        triplets = []

        for i, j, dist in zip(coo.row, coo.col, coo.data):
            triplets.append((int(i), int(j), float(dist)))

        return triplets


# Factory functions for specific dimensions (matching C++ interface)
def CKDTree2d(points: np.ndarray) -> CKDTreeWrapper:
    """
    Create a 2D KDTree wrapper.

    Args:
        points: numpy array of shape (N, 2)

    Returns:
        CKDTreeWrapper instance with d=2
    """
    if points.shape[1] != 2:
        raise ValueError("CKDTree2d requires points with 2 columns")
    return CKDTreeWrapper(points)


def CKDTree3d(points: np.ndarray) -> CKDTreeWrapper:
    """
    Create a 3D KDTree wrapper.

    Args:
        points: numpy array of shape (N, 3)

    Returns:
        CKDTreeWrapper instance with d=3
    """
    if points.shape[1] != 3:
        raise ValueError("CKDTree3d requires points with 3 columns")
    return CKDTreeWrapper(points)


def CKDTreeDyn(points: np.ndarray) -> CKDTreeWrapper:
    """
    Create a dynamic-dimension KDTree wrapper.

    Args:
        points: numpy array of shape (N, D) for any D

    Returns:
        CKDTreeWrapper instance with d=D
    """
    return CKDTreeWrapper(points)


# For backward compatibility, also create the wrapper class alias
CKDTree2dWrapper = CKDTreeWrapper


def test_wrapper():
    """Test the wrapper to ensure it works correctly for different dimensions"""
    print("Testing CKDTreeWrapper for different dimensions...")

    # Test 2D
    print("\n=== Testing 2D ===")
    points_2d = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    tree_2d = CKDTree2d(points_2d)

    print(f"2D: n = {tree_2d.n}, d = {tree_2d.d}")
    assert tree_2d.n == 4
    assert tree_2d.d == 2

    query_point_2d = np.array([0.5, 0.5])
    distances, indices = tree_2d.query(query_point_2d, k=2, return_distance=True)
    print(f"2D Query result: distances={distances}, indices={indices}")

    # Test 3D
    print("\n=== Testing 3D ===")
    points_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    tree_3d = CKDTree3d(points_3d)

    print(f"3D: n = {tree_3d.n}, d = {tree_3d.d}")
    assert tree_3d.n == 4
    assert tree_3d.d == 3

    query_point_3d = np.array([0.5, 0.5, 0.5])
    distances, indices = tree_3d.query(query_point_3d, k=2, return_distance=True)
    print(f"3D Query result: distances={distances}, indices={indices}")

    # Test dynamic dimension
    print("\n=== Testing Dynamic Dimension ===")
    points_4d = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    tree_4d = CKDTreeDyn(points_4d)

    print(f"4D: n = {tree_4d.n}, d = {tree_4d.d}")
    assert tree_4d.n == 3
    assert tree_4d.d == 4

    query_point_4d = np.array([0.5, 0.5, 0.5, 0.5])
    distances, indices = tree_4d.query(query_point_4d, k=2, return_distance=True)
    print(f"4D Query result: distances={distances}, indices={indices}")

    # Test query_ball_point for different dimensions
    print("\n=== Testing query_ball_point ===")
    indices_2d = tree_2d.query_ball_point(query_point_2d, r=1.5)
    print(f"2D Ball query result: {indices_2d}")

    indices_3d = tree_3d.query_ball_point(query_point_3d, r=1.5)
    print(f"3D Ball query result: {indices_3d}")

    # Test query_pairs
    print("\n=== Testing query_pairs ===")
    pairs_2d = tree_2d.query_pairs(r=2.0)
    print(f"2D Pairs result: {pairs_2d}")

    pairs_3d = tree_3d.query_pairs(r=2.0)
    print(f"3D Pairs result: {pairs_3d}")

    # Test sparse_distance_matrix
    print("\n=== Testing sparse_distance_matrix ===")
    other_points_2d = points_2d + 0.1
    other_tree_2d = CKDTree2d(other_points_2d)
    triplets_2d = tree_2d.sparse_distance_matrix(other_tree_2d, max_distance=2.0)
    print(f"2D Sparse matrix result: {triplets_2d}")

    other_points_3d = points_3d + 0.1
    other_tree_3d = CKDTree3d(other_points_3d)
    triplets_3d = tree_3d.sparse_distance_matrix(other_tree_3d, max_distance=2.0)
    print(f"3D Sparse matrix result: {triplets_3d}")

    # Test error handling
    print("\n=== Testing Error Handling ===")
    try:
        CKDTree2d(points_3d)  # Should fail - 3D points for 2D tree
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    try:
        CKDTree3d(points_2d)  # Should fail - 2D points for 3D tree
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    try:
        tree_2d.query(np.array([0.5]))  # Should fail - wrong dimension
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_wrapper()
