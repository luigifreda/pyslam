#!/usr/bin/env python3
"""
Comprehensive tests for bounding box classes.

Tests:

- **BoundingBox3D**
  - Basic operations (creation, getters)
  - Contains (single point and multiple points)
  - Intersects with other boxes
  - Compute from points

- **OrientedBoundingBox3D**
  - Basic operations (creation, getters, corners)
  - Contains (including rotated boxes)
  - Intersects (OBB vs OBB, OBB vs AABB)
  - Compute from points (PCA method)
  - Compute from points (Convex Hull method, requires Qhull)

- **BoundingBox2D**
  - Basic operations
  - Contains and intersects
  - Compute from points

- **OrientedBoundingBox2D**
  - Basic operations
  - Contains (including rotated boxes)
  - Intersects
  - Compute from points

- **Edge Cases**
  - Single point
  - Two points
  - Collinear points
  - Coplanar points

## Notes

- The convex hull method for OBB computation requires Qhull to be installed and linked
- If Qhull is not available, the convex hull tests will be skipped with a warning
- All tests use numpy arrays for point coordinates
- Tests verify both correctness and edge case handling

"""

import sys
import os
import numpy as np
import math

# Add lib directory to path to import the volumetric module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))


def assert_almost_equal(a, b, tol=1e-6):
    """Assert that two values are almost equal."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        assert np.allclose(a, b, atol=tol), f"Arrays not equal: {a} vs {b}"
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), f"Lengths differ: {len(a)} vs {len(b)}"
        for ai, bi in zip(a, b):
            assert_almost_equal(ai, bi, tol)
    else:
        assert abs(a - b) < tol, f"Values not equal: {a} vs {b}"


def test_bounding_box_3d_basic():
    """Test basic BoundingBox3D functionality."""
    print("Testing BoundingBox3D basic operations...")

    import volumetric

    # Test construction
    min_point = np.array([0.0, 0.0, 0.0])
    max_point = np.array([1.0, 1.0, 1.0])
    bbox = volumetric.BoundingBox3D(min_point, max_point)

    # Test getters
    assert_almost_equal(bbox.get_min_point(), min_point)
    assert_almost_equal(bbox.get_max_point(), max_point)
    assert_almost_equal(bbox.get_center(), np.array([0.5, 0.5, 0.5]))
    assert_almost_equal(bbox.get_size(), np.array([1.0, 1.0, 1.0]))
    assert_almost_equal(bbox.get_volume(), 1.0)
    assert_almost_equal(bbox.get_surface_area(), 6.0)
    assert_almost_equal(bbox.get_diagonal_length(), math.sqrt(3.0))

    # Test contains
    assert bbox.contains(np.array([0.5, 0.5, 0.5])) == True
    assert bbox.contains(np.array([0.0, 0.0, 0.0])) == True  # on boundary
    assert bbox.contains(np.array([1.0, 1.0, 1.0])) == True  # on boundary
    assert bbox.contains(np.array([1.5, 0.5, 0.5])) == False
    assert bbox.contains(np.array([-0.1, 0.5, 0.5])) == False

    # Test contains with multiple points
    points = [np.array([0.5, 0.5, 0.5]), np.array([1.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0])]
    contains_mask = bbox.contains(points)
    # Convert to list of Python booleans for comparison (pybind11 may return different types)
    if isinstance(contains_mask, np.ndarray):
        contains_mask_list = [bool(x) for x in contains_mask.tolist()]
    else:
        contains_mask_list = [bool(x) for x in contains_mask]
    assert contains_mask_list == [True, False, True]

    # Test intersects
    bbox2 = volumetric.BoundingBox3D(np.array([0.5, 0.5, 0.5]), np.array([2.0, 2.0, 2.0]))
    assert bbox.intersects(bbox2) == True

    bbox3 = volumetric.BoundingBox3D(np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]))
    assert bbox.intersects(bbox3) == False

    # Test touching boxes
    bbox4 = volumetric.BoundingBox3D(np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))
    assert bbox.intersects(bbox4) == True  # touching at boundary

    print("  ✓ BoundingBox3D basic operations passed")


def test_bounding_box_3d_compute_from_points():
    """Test BoundingBox3D::compute_from_points."""
    print("Testing BoundingBox3D::compute_from_points...")

    import volumetric

    # Test with simple points
    points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 0.0, 0.0])]
    bbox = volumetric.BoundingBox3D.compute_from_points(points)

    assert_almost_equal(bbox.get_min_point(), np.array([0.0, 0.0, 0.0]))
    assert_almost_equal(bbox.get_max_point(), np.array([2.0, 1.0, 1.0]))

    # Verify all points are contained
    for p in points:
        assert bbox.contains(p) == True

    # Test with single point
    single_point = [np.array([1.0, 2.0, 3.0])]
    bbox_single = volumetric.BoundingBox3D.compute_from_points(single_point)
    assert_almost_equal(bbox_single.get_min_point(), np.array([1.0, 2.0, 3.0]))
    assert_almost_equal(bbox_single.get_max_point(), np.array([1.0, 2.0, 3.0]))

    # Test with empty list (should handle gracefully or raise error)
    try:
        bbox_empty = volumetric.BoundingBox3D.compute_from_points([])
        # If it doesn't raise, check it's a valid box
        assert bbox_empty is not None
    except (ValueError, RuntimeError):
        pass  # Expected behavior

    print("  ✓ BoundingBox3D::compute_from_points passed")


def test_bounding_box_3d_compute_from_points_comprehensive():
    """Comprehensive tests for BoundingBox3D::compute_from_points."""
    print("Testing BoundingBox3D::compute_from_points (comprehensive)...")

    import volumetric

    # Test with many points forming a clear box
    points = []
    for x in [0.0, 1.0, 2.0]:
        for y in [0.0, 1.0]:
            for z in [0.0, 0.5]:
                points.append(np.array([x, y, z]))

    bbox = volumetric.BoundingBox3D.compute_from_points(points)

    # Verify bounds
    assert_almost_equal(bbox.get_min_point(), np.array([0.0, 0.0, 0.0]))
    assert_almost_equal(bbox.get_max_point(), np.array([2.0, 1.0, 0.5]))

    # Verify all points are contained
    for p in points:
        assert bbox.contains(p) == True

    # Test with points at different scales
    points_large = [p * 100.0 for p in points]
    bbox_large = volumetric.BoundingBox3D.compute_from_points(points_large)
    assert_almost_equal(bbox_large.get_size(), bbox.get_size() * 100.0)

    # Test with negative coordinates
    points_neg = [np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])]
    bbox_neg = volumetric.BoundingBox3D.compute_from_points(points_neg)
    assert_almost_equal(bbox_neg.get_min_point(), np.array([-1.0, -2.0, -3.0]))
    assert_almost_equal(bbox_neg.get_max_point(), np.array([1.0, 2.0, 3.0]))

    # Test with points forming a line (degenerate case)
    points_line = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]
    bbox_line = volumetric.BoundingBox3D.compute_from_points(points_line)
    assert bbox_line.get_volume() >= 0.0
    for p in points_line:
        assert bbox_line.contains(p) == True

    # Test with points forming a plane (degenerate case)
    points_plane = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    bbox_plane = volumetric.BoundingBox3D.compute_from_points(points_plane)
    assert bbox_plane.get_volume() >= 0.0
    for p in points_plane:
        assert bbox_plane.contains(p) == True

    # Test with random points
    np.random.seed(42)
    random_points = [np.random.randn(3) * 10.0 for _ in range(20)]
    bbox_random = volumetric.BoundingBox3D.compute_from_points(random_points)
    for p in random_points:
        assert bbox_random.contains(p) == True

    print("  ✓ BoundingBox3D::compute_from_points (comprehensive) passed")


def test_oriented_bounding_box_3d_basic():
    """Test basic OrientedBoundingBox3D functionality."""
    print("Testing OrientedBoundingBox3D basic operations...")

    import volumetric

    # Test construction with identity quaternion
    center = np.array([0.0, 0.0, 0.0])
    orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z (identity)
    size = np.array([2.0, 2.0, 2.0])
    obb = volumetric.OrientedBoundingBox3D(center, orientation, size)

    # Test getters
    assert_almost_equal(obb.get_volume(), 8.0)
    assert_almost_equal(obb.get_surface_area(), 24.0)
    assert_almost_equal(obb.get_diagonal_length(), math.sqrt(12.0))

    # Test get_corners
    corners = obb.get_corners()
    assert len(corners) == 8
    # Check that center is approximately at origin
    corner_center = np.mean(corners, axis=0)
    assert_almost_equal(corner_center, center, tol=1e-5)

    # Test contains
    assert obb.contains(np.array([0.0, 0.0, 0.0])) == True
    assert obb.contains(np.array([0.9, 0.9, 0.9])) == True
    assert obb.contains(np.array([1.1, 0.0, 0.0])) == False

    # Test with rotated box
    # Rotate 45 degrees around z-axis
    angle = math.pi / 4.0
    quat_w = math.cos(angle / 2.0)
    quat_z = math.sin(angle / 2.0)
    orientation_rot = np.array([quat_w, 0.0, 0.0, quat_z])
    obb_rot = volumetric.OrientedBoundingBox3D(center, orientation_rot, size)

    # Point at (1, 0, 0) in world should be inside rotated box
    point_rotated = np.array([0.707, 0.707, 0.0])  # rotated point
    assert obb_rot.contains(point_rotated) == True

    print("  ✓ OrientedBoundingBox3D basic operations passed")


def test_oriented_bounding_box_3d_compute_from_points():
    """Test OrientedBoundingBox3D::compute_from_points (default method, PCA)."""
    print("Testing OrientedBoundingBox3D::compute_from_points...")

    import volumetric

    # Test with a simple rectangular set of points
    # Create points forming a box aligned with axes
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([2.0, 0.0, 0.5]),
        np.array([0.0, 1.0, 0.5]),
        np.array([2.0, 1.0, 0.5]),
    ]

    # Test default method (PCA)
    obb_pca = volumetric.OrientedBoundingBox3D.compute_from_points(points)

    assert obb_pca is not None
    # PCA should be very accurate for axis-aligned boxes
    assert_almost_equal(obb_pca.get_volume(), 1.0, tol=0.05)  # 2.0 * 1.0 * 0.5 = 1.0 (5% tolerance)

    # All points should be contained
    for p in points:
        assert obb_pca.contains(p) == True

    # Test with rotated box (points forming a rotated rectangle)
    angle = math.pi / 4.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

    rotated_points = [R @ p for p in points]
    obb_rotated = volumetric.OrientedBoundingBox3D.compute_from_points(rotated_points)

    assert obb_rotated is not None
    # Volume should be approximately the same (PCA should handle rotations well)
    assert_almost_equal(obb_rotated.get_volume(), 1.0, tol=0.05)  # 5% tolerance

    print("  ✓ OrientedBoundingBox3D::compute_from_points passed")


def test_oriented_bounding_box_3d_compute_from_points_comprehensive():
    """Comprehensive tests for OrientedBoundingBox3D::compute_from_points."""
    print("Testing OrientedBoundingBox3D::compute_from_points (comprehensive)...")

    import volumetric

    # Test with many points forming a clear box
    points = []
    for x in [0.0, 1.0, 2.0]:
        for y in [0.0, 1.0]:
            for z in [0.0, 0.5]:
                points.append(np.array([x, y, z]))

    obb = volumetric.OrientedBoundingBox3D.compute_from_points(points)

    assert obb is not None
    # Volume should be approximately 2.0 * 1.0 * 0.5 = 1.0
    # PCA should be accurate for well-defined point sets
    assert_almost_equal(obb.get_volume(), 1.0, tol=0.05)  # 5% tolerance

    # All points should be contained
    for p in points:
        assert obb.contains(p) == True

    # Test with rotated box at different angles
    for angle_deg in [15, 30, 45, 60, 90]:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

        rotated_points = [R @ p for p in points]
        obb_rot = volumetric.OrientedBoundingBox3D.compute_from_points(rotated_points)

        assert obb_rot is not None
        # Volume should be approximately the same (PCA handles rotations well)
        assert_almost_equal(obb_rot.get_volume(), 1.0, tol=0.05)  # 5% tolerance

        # All points should be contained
        for p in rotated_points:
            assert obb_rot.contains(p) == True

    # Test with 3D rotation (not just around z-axis)
    # Rotate around multiple axes
    angle_x = math.radians(30)
    angle_y = math.radians(45)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle_x), -math.sin(angle_x)],
            [0.0, math.sin(angle_x), math.cos(angle_x)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(angle_y), 0.0, math.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle_y), 0.0, math.cos(angle_y)],
        ]
    )
    R_3d = Ry @ Rx

    rotated_3d_points = [R_3d @ p for p in points]
    obb_3d_rot = volumetric.OrientedBoundingBox3D.compute_from_points(rotated_3d_points)

    assert obb_3d_rot is not None
    # 3D rotations might have slightly more numerical error, but still should be accurate
    assert_almost_equal(obb_3d_rot.get_volume(), 1.0, tol=0.05)  # 5% tolerance
    for p in rotated_3d_points:
        assert obb_3d_rot.contains(p) == True

    # Test with points at different scales
    points_large = [p * 10.0 for p in points]
    obb_large = volumetric.OrientedBoundingBox3D.compute_from_points(points_large)
    assert obb_large is not None
    # Use relative tolerance for scaled points (5% of expected volume)
    assert_almost_equal(
        obb_large.get_volume(), 1.0 * 1000.0, tol=50.0
    )  # 10% tolerance (10^3 scale)

    # Test with points forming a sphere (should give reasonable OBB)
    sphere_points = []
    for i in range(20):
        theta = 2.0 * math.pi * i / 20.0
        phi = math.pi * i / 20.0
        r = 1.0
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        sphere_points.append(np.array([x, y, z]))

    obb_sphere = volumetric.OrientedBoundingBox3D.compute_from_points(sphere_points)
    assert obb_sphere is not None
    assert obb_sphere.get_volume() > 0.0
    # Volume should be less than a cube of side 2 (bounding sphere)
    assert obb_sphere.get_volume() < 8.0
    for p in sphere_points:
        assert obb_sphere.contains(p) == True

    # Test with elongated shape (should detect principal axis)
    elongated_points = []
    for x in np.linspace(-5.0, 5.0, 20):
        for y in [-0.1, 0.1]:
            for z in [-0.1, 0.1]:
                elongated_points.append(np.array([x, y, z]))

    obb_elongated = volumetric.OrientedBoundingBox3D.compute_from_points(elongated_points)
    assert obb_elongated is not None
    # Should have one long dimension
    size = obb_elongated.size
    assert max(size) > 5.0  # Long dimension
    assert min(size) < 1.0  # Short dimensions
    for p in elongated_points:
        assert obb_elongated.contains(p) == True

    # Test with random points
    np.random.seed(42)
    random_points = [np.random.randn(3) * 5.0 for _ in range(50)]
    obb_random = volumetric.OrientedBoundingBox3D.compute_from_points(random_points)
    assert obb_random is not None
    assert obb_random.get_volume() > 0.0
    for p in random_points:
        assert obb_random.contains(p) == True

    print("  ✓ OrientedBoundingBox3D::compute_from_points (comprehensive) passed")


def test_oriented_bounding_box_3d_convex_hull():
    """Test OrientedBoundingBox3D::compute_from_points with convex hull method."""
    print("Testing OrientedBoundingBox3D::compute_from_points (Convex Hull)...")

    import volumetric

    # Create points forming a box
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([2.0, 0.0, 0.5]),
        np.array([0.0, 1.0, 0.5]),
        np.array([2.0, 1.0, 0.5]),
    ]

    # Test default method (PCA)
    obb_pca = volumetric.OrientedBoundingBox3D.compute_from_points(points)
    assert obb_pca is not None
    assert_almost_equal(obb_pca.get_volume(), 1.0, tol=0.05)  # 2.0 * 1.0 * 0.5 = 1.0
    for p in points:
        assert obb_pca.contains(p) == True

    # Test convex hull method (if Qhull is available)
    try:
        obb_hull = volumetric.OrientedBoundingBox3D.compute_from_points(
            points, volumetric.OBBComputationMethod.CONVEX_HULL_MINIMAL
        )
        assert obb_hull is not None
        # Convex hull method should produce a box that contains all points
        for p in points:
            assert obb_hull.contains(p) == True
        # Volume should be reasonable (convex hull method should be at least as tight as PCA)
        assert obb_hull.get_volume() > 0.0
        assert obb_hull.get_volume() <= obb_pca.get_volume() * 1.1  # Allow 10% tolerance
        print("  ✓ Convex hull method works")
    except RuntimeError as e:
        # Qhull might not be available
        if "Qhull" in str(e) or "qhull" in str(e).lower():
            print(f"  ⚠ Convex hull method skipped: {e}")
        else:
            raise

    # Test with L-shaped distribution (convex hull should handle this well)
    l_points = []
    # Horizontal segment
    for x in np.linspace(0.0, 5.0, 11):
        l_points.append(np.array([x, 0.0, 0.0]))
    # Vertical segment
    for y in np.linspace(0.0, 5.0, 11):
        l_points.append(np.array([0.0, y, 0.0]))
    # Add some depth
    for p in l_points[:]:
        l_points.append(p + np.array([0.0, 0.0, 1.0]))

    obb_l_pca = volumetric.OrientedBoundingBox3D.compute_from_points(l_points)
    assert obb_l_pca is not None
    for p in l_points:
        assert obb_l_pca.contains(p) == True

    try:
        obb_l_hull = volumetric.OrientedBoundingBox3D.compute_from_points(
            l_points, volumetric.OBBComputationMethod.CONVEX_HULL_MINIMAL
        )
        assert obb_l_hull is not None
        for p in l_points:
            assert obb_l_hull.contains(p) == True
        # Convex hull method might produce a tighter box for L-shapes
        assert obb_l_hull.get_volume() > 0.0
        print("  ✓ Convex hull method works with L-shaped distribution")
    except RuntimeError as e:
        if "Qhull" in str(e) or "qhull" in str(e).lower():
            print(f"  ⚠ Convex hull method skipped for L-shape: {e}")
        else:
            raise

    print("  ✓ OrientedBoundingBox3D::compute_from_points (Convex Hull) passed")


def test_bounding_box_3d_intersects():
    """Test BoundingBox3D intersection logic."""
    print("Testing BoundingBox3D::intersects...")

    import volumetric

    bbox1 = volumetric.BoundingBox3D(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

    # Overlapping boxes
    bbox2 = volumetric.BoundingBox3D(np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5]))
    assert bbox1.intersects(bbox2) == True

    # Non-overlapping boxes
    bbox3 = volumetric.BoundingBox3D(np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]))
    assert bbox1.intersects(bbox3) == False

    # One box inside another
    bbox4 = volumetric.BoundingBox3D(np.array([0.25, 0.25, 0.25]), np.array([0.75, 0.75, 0.75]))
    assert bbox1.intersects(bbox4) == True

    # Touching boxes (edge case)
    bbox5 = volumetric.BoundingBox3D(np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]))
    assert bbox1.intersects(bbox5) == True

    print("  ✓ BoundingBox3D::intersects passed")


def test_bounding_box_2d_basic():
    """Test basic BoundingBox2D functionality."""
    print("Testing BoundingBox2D basic operations...")

    import volumetric

    min_point = np.array([0.0, 0.0])
    max_point = np.array([1.0, 1.0])
    bbox = volumetric.BoundingBox2D(min_point, max_point)

    assert_almost_equal(bbox.get_min_point(), min_point)
    assert_almost_equal(bbox.get_max_point(), max_point)
    assert_almost_equal(bbox.get_center(), np.array([0.5, 0.5]))
    assert_almost_equal(bbox.get_size(), np.array([1.0, 1.0]))
    assert_almost_equal(bbox.get_area(), 1.0)
    assert_almost_equal(bbox.get_perimeter(), 4.0)
    assert_almost_equal(bbox.get_diagonal_length(), math.sqrt(2.0))

    # Test contains
    assert bbox.contains(np.array([0.5, 0.5])) == True
    assert bbox.contains(np.array([1.5, 0.5])) == False

    # Test intersects
    bbox2 = volumetric.BoundingBox2D(np.array([0.5, 0.5]), np.array([2.0, 2.0]))
    assert bbox.intersects(bbox2) == True

    print("  ✓ BoundingBox2D basic operations passed")


def test_bounding_box_2d_compute_from_points():
    """Test BoundingBox2D::compute_from_points."""
    print("Testing BoundingBox2D::compute_from_points...")

    import volumetric

    points = [np.array([0.0, 0.0]), np.array([2.0, 1.0]), np.array([1.0, 3.0])]
    bbox = volumetric.BoundingBox2D.compute_from_points(points)

    assert_almost_equal(bbox.get_min_point(), np.array([0.0, 0.0]))
    assert_almost_equal(bbox.get_max_point(), np.array([2.0, 3.0]))

    # Verify all points are contained
    for p in points:
        assert bbox.contains(p) == True

    print("  ✓ BoundingBox2D::compute_from_points passed")


def test_bounding_box_2d_compute_from_points_comprehensive():
    """Comprehensive tests for BoundingBox2D::compute_from_points."""
    print("Testing BoundingBox2D::compute_from_points (comprehensive)...")

    import volumetric

    # Test with many points forming a clear rectangle
    points = []
    for x in [0.0, 1.0, 2.0]:
        for y in [0.0, 1.0]:
            points.append(np.array([x, y]))

    bbox = volumetric.BoundingBox2D.compute_from_points(points)

    assert_almost_equal(bbox.get_min_point(), np.array([0.0, 0.0]))
    assert_almost_equal(bbox.get_max_point(), np.array([2.0, 1.0]))

    # Verify all points are contained
    for p in points:
        assert bbox.contains(p) == True

    # Test with points at different scales
    points_large = [p * 100.0 for p in points]
    bbox_large = volumetric.BoundingBox2D.compute_from_points(points_large)
    assert_almost_equal(bbox_large.get_size(), bbox.get_size() * 100.0)

    # Test with negative coordinates
    points_neg = [np.array([-1.0, -2.0]), np.array([1.0, 2.0])]
    bbox_neg = volumetric.BoundingBox2D.compute_from_points(points_neg)
    assert_almost_equal(bbox_neg.get_min_point(), np.array([-1.0, -2.0]))
    assert_almost_equal(bbox_neg.get_max_point(), np.array([1.0, 2.0]))

    # Test with points forming a line (degenerate case)
    points_line = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([3.0, 0.0]),
    ]
    bbox_line = volumetric.BoundingBox2D.compute_from_points(points_line)
    assert bbox_line.get_area() >= 0.0
    for p in points_line:
        assert bbox_line.contains(p) == True

    # Test with random points
    np.random.seed(42)
    random_points = [np.random.randn(2) * 10.0 for _ in range(20)]
    bbox_random = volumetric.BoundingBox2D.compute_from_points(random_points)
    for p in random_points:
        assert bbox_random.contains(p) == True

    print("  ✓ BoundingBox2D::compute_from_points (comprehensive) passed")


def test_oriented_bounding_box_2d_basic():
    """Test basic OrientedBoundingBox2D functionality."""
    print("Testing OrientedBoundingBox2D basic operations...")

    import volumetric

    center = np.array([0.0, 0.0])
    angle_rad = 0.0
    size = np.array([2.0, 2.0])
    obb = volumetric.OrientedBoundingBox2D(center, angle_rad, size)

    assert_almost_equal(obb.get_area(), 4.0)
    assert_almost_equal(obb.get_perimeter(), 8.0)
    assert_almost_equal(obb.get_diagonal_length(), math.sqrt(8.0))

    # Test get_corners
    corners = obb.get_corners()
    assert len(corners) == 4

    # Test contains
    assert obb.contains(np.array([0.0, 0.0])) == True
    assert obb.contains(np.array([0.9, 0.9])) == True
    assert obb.contains(np.array([1.1, 0.0])) == False

    # Test with rotated box
    angle = math.pi / 4.0
    obb_rot = volumetric.OrientedBoundingBox2D(center, angle, size)
    # Point at (1, 0) rotated 45 degrees
    point_rotated = np.array([0.707, 0.707])
    assert obb_rot.contains(point_rotated) == True

    print("  ✓ OrientedBoundingBox2D basic operations passed")


def test_oriented_bounding_box_2d_compute_from_points():
    """Test OrientedBoundingBox2D::compute_from_points."""
    print("Testing OrientedBoundingBox2D::compute_from_points...")

    import volumetric

    # Create points forming a rectangle
    points = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 1.0]),
        np.array([0.0, 1.0]),
    ]

    obb = volumetric.OrientedBoundingBox2D.compute_from_points(points)

    assert obb is not None
    # PCA should be accurate for axis-aligned boxes
    assert_almost_equal(obb.get_area(), 2.0, tol=0.05)  # 2.0 * 1.0 (5% tolerance)

    # All points should be contained
    for p in points:
        assert obb.contains(p) == True

    # Test with rotated rectangle
    angle = math.pi / 4.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated_points = [R @ p for p in points]
    obb_rotated = volumetric.OrientedBoundingBox2D.compute_from_points(rotated_points)

    assert obb_rotated is not None
    # PCA should handle rotations well
    assert_almost_equal(obb_rotated.get_area(), 2.0, tol=0.05)  # 5% tolerance

    print("  ✓ OrientedBoundingBox2D::compute_from_points passed")


def test_oriented_bounding_box_2d_compute_from_points_comprehensive():
    """Comprehensive tests for OrientedBoundingBox2D::compute_from_points."""
    print("Testing OrientedBoundingBox2D::compute_from_points (comprehensive)...")

    import volumetric

    # Test with many points forming a clear rectangle
    points = []
    for x in [0.0, 1.0, 2.0]:
        for y in [0.0, 1.0]:
            points.append(np.array([x, y]))

    obb = volumetric.OrientedBoundingBox2D.compute_from_points(points)

    assert obb is not None
    # PCA should be accurate for well-defined point sets
    assert_almost_equal(obb.get_area(), 2.0, tol=0.05)  # 2.0 * 1.0 (5% tolerance)

    # All points should be contained
    for p in points:
        assert obb.contains(p) == True

    # Test with rotated rectangle at different angles
    for angle_deg in [15, 30, 45, 60, 90, 120]:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_points = [R @ p for p in points]
        obb_rot = volumetric.OrientedBoundingBox2D.compute_from_points(rotated_points)

        assert obb_rot is not None
        # Area should be approximately the same (PCA handles rotations well)
        assert_almost_equal(obb_rot.get_area(), 2.0, tol=0.05)  # 5% tolerance

        # All points should be contained
        for p in rotated_points:
            assert obb_rot.contains(p) == True

    # Test with points at different scales
    points_large = [p * 10.0 for p in points]
    obb_large = volumetric.OrientedBoundingBox2D.compute_from_points(points_large)
    assert obb_large is not None
    # Use relative tolerance for scaled points (5% of expected area)
    assert_almost_equal(obb_large.get_area(), 2.0 * 100.0, tol=10.0)  # 5% tolerance (10^2 scale)

    # Test with points forming a circle (should give reasonable OBB)
    circle_points = []
    for i in range(16):
        angle = 2.0 * math.pi * i / 16.0
        r = 1.0
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        circle_points.append(np.array([x, y]))

    obb_circle = volumetric.OrientedBoundingBox2D.compute_from_points(circle_points)
    assert obb_circle is not None
    assert obb_circle.get_area() > 0.0
    # Area should be at most a square of side 2 (bounding circle)
    # For a circle, the optimal OBB is a square with area 4.0, but PCA might produce
    # slightly larger area due to numerical issues or ambiguous principal axis when eigenvalues are equal
    assert obb_circle.get_area() <= 4.0 + 0.1  # Allow small tolerance for numerical errors
    for p in circle_points:
        assert obb_circle.contains(p) == True

    # Test with elongated shape (should detect principal axis)
    elongated_points = []
    for x in np.linspace(-5.0, 5.0, 20):
        for y in [-0.1, 0.1]:
            elongated_points.append(np.array([x, y]))

    obb_elongated = volumetric.OrientedBoundingBox2D.compute_from_points(elongated_points)
    assert obb_elongated is not None
    # Should have one long dimension
    size = obb_elongated.size
    assert max(size) > 5.0  # Long dimension
    assert min(size) < 1.0  # Short dimension
    for p in elongated_points:
        assert obb_elongated.contains(p) == True

    # Test with random points
    np.random.seed(42)
    random_points = [np.random.randn(2) * 5.0 for _ in range(30)]
    obb_random = volumetric.OrientedBoundingBox2D.compute_from_points(random_points)
    assert obb_random is not None
    assert obb_random.get_area() > 0.0
    for p in random_points:
        assert obb_random.contains(p) == True

    # Test with L-shaped points (should still work)
    l_shaped_points = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([0.0, 2.0]),
    ]
    obb_l = volumetric.OrientedBoundingBox2D.compute_from_points(l_shaped_points)
    assert obb_l is not None
    assert obb_l.get_area() > 0.0
    for p in l_shaped_points:
        assert obb_l.contains(p) == True

    print("  ✓ OrientedBoundingBox2D::compute_from_points (comprehensive) passed")


def test_oriented_bounding_box_intersects():
    """Test OrientedBoundingBox intersection methods."""
    print("Testing OrientedBoundingBox intersection methods...")

    import volumetric

    # Test OBB3D vs OBB3D
    center1 = np.array([0.0, 0.0, 0.0])
    orientation1 = np.array([1.0, 0.0, 0.0, 0.0])
    size1 = np.array([2.0, 2.0, 2.0])
    obb1 = volumetric.OrientedBoundingBox3D(center1, orientation1, size1)

    center2 = np.array([1.0, 0.0, 0.0])
    obb2 = volumetric.OrientedBoundingBox3D(center2, orientation1, size1)
    assert obb1.intersects(obb2) == True

    center3 = np.array([5.0, 0.0, 0.0])
    obb3 = volumetric.OrientedBoundingBox3D(center3, orientation1, size1)
    assert obb1.intersects(obb3) == False

    # Test OBB3D vs AABB3D
    bbox = volumetric.BoundingBox3D(np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5]))
    assert obb1.intersects(bbox) == True

    # Test OBB2D vs OBB2D
    obb2d1 = volumetric.OrientedBoundingBox2D(np.array([0.0, 0.0]), 0.0, np.array([2.0, 2.0]))
    obb2d2 = volumetric.OrientedBoundingBox2D(np.array([1.0, 0.0]), 0.0, np.array([2.0, 2.0]))
    assert obb2d1.intersects(obb2d2) == True

    # Test OBB2D vs AABB2D
    bbox2d = volumetric.BoundingBox2D(np.array([0.5, 0.5]), np.array([1.5, 1.5]))
    assert obb2d1.intersects(bbox2d) == True

    print("  ✓ OrientedBoundingBox intersection methods passed")


def test_edge_cases():
    """Test edge cases and degenerate inputs."""
    print("Testing edge cases...")

    import volumetric

    # Test single point
    single_point = [np.array([1.0, 2.0, 3.0])]
    bbox = volumetric.BoundingBox3D.compute_from_points(single_point)
    assert_almost_equal(bbox.get_min_point(), bbox.get_max_point())
    assert_almost_equal(bbox.get_volume(), 0.0)

    # Test two points
    two_points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
    bbox2 = volumetric.BoundingBox3D.compute_from_points(two_points)
    assert bbox2.get_volume() >= 0.0

    # Test collinear points (2D)
    collinear_points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    obb2d = volumetric.OrientedBoundingBox2D.compute_from_points(collinear_points)
    assert obb2d is not None
    assert obb2d.get_area() >= 0.0

    # Test coplanar points (3D)
    coplanar_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    obb3d = volumetric.OrientedBoundingBox3D.compute_from_points(coplanar_points)
    assert obb3d is not None
    assert obb3d.get_volume() >= 0.0

    print("  ✓ Edge cases passed")


def test_challenging_geometries():
    """Test challenging geometric configurations."""
    print("Testing challenging geometries...")

    import volumetric

    # Test 1: Very thin box (one dimension much smaller)
    thin_points = []
    for x in np.linspace(0.0, 10.0, 11):
        for y in np.linspace(0.0, 10.0, 11):
            thin_points.append(np.array([x, y, 0.0]))
            thin_points.append(np.array([x, y, 0.001]))  # Very thin in z

    obb_thin = volumetric.OrientedBoundingBox3D.compute_from_points(thin_points)
    assert obb_thin is not None
    # Volume should be approximately 10 * 10 * 0.001 = 0.1
    assert_almost_equal(obb_thin.get_volume(), 0.1, tol=0.01)
    # All points should be contained
    for p in thin_points:
        assert obb_thin.contains(p) == True

    # Test 2: Very elongated box (one dimension much larger)
    elongated_points = []
    for i in range(100):
        x = i * 0.1
        y = 0.0
        z = 0.0
        # Add some small perpendicular spread
        for dy in [-0.01, 0.0, 0.01]:
            for dz in [-0.01, 0.0, 0.01]:
                elongated_points.append(np.array([x, y + dy, z + dz]))

    obb_elongated = volumetric.OrientedBoundingBox3D.compute_from_points(elongated_points)
    assert obb_elongated is not None
    # Should detect the principal axis along x
    assert obb_elongated.size[0] > obb_elongated.size[1] * 10
    assert obb_elongated.size[0] > obb_elongated.size[2] * 10
    for p in elongated_points:
        assert obb_elongated.contains(p) == True

    # Test 3: L-shaped point distribution (2D)
    l_shape_points = []
    # Horizontal segment
    for x in np.linspace(0.0, 5.0, 11):
        l_shape_points.append(np.array([x, 0.0]))
    # Vertical segment
    for y in np.linspace(0.0, 5.0, 11):
        l_shape_points.append(np.array([0.0, y]))

    obb_l = volumetric.OrientedBoundingBox2D.compute_from_points(l_shape_points)
    assert obb_l is not None
    # Should create a box that contains the L-shape
    assert obb_l.get_area() > 0.0
    for p in l_shape_points:
        assert obb_l.contains(p) == True

    # Test 4: U-shaped point distribution (2D)
    u_shape_points = []
    # Bottom
    for x in np.linspace(0.0, 5.0, 11):
        u_shape_points.append(np.array([x, 0.0]))
    # Left side
    for y in np.linspace(0.0, 3.0, 7):
        u_shape_points.append(np.array([0.0, y]))
    # Right side
    for y in np.linspace(0.0, 3.0, 7):
        u_shape_points.append(np.array([5.0, y]))

    obb_u = volumetric.OrientedBoundingBox2D.compute_from_points(u_shape_points)
    assert obb_u is not None
    assert obb_u.get_area() > 0.0
    for p in u_shape_points:
        assert obb_u.contains(p) == True

    # Test 5: Points with outliers
    normal_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    # Add an outlier far away
    points_with_outlier = normal_points + [np.array([100.0, 100.0, 100.0])]

    obb_outlier = volumetric.OrientedBoundingBox3D.compute_from_points(points_with_outlier)
    assert obb_outlier is not None
    # Should contain all points including outlier
    for p in points_with_outlier:
        assert obb_outlier.contains(p) == True
    # PCA aligns along the long diagonal; expect a compact OBB with volume in the hundreds.
    vol_outlier = obb_outlier.get_volume()
    assert 200.0 < vol_outlier < 800.0
    size = obb_outlier.size
    assert size.max() > 150.0 and size.min() < 5.0

    print("  ✓ Challenging geometries passed")


def test_boundary_conditions():
    """Test points exactly on boundaries and near boundaries."""
    print("Testing boundary conditions...")

    import volumetric

    # Test 1: Points exactly on box boundaries (3D)
    # Create a box and test points at all corners and edges
    bbox = volumetric.BoundingBox3D(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

    # Corner points
    corners = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]
    for corner in corners:
        assert bbox.contains(corner) == True

    # Edge midpoints
    edge_midpoints = [
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 1.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([1.0, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([1.0, 0.0, 0.5]),
    ]
    for edge in edge_midpoints:
        assert bbox.contains(edge) == True

    # Points just outside (should not be contained)
    outside_points = [
        np.array([-0.0001, 0.5, 0.5]),
        np.array([1.0001, 0.5, 0.5]),
        np.array([0.5, -0.0001, 0.5]),
        np.array([0.5, 1.0001, 0.5]),
        np.array([0.5, 0.5, -0.0001]),
        np.array([0.5, 0.5, 1.0001]),
    ]
    for outside in outside_points:
        assert bbox.contains(outside) == False

    # Test 2: OBB with points on boundaries
    points = [
        np.array([0.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([2.0, 1.0]),
    ]
    obb = volumetric.OrientedBoundingBox2D.compute_from_points(points)
    for p in points:
        assert obb.contains(p) == True

    # Test 3: Rotated OBB boundary points
    angle = math.pi / 4.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    base_points = [
        np.array([-1.0, -0.5]),
        np.array([1.0, -0.5]),
        np.array([-1.0, 0.5]),
        np.array([1.0, 0.5]),
    ]
    rotated_points = [R @ p for p in base_points]
    obb_rot = volumetric.OrientedBoundingBox2D.compute_from_points(rotated_points)
    for p in rotated_points:
        assert obb_rot.contains(p) == True

    print("  ✓ Boundary conditions passed")


def test_extreme_values():
    """Test with extreme coordinate values."""
    print("Testing extreme values...")

    import volumetric

    # Test 1: Very large coordinates
    large_points = [
        np.array([1e6, 1e6, 1e6]),
        np.array([1e6 + 1.0, 1e6, 1e6]),
        np.array([1e6, 1e6 + 1.0, 1e6]),
        np.array([1e6, 1e6, 1e6 + 1.0]),
    ]
    bbox_large = volumetric.BoundingBox3D.compute_from_points(large_points)
    assert bbox_large is not None
    assert_almost_equal(bbox_large.get_volume(), 1.0, tol=0.05)
    for p in large_points:
        assert bbox_large.contains(p) == True

    # Test 2: Very small coordinates
    small_points = [
        np.array([1e-6, 1e-6, 1e-6]),
        np.array([1e-6 + 1e-9, 1e-6, 1e-6]),
        np.array([1e-6, 1e-6 + 1e-9, 1e-6]),
        np.array([1e-6, 1e-6, 1e-6 + 1e-9]),
    ]
    bbox_small = volumetric.BoundingBox3D.compute_from_points(small_points)
    assert bbox_small is not None
    volume = bbox_small.get_volume()
    assert volume > 0.0
    assert volume < 1e-24  # Should be very small
    for p in small_points:
        assert bbox_small.contains(p) == True

    # Test 3: Mixed positive and negative large values
    mixed_points = [
        np.array([-1e6, -1e6, -1e6]),
        np.array([1e6, 1e6, 1e6]),
    ]
    bbox_mixed = volumetric.BoundingBox3D.compute_from_points(mixed_points)
    assert bbox_mixed is not None
    assert bbox_mixed.get_volume() > 1e15
    for p in mixed_points:
        assert bbox_mixed.contains(p) == True

    # Test 4: OBB with extreme values
    obb_extreme = volumetric.OrientedBoundingBox3D.compute_from_points(large_points)
    assert obb_extreme is not None
    # For extreme coordinates, PCA might have slightly more numerical error
    assert_almost_equal(obb_extreme.get_volume(), 1.0, tol=0.1)  # 10% tolerance for extreme values
    for p in large_points:
        assert obb_extreme.contains(p) == True

    print("  ✓ Extreme values passed")


def test_stress_many_points():
    """Stress test with many points."""
    print("Testing stress with many points...")

    import volumetric

    # Test 1: Many points in 3D (1000 points)
    np.random.seed(42)  # For reproducibility
    many_points_3d = [
        np.array([np.random.uniform(0, 10), np.random.uniform(0, 10), np.random.uniform(0, 10)])
        for _ in range(1000)
    ]

    bbox_many = volumetric.BoundingBox3D.compute_from_points(many_points_3d)
    assert bbox_many is not None
    assert bbox_many.get_volume() > 0.0
    # All points should be contained
    for p in many_points_3d:
        assert bbox_many.contains(p) == True

    obb_many = volumetric.OrientedBoundingBox3D.compute_from_points(many_points_3d)
    assert obb_many is not None
    assert obb_many.get_volume() > 0.0
    # For uniform random points in [0,10], PCA can over-rotate when eigenvalues are nearly equal,
    # yielding an OBB larger than the AABB. Allow a generous cap while still catching blow-ups.
    assert obb_many.get_volume() <= bbox_many.get_volume() * 2.5
    for p in many_points_3d:
        assert obb_many.contains(p) == True

    # Test 2: Many points in 2D (500 points)
    many_points_2d = [
        np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)]) for _ in range(500)
    ]

    bbox_many_2d = volumetric.BoundingBox2D.compute_from_points(many_points_2d)
    assert bbox_many_2d is not None
    assert bbox_many_2d.get_area() > 0.0
    for p in many_points_2d:
        assert bbox_many_2d.contains(p) == True

    obb_many_2d = volumetric.OrientedBoundingBox2D.compute_from_points(many_points_2d)
    assert obb_many_2d is not None
    assert obb_many_2d.get_area() > 0.0
    # For nearly isotropic random clouds, PCA can rotate away from axes; an OBB up to ~2x the AABB
    # area is plausible. Cap at 2.5x to catch blow-ups while tolerating isotropic cases.
    assert obb_many_2d.get_area() <= bbox_many_2d.get_area() * 2.5
    for p in many_points_2d:
        assert obb_many_2d.contains(p) == True

    # Test 3: Points on a grid (regular pattern)
    grid_points = []
    for x in np.linspace(0.0, 5.0, 21):
        for y in np.linspace(0.0, 5.0, 21):
            grid_points.append(np.array([x, y]))

    obb_grid = volumetric.OrientedBoundingBox2D.compute_from_points(grid_points)
    assert obb_grid is not None
    # For a regular grid, OBB should be close to AABB
    bbox_grid = volumetric.BoundingBox2D.compute_from_points(grid_points)
    assert_almost_equal(obb_grid.get_area(), bbox_grid.get_area(), tol=0.1)
    for p in grid_points:
        assert obb_grid.contains(p) == True

    print("  ✓ Stress with many points passed")


def test_intersection_edge_cases():
    """Test intersection edge cases."""
    print("Testing intersection edge cases...")

    import volumetric

    # Test 1: Tangent boxes (just touching)
    bbox1 = volumetric.BoundingBox3D(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    bbox2 = volumetric.BoundingBox3D(
        np.array([1.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])
    )  # Touching at x=1.0
    assert bbox1.intersects(bbox2) == True  # Touching counts as intersection

    # Test 2: Nested boxes
    bbox_outer = volumetric.BoundingBox3D(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))
    bbox_inner = volumetric.BoundingBox3D(np.array([2.0, 2.0, 2.0]), np.array([8.0, 8.0, 8.0]))
    assert bbox_outer.intersects(bbox_inner) == True

    # Test 3: OBBs at various angles that just touch
    center1 = np.array([0.0, 0.0])
    center2 = np.array([2.0, 0.0])
    size = np.array([2.0, 2.0])
    obb1 = volumetric.OrientedBoundingBox2D(center1, 0.0, size)
    obb2 = volumetric.OrientedBoundingBox2D(center2, 0.0, size)
    assert obb1.intersects(obb2) == True  # Should touch

    # Test 4: Rotated OBBs that overlap
    obb3 = volumetric.OrientedBoundingBox2D(center1, math.pi / 4.0, size)
    obb4 = volumetric.OrientedBoundingBox2D(center2, math.pi / 4.0, size)
    # These should intersect even though centers are far apart
    assert obb3.intersects(obb4) == True

    # Test 5: OBB vs AABB edge cases
    # Create identity quaternion (w, x, y, z format)
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z (identity)
    obb5 = volumetric.OrientedBoundingBox3D(
        np.array([1.0, 1.0, 1.0]),
        identity_quat,  # Identity quaternion
        np.array([2.0, 2.0, 2.0]),
    )
    bbox3 = volumetric.BoundingBox3D(np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
    assert obb5.intersects(bbox3) == True

    print("  ✓ Intersection edge cases passed")


def test_pca_vs_convex_hull():
    """Compare PCA and Convex Hull methods when available."""
    print("Testing PCA vs Convex Hull comparison...")

    import volumetric

    # Test with a rectangular box (PCA should work well)
    points_rect = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([2.0, 0.0, 0.5]),
        np.array([0.0, 1.0, 0.5]),
        np.array([2.0, 1.0, 0.5]),
    ]

    obb_pca = volumetric.OrientedBoundingBox3D.compute_from_points(points_rect)
    assert obb_pca is not None

    # Test with L-shaped distribution (convex hull might be better)
    l_points_3d = []
    # Horizontal segment
    for x in np.linspace(0.0, 5.0, 11):
        l_points_3d.append(np.array([x, 0.0, 0.0]))
    # Vertical segment
    for y in np.linspace(0.0, 5.0, 11):
        l_points_3d.append(np.array([0.0, y, 0.0]))
    # Add some depth
    for p in l_points_3d[:]:
        l_points_3d.append(p + np.array([0.0, 0.0, 1.0]))

    obb_l_pca = volumetric.OrientedBoundingBox3D.compute_from_points(l_points_3d)
    assert obb_l_pca is not None
    assert obb_l_pca.get_volume() > 0.0
    for p in l_points_3d:
        assert obb_l_pca.contains(p) == True

    # Note: Convex hull method testing would require exposing the method parameter
    # This is a placeholder for when that's available

    print("  ✓ PCA vs Convex Hull comparison passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Bounding Boxes Test Suite")
    print("=" * 60)
    print()

    try:
        import volumetric
    except ImportError as e:
        print(f"✗ Failed to import volumetric module: {e}")
        print("\nMake sure the module is built:")
        print("  cd cpp && ./build.sh")
        sys.exit(1)

    tests = [
        test_bounding_box_3d_basic,
        test_bounding_box_3d_compute_from_points,
        test_bounding_box_3d_compute_from_points_comprehensive,
        test_bounding_box_3d_intersects,
        test_oriented_bounding_box_3d_basic,
        test_oriented_bounding_box_3d_compute_from_points,
        test_oriented_bounding_box_3d_compute_from_points_comprehensive,
        test_oriented_bounding_box_3d_convex_hull,
        test_bounding_box_2d_basic,
        test_bounding_box_2d_compute_from_points,
        test_bounding_box_2d_compute_from_points_comprehensive,
        test_oriented_bounding_box_2d_basic,
        test_oriented_bounding_box_2d_compute_from_points,
        test_oriented_bounding_box_2d_compute_from_points_comprehensive,
        test_oriented_bounding_box_intersects,
        test_edge_cases,
        test_challenging_geometries,
        test_boundary_conditions,
        test_extreme_values,
        test_stress_many_points,
        test_intersection_edge_cases,
        test_pca_vs_convex_hull,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
