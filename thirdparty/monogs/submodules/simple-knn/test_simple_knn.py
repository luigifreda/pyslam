#!/usr/bin/env python3
"""
Test script for simple_knn module.

This script tests the distCUDA2 function which computes the mean squared
distance from each point to its k nearest neighbors.
"""

import sys
import numpy as np
import torch

# Try to import simple_knn
try:
    from simple_knn._C import distCUDA2

    print("✓ Successfully imported simple_knn")
except ImportError as e:
    print(f"✗ Failed to import simple_knn: {e}")
    print("\nMake sure you have:")
    print("1. Built the extension: python setup.py install or pip install -e .")
    print("2. CUDA and PyTorch are properly installed")
    sys.exit(1)


def test_basic_functionality():
    """Test basic functionality with a simple point cloud."""
    print("\n=== Test 1: Basic Functionality ===")

    # Create a simple point cloud: a cube of points
    num_points = 100
    points = torch.rand(num_points, 3, device="cuda", dtype=torch.float32) * 2.0 - 1.0

    print(f"Input: {num_points} points")
    print(f"Points shape: {points.shape}")
    print(f"Points device: {points.device}")
    print(f"Points dtype: {points.dtype}")

    try:
        result = distCUDA2(points)
        print(f"✓ distCUDA2 executed successfully")
        print(f"Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")
        print(f"Output device: {result.device}")
        print(f"Output min: {result.min().item():.6f}")
        print(f"Output max: {result.max().item():.6f}")
        print(f"Output mean: {result.mean().item():.6f}")

        # Check that output is reasonable (positive distances)
        assert result.min() >= 0, "Distances should be non-negative"
        assert result.shape[0] == num_points, f"Output should have {num_points} elements"
        print("✓ All assertions passed")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_deterministic_points():
    """Test with deterministic point cloud."""
    print("\n=== Test 2: Deterministic Points ===")

    # Create a grid of points
    grid_size = 5
    x = torch.linspace(-1, 1, grid_size, device="cuda")
    y = torch.linspace(-1, 1, grid_size, device="cuda")
    z = torch.linspace(-1, 1, grid_size, device="cuda")

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).float()

    num_points = points.shape[0]
    print(f"Input: {num_points} points in a {grid_size}x{grid_size}x{grid_size} grid")

    try:
        result = distCUDA2(points)
        print(f"✓ distCUDA2 executed successfully")
        print(f"Output shape: {result.shape}")
        print(f"Output range: [{result.min().item():.6f}, {result.max().item():.6f}]")

        # For a regular grid, distances should be relatively uniform
        std = result.std().item()
        print(f"Output std: {std:.6f}")

        assert result.min() >= 0, "Distances should be non-negative"
        print("✓ All assertions passed")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test 3: Edge Cases ===")

    # Test with small number of points
    print("Testing with 3 points...")
    points_small = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda", dtype=torch.float32
    )
    try:
        result_small = distCUDA2(points_small)
        print(f"✓ Small point cloud works: shape {result_small.shape}")
        print(f"  Distances: {result_small.cpu().numpy()}")
    except Exception as e:
        print(f"✗ Error with small point cloud: {e}")
        return False

    # Test with many points
    print("Testing with 10000 points...")
    points_large = torch.rand(10000, 3, device="cuda", dtype=torch.float32) * 10.0
    try:
        result_large = distCUDA2(points_large)
        print(f"✓ Large point cloud works: shape {result_large.shape}")
        print(f"  Mean distance: {result_large.mean().item():.6f}")
    except Exception as e:
        print(f"✗ Error with large point cloud: {e}")
        return False

    print("✓ All edge cases passed")
    return True


def test_compare_with_known_values():
    """Compare results with expected behavior."""
    print("\n=== Test 4: Comparison with Known Values ===")

    # Create points that are far apart
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    print("Testing with points that are 10 units apart...")
    try:
        result = distCUDA2(points)
        print(f"✓ Computation successful")
        print(f"Distances: {result.cpu().numpy()}")

        # The distances should be relatively large (around 100-200 for squared distances)
        # since points are 10 units apart, squared distance is ~100
        print(f"Expected: Large values (squared distances ~100)")
        print(f"Got: Mean = {result.mean().item():.2f}")

        # Check that all distances are positive
        assert (result >= 0).all(), "All distances should be non-negative"
        print("✓ Results are reasonable")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Test performance with a reasonable point cloud size."""
    print("\n=== Test 5: Performance Test ===")

    num_points = 1000
    points = torch.rand(num_points, 3, device="cuda", dtype=torch.float32) * 10.0

    print(f"Testing with {num_points} points...")

    # Warm up
    _ = distCUDA2(points)
    torch.cuda.synchronize()

    # Time it
    import time

    num_iterations = 10
    start_time = time.time()
    for _ in range(num_iterations):
        result = distCUDA2(points)
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    print(f"✓ Average time per call: {avg_time*1000:.2f} ms")
    print(f"  ({num_points/avg_time:.0f} points/second)")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing simple_knn module")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. This module requires CUDA.")
        sys.exit(1)

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ PyTorch version: {torch.__version__}")

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Deterministic Points", test_deterministic_points),
        ("Edge Cases", test_edge_cases),
        ("Known Values", test_compare_with_known_values),
        ("Performance", test_performance),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
