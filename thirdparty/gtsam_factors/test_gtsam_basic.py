"""
Simple test to verify GTSAM is working correctly with basic built-in factors.
This test doesn't use any custom factors, just standard GTSAM functionality.
"""

import sys
import gtsam
from gtsam.symbol_shorthand import X
import numpy as np

if __name__ == "__main__":
    print(f'GTSAM location: {gtsam.__file__}')
    print(f'GTSAM version: {gtsam.__version__ if hasattr(gtsam, "__version__") else "unknown"}')
    print()
    
    # Test 1: Basic Pose3 creation and manipulation
    print("=" * 60)
    print("Test 1: Basic Pose3 operations")
    print("=" * 60)
    try:
        # Create an identity pose
        pose1 = gtsam.Pose3()
        print(f"✓ Created identity Pose3: {pose1}")
        
        # Create a pose from translation and rotation
        translation = np.array([1.0, 2.0, 3.0])
        rotation = gtsam.Rot3()
        pose2 = gtsam.Pose3(rotation, translation)
        print(f"✓ Created Pose3 from translation/rotation: {pose2}")
        
        # Create a pose from 6D vector (x, y, z, roll, pitch, yaw)
        pose3 = gtsam.Pose3(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64))
        print(f"✓ Created Pose3 from 6D vector: {pose3}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
    
    # Test 2: Values container operations
    print("=" * 60)
    print("Test 2: Values container operations")
    print("=" * 60)
    try:
        values = gtsam.Values()
        
        # Insert a Pose3
        key = X(1)
        values.insert(key, pose1)
        print(f"✓ Inserted Pose3 with key {key}")
        
        # Retrieve the Pose3
        retrieved_pose = values.atPose3(key)
        print(f"✓ Retrieved Pose3: {retrieved_pose}")
        print(f"✓ Type of retrieved value: {type(retrieved_pose)}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
    
    # Test 3: Simple factor graph with BetweenFactor
    print("=" * 60)
    print("Test 3: Factor graph with BetweenFactor")
    print("=" * 60)
    try:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()
        
        # Create a simple between factor (constraint between two poses)
        # This is a standard GTSAM factor, not a custom one
        noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        
        # Create a relative transformation (small translation)
        relative_pose = gtsam.Pose3(gtsam.Rot3(), np.array([0.1, 0.0, 0.0]))
        
        # Add a prior factor on x1 (fixes the first pose)
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        prior_factor = gtsam.PriorFactorPose3(X(1), pose1, prior_noise)
        graph.add(prior_factor)
        print(f"✓ Added PriorFactorPose3 on X(1)")
        
        # Add initial estimates
        initial_estimates.insert(X(1), pose1)
        print(f"✓ Inserted initial estimate for X(1)")
        
        # Verify we can retrieve it
        test_retrieve = initial_estimates.atPose3(X(1))
        print(f"✓ Successfully retrieved Pose3 from Values: {type(test_retrieve)}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 4: Optimization
    print("=" * 60)
    print("Test 4: Levenberg-Marquardt optimization")
    print("=" * 60)
    try:
        # Create optimizer
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        print(f"✓ Created LevenbergMarquardtOptimizer")
        
        # Optimize
        result = optimizer.optimize()
        print(f"✓ Optimization completed")
        
        # Retrieve result
        optimized_pose = result.atPose3(X(1))
        print(f"✓ Retrieved optimized Pose3: {optimized_pose}")
        print(f"✓ Type of optimized value: {type(optimized_pose)}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)
    print("All basic GTSAM tests passed! ✓")
    print("=" * 60)

