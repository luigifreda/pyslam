"""
Debug test to understand what's stored in Values.
"""

import sys
sys.path.append("./lib")

import gtsam
from gtsam.symbol_shorthand import X
import numpy as np

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Insert Pose3 using Python insert()")
    print("=" * 60)
    values1 = gtsam.Values()
    init_pose = gtsam.Pose3(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    values1.insert(X(1), init_pose)
    
    try:
        value1 = values1.atPose3(X(1))
        print(f"✓ Retrieved using atPose3(): {type(value1)}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print()
    print("=" * 60)
    print("Test 2: Check if we can use at<> template method")
    print("=" * 60)
    # Try to see if there's a way to access the template method
    # This might not work from Python, but let's see
    
    print("=" * 60)
    print("Test 3: Create graph with standard GTSAM factors")
    print("=" * 60)
    graph = gtsam.NonlinearFactorGraph()
    noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    prior_factor = gtsam.PriorFactorPose3(X(1), init_pose, noise_model)
    graph.add(prior_factor)
    
    try:
        error = graph.error(values1)
        print(f"✓ Graph error with standard factor: {error}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test 4: Try optimizer with standard factors")
    print("=" * 60)
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values1, params)
        print(f"✓ Optimizer creation with standard factors succeeded")
        result = optimizer.optimize()
        print(f"✓ Optimization completed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

