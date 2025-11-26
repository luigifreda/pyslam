"""
Debug test to isolate the issue with custom factors and Values container.
"""

import sys
sys.path.append("./lib")

import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import gtsam_factors

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Create Values and insert Pose3")
    print("=" * 60)
    initial_estimates = gtsam.Values()
    init_pose = gtsam.Pose3(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    initial_estimates.insert(X(1), init_pose)
    
    # Test retrieval
    value1 = initial_estimates.atPose3(X(1))
    print(f"✓ Retrieved Pose3: {type(value1)}")
    print(f"✓ Value type check: {isinstance(value1, gtsam.Pose3)}")
    print()
    
    print("=" * 60)
    print("Step 2: Create graph and add custom factors")
    print("=" * 60)
    graph = gtsam.NonlinearFactorGraph()
    
    K_mono = gtsam.Cal3_S2(600, 600, 0, 320, 240)
    K_stereo = gtsam.Cal3_S2Stereo(600, 600, 0, 320, 240, 0.1)
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    stereo_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    
    P = np.array([1, 2, 10], dtype=np.float64).reshape(3, 1)
    p_2d = np.array([320, 240], dtype=np.float64).reshape(2, 1)
    p_stereo = gtsam.StereoPoint2(145, 140, 120)
    
    # Add custom factors
    factor_mono = gtsam_factors.ResectioningFactor(noise_model, X(1), K_mono, p_2d, P)
    graph.add(factor_mono)
    print(f"✓ Added ResectioningFactor")
    
    factor_stereo = gtsam_factors.ResectioningFactorStereo(stereo_noise_model, X(1), K_stereo, p_stereo, P)
    graph.add(factor_stereo)
    print(f"✓ Added ResectioningFactorStereo")
    print()
    
    print("=" * 60)
    print("Step 3: Check Values after adding factors")
    print("=" * 60)
    # Try to retrieve again after adding factors
    try:
        value2 = initial_estimates.atPose3(X(1))
        print(f"✓ Retrieved Pose3 after adding factors: {type(value2)}")
        print(f"✓ Value type check: {isinstance(value2, gtsam.Pose3)}")
    except Exception as e:
        print(f"✗ Failed to retrieve after adding factors: {e}")
    print()
    
    print("=" * 60)
    print("Step 4: Try to evaluate factor error (this accesses Values)")
    print("=" * 60)
    try:
        # Evaluate error using the pose from Values
        pose = initial_estimates.atPose3(X(1))
        error = factor_mono.evaluateError(pose)
        print(f"✓ Factor error evaluation succeeded: {error.shape}")
    except Exception as e:
        print(f"✗ Factor error evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    print("=" * 60)
    print("Step 5: Try to compute graph error (this accesses Values)")
    print("=" * 60)
    try:
        error = graph.error(initial_estimates)
        print(f"✓ Graph error computation succeeded: {error}")
    except Exception as e:
        print(f"✗ Graph error computation failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    print("=" * 60)
    print("Step 6: Try to create optimizer")
    print("=" * 60)
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        print(f"✓ Optimizer creation succeeded")
    except Exception as e:
        print(f"✗ Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    print("=" * 60)
    print("Step 7: Check if Values was modified")
    print("=" * 60)
    try:
        value3 = initial_estimates.atPose3(X(1))
        print(f"✓ Final retrieval: {type(value3)}")
        print(f"✓ Value type check: {isinstance(value3, gtsam.Pose3)}")
    except Exception as e:
        print(f"✗ Final retrieval failed: {e}")
        import traceback
        traceback.print_exc()

