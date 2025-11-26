"""
Test workaround: Try to fix Values by re-inserting the value.
"""

import sys
sys.path.append("./lib")

import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import gtsam_factors

if __name__ == "__main__":
    print("=" * 60)
    print("Test: Workaround by clearing and re-inserting")
    print("=" * 60)
    
    # Create graph and factors first
    graph = gtsam.NonlinearFactorGraph()
    K_mono = gtsam.Cal3_S2(600, 600, 0, 320, 240)
    K_stereo = gtsam.Cal3_S2Stereo(600, 600, 0, 320, 240, 0.1)
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    stereo_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    
    P = np.array([1, 2, 10], dtype=np.float64).reshape(3, 1)
    p_2d = np.array([320, 240], dtype=np.float64).reshape(2, 1)
    p_stereo = gtsam.StereoPoint2(145, 140, 120)

    factor_mono = gtsam_factors.ResectioningFactor(noise_model, X(1), K_mono, p_2d, P)
    graph.add(factor_mono)
    
    factor_stereo = gtsam_factors.ResectioningFactorStereo(stereo_noise_model, X(1), K_stereo, p_stereo, P)
    graph.add(factor_stereo)
    
    # Create Values and insert pose
    initial_estimates = gtsam.Values()
    init_pose = gtsam.Pose3(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    
    # Try workaround: insert, then retrieve and re-insert
    initial_estimates.insert(X(1), init_pose)
    retrieved = initial_estimates.atPose3(X(1))
    print(f"✓ First insert and retrieve worked: {type(retrieved)}")
    
    # Clear and re-insert
    initial_estimates.clear()
    initial_estimates.insert(X(1), retrieved)
    print(f"✓ Re-inserted after retrieval")
    
    # Try graph error
    try:
        error = graph.error(initial_estimates)
        print(f"✓ Graph error computation succeeded: {error}")
    except Exception as e:
        print(f"✗ Graph error computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Try optimizer
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        print(f"✓ Optimizer creation succeeded")
        result = optimizer.optimize()
        print(f"✓ Optimization completed")
        print("Optimized Pose:", result.atPose3(X(1)))
    except Exception as e:
        print(f"✗ Optimizer failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

