"""
Test to check if the issue is with type storage/retrieval.
"""

import sys
sys.path.append("./lib")

import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import gtsam_factors

if __name__ == "__main__":
    print("=" * 60)
    print("Test: Check Values type storage")
    print("=" * 60)
    
    # Create Values and insert Pose3
    values = gtsam.Values()
    init_pose = gtsam.Pose3(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    values.insert(X(1), init_pose)
    
    # Try to retrieve using helper function
    try:
        retrieved = gtsam_factors.get_pose3(values, X(1))
        print(f"✓ Helper function get_pose3() worked: {type(retrieved)}")
    except Exception as e:
        print(f"✗ Helper function get_pose3() failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to retrieve using atPose3
    try:
        retrieved2 = values.atPose3(X(1))
        print(f"✓ atPose3() worked: {type(retrieved2)}")
    except Exception as e:
        print(f"✗ atPose3() failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test: Try with custom factors")
    print("=" * 60)
    
    # Create graph with custom factors
    graph = gtsam.NonlinearFactorGraph()
    K_mono = gtsam.Cal3_S2(600, 600, 0, 320, 240)
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    P = np.array([1, 2, 10], dtype=np.float64).reshape(3, 1)
    p_2d = np.array([320, 240], dtype=np.float64).reshape(2, 1)
    
    factor = gtsam_factors.ResectioningFactor(noise_model, X(1), K_mono, p_2d, P)
    graph.add(factor)
    
    # Try graph.error
    try:
        error = graph.error(values)
        print(f"✓ graph.error() worked: {error}")
    except Exception as e:
        print(f"✗ graph.error() failed: {e}")
        import traceback
        traceback.print_exc()

