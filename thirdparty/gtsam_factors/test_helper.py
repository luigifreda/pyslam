"""
Test using the helper function to insert Pose3 values.
"""

import sys
sys.path.append("./lib")

import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import gtsam_factors

if __name__ == "__main__":
    print("=" * 60)
    print("Test using helper function to insert Pose3")
    print("=" * 60)
    
    # Create a NonlinearFactorGraph to hold the factors
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
        
    # Monocular Camera Calibration
    K_mono = gtsam.Cal3_S2(600, 600, 0, 320, 240)
    K_stereo = gtsam.Cal3_S2Stereo(600, 600, 0, 320, 240, 0.1)

    # Noise Model
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    stereo_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    
    # Define a known 3D point
    P = np.array([1, 2, 10], dtype=np.float64).reshape(3, 1)
    p_2d = np.array([320, 240], dtype=np.float64).reshape(2, 1)
    p_stereo = gtsam.StereoPoint2(145, 140, 120)

    # Add custom factors
    factor_mono = gtsam_factors.ResectioningFactor(noise_model, X(1), K_mono, p_2d, P)
    graph.add(factor_mono)
    print(f'✓ Added ResectioningFactor')
    
    factor_stereo = gtsam_factors.ResectioningFactorStereo(stereo_noise_model, X(1), K_stereo, p_stereo, P)
    graph.add(factor_stereo)
    print(f'✓ Added ResectioningFactorStereo')
                    
    # Initial Estimates - try using helper function
    init_pose = gtsam.Pose3(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    gtsam_factors.insert_pose3(initial_estimates, X(1), init_pose)
    print(f'✓ Inserted Pose3 using helper function')
    
    # Verify we can retrieve it
    value = initial_estimates.atPose3(X(1))
    print(f'✓ Retrieved Pose3: {type(value)}')
                
    # Try to compute graph error
    try:
        error = graph.error(initial_estimates)
        print(f'✓ Graph error computation succeeded: {error}')
    except Exception as e:
        print(f'✗ Graph error computation failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Levenberg-Marquardt Optimizer Setup
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        print(f'✓ Optimizer creation succeeded')
        
        # Perform Optimization
        result = optimizer.optimize()
        print(f'✓ Optimization completed')
        
        # Print the result
        optimized_pose = result.atPose3(X(1))
        print(f'✓ Optimized Pose retrieved: {type(optimized_pose)}')
        print("Optimized Pose:", optimized_pose)
    except Exception as e:
        print(f'✗ Optimizer failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

