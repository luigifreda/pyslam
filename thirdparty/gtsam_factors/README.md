# gtsam_factors 

**Author**: Luigi Freda

This package contains a custom **pybind11** module for **pySLAM**, wrapping additional **GTSAM** helpers and factors that are not included in the original codebase. Specifically, there are factors for handling **Similarity3** transformations and camera resectioning in SLAM. The factors are designed to be used in non-linear optimization for pose estimation and 3D point positioning.


## Build 

First build and install gtsam along with its Python bindings:
```
<pyslam_root>/install_gtsam.h
```

Then, open a terminal in this folder and run 
```
. <pyslam_root>/pyenv-activate.sh
./buil.sh
```

## Overview 

This is a list of the key classes and helpers. 

### **SwitchableRobustNoiseModel**
- A custom noise model that can switch between **robust** and **diagonal** models.
- Uses **Huber loss** for the robust model to handle outliers effectively.
- Can switch active noise models dynamically.

### **ResectioningFactor**
- Projects 3D world points onto a 2D image plane using a camera model.
- Computes the error between projected 3D points and observed 2D points.
- Supports optional Jacobian computation for optimization.
- **Weight**: A weight can be applied to the error, allowing you to disable or enable the factor dynamically.

### **ResectioningFactorStereo**
- A stereo version of the `ResectioningFactor`, using a **StereoCamera** for handling stereo vision.
- Projects a 3D world point onto two 2D points (left and right) and computes the error.
- **Weight**: Similar to the monocular version, a weight can be applied to enable/disable the factor dynamically.

### **WeightedGenericProjectionFactor**
- A generic factor for projecting 3D landmarks onto a 2D plane, considering an optional **weight** to control the importance of the factor.
- Handles cheirality issues and supports Jacobian computation for both camera pose and landmark.
- **Weight**: The weight dynamically adjusts the importance of the factor and can effectively disable it if the weight is set to zero or a very small value.

### **PriorFactorSimilarity3**
- Applies a **Similarity3** prior to penalize deviation from a given transformation.
- Defines an error function using the **Logmap** of the transformation error.
- Computes Jacobians for optimization using numerical derivatives.

### **PriorFactorSimilarity3ScaleOnly**
- A variant of the `PriorFactorSimilarity3`, which penalizes only the scale difference in a **Similarity3** transformation.
- Computes Jacobians for the scale error.

### **SimResectioningFactor**
- A custom factor for optimizing the similarity transformation (Sim3) using 2D-3D resectioning.
- Projects the transformed 3D point using a camera calibration matrix, and compares it with the observed 2D point.
- **Weight**: Similar to other resectioning factors, a weight can be applied to enable/disable this factor.

### **SimInvResectioningFactor**
- Similar to `SimResectioningFactor`, but it computes the error with the inverse of the similarity transformation.
- Useful for scenarios where the transformation needs to be inverted.
- **Weight**: A weight can be applied to control the factor's influence on the optimization process.

### **BetweenFactorSimilarity3**
- A factor that computes the error between two **Similarity3** poses (i.e., relative transformation between two keyframes).
- The error is the **Logmap** of the relative transformation between two `Similarity3` poses.

### **Numerical Autodiff Helpers**
- Provides helper functions for **numerical derivatives** and **autodifferentiation** (autodiff) for the error functions of **Similarity3** factors.
- The function `numericalDerivative11General` is used to compute Jacobians for any factor, with a specialization for **Similarity3** to **Vector2**. 
- These autodiff helpers facilitate the efficient computation of Jacobians, ensuring correct gradient propagation for optimization.

## Utility Functions

- **`insertSimilarity3`**: Inserts a **Similarity3** transformation into GTSAM's `Values` container.
- **`getSimilarity3`**: Retrieves a **Similarity3** transformation from `Values`.

## Serialization and Cloning
- Each factor includes serialization support for saving/loading factor data.
- The factors implement the `clone()` method to create deep copies of themselves for optimization.

## Custom Optimizer

- **LevenbergMarquardtOptimizerG2o**: A custom optimizer that wraps GTSAM's Levenberg-Marquardt optimization to experiment with lambda policies.
