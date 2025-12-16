# pybind11 binding for C++ pyslam utils

## Available Modules

### `cv2_pyslam_module`

OpenCV feature detector bindings. Provides `MSDDetector` (Maximal Self-Dissimilarity) for robust feature detection.

### `glutils`

OpenGL rendering utilities for visualization. Includes functions for drawing point clouds, meshes, camera frustums, trajectories, and line segments. Supports colored rendering and camera image overlays.

### `pnpsolver`

Perspective-n-Point (PnP) pose estimation solvers with RANSAC support. Provides `PnPsolver` and `MLPnPsolver` classes for robust camera pose estimation from 2D-3D correspondences.

### `sim3solver`

Similarity transformation (Sim(3)) solvers for 3D point cloud registration. Includes `Sim3Solver` and `Sim3PointRegistrationSolver` for estimating scale, rotation, and translation between point sets.

### `trajectory_tools`

Trajectory alignment and association utilities. Provides functions for finding timestamp associations between trajectories, aligning 3D points using SVD, and incremental trajectory alignment with ground truth.

### `pyslam_utils`

General SLAM utilities including feature matching filters, patch extraction from images, and epipolar constraint filtering. Supports both OpenCV and NumPy array interfaces.

### `color_utils`

Color conversion utilities for ID visualization. Provides `IdsColorTable` class for converting arbitrary integer IDs to RGB colors using a hash-based color table. Supports both 1D and 2D input arrays, handles unlabeled instances, and supports BGR/RGB output formats.

### `volumetric`

Volumetric mapping and integration. Provides voxel grid data structures for TSDF (Truncated Signed Distance Function) mapping, semantic voxel grids, camera frustum culling, and TBB-based parallel processing utilities.

See [volumetric/README.md](volumetric/README.md) for detailed documentation.
