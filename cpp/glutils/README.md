# GLUtils Module

OpenGL rendering utilities for 3D visualization in PYSLAM. This module provides Python bindings for drawing point clouds, meshes, camera frustums, trajectories, bounding boxes, and other 3D primitives using OpenGL.

## Module Structure

The module is organized into the following files:

- **`glutils_common.h`** - Shared types, constants, and common includes
- **`glutils_drawing.h`** - Low-level OpenGL drawing functions (internal `glutils_detail` namespace)
- **`glutils_utils.h`** - Helper utility functions (pose matrix extraction, alignment computation)
- **`glutils_bindings.h/cpp`** - Python wrapper functions that interface between NumPy arrays and OpenGL
- **`glutils_camera.h`** - Camera image classes for texture-based camera visualization
- **`glutils_module.cpp`** - Pybind11 module definition and bindings

## Available Functions

### Point Cloud Rendering

- **`DrawPoints(points)`** - Draw a point cloud from an N×3 array of 3D points
- **`DrawPoints(points, colors)`** - Draw a colored point cloud with per-point RGB colors (N×3 arrays)

### Mesh Rendering

- **`DrawMesh(vertices, triangles, colors, wireframe=False)`** - Draw a colored triangular mesh
  - `vertices`: N×3 array of vertex positions
  - `triangles`: M×3 array of triangle indices
  - `colors`: N×3 array of vertex colors (RGB)
  - `wireframe`: Boolean to toggle wireframe mode

- **`DrawMonochromeMesh(vertices, triangles, color, wireframe=False)`** - Draw a mesh with a single color
  - `color`: RGB color array `[r, g, b]` (float values 0.0-1.0)

### Camera Visualization

- **`DrawCamera(pose, w=1.0, h_ratio=0.75, z_ratio=0.6)`** - Draw a single camera frustum
  - `pose`: 4×4 transformation matrix (camera-to-world)
  - `w`, `h_ratio`, `z_ratio`: Frustum dimensions

- **`DrawCameras(poses, w=1.0, h_ratio=0.75, z_ratio=0.6)`** - Draw multiple camera frustums
  - `poses`: N×4×4 array of camera poses

### Line and Trajectory Rendering

- **`DrawLine(points, point_size=0.0)`** - Draw a connected line strip from an N×3 point array
  - `point_size`: Size of point markers (0.0 to disable markers)

- **`DrawLines(points, point_size=0.0)`** - Draw line segments from an N×6 array (each row: [x1,y1,z1,x2,y2,z2])

- **`DrawLines2(points1, points2, point_size=0.0)`** - Draw lines connecting two point sets (N×3 arrays)

- **`DrawTrajectory(points, point_size=0.0)`** - Draw a trajectory as a connected line with optional markers

### Bounding Boxes and Primitives

- **`DrawBoxes(poses, sizes, line_width=1.0)`** - Draw axis-aligned bounding boxes
  - `poses`: N×4×4 transformation matrices
  - `sizes`: N×3 array of box dimensions (width, height, depth)
  - `line_width`: Width of box edges

- **`DrawPlane(num_divs=200, div_size=10.0, scale=1.0)`** - Draw a grid plane (x-z plane at y=0)
  - `num_divs`: Number of grid divisions
  - `div_size`: Size of each grid cell
  - `scale`: Overall scale factor

## Classes

### CameraImage

Represents a camera with an associated image texture that can be rendered in 3D space.

**Constructor:**
```python
CameraImage(image, pose, id, scale=1.0, h_ratio=0.75, z_ratio=0.6, color=[0.0, 1.0, 0.0])
```
- `image`: NumPy array (H×W or H×W×3) - grayscale or RGB image
- `pose`: 4×4 transformation matrix (float32 or float64)
- `id`: Unique identifier for the camera
- `scale`, `h_ratio`, `z_ratio`: Frustum dimensions
- `color`: RGB color for frustum wireframe

**Methods:**
- **`draw()`** - Draw the camera with its stored pose
- **`drawPose(pose)`** - Draw the camera with a new pose (4×4 matrix)
- **`setPose(pose)`** - Update the camera's pose (float32 or float64)
- **`setTransparent(transparent)`** - Toggle texture rendering (True = wireframe only)

**Properties:**
- **`id`** - Camera identifier

### CameraImages

Container for managing multiple `CameraImage` objects.

**Methods:**
- **`add(image, pose, id, scale=1.0, h_ratio=0.75, z_ratio=0.6, color=[0.0, 1.0, 0.0])`** - Add a camera image
- **`draw()`** - Draw all cameras with their stored poses
- **`drawPoses(poses)`** - Draw all cameras with new poses (N×4×4 array)
- **`clear()`** - Remove all cameras
- **`erase(id)`** - Remove a camera by ID
- **`size()`** - Get number of cameras
- **`setTransparent(id, transparent)`** - Set transparency for a specific camera
- **`setAllTransparent(transparent)`** - Set transparency for all cameras
- **`__getitem__(i)`** - Access camera by index
- **`__len__()`** - Get number of cameras

## Usage Example

```python
import numpy as np
import glutils

# Draw a point cloud
points = np.random.rand(1000, 3) * 10
colors = np.random.rand(1000, 3)
glutils.DrawPoints(points, colors)

# Draw a camera trajectory
trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1]])
glutils.DrawTrajectory(trajectory, point_size=5.0)

# Draw camera frustums
poses = np.array([np.eye(4), np.eye(4)])  # Two cameras
glutils.DrawCameras(poses, w=0.5)

# Create a camera image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
pose = np.eye(4)
camera = glutils.CameraImage(image, pose, id=0)
camera.draw()

# Manage multiple cameras
cameras = glutils.CameraImages()
cameras.add(image, pose, id=0)
cameras.add(image, pose, id=1)
cameras.draw()
```

## Notes

- All point/vertex arrays must be N×3 (or N×6 for line segments)
- All pose matrices must be 4×4
- Colors are RGB in range [0.0, 1.0] for float arrays, or [0, 255] for uint8 images
- The module uses OpenGL immediate mode (legacy OpenGL)
- GIL (Global Interpreter Lock) is released during OpenGL calls for better performance
