# GLUtils Module

OpenGL rendering utilities for 3D visualization in PYSLAM. This module provides Python bindings for drawing point clouds, meshes, camera frustums, trajectories, bounding boxes, and other 3D primitives using OpenGL.


<!-- TOC -->

- [GLUtils Module](#glutils-module)
  - [Module Structure](#module-structure)
  - [Available Functions](#available-functions)
    - [Point Cloud Rendering](#point-cloud-rendering)
    - [Mesh Rendering](#mesh-rendering)
    - [Camera Visualization](#camera-visualization)
    - [Line and Trajectory Rendering](#line-and-trajectory-rendering)
    - [Bounding Boxes and Primitives](#bounding-boxes-and-primitives)
    - [Object Rendering](#object-rendering)
  - [Classes](#classes)
    - [CameraImage](#cameraimage)
    - [CameraImages](#cameraimages)
    - [GlMeshT / GlMeshDirectT](#glmesht--glmeshdirectt)
    - [GlPointCloudT / GlPointCloudDirectT](#glpointcloudt--glpointclouddirectt)
    - [GlObjectT / GlObjectSetT](#globjectt--globjectsett)
  - [Usage Example](#usage-example)
  - [Notes](#notes)

<!-- /TOC -->


## Module Structure

The module is organized into the following files:

- **`glutils_gl_includes.h`** - Shared types, constants, and common includes
- **`glutils_drawing.h`** - Low-level OpenGL drawing functions (internal `glutils_detail` namespace)
- **`glutils_utils.h`** - Helper utility functions (pose matrix extraction, alignment computation)
- **`glutils_bindings.h/cpp`** - Python wrapper functions that interface between NumPy arrays and OpenGL
- **`glutils_bindings_common.h`** - Common pybind11 array aliases
- **`glutils_bindings_utils.h`** - Array validation helpers and packed vector accessors
- **`glutils_opaque_types.h`** - Opaque type declarations for pybind11
- **`glutils_camera.h`** - Camera image classes for texture-based camera visualization
- **`glmesh.h`** - GPU-backed mesh helpers (VBO/VAO-based) for fast drawing
- **`glmesh_bindings.h`** - Pybind11 wrappers for `GlMesh*` helpers
- **`glpoint_cloud.h`** - GPU-backed point cloud helpers (VBO-based) for fast drawing
- **`glpoint_cloud_bindings.h`** - Pybind11 wrappers for `GlPointCloud*` helpers
- **`globject.h`** - GPU-backed object rendering helpers
- **`globject_bindings.h`** - Pybind11 wrappers for `GlObject*` helpers
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

### Object Rendering

- **`DrawObjectData(object_data)`** - Draw a `volumetric.ObjectData` point cloud with optional colors
  - `object_data`: `volumetric.ObjectData` instance (points and optional colors)

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

### GlMeshT / GlMeshDirectT

GPU-backed mesh renderers using VBOs (and a VAO for `GlMeshT`).

- **`GlMeshT<VertexT, ColorT>`** - Stores vertices/colors on CPU, uploads on `UpdateGPU()`
  - `SetVertices(vertices_xyz, vertex_count)` - Set N vertex positions (N×3)
  - `SetColors(colors_rgb)` - Set per-vertex colors (N×3) or clear if null
  - `SetTriangles(tri_idx, tri_count)` - Set triangle indices (M×3)
  - `ReserveGPU(max_vertices, max_indices)` - Pre-allocate GPU buffers
  - `UpdateGPU()` - Upload dirty buffers
  - `Draw(wireframe)` - Draw triangles
- **`GlMeshDirectT<VertexT, ColorT>`** - Uploads data directly on `Update()`
  - `Update(vertices_xyz, colors_rgb, tri_idx, vertex_count, tri_count)`
  - `Draw(wireframe)` - Draw using already-uploaded buffers

**Typedefs:** `GlMeshF`, `GlMeshD`, `GlMeshDirectF`, `GlMeshDirectD`

### GlPointCloudT / GlPointCloudDirectT

GPU-backed point cloud renderers using VBOs.

- **`GlPointCloudT<PointT, ColorT>`** - Stores points/colors on CPU, uploads on `UpdateGPU()`
  - `Set(points, colors, point_count)` - Set points and optional colors (N×3)
  - `SetPoints(points, point_count)` - Set points only
  - `SetColors(colors)` - Set colors (N×3)
  - `ClearColors()` - Remove colors
  - `UpdateGPU()` - Upload dirty buffers
  - `Draw()` - Draw points
- **`GlPointCloudDirectT<PointT, ColorT>`** - Uploads data directly on `Update()`
  - `Update(points, colors, point_count)`
  - `Draw()` - Draw using already-uploaded buffers

**Typedefs:** `GlPointCloudF`, `GlPointCloudD`, `GlPointCloudDirectF`, `GlPointCloudDirectD`

### GlObjectT / GlObjectSetT

GPU-backed object renderers using VBOs. Supports drawing points, optional per-point colors,
per-class/per-object color modes, and optional bounding boxes.

- **`GlObjectT<PointT, ColorT>`** - Stores a single object on the GPU
  - `update(points, colors, point_count)` - Upload point data directly
  - `set_object_id_color(color)` - Set color used in OBJECT_ID mode
  - `set_class_id_color(color)` - Set color used in CLASS mode
  - `set_color_draw_mode(color_draw_mode)` - Select color mode (`POINTS`, `CLASS`, `OBJECT_ID`)
  - `set_bounding_box(box_matrix, box_size)` - Set oriented bounding box for rendering
  - `draw()` - Draw the object
- **`GlObjectSetT<PointT, ColorT>`** - Manages a set of objects
  - `update(object_data_list, class_id_colors, object_id_colors)` - Update all objects from
    `volumetric.ObjectData` and color lists
  - `draw()` - Draw all objects

**Typedefs:** `GlObjectF`, `GlObjectD`, `GlObjectsF`, `GlObjectsD`

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
- For zero-copy paths, pass contiguous NumPy arrays with matching dtype; otherwise bindings may
  perform a copy to cast input data.
