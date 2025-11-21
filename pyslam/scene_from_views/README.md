# Scene From Views

This directory contains the implementation of the `scene_from_views` factory, which provides a **unified interface for 3D scene reconstruction from multiple views**. The factory follows a modular architecture that allows easy integration of different reconstruction models while maintaining a consistent API.

## Architecture

### Base Class: `SceneFromViewsBase`

The `SceneFromViewsBase` class (`scene_from_views_base.py`) serves as the abstract base class that defines the common interface for all scene reconstruction implementations. It provides:

- **Shared Pipeline**: Implements a unified `reconstruct()` method that follows a consistent three-step pipeline:
  1. `preprocess_images()`: Preprocess input images for the specific model
  2. `infer()`: Run model inference on preprocessed images
  3. `postprocess_results()`: Convert raw model output to standardized `SceneFromViewsResult`

- **Abstract Methods**: Defines abstract methods that all derived classes must implement:
  - `preprocess_images()`: Preprocess input images for the specific model (accepts `**kwargs` for flexibility)
  - `infer()`: Run inference on preprocessed images and return raw model output (accepts `**kwargs` for model-specific parameters)
  - `postprocess_results()`: Convert raw model output to standardized `SceneFromViewsResult` (accepts `**kwargs` for postprocessing options)

### Result Structure: `SceneFromViewsResult`

`SceneFromViewsResult` is the class that standardizes the output format across all models, containing:
  - `global_point_cloud`: Merged point cloud in world coordinates
  - `global_mesh`: Merged mesh in world coordinates
  - `camera_poses`: List of camera-to-world transformation matrices (4x4)
  - `processed_images`: List of processed images used by the model
  - `depth_predictions`: List of depth maps (H, W) if available
  - `point_clouds`: List of per-view point clouds
  - `intrinsics`: List of camera intrinsic matrices (3x3) if available
  - `confidences`: List of confidence maps (H, W) if available

### Factory Pattern: `scene_from_views_factory`

The factory function (`scene_from_views_factory.py`) provides a centralized way to instantiate different reconstruction models:

```python
from pyslam.scene_from_views import scene_from_views_factory, SceneFromViewsType

# Create a reconstructor
scene_reconstructor = scene_from_views_factory(
    scene_from_views_type=SceneFromViewsType.DUST3R,
    device='cuda',
    **model_specific_kwargs
)
```

The factory handles:
- Dynamic import of model-specific classes
- Device management (CPU/CUDA)
- Parameter passing to model constructors
- Error handling for missing dependencies

### Derived Implementations

Each derived class extends `SceneFromViewsBase` and implements the three specialized methods (`preprocess_images()`, `infer()`, and `postprocess_results()`). The shared `reconstruct()` method in the base class orchestrates these steps, ensuring a consistent reconstruction pipeline across all models:

1. **`SceneFromViewsDust3r`** (`scene_from_views_dust3r.py`)
   - Wraps DUSt3R for image-based 3D reconstruction
   - Handles multi-view reconstruction with pose estimation derived from pointmaps
   - Uses a global alignment optimization stage for merging views into a common frame

2. **`SceneFromViewsMast3r`** (`scene_from_views_mast3r.py`)
   - Wraps MASt3R (Grounding Image Matching in 3D with MASt3R)
   - Provides sparse global alignment for multi-view consistency
   - Can optionally use TSDF-based fusion / refinement in the implementation

3. **`SceneFromViewsDepthAnythingV3`** (`scene_from_views_depth_anything_v3.py`)
   - Wraps Depth Anything 3 (DA3) for monocular depth estimation
   - Can optionally provide camera poses and intrinsics (when supported by the underlying model)
   - Supports both standard (relative) and metric depth models

4. **`SceneFromViewsMvdust3r`** (`scene_from_views_mvdust3r.py`)
   - Wraps the multi-view DUSt3R variant MV-DUSt3R / MV-DUSt3R+
   - Processes multiple views simultaneously in a single forward pass
   - Uses a multi-view decoder to jointly reason over all input views

5. **`SceneFromViewsVggt`** (`scene_from_views_vggt.py`)
   - Wraps VGGT (Visual Geometry Grounded Transformer)
   - Supports using the model’s pointmap and depthmap predictions
   - Provides joint prediction of camera parameters and dense scene geometry


### Capability Quick Reference

- **End-to-end multi-view reconstruction** (poses + fused geometry directly from images): `SceneFromViewsDust3r`, `SceneFromViewsMast3r`, `SceneFromViewsMvdust3r`, `SceneFromViewsVggt`. 
- **Single-view depth-first pipeline with optional poses/intrinsics**: `SceneFromViewsDepthAnythingV3`
- **Global alignment optimization stage for merging views**: `SceneFromViewsDust3r` (dense alignment) and `SceneFromViewsMast3r` (sparse alignment variant)

Note that DUSt3R and MASt3R are **pairwise models**: they take two images at a time. Multi-view end-to-end reconstruction is achieved by running them on many image pairs and performing a global alignment / optimization over all pairwise pointmaps.”

### Type Enumeration: `SceneFromViewsType`

The `SceneFromViewsType` enum (`scene_from_views_types.py`) defines all supported model types:
- `DEPTH_ANYTHING_V3`
- `MAST3R`
- `MVDUST3R`
- `VGGT`
- `DUST3R`

## Supported Models and Reference Papers

### 1. DUSt3R (Geometric 3D Vision Made Easy)

**Description**: DUSt3R is a framework for dense 3D reconstruction from arbitrary image collections without requiring known camera intrinsics or poses. It predicts dense 3D pointmaps for image pairs; from these pointmaps, depth maps and relative/absolute camera parameters can be recovered.

**Key Features** (as used in this framework):
- Predicts dense 3D pointmaps from image pairs, from which depth and camera poses can be recovered
- Global alignment optimization to merge pairwise reconstructions into a common coordinate frame
- Support for various scene graph / pairing strategies (e.g. complete, sliding-window) in the reference implementation
- Confidence-aware regression and confidence-based filtering of low-quality regions

**Reference Paper**:  
Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud. _"DUSt3R: Geometric 3D Vision Made Easy."_ *CVPR 2024*.  
[Paper](https://arxiv.org/abs/2312.14132) | [Project page](https://europe.naverlabs.com/blog/dust3r/) 

**Implementation**: `SceneFromViewsDust3r`

---

### 2. MASt3R (Grounding Image Matching in 3D with MASt3R)

**Description**: MASt3R builds on DUSt3R by adding a 3D-aware dense matching head on top of the pointmap-based reconstruction. It produces both dense 3D pointmaps and dense local features for highly accurate matching, and the reference implementation provides tools for multi-view reconstruction and alignment.

**Key Features** (paper + official implementation):
- Joint 3D reconstruction and dense matching from image pairs
- Coarse-to-fine matching scheme with a fast reciprocal matching algorithm
- Confidence-aware pointmap regression with optional metric predictions
- Sparse global alignment and optional TSDF-based fusion utilities in the reference codebase

**Reference Paper**:  
Vincent Leroy, Yohann Cabon, Jerome Revaud. _"Grounding Image Matching in 3D with MASt3R."_ *ECCV 2024*.  
[Paper](https://arxiv.org/abs/2406.09756) | [Project page](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/) 

**Implementation**: `SceneFromViewsMast3r`

---

### 3. Depth Anything V3

**Description**: Depth Anything 3 is a monocular depth estimation model family that can produce both relative and metric depth. The official implementation also supports estimating camera parameters (extrinsics and intrinsics) from one or more images, which can be used for reconstruction pipelines.

**Key Features**:
- High-quality monocular depth estimation from single or multiple views
- Optional camera pose and intrinsic parameter estimation (where model variant supports it)
- Support for metric depth models (e.g. DA3 Metric series)
- Designed to work across diverse scenes and domains

**Reference Paper**:  
Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang. _"Depth Anything 3: Recovering the Visual Space from Any Views."_, 2025.  
[Paper](https://arxiv.org/pdf/2511.10647) | [Repository](https://github.com/ByteDance-Seed/Depth-Anything-3)

**Implementation**: `SceneFromViewsDepthAnythingV3`

---

### 4. MVDust3r (Multi-view DUSt3R)

**Description**: MV-DUSt3R and its enhanced variant MV-DUSt3R+ extend DUSt3R to the multi-view setting. They process multiple unposed RGB views in a single feed-forward pass, using multi-view decoder blocks (and cross-reference-view blocks in MV-DUSt3R+) to jointly reason about all input views and reconstruct the scene.

**Key Features**:
- Single-stage feed-forward multi-view reconstruction from sparse pose-free RGB views
- Multi-view transformer-style decoder that exchanges information across views
- Designed to handle up to tens of views in one forward pass
- Support for multiple checkpoints / variants (e.g. MVD.pth, MVDp_s1.pth, MVDp_s2.pth) in the official release

**Reference Paper**:  
Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, Zhicheng Yan, _"MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds"_
[Paper](https://arxiv.org/abs/2412.06974) | [Repository](https://github.com/facebookresearch/mvdust3r)

**Implementation**: `SceneFromViewsMvdust3r`

---

### 5. VGGT (Visual Geometry Grounded Transformer)

**Description**: VGGT is a large feed-forward transformer for 3D reconstruction that directly predicts camera parameters, pointmaps, depth maps, and 3D point tracks from one or many input views. It performs multi-view reconstruction in a single forward pass and achieves state-of-the-art results on several 3D tasks.

**Key Features**:
- Transformer-based feed-forward architecture
- Joint prediction of camera parameters (intrinsics + extrinsics), depth maps, pointmaps, and 3D tracks
- Handles from single-view to hundreds of views in one pass
- Strong performance on multi-view depth estimation, dense point cloud reconstruction, camera estimation, and tracking

**Reference Paper**:  
_"VGGT: Visual Geometry Grounded Transformer"_
[Paper](https://arxiv.org/abs/2503.11651) | [VGGT repository](https://github.com/facebookresearch/vggt).

**Implementation**: `SceneFromViewsVggt`

---

## Usage Example

See the main script `main_scene_from_views.py`.

```python
from pyslam.scene_from_views import scene_from_views_factory, SceneFromViewsType
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Create reconstructor using factory
reconstructor = scene_from_views_factory(
    scene_from_views_type=SceneFromViewsType.DUST3R,
    device='cuda',
    inference_size=512,
    min_conf_thr=20.0
)

# Reconstruct scene
result = reconstructor.reconstruct(images, as_pointcloud=True)

# Access results
print(f"Point cloud has {len(result.global_point_cloud.vertices)} points")
print(f"Estimated {len(result.camera_poses)} camera poses")
```

## Extending the Framework

To add a new reconstruction model:

1. **Create a new class** inheriting from `SceneFromViewsBase`:
   ```python
   class SceneFromViewsNewModel(SceneFromViewsBase):
       def __init__(self, device=None, **kwargs):
           super().__init__(device=device, **kwargs)
           # Initialize your model
       
       def preprocess_images(self, images, **kwargs):
           """
           Preprocess input images for your model.
           
           Args:
               images: List of input images
               **kwargs: Additional preprocessing parameters
           
           Returns:
               Preprocessed images in the format expected by your model
           """
           # Implement preprocessing (e.g., resize, normalize, convert format)
           return processed_images
       
       def infer(self, processed_images, **kwargs):
           """
           Run inference on preprocessed images.
           
           Args:
               processed_images: Preprocessed images from preprocess_images()
               **kwargs: Additional inference parameters
           
           Returns:
               Raw output from your model (format is model-specific)
           """
           # Run your model inference
           raw_output = self.model(processed_images, **kwargs)
           return raw_output
       
       def postprocess_results(
           self, raw_output, images, processed_images, as_pointcloud=True, **kwargs
       ):
           """
           Convert raw model output to SceneFromViewsResult.
           
           Args:
               raw_output: Raw output from infer()
               images: Original input images (for reference)
               processed_images: Preprocessed images used for inference
               as_pointcloud: Whether to return point cloud or mesh
               **kwargs: Additional postprocessing parameters
           
           Returns:
               SceneFromViewsResult: Standardized reconstruction results
           """
           # Extract and format results
           # Convert to SceneFromViewsResult format
           return SceneFromViewsResult(
               global_point_cloud=...,
               global_mesh=...,
               camera_poses=...,
               processed_images=...,
               depth_predictions=...,
               point_clouds=...,
               intrinsics=...,
               confidences=...,
           )
   ```

   **Note**: You don't need to implement `reconstruct()` - it's automatically provided by the base class and follows the pipeline: `preprocess_images()` → `infer()` → `postprocess_results()`.

2. **Add the model type** to `SceneFromViewsType` enum

3. **Register in factory** by adding a case in `scene_from_views_factory()`

4. **Follow the interface** defined by `SceneFromViewsBase` to ensure compatibility

<!-- ## Architecture Benefits

The shared pipeline pattern provides several advantages:

- **Consistency**: All models follow the same three-step reconstruction process
- **Maintainability**: Changes to the pipeline flow only need to be made in the base class
- **Flexibility**: Each step can accept model-specific parameters via `**kwargs`
- **Extensibility**: New models only need to implement the three specialized methods
- **Separation of Concerns**: Preprocessing, inference, and postprocessing are clearly separated

## Notes

All models return results in the standardized `SceneFromViewsResult` format:
- Camera poses are stored as camera-to-world (c2w) transformation matrices (4x4)
- Point clouds and meshes are in world coordinates
- Models handle device management (CPU/CUDA) automatically
- Each model may have specific dependencies that need to be installed separately
- The `reconstruct()` method accepts `**kwargs` which are passed through to all three pipeline steps, allowing runtime parameter overrides -->
