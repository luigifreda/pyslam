# Scene From Views

<p align="center">
  <img src="../images/scene_from_views.png" alt="3D Sparse Semantic Mapping" height="300"/>
</p>


This directory contains the implementation of the `scene_from_views` factory, which provides a **unified interface for end-to-end 3D scene reconstruction from multiple views**. The factory follows a modular architecture that allows easy integration of different reconstruction models while maintaining a consistent API. This document is a work in progress.


<!-- TOC -->

- [Scene From Views](#scene-from-views)
  - [Architecture](#architecture)
    - [Base Class: `SceneFromViewsBase`](#base-class-scenefromviewsbase)
    - [Result Structure: `SceneFromViewsResult`](#result-structure-scenefromviewsresult)
    - [Factory Pattern: `scene_from_views_factory`](#factory-pattern-scene_from_views_factory)
    - [Derived Implementations](#derived-implementations)
    - [Capability Quick Reference](#capability-quick-reference)
    - [Type Enumeration: `SceneFromViewsType`](#type-enumeration-scenefromviewstype)
  - [Supported Models and Reference Papers](#supported-models-and-reference-papers)
    - [DUSt3R (Geometric 3D Vision Made Easy)](#dust3r-geometric-3d-vision-made-easy)
    - [MASt3R (Grounding Image Matching in 3D with MASt3R)](#mast3r-grounding-image-matching-in-3d-with-mast3r)
    - [Depth Anything V3](#depth-anything-v3)
    - [MVDust3r (Multi-view DUSt3R)](#mvdust3r-multi-view-dust3r)
    - [VGGT (Visual Geometry Grounded Transformer)](#vggt-visual-geometry-grounded-transformer)
    - [VGGT Robust (Outlier-Aware VGGT)](#vggt-robust-outlier-aware-vggt)
    - [FAST3R](#fast3r)
  - [Scene Optimization Modules](#scene-optimization-modules)
    - [Overview](#overview)
    - [Available Optimizers](#available-optimizers)
      - [DenseSceneOptimizer](#densesceneoptimizer)
      - [SparseSceneOptimizer](#sparsesceneoptimizer)
    - [Configuration Dictionary](#configuration-dictionary)
      - [DenseSceneOptimizer Parameters](#densesceneoptimizer-parameters)
      - [SparseSceneOptimizer Parameters](#sparsesceneoptimizer-parameters)
    - [Global Scene Optimization Details](#global-scene-optimization-details)
      - [Dense Scene Optimization (global\_aligner)](#dense-scene-optimization-global_aligner)
      - [Sparse Scene Optimization (sparse\_scene\_optimizer)](#sparse-scene-optimization-sparse_scene_optimizer)
      - [Key Differences Summary](#key-differences-summary)
    - [TSDF Post-Processing](#tsdf-post-processing)
    - [Unified Optimizer Interface](#unified-optimizer-interface)
      - [Key Components](#key-components)
      - [Usage Examples](#usage-examples)
      - [Benefits](#benefits)
      - [Architecture](#architecture-1)
      - [Implementation Details](#implementation-details)
  - [Usage Examples](#usage-examples-1)
    - [Basic Usage](#basic-usage)
    - [Using Dense Scene Optimizer with Dust3r](#using-dense-scene-optimizer-with-dust3r)
    - [Using Sparse Scene Optimizer with MASt3r](#using-sparse-scene-optimizer-with-mast3r)
    - [Using Optimizer as Post-Processing](#using-optimizer-as-post-processing)
    - [Using Optimizer in reconstruct() call](#using-optimizer-in-reconstruct-call)
    - [Using VGGT Robust with outlier rejection](#using-vggt-robust-with-outlier-rejection)
    - [Using FAST3R for large-scale reconstruction](#using-fast3r-for-large-scale-reconstruction)
  - [Extending the Framework](#extending-the-framework)
  - [Notes and Best Practices](#notes-and-best-practices)

<!-- /TOC -->

---

## Architecture


<p align="center">
<img src="./images/scene_from_views.png" alt="3D Scene From Views" /> 
</p>

This diagram illustrates the architecture of the *Scene from Views* module, which provides a unified interface for 3D scene reconstruction from multiple views. At its core, the `scene_from_views_factory` instantiates specific reconstruction models based on the selected `SceneFromViewsType`, such as `DUST3R`, `MAST3R`, `DEPTH_ANYTHING_V3`, `MVDUST3R`, `VGGT`, `VGGT_ROBUST`, and `FAST3R`.

Each type creates a corresponding implementation (e.g., `SceneFromViewsDust3r`, `SceneFromViewsMast3r`, `SceneFromViewsDepthAnythingV3`, `SceneFromViewsMvdust3r`, `SceneFromViewsVggt`, `SceneFromViewsVggtRobust`, `SceneFromViewsFast3r`), all inheriting from a common `SceneFromViewsBase`. This base class implements a unified three-step reconstruction pipeline: `preprocess_images()` prepares input images for the specific model, `infer()` runs model inference, and `postprocess_results()` converts raw model output to a standardized `SceneFromViewsResult` format containing merged point clouds, meshes, camera poses, and optional depth maps or intrinsics.

The module supports both pairwise models (DUSt3R, MASt3R) that process image pairs and perform global alignment, as well as multi-view models (MV-DUSt3R, VGGT, FAST3R) that process multiple views simultaneously in a single forward pass. This modular design enables flexible integration of diverse 3D reconstruction techniques while maintaining a consistent API across different model architectures.

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

- **Optimizer Post-Processing**: Supports optional scene optimizer post-processing via the `optimizer` parameter in `reconstruct()` or the `apply_optimizer_postprocessing()` method.

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
   - Default optimizer: `DenseSceneOptimizer` (dense alignment)

2. **`SceneFromViewsMast3r`** (`scene_from_views_mast3r.py`)
   - Wraps MASt3R (Grounding Image Matching in 3D with MASt3R)
   - Provides sparse global alignment for multi-view consistency
   - Can optionally use TSDF-based fusion / refinement in the implementation
   - Default optimizer: `SparseSceneOptimizer` (sparse alignment)

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
   - Supports using the model's pointmap and depthmap predictions
   - Provides joint prediction of camera parameters and dense scene geometry

6. **`SceneFromViewsVggtRobust`** (`scene_from_views_vggt_robust.py`)
   - Wraps VGGT with anchor-based outlier view rejection (mirrors the `robust_vggt` demo)
   - Scores each view via global attention maps and cosine similarity to an anchor, then drops low-scoring views before the final forward pass
   - Supports background masking (black/white/sky flags), percentile-based confidence filtering, and anchor selection (first frame or most central)
   - Exposes rejected and survivor indices in the returned `SceneFromViewsResult`

7. **`SceneFromViewsFast3r`** (`scene_from_views_fast3r.py`)
   - Wraps Fast3R (Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass)
   - Processes 1000+ images in a single forward pass for large-scale reconstruction
   - Performs multi-view reconstruction with pose estimation using PnP
   - Uses global alignment to merge local point clouds into a unified coordinate frame
   - Supports confidence-based filtering and automatic camera pose estimation
   - **Note**: Requires significant GPU memory for large image sets

### Capability Quick Reference

- **End-to-end multi-view reconstruction** (poses + fused geometry directly from images): `SceneFromViewsDust3r`, `SceneFromViewsMast3r`, `SceneFromViewsMvdust3r`, `SceneFromViewsVggt`, `SceneFromViewsVggtRobust`, `SceneFromViewsFast3r`.
- **Large-scale reconstruction** (1000+ images in one pass): `SceneFromViewsFast3r` (designed for processing very large image collections efficiently)
- **Robust view filtering / outlier rejection**: `SceneFromViewsVggtRobust` (anchor-based attention + cosine scoring that discards low-confidence views before reconstruction)
- **Single-view depth-first pipeline with optional poses/intrinsics**: `SceneFromViewsDepthAnythingV3`
- **Global alignment optimization stage for merging views**: `SceneFromViewsDust3r` (dense alignment) and `SceneFromViewsMast3r` (sparse alignment variant)

Note that DUSt3R and MASt3R are **pairwise models**: they take two images at a time. Multi-view end-to-end reconstruction is achieved by running them on many image pairs and performing a global alignment / optimization over all pairwise pointmaps. Fast3R, MV-DUSt3R, and VGGT are **multi-view models** that process all images simultaneously in a single forward pass.

### Type Enumeration: `SceneFromViewsType`

The `SceneFromViewsType` enum (`scene_from_views_types.py`) defines all supported model types:
- `DEPTH_ANYTHING_V3`
- `MAST3R`
- `MVDUST3R`
- `VGGT`
- `VGGT_ROBUST`
- `DUST3R`
- `FAST3R`

---

## Supported Models and Reference Papers

### DUSt3R (Geometric 3D Vision Made Easy)

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



### MASt3R (Grounding Image Matching in 3D with MASt3R)

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


### Depth Anything V3

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



### MVDust3r (Multi-view DUSt3R)

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



### VGGT (Visual Geometry Grounded Transformer)

**Description**: VGGT is a large feed-forward transformer for 3D reconstruction that directly predicts camera parameters, pointmaps, depth maps, and 3D point tracks from one or many input views. It performs multi-view reconstruction in a single forward pass and achieves state-of-the-art results on several 3D tasks.

**Key Features**:
- Transformer-based feed-forward architecture
- Joint prediction of camera parameters (intrinsics + extrinsics), depth maps, pointmaps, and 3D tracks
- Handles from single-view to hundreds of views in one pass
- Strong performance on multi-view depth estimation, dense point cloud reconstruction, camera estimation, and tracking

**Reference Paper**:  
_"VGGT: Visual Geometry Grounded Transformer"_  
[Paper](https://arxiv.org/abs/2503.11651) | [VGGT repository](https://github.com/facebookresearch/vggt)

**Implementation**: `SceneFromViewsVggt`

### VGGT Robust (Outlier-Aware VGGT)

**Description**: A VGGT variant that mirrors the `robust_vggt` demo by scoring each view against an anchor frame and discarding outlier views before the final forward pass. Anchor selection can be the first frame or the most "central" frame based on token similarity, and scores blend global attention responses with cosine similarity.

**Key Features**:
- Anchor-based view scoring with tunable weights (`attn_a`, `cos_a`) and a rejection threshold (`rej_thresh`); optional anchor-relative normalization via `use_anchor_relative_score`
- Flexible anchor choice (`use_most_central_as_reference`) and percentile-based confidence filtering of 3D points (`conf_thres`)
- Optional masking helpers for black/white backgrounds or sky regions, plus padding/resizing to square inputs (`target_size`)
- Returns metadata about dropped views through `result.rejected_indices` and `result.survivor_indices`
- Default checkpoint: `facebook/VGGT-1B` (override with `model_id`)

**Reference Paper**:
"Emergent Outlier View Rejection in Visual Geometry Grounded Transformers"
[Paper](https://arxiv.org/abs/2512.04012) | [VGGT Robust repository](https://github.com/cvlab-kaist/RobustVGGT)

**Implementation**: `SceneFromViewsVggtRobust` (`scene_from_views_vggt_robust.py`, depends on `thirdparty/vggt_robust`)

### FAST3R

**Description**: Fast3R is a transformer-based model that can process 1000+ images in a single forward pass for large-scale 3D reconstruction. It extends DUSt3R's architecture to handle massive image collections efficiently, making it ideal for reconstructing scenes from video sequences or large photo collections.

**Key Features**:
- Single forward pass processing of 1000+ images simultaneously
- Multi-view transformer architecture that jointly reasons over all input views
- Automatic camera pose estimation using PnP (Perspective-n-Point) algorithm
- Global alignment of local point clouds into a unified coordinate frame
- Confidence-aware point cloud filtering and depth estimation
- Support for images with different aspect ratios and resolutions
- Efficient memory management for large-scale reconstruction

**Key Features** (as used in this framework):
- Processes all input images in a single forward pass (no pairwise processing needed)
- Estimates camera-to-world poses using PnP with configurable iterations
- Aligns local 3D points to global coordinate frame using confidence-based filtering
- Extracts dense point clouds, depth maps, and confidence maps per view
- Supports both Hugging Face model loading and local checkpoint loading
- Configurable image size (224 or 512), confidence thresholds, and focal length estimation methods

**Reference Paper**:  
"Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass." 
[Paper](https://arxiv.org/abs/2503.11651) | [Repository](https://github.com/facebookresearch/fast3r) | [Project page](https://fast3r-3d.github.io/)

**Implementation**: `SceneFromViewsFast3r`

**Note**: Fast3R requires significant GPU memory, especially when processing large numbers of images. The model automatically downloads pre-trained weights from Hugging Face (default: `jedyang97/Fast3R_ViT_Large_512`) or can load from a local checkpoint directory.

---

## Scene Optimization Modules

### Overview

The scene optimization modules (`optimizers/` directory) provide wrapper classes for different scene optimization methods that can be used with `SceneFromViewsDust3r`, `SceneFromViewsMast3r`, and optionally as post-processing with other `SceneFromViews` classes.

All optimizers inherit from `SceneOptimizerBase`, making them interchangeable and ensuring a consistent interface.

### Available Optimizers

#### DenseSceneOptimizer

Wrapper for Dust3r's `global_aligner` optimizer. Performs dense optimization on full depth maps and point clouds using a single-stage approach.

**How it works:**
- Optimizes **every pixel** in the depth maps directly
- Works on dense point clouds where each pixel has a corresponding 3D point
- Single-stage optimization that aligns all pairwise 3D predictions simultaneously
- Optimizes camera poses, dense depth maps (per-pixel), and intrinsics jointly

**Best for:**
- Faster processing
- Good initial depth estimates
- Simpler optimization requirements

**Usage:**
```python
from pyslam.scene_from_views import SceneFromViewsDust3r

# Default (uses dense_scene_optimizer)
reconstructor = SceneFromViewsDust3r()

# Explicit configuration
reconstructor = SceneFromViewsDust3r(
    optimizer_config={
        "type": "dense_scene_optimizer",  # or "global_aligner" for backward compatibility
        "niter": 300,
        "schedule": "linear",
        "lr": 0.01,
    }
)
```

#### SparseSceneOptimizer

Wrapper for MASt3r's `sparse_scene_optimizer`. Performs sparse optimization on feature correspondences using a two-stage approach (coarse + fine).

**How it works:**
- Optimizes on **subsampled pixels/features** rather than every pixel
- Uses a `subsample` parameter (default: 8) to extract features at regular intervals (e.g., every 8th pixel)
- Works on sparse correspondences between matching features across image pairs
- Two-stage optimization:
  - **Coarse stage**: Aligns sparse 3D correspondences
  - **Fine stage**: Refines with 2D reprojection errors
- After optimization, dense depth maps are generated from the sparse depth anchors

**Best for:**
- Higher accuracy requirements
- Challenging scenes with occlusions
- When feature correspondences are reliable

**Usage:**
```python
from pyslam.scene_from_views import SceneFromViewsMast3r

# Default (uses sparse_scene_optimizer)
reconstructor = SceneFromViewsMast3r()

# Explicit configuration
reconstructor = SceneFromViewsMast3r(
    optimizer_config={
        "type": "sparse_scene_optimizer",
        "subsample": 8,
        "lr1": 0.07,
        "niter1": 500,
        "lr2": 0.014,
        "niter2": 200,
        "matching_conf_thr": 5.0,
        "shared_intrinsics": False,
        "optim_level": "refine+depth",
        "kinematic_mode": "hclust-ward",
    }
)
```

### Configuration Dictionary

The `optimizer_config` parameter accepts a dictionary with the following structure:

```python
optimizer_config = {
    "type": "dense_scene_optimizer" | "sparse_scene_optimizer",
    # Additional parameters specific to the optimizer type
    ...
}
```

**Note**: `"global_aligner"` is supported as an alias for `"dense_scene_optimizer"` for backward compatibility.

#### DenseSceneOptimizer Parameters

- `niter` (int, default: 300): Number of optimization iterations
- `schedule` (str, default: "linear"): Learning rate schedule ('linear', 'cosine', etc.)
- `lr` (float, default: 0.01): Learning rate

#### SparseSceneOptimizer Parameters

- `subsample` (int, default: 8): Subsampling factor for feature extraction
- `lr1` (float, default: 0.07): Coarse learning rate
- `niter1` (int, default: 500): Number of coarse iterations
- `lr2` (float, default: 0.014): Fine learning rate
- `niter2` (int, default: 200): Number of fine iterations
- `matching_conf_thr` (float, default: 5.0): Matching confidence threshold
- `shared_intrinsics` (bool, default: False): Whether to use shared intrinsics
- `optim_level` (str, default: "refine+depth"): Optimization level ('coarse', 'refine', 'refine+depth')
- `kinematic_mode` (str, default: "hclust-ward"): Kinematic chain mode ('mst', 'hclust-ward', 'hclust-complete', etc.)

### Global Scene Optimization Details

Both `global_aligner()` (Dust3r) and `sparse_scene_optimizer()` (MASt3r) are methods for aligning multiple camera views into a consistent 3D scene, but they use fundamentally different approaches.

#### Dense Scene Optimization (global_aligner)

**Location**: `pyslam.scene_from_views.optimizers.dense_scene_optimizer` (consolidates code from `dust3r.cloud_opt.global_aligner`)

**Approach**: Dense optimization on full depth maps and point clouds

**Key Characteristics**:
- Works directly on dense depth maps and 3D point clouds from model output
- Single-stage optimization with a unified loss function
- Optimizes:
  - Camera poses (extrinsics)
  - Dense depth maps (per-pixel)
  - Camera intrinsics (focal length, principal point)
- Loss function: Aligns pairwise 3D predictions by minimizing distance between corresponding 3D points
- Simpler and faster approach
- Good for scenes with good initial depth estimates

**Optimization Process**:
1. Takes pairwise model outputs (3D points + confidence maps)
2. Initializes global camera poses using MST (minimum spanning tree)
3. Optimizes all parameters jointly in a single pass
4. Uses adaptive learning rate scheduling

**When to use**:
- Faster processing needed
- Good initial depth estimates from model
- Simpler optimization requirements

#### Sparse Scene Optimization (sparse_scene_optimizer)

**Location**: `pyslam.scene_from_views.optimizers.sparse_scene_optimizer` (wrapper for `mast3r.cloud_opt.sparse_ga.sparse_scene_optimizer`)

**Approach**: Sparse optimization on feature correspondences

**Key Characteristics**:
- Works on sparse correspondences extracted from feature descriptors
- Two-stage optimization:
  - **Coarse stage**: 3D point matching loss (`loss_3d`) - aligns sparse 3D correspondences
  - **Fine stage**: 2D reprojection loss (`loss_2d`) - refines with pixel-level reprojection
- Uses kinematic chain (MST or hierarchical clustering) to build camera relationships
- Optimizes:
  - Camera poses (as relative poses in kinematic chain)
  - Sparse depth anchors (subsampled depth values)
  - Camera intrinsics (focal length, principal point)
- More sophisticated optimization with multiple loss terms
- Potentially more accurate but slower

**Optimization Process**:
1. Extracts sparse correspondences using feature matching (fast reciprocal NNs)
2. Builds canonical views per image (averages multiple pairwise predictions)
3. Constructs kinematic chain (MST or hierarchical clustering) from pairwise scores
4. Coarse optimization: Aligns 3D correspondences
5. Fine optimization: Refines with 2D reprojection errors
6. Dense depth maps are generated from sparse anchors after optimization

**When to use**:
- Higher accuracy needed
- Challenging scenes with occlusions or textureless regions
- When feature correspondences are reliable
- Can handle more complex camera configurations

#### Key Differences Summary

| Aspect | DenseSceneOptimizer (global_aligner) | SparseSceneOptimizer (sparse_scene_optimizer) |
|--------|-------------------------------------|-----------------------------------------------|
| **Data representation** | Dense depth maps | Sparse correspondences |
| **Optimization stages** | Single stage | Two stages (coarse + fine) |
| **Loss functions** | 3D point alignment | 3D matching + 2D reprojection |
| **Camera representation** | Absolute poses | Relative poses (kinematic chain) |
| **Depth representation** | Full dense maps | Sparse anchors (densified later) |
| **Speed** | Faster | Slower (more computation) |
| **Accuracy** | Good | Potentially better |
| **Complexity** | Simpler | More sophisticated |

### TSDF Post-Processing

**Purpose**: Post-processing step that refines depth maps using TSDF (Truncated Signed Distance Function) fusion

**How it works**:
- Builds a TSDF volume from all camera views
- For each pixel, searches along the ray to find the zero-level set of TSDF
- Refines depth estimates by finding surfaces in the TSDF volume
- Helps remove outliers and improve depth consistency

**When to use**:
- When depth maps have noise or outliers
- To improve surface consistency across views
- When `TSDF_thresh > 0` (disabled when `TSDF_thresh = 0`)

**Note**: TSDFPostProcess is located in `pyslam.scene_from_views.optimizers.tsdf_postprocess` and is optional. It can be disabled by setting `TSDF_thresh=0` or `use_tsdf=False` in `SceneFromViewsMast3r`. The TSDF post-processing code was originally from MASt3r's `tsdf_optimizer` module.

### Unified Optimizer Interface

The unified optimizer interface allows you to use any optimizer (dense or sparse) with any model output format (Dust3r, MASt3r, etc.). Format conversion from model-specific outputs to the unified `SceneOptimizerInput` format is handled by the `create_optimizer_input()` methods in each `SceneFromViews` class.

#### Key Components

**SceneOptimizerInput**

A unified input structure that can represent model outputs from any method. The structure uses `pair_predictions` (a list of `PairPrediction` objects) as the core data format:

```python
from pyslam.scene_from_views.optimizers import SceneOptimizerInput, PairPrediction

# SceneOptimizerInput structure:
optimizer_input = SceneOptimizerInput(
    images=images,  # List of [H, W, 3] RGB images
    pair_predictions=pair_predictions,  # List of PairPrediction objects
    pairs=pairs,  # List of image pair metadata
    filelist=filelist,  # Optional list of image file names
    cache_dir=cache_dir,  # Optional cache directory (for sparse optimizer)
    pairs_output=pairs_output,  # Optional canonical view data (for sparse optimizer)
)
```

**Note**: You typically don't construct `SceneOptimizerInput` directly. Instead, use the `create_optimizer_input()` method provided by each `SceneFromViews` class to convert model output into the unified format.

**SceneOptimizerOutput**

A unified output structure from scene optimizers:

```python
from pyslam.scene_from_views.optimizers import SceneOptimizerOutput

optimizer_output = SceneOptimizerOutput(
    scene=scene,
    optimizer_type="dense_scene_optimizer",
    rgb_imgs=rgb_imgs,
    focals=focals,
    cams2world=cams2world,
    pts3d=pts3d,
    confs=confs,
)
```

**Creating Optimizer Input**

Each `SceneFromViews` class provides a `create_optimizer_input()` method to convert model output into a unified `SceneOptimizerInput`:

```python
from pyslam.scene_from_views import SceneFromViewsDust3r, SceneFromViewsMast3r

# Create input from Dust3r output
dust3r_reconstructor = SceneFromViewsDust3r()
optimizer_input = dust3r_reconstructor.create_optimizer_input(
    raw_output=output,
    pairs=pairs,
    processed_images=imgs_preproc,
    filelist=filelist,  # optional
)

# Create input from MASt3r output
mast3r_reconstructor = SceneFromViewsMast3r()
optimizer_input = mast3r_reconstructor.create_optimizer_input(
    raw_output=pairs_output,
    pairs=pairs,
    processed_images=imgs_preproc,
    filelist=filelist,  # required
    cache_dir=cache_dir,  # required
)
```

#### Usage Examples

**Example 1: Using Unified Interface with Any Optimizer**

```python
from pyslam.scene_from_views import SceneFromViewsDust3r
from pyslam.scene_from_views.optimizers import scene_optimizer_factory

# Create Dust3r reconstructor
reconstructor = SceneFromViewsDust3r()

# Run inference (this would normally be done internally)
# ... get output, pairs, processed_images ...

# Create optimizer
optimizer = scene_optimizer_factory(
    optimizer_config={"type": "dense_scene_optimizer"},
)

# Create unified input from Dust3r output using the reconstructor's method
optimizer_input = reconstructor.create_optimizer_input(
    raw_output=output,
    pairs=pairs,
    processed_images=processed_images,
)

# Optimize using unified interface (works with any optimizer)
optimizer_output = optimizer.optimize(
    optimizer_input=optimizer_input,
    verbose=True,
    niter=500,
)

# Access results
scene = optimizer_output.scene
rgb_imgs = optimizer_output.rgb_imgs
focals = optimizer_output.focals
cams2world = optimizer_output.cams2world
```

**Example 2: Using Dense Optimizer with MASt3r Output**

```python
from pyslam.scene_from_views import SceneFromViewsMast3r
from pyslam.scene_from_views.optimizers import scene_optimizer_factory

# Create MASt3r reconstructor
reconstructor = SceneFromViewsMast3r()

# Run inference (this would normally be done internally)
# ... get pairs_output, pairs, processed_images, filelist, cache_dir ...

# Create unified input using the reconstructor's method
optimizer_input = reconstructor.create_optimizer_input(
    raw_output=pairs_output,
    pairs=pairs,
    processed_images=processed_images,
    filelist=filelist,
    cache_dir=cache_dir,
)

# Create dense optimizer
optimizer = scene_optimizer_factory(
    optimizer_config={"type": "dense_scene_optimizer"},
)

# Optimize (will automatically convert format if supported)
optimizer_output = optimizer.optimize(
    optimizer_input=optimizer_input,
    verbose=True,
)
```

**Example 3: Using Sparse Optimizer with Dust3r Output**

```python
from pyslam.scene_from_views import SceneFromViewsDust3r
from pyslam.scene_from_views.optimizers import scene_optimizer_factory

# Create Dust3r reconstructor
reconstructor = SceneFromViewsDust3r()

# Run inference
# ... get output, pairs, processed_images ...

# Create unified input using the reconstructor's method
optimizer_input = reconstructor.create_optimizer_input(
    raw_output=output,
    pairs=pairs,
    processed_images=processed_images,
)

# Create sparse optimizer
optimizer = scene_optimizer_factory(
    optimizer_config={"type": "sparse_scene_optimizer"},
)

# Optimize (will automatically convert format if supported)
optimizer_output = optimizer.optimize(
    optimizer_input=optimizer_input,
    verbose=True,
)
```

#### Benefits

1. **Unified Interface**: All optimizers use the same `optimize()` method that accepts `SceneOptimizerInput`
2. **Standardized Format**: Both optimizers work on `pair_predictions` (list of `PairPrediction` objects), eliminating format-specific code
3. **Type Safety**: Clear input/output types with `SceneOptimizerInput` and `SceneOptimizerOutput`
4. **Easy Integration**: Each `SceneFromViews` class provides `create_optimizer_input()` to convert model output to the unified format
5. **Modular Design**: Optimizers are self-contained and can be used independently or as post-processing steps

#### Architecture

The unified interface is built into `SceneOptimizerBase`:

- `optimize()`: Accepts `SceneOptimizerInput`, returns `SceneOptimizerOutput`
- `extract_results()`: Accepts `SceneOptimizerOutput`, returns updated `SceneOptimizerOutput`

#### Implementation Details

Both `DenseSceneOptimizer` and `SparseSceneOptimizer` operate directly on `SceneOptimizerInput` with `pair_predictions`. The format conversion is handled by the `create_optimizer_input()` methods in each `SceneFromViews` class, which convert model-specific outputs into the unified `PairPrediction` format.

**Code Organization**:
- Dense optimizer code is consolidated in `dense_scene_optimizer.py` and `sparse_scene_optimizer_helpers.py` (shared utilities)
- Sparse optimizer code is in `sparse_scene_optimizer.py`
- TSDF post-processing is in `tsdf_postprocess.py` (can be used with any optimizer)
- Common utilities (learning rate schedules, point cloud cleaning) are in `optimizer_utils.py`

---

## Usage Examples

### Basic Usage

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

### Using Dense Scene Optimizer with Dust3r

```python
from pyslam.scene_from_views import SceneFromViewsDust3r
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Create reconstructor with dense_scene_optimizer
reconstructor = SceneFromViewsDust3r(
    optimizer_config={
        "type": "dense_scene_optimizer",
        "niter": 500,
        "lr": 0.02,
        "schedule": "cosine",
    }
)

# Reconstruct scene
result = reconstructor.reconstruct(images)
```

### Using Sparse Scene Optimizer with MASt3r

```python
from pyslam.scene_from_views import SceneFromViewsMast3r
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Create reconstructor with sparse_scene_optimizer
reconstructor = SceneFromViewsMast3r(
    optimizer_config={
        "type": "sparse_scene_optimizer",
        "lr1": 0.1,
        "niter1": 600,
        "lr2": 0.02,
        "niter2": 300,
        "kinematic_mode": "mst",
    }
)

# Reconstruct scene
result = reconstructor.reconstruct(images)
```

### Using Optimizer as Post-Processing

```python
from pyslam.scene_from_views import SceneFromViewsDepthAnythingV3
from pyslam.scene_from_views.optimizers import DenseSceneOptimizer
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Create initial reconstructor (without optimizer)
reconstructor = SceneFromViewsDepthAnythingV3()

# Get initial result
result = reconstructor.reconstruct(images)

# Apply optimizer as post-processing
optimizer = DenseSceneOptimizer(device="cuda")
optimized_result = reconstructor.apply_optimizer_postprocessing(
    result, optimizer, processed_images=reconstructor.preprocess_images(images)
)
```

### Using Optimizer in reconstruct() call

```python
from pyslam.scene_from_views import SceneFromViewsDepthAnythingV3
from pyslam.scene_from_views.optimizers import SparseSceneOptimizer
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Create reconstructor
reconstructor = SceneFromViewsDepthAnythingV3()

# Reconstruct with optimizer post-processing
optimizer = SparseSceneOptimizer(device="cuda")
result = reconstructor.reconstruct(
    images,
    optimizer=optimizer,
    optimizer_kwargs={"lr1": 0.1, "niter1": 600}
)
```

### Using VGGT Robust with outlier rejection

```python
from pyslam.scene_from_views import SceneFromViewsVggtRobust
import cv2

# Load images
images = [cv2.imread(f"image_{i}.jpg") for i in range(6)]

# Create reconstructor with anchor-based view rejection
reconstructor = SceneFromViewsVggtRobust(
    rej_thresh=0.25,                  # drop views below this combined score
    attn_a=0.5,                       # weight for attention-based score
    cos_a=0.5,                        # weight for cosine similarity score
    use_most_central_as_reference=True,  # choose anchor automatically
)

# Reconstruct scene (low-scoring views are skipped automatically)
result = reconstructor.reconstruct(images, as_pointcloud=True)

print("Rejected view indices:", getattr(result, "rejected_indices", []))
print("Used view indices:", getattr(result, "survivor_indices", []))
```

### Using FAST3R for large-scale reconstruction

```python
from pyslam.scene_from_views import SceneFromViewsFast3r
import cv2

# Load many images (Fast3R can handle 1000+ images in one pass)
images = [cv2.imread(f"image_{i}.jpg") for i in range(100)]

# Create reconstructor with custom settings
reconstructor = SceneFromViewsFast3r(
    device='cuda',
    checkpoint_dir="jedyang97/Fast3R_ViT_Large_512",  # or path to local checkpoint
    image_size=512,                                    # 224 or 512
    min_conf_thr_percentile=10,                       # confidence threshold percentile
    niter_PnP=100,                                     # PnP iterations for pose estimation
    focal_length_estimation_method="first_view_from_global_head",
)

# Reconstruct scene (all images processed in one forward pass)
result = reconstructor.reconstruct(images, as_pointcloud=True)

# Access results
print(f"Point cloud has {len(result.global_point_cloud.vertices)} points")
print(f"Estimated {len(result.camera_poses)} camera poses")
print(f"Processed {len(result.processed_images)} images")

# Using with factory pattern
from pyslam.scene_from_views import scene_from_views_factory, SceneFromViewsType

reconstructor = scene_from_views_factory(
    scene_from_views_type=SceneFromViewsType.FAST3R,
    device='cuda',
    image_size=512,
    min_conf_thr_percentile=10,
)

result = reconstructor.reconstruct(images)
```

**Note**: Fast3R requires significant GPU memory. For very large image sets (1000+ images), ensure you have sufficient GPU memory or process images in batches.

---

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

---

## Notes and Best Practices

- **Base Class**: All optimizers inherit from `SceneOptimizerBase`, ensuring a consistent interface and making them interchangeable.

- **Default behavior**: If `optimizer_config` is `None`, each class uses its native optimizer:
  - `SceneFromViewsDust3r` defaults to `dense_scene_optimizer`
  - `SceneFromViewsMast3r` defaults to `sparse_scene_optimizer`

- **Cross-optimizer usage**: Both optimizers can work with any `SceneFromViews` class that provides `create_optimizer_input()`. However, note that:
  - `SparseSceneOptimizer` requires `pairs_output` (canonical view data) which is only available from `SceneFromViewsMast3r`
  - `DenseSceneOptimizer` works with `pair_predictions` which can be created from any model output
  - For best results, use the native optimizer for each class (`dense_scene_optimizer` with Dust3r, `sparse_scene_optimizer` with MASt3r)

- **Post-processing mode**: Optimizers can be used as post-processing steps on results from other `SceneFromViews` classes. Use the `optimizer` parameter in `reconstruct()` or call `apply_optimizer_postprocessing()` directly. Note that not all optimizers support post-processing from `SceneFromViewsResult` (they may require model-specific output formats).

- **TSDF Post-processing**: TSDF post-processing can be enabled/disabled independently using the `use_tsdf` and `TSDF_thresh` parameters in `SceneFromViewsMast3r`.

- **Backward compatibility**: The name `"global_aligner"` is still supported as an alias for `"dense_scene_optimizer"` for backward compatibility.

- **Camera poses**: All models return camera poses as camera-to-world (c2w) transformation matrices (4x4).

- **Point clouds and meshes**: All results are in world coordinates.

- **Device management**: Models handle device management (CPU/CUDA) automatically.

- **Dependencies**: Each model may have specific dependencies that need to be installed separately.

- **Parameter passing**: The `reconstruct()` method accepts `**kwargs` which are passed through to all three pipeline steps, allowing runtime parameter overrides.
