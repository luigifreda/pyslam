# Volumetric Mapping Module

This module provides efficient C++ implementations for volumetric data structures used in SLAM (Simultaneous Localization and Mapping) applications. It offers voxel grid representations with support for point cloud integration, color information, and semantic segmentation.



<p align="center">
  <img src="../images/dense-reconstruction2.png" height="300" /> 
</p>

<p align="center">
  <img src="../images/dense-reconstruction-composition.gif"
       alt="pySLAM - Dense reconstruction - Gaussian Splatting"
       height="300">
</p>


<!-- TOC -->

- [Volumetric Mapping Module](#volumetric-mapping-module)
  - [Overview](#overview)
  - [Key Components](#key-components)
    - [Core Data Structures](#core-data-structures)
      - [Semantic Data Type Comparison](#semantic-data-type-comparison)
    - [Grid Implementations](#grid-implementations)
    - [Semantic Extensions](#semantic-extensions)
    - [Utilities](#utilities)
  - [Features](#features)
  - [Usage](#usage)

<!-- /TOC -->


## Overview

The volumetric module implements sparse voxel grids using hash-based data structures, optimized for real-time SLAM applications. It supports both direct voxel hashing and block-based hashing strategies, with optional semantic information (class IDs, instance/object IDs) and probabilistic fusion.

## Key Components

### Core Data Structures

The voxel data types are now defined with templates to allow compile-time customization of the stored fields (e.g., color, semantic IDs, probabilistic fusion state) while preserving the same high-level grid APIs.

- **`voxel_data.h`**: Defines voxel data types (template-based):
  - `VoxelData`: Basic voxel storing position and color averages
  - `VoxelSemanticData`: Semantic voxel with instance/class IDs and confidence counter
  - `VoxelSemanticDataProbabilistic`: Probabilistic semantic voxel using Bayesian fusion in log-space

- **`voxel_data_semantic.h`**: Extended semantic voxel data types (template-based):
  - `VoxelSemanticData2`: Semantic voxel with separate confidence counters for object and class IDs
  - `VoxelSemanticDataProbabilistic2`: Probabilistic semantic voxel with independent marginal distributions (assumes independence between object and class IDs)


**Semantic fields**
  - `class_id=0` background, `class_id=-1` invalid, `class_id>0` actual semantic class
  - `object_id=0` no specific object, `object_id=-1` invalid, `object_id>0` specific object
  - The code relies on `<0` checks, so `object_id`/instance IDs must remain signed

#### Semantic Data Type Comparison

**Probabilistic Variants:**
- **`VoxelSemanticDataProbabilistic`** (recommended): Uses joint probability distribution P(object_id, class_id), preserving correlations between object and class IDs. More efficient and provides better behavior.
- **`VoxelSemanticDataProbabilistic2`**: Uses separate marginal distributions P(object_id) and P(class_id), assuming independence. This approach has limitations:
  - **Independence assumption**: Object and class IDs are often correlated, making the factorization P(obj, cls) = P(obj) × P(cls) incorrect
  - **Update splitting**: Splitting log probabilities in half dilutes the signal when marginals are shared across pairs
  - **Over-normalization**: Normalizing over all possible pairs (not just observed ones) makes probabilities artificially small

**Voting Variants:**
- **`VoxelSemanticData`**: Uses a single confidence counter for the (object_id, class_id) pair
- **`VoxelSemanticData2`**: Uses separate confidence counters for object_id and class_id independently

### Grid Implementations

- **`voxel_grid.h`**: Direct voxel hashing grid (`VoxelGridT`)
  - Hash map from voxel coordinates to voxel data
  - SIMD optimizations (AVX2, SSE4.1) for fast integration
  - TBB parallelization support

- **`voxel_block_grid.h`**: Block-based voxel grid (`VoxelBlockGridT`)
  - Divides space into fixed-size blocks (N×N×N voxels)
  - More memory-efficient for sparse data
  - Thread-safe with per-block mutexes

### Semantic Extensions

- **`voxel_semantic_grid.h`**: Semantic grid with direct hashing (`VoxelSemanticGridT`)
- **`voxel_block_semantic_grid.h`**: Semantic grid with block-based hashing (`VoxelBlockSemanticGridT`)

Both support:
- Instance and class ID tracking
- Segment merging and removal
- Confidence-based filtering
- Multiple semantic data variants (standard, probabilistic, and their variants with separate counters/marginals)

### Utilities

- **`voxel_hashing.h`**: Hashing functions and coordinate transformations
  - `VoxelKey`: Direct voxel coordinate hashing
  - `BlockKey` / `LocalVoxelKey`: Block-based coordinate system
  - Efficient floor division for negative coordinates

- **`voxel_block.h`**: Block structure containing a 3D array of voxels

- **`tbb_utils.h`**: TBB (Threading Building Blocks) thread management utilities

- **`image_utils.h`**: Image processing utilities
  - `check_image_size`: Validates image dimensions against expected size
  - `convert_image_type_if_needed`: Converts OpenCV image types when needed
  - `remap_instance_ids`: Remaps instance IDs using a provided mapping

- **`voxel_grid_carving.h`**: Depth-based voxel carving
  - `carve`: Removes voxels inconsistent with depth images (carves voxels in front of observed depth)

- **`voxel_semantic_data_association.h`**: Semantic data association utilities
  - `assign_object_ids_to_instance_ids`: Assigns object IDs to instance IDs using voting mechanism with depth filtering and optional carving

- **`volumetric_module.cpp`**: Python bindings (pybind11) for all grid types and utilities

## Features

- **Sparse Representation**: Only stores occupied voxels, memory-efficient for large scenes
- **Multi-threading**: Optional TBB parallelization for faster point cloud integration
- **SIMD Optimizations**: Optional vectorized operations for float/double point processing (for `voxel_grid.h`)
- **Type Safety**: C++20 concepts ensure compile-time type checking
- **Python Integration**: Full pybind11 bindings for Python usage
- **Semantic Support**: Instance segmentation and class labeling with probabilistic fusion
- **Depth-based Carving**: Removes inconsistent voxels based on depth observations
- **Semantic Association**: Voting-based assignment of object IDs to instance IDs with depth filtering
- **Multiple Semantic Variants**: Support for different semantic fusion strategies (voting, probabilistic joint, probabilistic marginal)

## Usage

The module is designed to be used through Python bindings. Grid types are exposed as:
- `VoxelGrid`: Basic voxel grid
- `VoxelBlockGrid`: Block-based voxel grid
- `VoxelSemanticGrid`: Semantic voxel grid (using `VoxelSemanticData`)
- `VoxelSemanticGrid2`: Semantic voxel grid with separate confidence counters (using `VoxelSemanticData2`)
- `VoxelSemanticGridProbabilistic`: Probabilistic semantic grid (using `VoxelSemanticDataProbabilistic`)
- `VoxelSemanticGridProbabilistic2`: Probabilistic semantic grid with marginal distributions (using `VoxelSemanticDataProbabilistic2`)
- `VoxelBlockSemanticGrid`: Block-based semantic grid
- `VoxelBlockSemanticGrid2`: Block-based semantic grid with separate confidence counters (just experimental)
- `VoxelBlockSemanticProbabilisticGrid`: Block-based probabilistic semantic grid
- `VoxelBlockSemanticProbabilistic2Grid`: Block-based probabilistic semantic grid with marginal distributions (just experimental)

All grids support:
- Point cloud integration with optional color and semantic information
- Depth-based carving via `carve()` method
- Semantic data association via `assign_object_ids_to_instance_ids()` function

Additional utilities available:
- `check_image_size()`: Validate image dimensions
- `convert_image_type_if_needed()`: Convert image types
- `remap_instance_ids()`: Remap instance IDs using a mapping








