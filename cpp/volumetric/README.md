# Volumetric Module

This module provides efficient C++ implementations for volumetric data structures used in SLAM (Simultaneous Localization and Mapping) applications. It offers voxel grid representations with support for point cloud integration, color information, and semantic segmentation.

## Overview

The volumetric module implements sparse voxel grids using hash-based data structures, optimized for real-time SLAM applications. It supports both direct voxel hashing and block-based hashing strategies, with optional semantic information (instance IDs, class IDs) and probabilistic fusion.

## Key Components

### Core Data Structures

- **`voxel_data.h`**: Defines voxel data types:
  - `VoxelData`: Basic voxel storing position and color averages
  - `VoxelSemanticData`: Semantic voxel with instance/class IDs and confidence counter
  - `VoxelSemanticDataProbabilistic`: Probabilistic semantic voxel using Bayesian fusion in log-space

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

### Utilities

- **`voxel_hashing.h`**: Hashing functions and coordinate transformations
  - `VoxelKey`: Direct voxel coordinate hashing
  - `BlockKey` / `LocalVoxelKey`: Block-based coordinate system
  - Efficient floor division for negative coordinates

- **`voxel_block.h`**: Block structure containing a 3D array of voxels

- **`tbb_utils.h`**: TBB (Threading Building Blocks) thread management utilities

- **`volumetric_grid_module.cpp`**: Python bindings (pybind11) for all grid types

## Features

- **Sparse Representation**: Only stores occupied voxels, memory-efficient for large scenes
- **Multi-threading**: Optional TBB parallelization for faster point cloud integration
- **SIMD Optimizations**: Vectorized operations for float/double point processing
- **Type Safety**: C++20 concepts ensure compile-time type checking
- **Python Integration**: Full pybind11 bindings for Python usage
- **Semantic Support**: Instance segmentation and class labeling with probabilistic fusion

## Usage

The module is designed to be used through Python bindings. Grid types are exposed as:
- `VoxelGrid`: Basic voxel grid
- `VoxelBlockGrid`: Block-based voxel grid
- `VoxelSemanticGrid`: Semantic voxel grid
- `VoxelSemanticGridProbabilistic`: Probabilistic semantic grid
- `VoxelBlockSemanticGrid`: Block-based semantic grid
- `VoxelBlockSemanticProbabilisticGrid`: Block-based probabilistic semantic grid

All grids support point cloud integration with optional color and semantic information.



