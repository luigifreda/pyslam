# Semantic pySLAM

Original document from [David](https://github.com/dvdmc) is available [here](https://docs.google.com/document/d/1MpLvLVx35Sr9fh6W-YcUlRjWY3dvww4ROGqWeOI_J_8/edit?tab=t.0#heading=h.65ibkflchz2i).


## Aim

Include a module to perform semantic mapping within pySLAM. This module will serve in prototyping, benchmarking and evaluation of semantic mapping methods.

## Quick test

Just change to the `semantics` branch. If pySLAM is already installed, just install the `transformers==4.51.0` and `f3rm` dependencies in the venv. Run the `main_slam.py` example to run Segformer trained on Cityscapes.

For testing the open-vocabulary feature: change the semantic mapping config in the `semantic_mapping_configs.py` file to:  
Then change your query word in `semantic_segmentation_clip.py` to your desire.

## Implemented features

- A semantic mapping module that assigns semantic features to keypoints on keyframes and uses them to assign semantic features to map points.
- The semantic mapping module acts **after** the local mapping module has refined a keyframe and **before** the loop closure and volumetric integration modules. But, since it can be run on a different thread, the latter is not ensured.
- The interface for the semantic mapping module is:  
  **IN**: KFs, **OUT**: semantic features for KPs and MPs.

### Semantic features types

- **Labels**: categorical labels  
- **Probability vectors**: with confidence or probability for each of the classes.  
- **Feature vectors**: with latent features (used for open-vocabulary semantic mapping)

### Dense vs Object-Based

- The semantic mapping module can potentially be “dense” or “object-based”. Both will maintain the same interface.
- **Dense** version:
  - Uses per-pixel semantic segmentation.
- **Object-based** (not implemented yet):
  - Generate, track, and maintain 3D segments as groups of points.
  - Features are assigned at object-level: multiple KPs or MPs share descriptors.
  - Approaches: project 2D masks or use DBSCAN for 3D clustering.

### Model Support

- Segmentation models have a base class with an `infer` interface returning semantic images.
- A `to_rgb` method converts semantic outputs into color maps.
- Implemented models:
  - DeepLabv3 (torchvision, pre-trained on COCO/VOC)
  - Segformer (transformers, pre-trained on Cityscapes or ADE20k)
  - Dense CLIP (from f3rm repo for open-vocabulary)

### Dataset Type Support

- Trained for a “dataset type” managing label-color mappings and equivalence mappings.
- Special dataset types for open-vocabulary:
  - **closed_set**: user-defined labels and generic color map
  - **feature_sim**: similarity map based on a query word

### Feature Fusion

- Features are fused from KPs into MPs using a fusion method.

### Visualizations

- Visualizations of:
  - Label color maps
  - Similarity heatmaps (open vocab)
- Available in both 2D and 3D viewers.

### Dataset Support

- **Scannet** is supported with GT pose and GT semantics.
- Evaluation done with Segformer on ADE20k + class mapping.

## Main code changes

- Added `semantics` and `Scannet` to README.
- Added `SCANNET_DATSET` to `config.yaml`.
- Added semantic parameters to `config_parameters.yaml`.
- Semantic module connected to volumetric integrator (**not used** currently).
- Added configs (feature tracker, loop detection, semantic mapping) to `Config` class for future `dump_config`.
- Integrated Scannet in:
  - `dataset.py`
  - `dataset_factory.py`
  - `dataset_types.py`
  - `ground_truth.py`
- Added `scannetv2-labels.combined.tsv` for label mapping to NYU40.
- Updated `main_slam.py`:
  - Instantiate `SemanticMappingConfig` based on SLAM dataset
  - Add semantic evaluation
- Created `semantics/` folder

### Data structures

- `frame.py`: Added `kps_sem`, `semantic_img`, `set_semantics()`
- `keyframe.py`: Added `kps_sem`
- `local_mapping.py`: Linked local mapping with semantic mapping
- `map_point.py`: Added `semantic_des`, `update_semantics()`
- `slam.py`: Instantiate `semantic_mapping`, manage lifecycle (`stop`, `reset`, etc.)
- `viewer3D.py`: Add semantic visualizations

## TODOs

- Add documentation (depends on integration strategy)
- Extensive tests with full pySLAM functionality (e.g. save/load maps)
- Investigate variation in KF count for LABEL vs PROBABILITY_VECTOR
- Handle unlabeled MPs (possibly added post-semantic mapping)
- Implement object-based semantic mapping
- Integrate semantics in volumetric mapping (Open3D-based — postponed)
- Add interaction in 3D viewer to change query word (open-vocab)
- Refactor `sem_des_to_rgb` vs `sem_img_to_rgb` (may be redundant)

## TODiscuss

- Should label images be `(H,W)` or `(H,W,1)`?
- How should configuration and factories be structured?
- How to structure `semantic_labels` and `semantic_utils`?
