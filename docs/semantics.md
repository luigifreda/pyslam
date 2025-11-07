# Semantic Mapping and Segmentation 

<!-- TOC -->

- [Semantic Mapping and Segmentation](#semantic-mapping-and-segmentation)
  - [Quick test](#quick-test)
  - [Sparse Semantic Mapping and Segmentation](#sparse-semantic-mapping-and-segmentation)
  - [Implemented features](#implemented-features)
    - [Semantic features types](#semantic-features-types)
    - [Dense vs Object-Based](#dense-vs-object-based)
    - [Supported Models](#supported-models)
    - [Dataset Type Support](#dataset-type-support)
    - [Feature Fusion](#feature-fusion)
    - [Visualizations](#visualizations)
    - [Dataset Support](#dataset-support)
  - [Semantic volumetric mapping](#semantic-volumetric-mapping)
  - [TODOs](#todos)

<!-- /TOC -->


The `semantics` module in pySLAM enables **semantic mapping** and **image segmentation** capabilities. It is designed to support rapid prototyping, benchmarking, and evaluation of semantic mapping methods within the SLAM pipeline.

## Quick test

- Enable sparse semantic mapping and segmentation by setting: ``
- Run the `main_slam.py` example to run Segformer (trained on Cityscapes) on the default KITTI video.

For testing the **open-vocabulary feature**: change the semantic mapping config in the `semantic_mapping_configs.py` file to:  
Then change your query word in `semantic_segmentation_clip.py` to your desire.


## Sparse Semantic Mapping and Segmentation

## Implemented features

- A semantic mapping module that assigns semantic features to keypoints on keyframes and uses them to assign semantic features to map points.
- The semantic mapping module acts **after** the local mapping module has refined a keyframe and **before** the loop closure and volumetric integration modules. But, since it can be run on a different thread, the latter is not ensured.
- The interface for the semantic mapping module is:  
  * **IN**: KFs, 
  * **OUT**: semantic features for KPs and MPs.

### Semantic features types

- **Labels**: categorical labels  
- **Probability vectors**: with confidence or probability for each of the classes.  
- **Feature vectors**: with latent features (used for open-vocabulary semantic mapping)

### Dense vs Object-Based

- The semantic mapping module can potentially be “dense” or “object-based”. Both will maintain the same interface.
- **Dense** version:
  - Uses per-pixel semantic segmentation.
- **Object-based** [*not implemented yet*]:
  - Generate, track, and maintain 3D segments as groups of points.
  - Features are assigned at object-level: multiple KPs or MPs share descriptors.
  - Approaches: project 2D masks or use DBSCAN for 3D clustering.

### Supported Models

- Segmentation models have a base class with an `infer` interface returning semantic images.
- A `to_rgb` method converts semantic outputs into color maps.
- Implemented models:
  - **DeepLabv3** (torchvision, pre-trained on COCO/VOC)
  - **Segformer** (transformers, pre-trained on Cityscapes or ADE20k)
  - **Dense CLIP** (from f3rm repo for open-vocabulary)

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


## Semantic volumetric mapping

Semantic volumetric mapping can be performed by setting: 
```python
kDoSparseSemanticMappingAndSegmentation=True #  enable sparse mapping and segmentation
kDoVolumetricIntegration = True # enable volumetric integration
kVolumetricIntegrationType = "VOXEL_SEMANTIC_GRID_PROBABILISTIC" # use semantic volumetric models like VOXEL_SEMANTIC_GRID_PROBABILISTIC and VOXEL_SEMANTIC_GRID
```


## TODOs

- Investigate variants in KF count for LABEL vs PROBABILITY_VECTOR
- Implement object-based semantic mapping
- Add interaction in 3D viewer to change query word (open-vocab)
- Refactor `sem_des_to_rgb` vs `sem_img_to_rgb` (may be redundant)
