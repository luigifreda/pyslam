# Semantic Mapping and Segmentation 

<!-- TOC -->

- [Semantic Mapping and Segmentation](#semantic-mapping-and-segmentation)
  - [Sparse Semantic Mapping and Segmentation](#sparse-semantic-mapping-and-segmentation)
    - [Quick test](#quick-test)
  - [Short description](#short-description)
    - [Semantic features types](#semantic-features-types)
    - [Dense vs Object-Based](#dense-vs-object-based)
    - [Supported Models](#supported-models)
    - [Dataset Type Support](#dataset-type-support)
    - [Feature Fusion](#feature-fusion)
    - [Visualizations](#visualizations)
  - [Volumetric Semantic mapping](#volumetric-semantic-mapping)
  - [TODOs](#todos)

<!-- /TOC -->


The `semantics` module in pySLAM enables **semantic mapping** and **image segmentation** capabilities. It is designed to support rapid prototyping, benchmarking, and evaluation of semantic mapping methods within the SLAM pipeline. If combined with the volumetric mapping module, the `semantics` module allows to get semantic volumetric mapping.


---
## Sparse Semantic Mapping and Segmentation

### Quick test

1. Enable sparse semantic mapping and segmentation by setting:
    ```python
    kDoSparseSemanticMappingAndSegmentation=True #  enable sparse mapping and segmentation
    ```
2. Run the `main_slam.py` example to run Segformer (trained on Cityscapes) on the default KITTI video.

For testing the **open-vocabulary feature**: change the semantic segmentation model to:
```python
kSemanticSegmentationType="CLIP" # in config_parameters.py
```
Then change your query word in `semantic_segmentation_clip.py` to your desire.


## Short description

The _sparse semantic mapping module_ assigns semantic features to keypoints on keyframes and uses them to assign semantic features to map points.

The semantic mapping module acts **after** the local mapping module has refined a keyframe and **before** the loop closure and volumetric integration modules. If semantic volumetric mapping is requested, the volumetric mapping waits the semantic prediction of a keyframe is available before processing it. 

In a few words, the semantic mapping module takes as input the extracted keyframes and returns as output the semantic features for KPs and MPs.

### Semantic features types

- **Labels**: categorical labels  
- **Probability vectors**: with confidence or probability for each of the classes.  
- **Feature vectors**: with latent features (used for open-vocabulary semantic mapping)

### Dense vs Object-Based

The semantic mapping module can potentially be “dense” or “object-based”. Both will maintain the same interface.
- **Dense** version:
  - Uses per-pixel semantic segmentation.
- **Object-based** [*NOT IMPLEMENTED YET*]:
  - Generate, track, and maintain 3D segments as groups of points.
  - Features are assigned at object-level: multiple KPs or MPs share descriptors.
  - Possible approaches: project 2D masks or use DBSCAN for 3D clustering.

### Supported Models

- Segmentation models are implemented on the top of a base class with an `infer` method returning semantic images.
- A `to_rgb` method converts semantic outputs into color maps.
- Implemented models:
  - **DeepLabv3** (torchvision, pre-trained on COCO/VOC)
  - **Segformer** (transformers, pre-trained on Cityscapes or ADE20k)
  - **Dense CLIP** (from f3rm repo for open-vocabulary)
  - **DETIC** (from https://github.com/facebookresearch/Detic)
  - **EOV-SEG** (from https://github.com/nhw649/EOV-Seg)
  - **ODISE** (from https://github.com/NVlabs/ODISE)

### Dataset Type Support

- Trained for a “dataset type” managing label-color mappings and equivalence mappings.
- Special dataset types for open-vocabulary:
  - **closed_set**: user-defined labels and generic color map
  - **feature_sim**: similarity map based on a query word

### Feature Fusion

- Features are fused from KPs into MPs using one of the available fusion methods.

### Visualizations

It is possible to visualize both in 2D and 3D the:
- Label color maps
- Similarity heatmaps (open vocab)

<!-- ### Dataset Support

- **Scannet** is supported with GT pose and GT semantics.
- Evaluation done with Segformer on ADE20k + class mapping. -->

---

## Volumetric Semantic mapping

Semantic volumetric mapping can be performed by setting: 
```python
kDoSparseSemanticMappingAndSegmentation=True #  enable sparse mapping and segmentation
kDoVolumetricIntegration = True # enable volumetric integration
kVolumetricIntegrationType = "VOXEL_SEMANTIC_PROBABILISTIC_GRID" # use semantic volumetric models like VOXEL_SEMANTIC_PROBABILISTIC_GRID and VOXEL_SEMANTIC_GRID
```

Further information about the volumetric integration models and SW architecture are available [here](../cpp/volumetric/README.md).

## TODOs

- [ ] Investigate variants in KF count for LABEL vs PROBABILITY_VECTOR
- [ ] Implement object-level semantic mapping
- [ ] Add interaction in 3D viewer to change query word (open-vocab)
