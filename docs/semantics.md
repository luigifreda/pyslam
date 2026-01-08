# Semantic Mapping and Segmentation 

<!-- TOC -->

- [Semantic Mapping and Segmentation](#semantic-mapping-and-segmentation)
    - [1. Sparse Semantic Mapping and Segmentation](#1-sparse-semantic-mapping-and-segmentation)
        - [1.1. Quick test](#11-quick-test)
    - [2. Short description](#2-short-description)
        - [2.1. Semantic features types](#21-semantic-features-types)
        - [2.2. Dense vs Object-Based](#22-dense-vs-object-based)
        - [2.3. Supported Models](#23-supported-models)
        - [2.4. Dataset Type Support](#24-dataset-type-support)
        - [2.5. Feature Fusion](#25-feature-fusion)
        - [2.6. Visualizations](#26-visualizations)
    - [3. Volumetric Semantic mapping](#3-volumetric-semantic-mapping)
    - [4. TODOs](#4-todos)

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
- **Object-based** [*WIP*]:
  - Generate, track, and maintain 3D segments as groups of points.
  - Features are assigned at object-level: multiple KPs or MPs share "object descriptors".
  - Possible approaches: project 2D masks and use label fusion and DBSCAN for 3D clustering (WIP).

### Supported Models

- Segmentation models are implemented on the top of a base class `SemanticSegmentationBase` with an `infer` method returning semantic images.
- A `to_rgb` method converts semantic outputs into color maps.

**Panoptic/Instance segmentation:**
  - `DETIC`: from https://github.com/facebookresearch/Detic
    - Object detection-based (CenterNet2 + CLIP), supports large vocabularies (LVIS/COCO/OpenImages/Objects365).
    - Architecture: Object detector (CenterNet2) detects individual objects first, then segments each detection.
    - Can output both *"instances"* (direct instance segmentation) and *"panoptic_seg"* formats.
    - Instance extraction: Direct from object detections - each detected object = one instance ID.
    - Result: Robust instance segmentation - each detected object gets a unique instance ID, even for multiple objects of the same category (e.g., two pillows = two separate instances).
  - `ODISE`: from https://github.com/NVlabs/ODISE
    - Diffusion-based panoptic segmentation, leverages diffusion models for segmentation.
    - Architecture: Panoptic segmentation model that segments image into regions first, then classifies regions.
    - Only outputs *"panoptic_seg"* format - instances extracted from panoptic segments via *"isthing"* flag.
    - Instance extraction: Derived from panoptic segments - one segment may contain multiple objects if model groups them together (e.g., spatially connected objects of same category).
    - Result: Instance segmentation may merge multiple objects of the same category into a single instance (e.g., two pillows may be detected as one "pillow" instance).
  - `EOV_SEG`: from https://github.com/nhw649/EOV-Seg
    - Dual-backbone (CNN + ViT) with CLIP, text-prompt driven open vocabulary.
    - Architecture: Panoptic segmentation model (similar to ODISE) - segments image into regions first.
    - Only outputs *"panoptic_seg"* format - instances extracted from panoptic segments via *"isthing"* flag.
    - Instance extraction: Same as `ODISE` - derived from panoptic segments, may group multiple objects.
    - Result: Similar to `ODISE`, instance segmentation may group multiple objects of the same category together (e.g., two pillows may be detected as one "pillow" instance).

**Semantic segmentation only:**
  - `DEEPLABV3`: from `torchvision`, pre-trained on COCO/VOC.
    - Semantic segmentation model from torchvision DeepLab's v3.
  - `SEGFORMER`: from `transformers`, pre-trained on Cityscapes or ADE20k.
    - Semantic segmentation model from transformer's Segformer.
  - `CLIP`: from `f3rm` package for open-vocabulary support.
    - Uses CLIP patch embeddings + text similarity to produce labels/probabilities (it is not a dedicated "segmentation head"). 


### Dataset Type Support

- Trained for a “dataset type” managing label-color mappings and equivalence mappings.
- Special dataset types for open-vocabulary:
  - **closed_set**: user-defined labels and generic color map
  - **feature_sim**: similarity map based on a query word

### Feature Fusion

- Features are fused from KPs into MPs using one of the fusion methods available [here](../pyslam/semantics/semantic_fusion_methods.py).

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
- [ ] Implement object-level semantic mapping [WIP]
- [ ] Add interaction in 3D viewer to change query word (open-vocab)
