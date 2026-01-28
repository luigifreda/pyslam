# Semantic Mapping and Segmentation 

The `semantics` module in pySLAM provides **sparse semantic mapping** and **image segmentation** within the SLAM pipeline. It is intended for rapid prototyping, benchmarking, and evaluation of semantic mapping and segmentation methods. When paired with the `volumetric` mapping module, it also enables **semantic volumetric mapping**.

<!-- TOC -->

- [Semantic Mapping and Segmentation](#semantic-mapping-and-segmentation)
  - [Sparse Semantic Mapping and Image Segmentation](#sparse-semantic-mapping-and-image-segmentation)
    - [Quick test](#quick-test)
    - [Short description](#short-description)
    - [Semantic feature types](#semantic-feature-types)
    - [Dense vs Object-Based](#dense-vs-object-based)
    - [Supported Models](#supported-models)
    - [Dataset Type Support](#dataset-type-support)
    - [Sparse Feature Fusion Methods](#sparse-feature-fusion-methods)
    - [Visualizations](#visualizations)
  - [Volumetric Semantic mapping](#volumetric-semantic-mapping)
  - [TODOs](#todos)

<!-- /TOC -->

---
## Sparse Semantic Mapping and Image Segmentation

### Quick test

1. Enable sparse semantic mapping and segmentation by setting:
    ```python
    kDoSparseSemanticMappingAndSegmentation=True #  enable sparse mapping and image segmentation
    ```
2. Run the `main_slam.py` example to run Segformer (trained on Cityscapes) on the default KITTI video.

For testing the **open-vocabulary feature**: change the semantic segmentation model to:
```python
kSemanticSegmentationType="CLIP" # in config_parameters.py
```
Then change your query word in `semantic_segmentation_clip.py` to your desire.


### Short description

The _sparse semantic mapping_ module:
- infers semantic information from each keyframe image,
- assigns semantic features to keyframe keypoints,
- and propagates those features to the corresponding sparse map points.

The semantic mapping module runs **after** local mapping refines a keyframe and **before** loop closure and volumetric integration. When `volumetric` mapping is enabled together with the `semantics` module, volumetric integration waits until a keyframe’s semantic prediction is available before processing it.

### Semantic feature types

- **Labels**: categorical class IDs; stored on keypoints/map points and used for fusion, visualization, and semantic weighting.
- **Probability vectors**: per-class confidence/probability scores; fused across observations to estimate robust semantics.
- **Feature vectors**: latent embeddings used for open-vocabulary semantic mapping and similarity-based inference.

### Dense vs Object-Based

The semantic mapping module can potentially be _“dense”_ or _“object-based”_; both are intended to expose the same interface.

The currently implemented version is **Dense**: it runs per-pixel semantic segmentation on keyframe images, assigns semantic descriptors to keypoints, and fuses them into the corresponding sparse map points.

An **Object-based** variant is currently implemented only in combination with the `volumetric` module, where object segments are grouped 3D points with shared semantic class and object IDs (which are still extracted from keyframe images and then backprojected and integrated into voxels).

<!--
- **Object-based** [*WIP*]:
  - Generate, track, and maintain 3D segments as groups of points.
  - Features are assigned at object-level: multiple KPs or MPs share "object descriptors".
  - Possible approaches: project 2D masks and use label fusion and DBSCAN for 3D clustering (WIP). -->

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


**Instance segmentation:**
  - `RFDETR`: from https://github.com/roboflow/rf-detr.git
    - RF-DETR instance segmentation; pretrained weights target COCO classes by default.
  - `YOLO`: from https://github.com/ultralytics/ultralytics

### Dataset Type Support

- Trained for a “dataset type” managing label-color mappings and equivalence mappings.
- Special dataset types for open-vocabulary:
  - **closed_set**: user-defined labels and generic color map
  - **feature_sim**: similarity map based on a query word

### Sparse Feature Fusion Methods

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

Semantic volumetric mapping is enabled by setting: 
```python
kDoSparseSemanticMappingAndSegmentation=True #  enable sparse mapping and segmentation
kDoVolumetricIntegration = True # enable volumetric integration
kVolumetricIntegrationType = "VOXEL_SEMANTIC_PROBABILISTIC_GRID" # use semantic volumetric models like
                                                                 # VOXEL_SEMANTIC_PROBABILISTIC_GRID and VOXEL_SEMANTIC_GRID
```

 When `volumetric` mapping and `semantic` module are both enabled, the volumetric mapping waits the semantic prediction of a keyframe is available before processing it.

Further information about the volumetric integration models and SW architecture are available [here](../cpp/volumetric/README.md).

## TODOs

- [ ] Investigate variants in KF count for `LABEL` vs `PROBABILITY_VECTOR`
- [ ] Implement object-level semantic mapping [WIP]
- [ ] Add interaction in 3D viewer to change query word (open-vocab)
