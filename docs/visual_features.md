# Visual Features, Matchers and Loop-Closure Descriptors

> **New in pySLAM v2.13.0 (2026-11-22).** This release adds new learned local features (DeDoDe v2, RDD), dense detector-free matchers (RoMa, tiny-RoMa) and DINOv2-based loop-closure descriptors (SALAD, BoQ). It also makes the existing feature pipeline faster and lighter (lazy imports, heap tuning, persistent inference threads), adds opt-in GPU accelerations, a one-command model installer, and a model catalog with checksum-pinned weights. Existing presets, configuration keys, serialized enum values and install scripts are unchanged: everything new is additive.

<!-- TOC -->

- [Visual Features, Matchers and Loop-Closure Descriptors](#visual-features-matchers-and-loop-closure-descriptors)
  - [What's new at a glance](#whats-new-at-a-glance)
  - [Quick start](#quick-start)
  - [New local features](#new-local-features)
    - [DeDoDe v2](#dedode-v2)
    - [RDD](#rdd)
  - [New detector-free matchers: RoMa and tiny-RoMa](#new-detector-free-matchers-roma-and-tiny-roma)
  - [New loop-closure descriptors: SALAD and BoQ](#new-loop-closure-descriptors-salad-and-boq)
  - [Faster and lighter by default](#faster-and-lighter-by-default)
    - [Lazy imports and heap tuning](#lazy-imports-and-heap-tuning)
    - [Persistent inference threads](#persistent-inference-threads)
    - [Vectorized keypoint conversion](#vectorized-keypoint-conversion)
  - [Optional accelerations](#optional-accelerations)
  - [Installing models](#installing-models)
  - [Model catalog and inspection tools](#model-catalog-and-inspection-tools)
  - [Adding a new method](#adding-a-new-method)
  - [Evaluated and not integrated](#evaluated-and-not-integrated)
  - [Fixes included in this release](#fixes-included-in-this-release)
  - [Validation and benchmarking (developer notes)](#validation-and-benchmarking-developer-notes)
  - [Model catalog](#model-catalog)

<!-- /TOC -->

## What's new at a glance

| Addition | Type | How to use | Notes |
| --- | --- | --- | --- |
| DeDoDe v2 | local feature (detector + descriptor) | `FeatureTrackerConfigs.DEDODE` | via kornia; accurate, heavy (~115 ms at 640x480) |
| RDD | local feature (detector + descriptor) | `FeatureTrackerConfigs.RDD` | deformable transformer; most accurate learned preset on the KITTI smoke run; heavy |
| RoMa | dense detector-free matcher | `FeatureTrackerConfigs.ROMA` | sub-pixel accurate (0.03-0.05 px), ~1 s/pair, 4.3 GB GPU |
| tiny-RoMa | dense detector-free matcher | `FeatureTrackerConfigs.TINY_ROMA` | 25-31 ms/pair, 0.4 px |
| SALAD | loop-closure global descriptor | `LoopDetectorConfigs.SALAD` | DINOv2 + optimal transport, 36 ms/frame |
| BoQ | loop-closure global descriptor | `LoopDetectorConfigs.BOQ` | DINOv2 + bag of queries, 38 ms/frame |
| Faster startup, less memory | default behavior | automatic | feature stack import ~4 s -> 0.3 s, peak RSS about halved |
| Persistent inference threads | default behavior | automatic (`PYSLAM_PERSISTENT_THREAD=0` to disable) | KITTI stereo tracking: ALIKED 197 -> 90 ms/frame |
| Optional accelerations | opt-in | environment variables | GPU NN matcher, batched LightGlue, torch.compile/TensorRT |
| Model installer | tool | `./scripts/install_visual_model.sh` | prefetches pinned code + weights, verifies checksums |
| Model catalog | tool | `python -m pyslam.visual_models check` | registry of all integrations, licenses, cached artifacts |

Speeds are measured on an RTX A3000 laptop GPU.

## Quick start

Select a new preset like any existing one, e.g. in `main_slam.py` / `main_vo.py` or in `config.yaml`:

```python
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

feature_tracker_config = FeatureTrackerConfigs.RDD        # or DEDODE, ALIKED, ...
loop_detection_config = LoopDetectorConfigs.SALAD         # or BOQ, MEGALOC, DBOW3, ...
```

Prefetch the code and weights of the new models once (optional: they are also fetched on first use):

```bash
./scripts/install_visual_model.sh            # all self-installing models
./scripts/install_visual_model.sh RDD SALAD  # selected models
```

Detector-free matchers (RoMa, tiny-RoMa, LoFTR, MASt3R) return matched keypoints without per-keypoint descriptors, so they are meant for `main_vo.py`, `main_feature_matching.py`, initialization and verification, not for the descriptor-based map of `main_slam.py` (see the warning in `feature_tracker_configs.py`).

## New local features

Both new features produce 2000 keypoints (the preset's `num_features`) with L2-normalized 256-D float descriptors, so they work with pyslam's BF/FLANN matchers, the descriptor-distance thresholds of the SLAM front end and the VPR loop detectors. Their `max_descriptor_distance` values were calibrated with the `main_feature_matching.py` rule (3 x sigma-MAD of matched descriptor distances) on 12 TUM/KITTI/EuRoC frame pairs.

### DeDoDe v2

`FeatureTrackerConfigs.DEDODE` uses the DeDoDe v2 detector (`L-C4-v2`: better under rotation, less keypoint clustering) and the DeDoDe `B` descriptor through kornia ([feature_dedode.py](../pyslam/local_features/feature_dedode.py)), so no new package is needed. Keypoints are the deterministic top-k of the detector score map. `max_descriptor_distance` = 1.76. Code: kornia (Apache-2.0); method and weights: Parskatt/DeDoDe (MIT).

| Preset (GPU, 60-frame smoke) | KITTI stereo track ms | TUM RGB-D track ms | TUM ATE |
| --- | --- | --- | --- |
| DEDODE | 536 | 159 | 4.8 mm |
| ALIKED | 90 | 45 | 5.3 mm |

DeDoDe is accurate but heavy: extraction alone takes about 115 ms at 640x480 and 185 ms at 1241x376 (fp16 is already used internally). Like ALIKED, it does not initialize within the 60-frame EuRoC smoke run, while ORB2 does.

### RDD

`FeatureTrackerConfigs.RDD` uses RDD (Chen et al., CVPR 2025, `xtcpete/rdd`, Apache-2.0), a deformable-transformer detector and descriptor ([feature_rdd.py](../pyslam/local_features/feature_rdd.py)). The pinned source snapshot is loaded under a private module name (its own package is named `src`), so nothing is pip-installed. Multi-scale deformable attention runs in pure PyTorch unless the optional CUDA op is compiled. The weights (`weights/rdd.pth`, retrained by the authors after their data-leakage fix) come from the authors' Google Drive folder and are verified by SHA-256 before loading. `max_descriptor_distance` = 1.41.

On the calibration pairs RDD returns 338-678 ratio-test matches per KITTI pair (DeDoDe: 211-359) with a mean MAGSAC inlier rate of 0.86. Extraction takes about 290 ms at 640x480 and 410 ms at 1241x376. All RDD extractors on a device share one model and one inference thread (two stereo instances ran out of GPU memory at KITTI resolution). SLAM smoke runs (60 frames, C++ core): TUM 305 ms/frame, ATE 5.4 mm; KITTI stereo 968 ms/frame, ATE 0.195 m with all rows accepted.

## New detector-free matchers: RoMa and tiny-RoMa

`FeatureTrackerConfigs.ROMA` and `FeatureTrackerConfigs.TINY_ROMA` are dense matchers from `Parskatt/RoMa` (MIT), loaded from a pinned source snapshot without installing `romatch` (`loguru` is routed to `logging` if absent; the optional `fused-local-corr` CUDA extension is not used). tiny-RoMa uses pyslam's vendored XFeat network instead of fetching it from torch.hub; RoMa downloads DINOv2 ViT-L/14 (1.2 GB) once.

Both are built on `DetectorFreeMatcher` ([feature_matcher_detector_free.py](../pyslam/local_features/feature_matcher_detector_free.py)), the new shared base for detector-free matchers: aspect-preserving resizing, sub-pixel keypoints in original image coordinates, confidence and bounds filtering, a match cap (`max_num_matches`, default `kNumFeatures`), optional MAGSAC filtering, row filtering for rectified stereo, and the usual `FeatureMatchingResult` with `cv2.KeyPoint` arrays.

| Matcher | Error on a known shift (real TUM frame) | Pair time | Peak GPU |
| --- | --- | --- | --- |
| RoMa | median 0.03-0.05 px, 100% < 1 px | 1.0-1.2 s | 4.3 GB |
| tiny-RoMa | median 0.42-0.44 px, 85-89% < 1 px | 25-31 ms | 0.3-0.4 GB |

Matches are sampled from the dense certainty map with `torch.multinomial` (as upstream), so outputs vary from run to run.

## New loop-closure descriptors: SALAD and BoQ

Two DINOv2-based place-recognition descriptors are available as loop detectors. Like MegaLoc, they need no vocabulary and run in the loop-detecting process. [global_feature_torchhub.py](../pyslam/loop_closing/global_feature_torchhub.py) provides one extractor for torch.hub models, pinned to an upstream commit:

| Model | Preset | Source (pinned) | Descriptor | License | GPU ms/frame |
| --- | --- | --- | --- | --- | --- |
| SALAD | `LoopDetectorConfigs.SALAD` | `serizba/salad@6aede13a`, release v1.0.0 | 8448 | GPL-3.0 | 36 |
| BoQ (DINOv2) | `LoopDetectorConfigs.BOQ` | `amaralibey/Bag-of-Queries@1a4965ea`, release v1.0 | 12288 | MIT | 38 |
| MegaLoc (existing) | `LoopDetectorConfigs.MEGALOC` | `gmberton/MegaLoc` | 8448 | MIT | 106 |

On TUM `long_office_household`, nearby frames score 0.82-0.90 (SALAD) and 0.69-0.84 (BoQ) cosine similarity, against 0.05-0.43 for distant places. Both pass headless ORB2 SLAM smoke runs on TUM and KITTI.

SALAD's own hub entry point imports its PyTorch Lightning training wrapper (and through it wandb, which fails with the pinned protobuf). pyslam therefore builds SALAD's backbone and aggregator from the pinned source, with the two needed files loaded under private module names, and loads the released checkpoint with `strict=True`.

## Faster and lighter by default

These changes keep keypoints, descriptors and matches bit-identical for the deterministic presets (checked on golden outputs for SIFT, XFeat, SuperPoint+LightGlue, ALIKED+LightGlue and LoFTR, on CPU and GPU).

### Lazy imports and heap tuning

`feature_manager.py` no longer imports its 19 learned/vendor wrappers (SuperPoint, DISK, ALIKED, R2D2, the TensorFlow models, ...) when the module is imported: each wrapper is imported the first time a feature manager needs it. `feature_matcher.py` likewise imports kornia, XFeat, LightGlue and the MASt3R preprocessing only in the matcher that uses them. OpenCV features (SIFT, ORB, AKAZE, ...) and ORB2 still come from the locally built OpenCV, unchanged.

Importing the feature stack went from ~4 s (TensorFlow, Keras, torch, kornia and LightGlue always loaded) to ~0.3 s. In the CPU pair benchmark, process startup fell from ~7.3 s to ~1.4 s and peak RSS from ~1.46 GB to ~0.7-0.8 GB. Full 60-frame SLAM smoke runs finished 4-34% faster, with about 0.9 GiB less memory per run.

The eager imports had also been hiding an allocator effect: OpenCV and numpy free tens of MB of image buffers per frame (SIFT at 640x480 releases ~70 MB per call), and glibc's defaults return that memory to the kernel and page-fault it back on the next frame (~17k minor page faults per SIFT call, ~10% of its time). Some eagerly imported modules had raised glibc's adaptive thresholds by accident. `pyslam/config.py` now sets them explicitly once per process ([malloc_tuning.py](../pyslam/utilities/malloc_tuning.py): 128 MB trim threshold, 32 MB mmap threshold; Linux/glibc only). Overrides: `PYSLAM_MALLOC_TUNING=0`, `PYSLAM_MALLOC_TRIM_MB`, `PYSLAM_MALLOC_MMAP_MB`.

### Persistent inference threads

Stereo frames are extracted on threads created per frame (in the Python `Frame` and in the C++ core). PyTorch keeps part of its CUDA state per thread (e.g. cuDNN execution-plan caches), so a network called from a new thread rebuilds it on every call: at KITTI resolution, DeDoDe went from 175 ms to 2.8 s and ALIKED from 20 to 45 ms per image. Torch-based learned detectors now run on one long-lived thread each ([persistent_thread.py](../pyslam/utilities/persistent_thread.py), `PersistentThreadAdaptor` in `feature_manager_adaptors.py`), with identical outputs. KITTI stereo median tracking (C++ core, GPU): ALIKED 197 -> 90 ms, XFeat 116-140 -> 95-97 ms, SuperPoint+LightGlue 119 -> 115 ms; TUM RGB-D is unchanged. Disable with `PYSLAM_PERSISTENT_THREAD=0`.

### Vectorized keypoint conversion

ALIKED keypoints are built with `cv2.KeyPoint.convert` instead of a Python loop: identical attributes, about 3x faster for 2000 keypoints.

## Optional accelerations

These options are **off by default**. They are enabled per run with environment variables (or the corresponding `Parameters` fields in `pyslam/config_parameters.py`). Their results can differ slightly from the default path; the differences are documented here and are accepted when enabling them.

| Option | Enable | What it does | Measured effect (RTX A3000) | Differences |
| --- | --- | --- | --- | --- |
| GPU nearest-neighbour matcher | `PYSLAM_GPU_NN_MATCHER=1` | BF/FLANN matching of float (L2) descriptors on the GPU, with the same one-to-one ratio-test semantics as the default C++ path | 5-10x faster per pair: ALIKED 4.4 -> 0.9 ms, DeDoDe 19.9 -> 2.1 ms | identical matches on all tested pairs; rounding can flip rare near-ties; exact instead of approximate when replacing FLANN |
| Batched LightGlue | `PYSLAM_LIGHTGLUE_BATCHED=1` | matches one keyframe against all candidate keyframes of `compute_frame_matches` in one masked forward | no speedup here: 10-20% slower than per-pair (pyslam's LightGlue has only 2 layers; masked attention cannot use the fused flash kernel) | no adaptive early stopping/pruning; 11 of 12 test pairs identical, one differed by 1 of 592 matches |
| torch.compile | `PYSLAM_TORCH_COMPILE=1` (backend: `PYSLAM_TORCH_COMPILE_BACKEND=inductor` or `tensorrt`) | compiles the SuperPoint, XFeat and ALIKED networks and LightGlue (upstream static-length compile) | no benefit here: extraction within ±5%, LightGlue matching 7.8 -> 11.3 ms (padding); 4-16 s compile at first call | 99.3-99.9% of keypoints identical (order can change); LightGlue pruning disabled |

SLAM smoke runs with each option enabled (60 frames, GPU): GPU NN matcher with ALIKED on KITTI stereo, all rows accepted, ATE 0.257 m (default 0.256 m), median tracking 74 ms (default 90 ms); batched LightGlue with SuperPoint on KITTI (Python core), all rows accepted, ATE 0.232 m; torch.compile with ALIKED on TUM, all rows accepted, ATE 5.4 mm, tracking 41 ms (default 45 ms). Evidence: `docs/benchmarks/visual_models/sequence-smoke-optional-accelerations-gpu.json`.

Notes:

- torch.compile's inductor backend starts parallel compile workers; set `TORCHINDUCTOR_COMPILE_THREADS=1` to limit their memory (one smoke run exceeded 13 GB aggregate RSS without it).
- The GPU matcher applies wherever the Python `FeatureMatcher` matches float descriptors: initialization, relocalization, loop closing, `main_feature_matching.py` and the Python core. The C++ core's internal projection/triangulation matching is not affected.
- Batched LightGlue applies to `compute_frame_matches` (Python-core local mapping, relocalization, direct loop closing). The C++ core's local mapping matches pairs natively.
- The TensorRT backend requires `torch_tensorrt`, which is not installed by pyslam; without it the option prints a warning and runs uncompiled. Any compilation failure also falls back to eager mode for the rest of the run.

## Installing models

Existing integrations keep their installers (`install_all*.sh`, `install_thirdparty.sh`, git submodules, pip pins), which are unchanged and protected by `tests/local_features/test_visual_model_contract.py`. Models integrated through pinned sources (DeDoDe, RDD, RoMa, tiny-RoMa, SALAD, BoQ, MegaLoc, LoFTR weights) are installed by the registry:

```bash
./scripts/install_visual_model.sh --list          # self-installing models and their groups
./scripts/install_visual_model.sh                 # all of them
./scripts/install_visual_model.sh ROMA SALAD      # selected models
./scripts/install_visual_model.sh --group vpr     # a group (members installed by legacy scripts are reported)
./scripts/install_visual_model.sh --dry-run       # plan and cache status only
python -m pyslam.visual_models check --artifacts  # per-artifact cache status (no download)
```

Installing a model constructs it once on the CPU with downloads allowed. This fetches the pinned source snapshot and all checkpoints (including indirect ones such as DINOv2) into the standard caches (`$TORCH_HOME/hub`, the Hugging Face cache). Each registered SHA-256 is then verified, and the command fails if one is missing or mismatched. No package is installed and no pin changes. Afterwards the models also run offline (`PYSLAM_OFFLINE=1`).

## Model catalog and inspection tools

`pyslam.visual_models` is a registry of every integration (local features, pair matchers, global descriptors) with its wrapper, dependency group, capabilities (e.g. per-keypoint descriptors), source revision, license notes and checksum-pinned weights. Importing it loads no inference framework.

```bash
python -m pyslam.visual_models check                       # all models
python -m pyslam.visual_models check SUPERPOINT
python -m pyslam.visual_models check LOFTR --kind pair_matcher
python -m pyslam.visual_models check --group vpr --device cuda
python -m pyslam.visual_models table                       # regenerates the table at the end of this page
python -m pyslam.visual_models prefetch LOFTR_OUTDOOR      # fetch one registered artifact into the resolver cache
```

`check` inspects metadata and files; it does not run a model or download weights. `pyslam.visual_models.weights.WeightResolver` resolves checksum-pinned artifacts (Hugging Face files pinned to a commit, torch.hub checkpoints, HTTPS mirrors) with an offline mode (`PYSLAM_OFFLINE=1`, `HF_HUB_OFFLINE=1`), a cache override (`PYSLAM_MODELS_DIR`), locking and atomic publication. Existing wrappers keep their original weight-loading code.

## Adding a new method

1. Review the upstream code and weight licenses separately. Record an immutable source revision and checksum-pinned weight sources.
2. Implement the wrapper on the existing interfaces: `BaseFeature2D` (local features), `FeatureMatcher` or `DetectorFreeMatcher` (matchers), `LoopDetectorVprBase` / `GlobalFeatureTorchHub` (global descriptors). Keep aspect ratio, original pixel coordinates, descriptor dtype/norm and feature limits; run torch inference through `PersistentThread` when it may be called from per-frame threads.
3. Append enum values (never renumber), add the preset, the registry spec, the `FeatureInfo` entries (calibrate `max_descriptor_distance`) and an `INSTALLERS` entry in `pyslam/visual_models/install.py`.
4. Add output-contract and equivalence tests, and record pair timings and SLAM smoke runs.
5. Regenerate the table below and update the README, TeX document and install docs.

## Evaluated and not integrated

- **EfficientLoFTR via transformers 5.24**: the port returns coarse-grid matches only (its first fine-matching stage gathers along the wrong tensor dimension), and pairs are misordered under general shifts. kornia LoFTR is accurate on the same test.
- **LiftFeat** (`lyp-deeplearning/LiftFeat`): the repository has no license, so its code and weights cannot be used by pyslam.

## Fixes included in this release

- `LoopDetectorConfigs.MEGALOC` raised a `TypeError` when selected (the factory built the feature extractor instead of `LoopDetectorMegaloc`); the detector imported its extractor through a non-existent top-level module; `GlobalFeatureMegaloc.dim` was 2048 instead of 8448, which broke `compute_features()`.
- CosPlace/EigenPlaces stored string device arguments unchanged (VPR patch fix).
- Stereo initialization (EuRoC): both stereo extractors receive the startup feature boost; stereo point support relaxes progressively after repeated failures (floor: twice the depth-PnP minimum); stereo/RGB-D BA acceptance compares the same reprojection metric before and after BA (monocular keeps its original comparison); reset restores the original checks. The 60-frame EuRoC runs now initialize in all four C++/Python x DBoW3/SAD combinations.
- The `FeatureMatcher.match()` if/elif chain is now one `_match_impl()` override per matcher (XFeat, LightGlue, LoFTR, MASt3R; BF/FLANN in the base class), with identical outputs.

## Validation and benchmarking (developer notes)

The tools live in [`benchmarking/`](../benchmarking/README.md):

```bash
python benchmarking/inventory_visual_models.py --output /tmp/visual-inventory.json
python benchmarking/benchmark_visual_models.py --presets ORB2 SIFT XFEAT --sizes kitti tum euroc \
  --cores cpp python --device cpu --iterations 20 --warmup 3 --threads 1 \
  --golden-dir /tmp/visual-golden-before --output /tmp/visual-before.json
# repeat after a change with identical options on the same machine, then:
python benchmarking/compare_benchmarks.py /tmp/visual-before.json /tmp/visual-after.json
python benchmarking/compare_visual_model_outputs.py /tmp/visual-golden-before/ORB2-tum-cpp-cpu.npz /tmp/visual-golden-after/ORB2-tum-cpp-cpu.npz
python benchmarking/benchmark_visual_slam.py --presets ORB2 --loop-presets DBOW3 SAD --frames 60 --output /tmp/visual-slam-smoke.json
```

- `benchmark_visual_models.py` runs pair micro-benchmarks per preset, resolution, core and device in isolated offline worker processes. It records median/p95 times, matches, synthetic inliers, peak memory, import/model-load times, golden outputs (`--golden-dir`), whether outputs are deterministic across repeated calls, and optional cProfile files.
- `compare_benchmarks.py` is the regression gate (exit 0 pass, 1 regression, 2 incomplete). Tolerances: per-frame times and fps 3% (`--tolerance`), import/model-load 10% (`--startup-tolerance`), peak memory 2% (`--memory-tolerance`). Counts must be exact for deterministic cases and within 3% (`--count-tolerance`) for non-deterministic ones (e.g. ORB2's default extractor, RoMa sampling).
- `benchmark_visual_slam.py` runs bounded headless `main_slam.py` cases on the datasets in `config.yaml` and records tracking timings, ATE, accepted trajectory rows, memory and artifacts. `--profile-dir` profiles every process.
- The enum/`FeatureInfo` contract is append-only: recorded names and values must not change and new members must use larger values (saved maps and the C++ core depend on them).
- Evidence files are in [`docs/benchmarks/visual_models/`](./benchmarks/visual_models/), with the [fixes ledger](./benchmarks/visual_models/fixes-ledger.md).

## Model catalog

<!-- BEGIN GENERATED VISUAL MODEL TABLE -->
| Model | Kind | Per-keypoint descriptors | Group | License |
| --- | --- | --- | --- | --- |
| AGAST | local_feature | no | core | existing integration (not re-reviewed) |
| AKAZE | local_feature | yes | core | existing integration (not re-reviewed) |
| ALIKED | local_feature | yes | core | existing integration (not re-reviewed) |
| BEBLID | local_feature | yes | core | existing integration (not re-reviewed) |
| BOOST_DESC | local_feature | yes | core | existing integration (not re-reviewed) |
| BRISK | local_feature | yes | core | existing integration (not re-reviewed) |
| CONTEXTDESC | local_feature | yes | tf_models | existing integration (not re-reviewed) |
| D2NET | local_feature | yes | core | existing integration (not re-reviewed) |
| DAISY | local_feature | yes | core | existing integration (not re-reviewed) |
| DEDODE | local_feature | yes | core | code Apache-2.0 (kornia); method and weights MIT (Parskatt/DeDoDe) |
| DELF | local_feature | yes | tf_models | existing integration (not re-reviewed) |
| DISK | local_feature | yes | core | existing integration (not re-reviewed) |
| FAST | local_feature | no | core | existing integration (not re-reviewed) |
| FREAK | local_feature | yes | core | existing integration (not re-reviewed) |
| GEODESC | local_feature | yes | tf_models | existing integration (not re-reviewed) |
| GFTT | local_feature | no | core | existing integration (not re-reviewed) |
| HARDNET | local_feature | yes | core | existing integration (not re-reviewed) |
| HL | local_feature | no | core | existing integration (not re-reviewed) |
| KAZE | local_feature | yes | core | existing integration (not re-reviewed) |
| KEYNET | local_feature | yes | tf_models | existing integration (not re-reviewed) |
| KEYNETAFFNETHARDNET | local_feature | yes | core | existing integration (not re-reviewed) |
| L2NET | local_feature | yes | core | existing integration (not re-reviewed) |
| LATCH | local_feature | yes | core | existing integration (not re-reviewed) |
| LFNET | local_feature | yes | tf_models | existing integration (not re-reviewed) |
| LIGHTGLUESIFT | local_feature | yes | core | existing integration (not re-reviewed) |
| LOGPOLAR | local_feature | yes | core | existing integration (not re-reviewed) |
| LUCID | local_feature | yes | core | existing integration (not re-reviewed) |
| MSD | local_feature | no | core | existing integration (not re-reviewed) |
| MSER | local_feature | no | core | existing integration (not re-reviewed) |
| ORB | local_feature | yes | core | existing integration (not re-reviewed) |
| ORB2 | local_feature | yes | core | existing integration (not re-reviewed) |
| R2D2 | local_feature | yes | core | existing integration (not re-reviewed) |
| RDD | local_feature | yes | core | Apache-2.0 (code); weights distributed by the authors with the repository |
| ROOT_SIFT | local_feature | yes | core | existing integration (not re-reviewed) |
| SHI_TOMASI | local_feature | no | core | existing integration (not re-reviewed) |
| SIFT | local_feature | yes | core | existing integration (not re-reviewed) |
| SOSNET | local_feature | yes | core | existing integration (not re-reviewed) |
| STAR | local_feature | no | core | existing integration (not re-reviewed) |
| SUPERPOINT | local_feature | yes | core | existing integration (not re-reviewed) |
| SURF | local_feature | yes | core | existing integration (not re-reviewed) |
| TFEAT | local_feature | yes | core | existing integration (not re-reviewed) |
| VGG | local_feature | yes | core | existing integration (not re-reviewed) |
| XFEAT | local_feature | yes | core | existing integration (not re-reviewed) |
| BF | pair_matcher | no | core | existing integration (not re-reviewed) |
| FLANN | pair_matcher | no | core | existing integration (not re-reviewed) |
| XFEAT | pair_matcher | no | core | existing integration (not re-reviewed) |
| LIGHTGLUE | pair_matcher | no | core | existing integration (not re-reviewed) |
| LOFTR | pair_matcher | no | dense_matchers | existing integration (not re-reviewed) |
| MAST3R | pair_matcher | no | dense_matchers | existing integration (not re-reviewed) |
| ROMA | pair_matcher | no | dense_matchers | code and weights MIT; DINOv2 Apache-2.0; tiny-RoMa uses pyslam's vendored XFeat (Apache-2.0) |
| TINY_ROMA | pair_matcher | no | dense_matchers | code and weights MIT; DINOv2 Apache-2.0; tiny-RoMa uses pyslam's vendored XFeat (Apache-2.0) |
| MEGALOC | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| COSPLACE | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| EIGENPLACES | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| NETVLAD | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| ALEXNET | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| SAD | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| HDC-DELF | global_descriptor | no | vpr | existing integration (not re-reviewed) |
| SALAD | global_descriptor | no | vpr | code and weights GPL-3.0 (same as pyslam); DINOv2 backbone Apache-2.0 |
| BOQ | global_descriptor | no | vpr | MIT; DINOv2 backbone Apache-2.0 |
<!-- END GENERATED VISUAL MODEL TABLE -->

Generated from the registry with `python -m pyslam.visual_models table`.
