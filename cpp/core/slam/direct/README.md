# Direct SLAM (KLT, hybrid, and semi-direct tracking)

See the [direct-SLAM guide](../../../docs/direct_slam.md) for the runner, architecture diagrams, complete parameter reference, and comparison with `main_slam.py`.

A KLT/hybrid direct-SLAM frontend and local bundle-adjustment backend compiled into `cpp_core`. It reuses the pyslam camera, settings, and datasets. This C++ map has no loop-closure or global pose-graph optimizer of its own. `main_slam_direct.py` reuses the same `Viewer3D` and `CvImageViewer` as `main_slam.py`. Unless `--no-loop-closing` is set or `Parameters.kUseLoopClosing` is false, the runner starts the standard `LoopClosing` thread on separately extracted descriptor keyframes and writes the corrected poses back into the direct map. See [`direct_loop_closing.py`](../../../pyslam/slam/direct_loop_closing.py).

Three modes, selected by `DIRECT_SLAM.mode`:

- `klt`: grid FAST or GFTT corners, pyramidal Lucas-Kanade, then a geometric reprojection pose.
- `hybrid`: sparse photometric alignment on patches with known depth (the optic-flow direct method), seeded and checked by geometric tracking. If corner flow fails, a photometric-only step requires support from at least 60% of the reference landmarks. The photometric residual is a 2D window around the projected center.
- `semi_direct`: a fronto-parallel 3D patch is aligned against the reference keyframe, each feature is then aligned in the current image, and the existing reprojection pose consumes those pixels. Unknown depth is an inverse-depth seed updated by epipolar search with the SVO depth filter (Gaussian × Beta, one-pixel measurement τ). `direct` remains a deprecated alias of `hybrid`.

"Direct" in the component name means that tracking and local-BA correspondences come from intensity on raw pixels instead of descriptor matching. This is what separates it from the sparse SLAM in `cpp/core/slam/`: both maps store world-frame landmarks observed by keyframes, but there a lost point can be matched again by its descriptor, while here a point's identity is the continuity of its track. `klt` minimizes reprojection error, so in the usual taxonomy it is an indirect method. `hybrid` adds a photometric step after that geometric solve. `semi_direct` is the SVO-style mode. See the [direct-SLAM guide](../../../docs/direct_slam.md#direct-slam-klt-hybrid-and-semi-direct-tracking) for details.

`klt` and `hybrid` need stereo or RGBD. `semi_direct` accepts those sensors and is the only mode that accepts a monocular camera; monocular scale follows `seed_min_depth` and `seed_max_depth`, and loop closing applies Sim(3) corrections, with a per-keyframe scale passed to `apply_keyframe_poses`. All three accept an undistorted pinhole or a raw-pixel camera (Kannala–Brandt, rational pinhole, double sphere). The native frontend does not remap images and rejects a distorted linear pinhole that does not project raw pixels. The `main_slam_direct.py` runner remaps distorted pinhole monocular/RGBD images and depth before tracking and loop closing, using the same intrinsics with zero distortion; stereo pinhole input must already be rectified.

## Layout

This subsystem lives in `cpp/core/slam/direct/`, next to the descriptor-based sparse SLAM in `cpp/core/slam/`. The two share the C++ module and the camera code but not their data structures: the sparse SLAM mirrors the Python classes in `pyslam/slam/`, while this one is C++ only.

Each tracking mode is a `PoseTracker` in its own folder (`klt/`, `hybrid/`, `semi_direct/`). `DirectSlam` in `core/` keeps only what the modes share. The building blocks (`features/`, `geometry/`, `lk/`, `photometric/`) are used by the modes and by `core/`. The include rules are:

- A building block includes, from `core/`, only the leaf headers `direct_params.h`, `image_utils.h`, `se3.h` and `live_point.h`. It never includes a mode.
- `lk/` (2D correspondences) and `photometric/` (intensity residuals) never include each other. `geometry/` includes no other building block.
- `core/` never includes `photometric/`. It includes a mode only in the `make_pose_tracker` factory (`core/pose_tracker.cpp`).
- `klt/` uses no photometric code. `hybrid/` is the only mode that includes another mode (`klt/`), because it combines the two families.

Stereo matching uses LK in every mode, because it is a depth measurement and not pose tracking.

| File | Role |
| --- | --- |
| `core/direct_frontend.h/.cpp` | `DirectSlam`: input checks, initialization from sensor depth, accept/reject, live points, keyframes, local BA, loop-closure poses |
| `core/pose_tracker.h/.cpp` | `PoseTracker` interface (track plus lifecycle hooks) and the `make_pose_tracker(mode)` factory |
| `core/monocular_initializer.h/.cpp` | Monocular two-view initialization: LK tracks, essential matrix, triangulation, map scale and inverse-depth priors |
| `core/depth_measurement.h/.cpp` | RGBD depth sampling, stereo LK triangulation, metric validity checks, and `measure_live_points` (re-measure live points in the current frame) |
| `core/direct_backend.h/.cpp` | Robust local g2o bundle adjustment |
| `core/direct_map.h` | Persistent landmarks, keyframe observations and corrected `Tcw` trajectory |
| `core/direct_params.h/.cpp` | `DIRECT_SLAM` settings |
| `core/image_utils.h/.cpp` | Shared gray-image helpers: `require_gray`, the two bilinear samplers (planar patches / depth filter vs. the photometric aligner, which differ only at the right/bottom edge), `build_pyramid`, `inside_border` |
| `core/live_point.h` | `LivePoint`: a landmark tracked from the last accepted image, and `make_live_point` for a newly created one |
| `core/se3.h` | Eigen left SE(3) increment (no Sophus) |
| `klt/klt_pose_tracker.h/.cpp` | `klt`: LK temporal tracks, then reprojection pose (indirect) |
| `hybrid/hybrid_pose_tracker.h/.cpp` | `hybrid`: the `klt` pose, then photometric refinement or fallback |
| `semi_direct/semi_direct_pose_tracker.h/.cpp` | `semi_direct` (SVO-style): planar-patch pose, per-feature alignment, reprojection pose, and inverse-depth seed lifecycle |
| `features/corner_detection.h/.cpp` | GFTT / grid FAST dispatch with border and occupancy masks; `OccupancyMask` keeps new corners away from existing features |
| `features/corner_grid.h/.cpp` | Grid FAST extraction: strongest corners per cell with the budget water-filled across cells, optional subpixel refinement |
| `geometry/reprojection_pose.h/.cpp` | PnP RANSAC seed plus robust reprojection Gauss-Newton (shared by all modes) |
| `geometry/ray_geometry.h/.cpp` | `lift_reference_point` (raw pixel and depth to a camera point) and `depth_uncertainty_one_pixel` (SVO τ) |
| `geometry/triangulation.h` | Midpoint triangulation of two bearings |
| `lk/lk_tracker.h/.cpp` | Pyramidal LK with a forward-backward check |
| `photometric/direct_aligner.h/.cpp` | Single-level and coarse-to-fine photometric alignment for `hybrid` |
| `photometric/planar_patch.h/.cpp` | Fronto-parallel patch pose and per-feature alignment for `semi_direct` |
| `photometric/depth_filter.h/.cpp` | SVO depth filter: Gaussian × Beta inverse-depth seeds, one-pixel τ, subpixel epipolar search, expiry |
| `py_module/direct_slam_module.h` | pybind11 `DirectSlam` |
| `tests_cpp/test_direct_slam.cpp` | Synthetic photometric, LK, stereo, RGBD, outlier, loss/recovery and settings checks; SVO depth-filter equivalence, τ geometry and semi-direct tracker regressions |
| `tests_cpp/test_direct_components.cpp` | Unit tests of the shared building blocks: image helpers (sampler edge behavior, pyramid sizes), occupancy masks, `measure_live_points` (RGBD, rectified stereo, right image ignored without stereo), map/landmark bookkeeping, and the tracker hooks (hybrid's depth refresh without BA, keyframe-image ownership per mode) |
| `tests_py/test_direct_slam.py` | Binding snapshot ownership, shapes, metadata and reset |
| `tests_py/test_semi_direct_slam.py` | `semi_direct` through the binding: mode aliases, monocular gating and two-view initialization, RGBD seeds where depth is missing |
| `tests_py/test_direct_loop_features.py`, `test_direct_loop_gate.py` | Loop-closing bridge (`direct_loop_closing.py`): deferred keyframe features, pose sync, reset, and the tracking-idle gate |
| `tests_py/test_direct_loop_monocular.py` | Monocular loop-closing bridge: landmark association and late refresh, delayed submission, point fusion, descriptor points kept on the direct landmarks; relocalization (recent keyframes, then the keyframe database) |
| `tests_py/test_main_slam_direct.py` | `main_slam_direct.py`: viewers stay open after the last frame until closed or `q`; monocular viewer scale; image undistortion for a distorted pinhole |

The sources are listed in `cpp/core/slam/CMakeLists.txt` and linked into `cpp_core`. Includes use the full path from `cpp/core`, for example `#include "slam/direct/core/direct_frontend.h"`. Python imports the class from `pyslam.slam.cpp`.

## Poses

`CameraPose` stores `Tcw` (world to camera), the same convention as `cpp/core/slam`. A live point keeps a stable feature ID, its pixel in the last accepted image and its position in the world. The current point is `X_c = Tcw * X_w`.

`se3.h` applies a left increment `T <- exp(xi) * T`. The two solvers use different `xi` orders and must not be mixed:

- Photometric alignment follows the Sophus order `xi = [tx, ty, tz, wx, wy, wz]` (`left_multiply_translation_rotation`).
- The geometric pose step uses `g2o_left_se3_jacobian`, so `xi = [wx, wy, wz, tx, ty, tz]` (`left_multiply_rotation_translation`).

The photometric Jacobian is `∂π/∂X` from the camera model times the left SE3 increment `[t, ω]`. Reference points are lifted once. A pyramid level scales the projected pixel and that Jacobian; it does not build another camera. Depth stays metric.

## Depth

Stereo correspondence is pyramidal Lucas–Kanade, not a descriptor match. A right camera then uses `Camera::triangulate_stereo_point`. A rectified stereo camera with no right camera (KITTI) uses `depth = bf / (u_l - u_r)` when the vertical mismatch is under 2 px and the disparity is positive. RGBD samples the depth image and multiplies by `Camera::depth_factor`, then lifts the raw pixel with `unproject_bearing`.

A new landmark needs a finite depth in `(seed_min_depth, seed_max_depth)` (default `(0.5, 80)` meters; `metric_point_ok`). An undistorted pinhole checks `z`; a raw-pixel camera keeps `z > 0` and applies that window to the ray length. `Camera::depth_threshold` is the sparse SLAM close/far split and is not used as a validity gate here.

## Tracking

`klt` and `hybrid` align each frame to the last accepted image, not to the oldest keyframe. `semi_direct` aligns to the reference keyframe with a planar patch, then refines each feature, and only then runs the reprojection pose. It does not fall back to Lucas–Kanade. The depth filter runs on the tracking thread after an accepted pose. A measured stereo or RGBD corner is still a landmark immediately; a corner without depth, and every monocular corner, is a seed. A monocular map starts from a two-view initialization (LK tracks, essential matrix, triangulation); the status stays `INIT` until it succeeds. Reprojection pose estimation uses PnP RANSAC followed by robust Gauss-Newton and a per-observation chi-square inlier mask. Rejected matches are removed from the live set. A constant-velocity prediction initializes Lucas-Kanade or the photometric pyramid. New corners are added when the live set falls below `num_features_needed_for_keyframe`, and a keyframe is stored with that pose. Keyframes record observations of persistent landmark IDs and trigger local bundle adjustment. With local BA enabled, a keyframe is also inserted every `max_keyframe_interval` frames so that stable tracks continue to constrain the backend. A failed frame reports `LOST`, leaves the last measured pose and map intact, and clears the motion prediction. Subsequent frames can recover against that retained reference. Recovery does not change the world origin; persistent loss needs an explicit reset. Hybrid mode robustly weights photometric residuals and uses the optimized landmark depths. When automatic local BA is disabled, it retains the previous per-frame sensor depth refresh.

The backend follows `slamplay/full_slam/vslam`: poses and shared landmarks are optimized together from left/right observations using Levenberg-Marquardt and Huber kernels. This implementation runs synchronously on keyframe insertion, avoiding concurrent map mutation during tracking and Python snapshots. Its active set is the latest `num_active_keyframes` keyframes (default 7), with the oldest pose fixed to remove the gauge freedom. Compared with vslam's distance-based keyframe eviction, this is a chronological window.

Right pixels are measured again at every keyframe, including existing landmarks; they are not synthesized from the estimated depth. Calibrated rigs use their full right-camera extrinsics and projection model. RGBD adds a measured depth edge with standard deviation `ba_depth_sigma * max(1, z*z)` meters. Stereo and RGBD keep metric scale constrained by those measurements. Monocular landmarks from seed promotion or two-view initialization add inverse-depth priors. Each prior is a binary edge between the landmark and its reference keyframe, constraining depth in that camera and supplying the scale gauge. Robust optimization is followed by outlier exclusion and refinement. Rejected left/right/depth observations are recorded separately, and failed, non-finite, disconnected or scale-unconstrained solves leave the map intact.

After a successful solve, corrected landmarks are copied into live tracking points, rejected current observations are removed, and poses/velocity are updated. Non-keyframe trajectory entries retain their transform relative to their reference keyframe and are reconstructed after BA. The viewer reads the complete optimized landmark map instead of caching stale coordinates. `bundle_adjustment()` also exposes a manual local solve and the viewer's BA button invokes it. `last_ba()` reports graph sizes, costs, outliers and whether the update was applied. Local BA reduces drift inside the active window. Loop closure is applied by the Python runner, not by this backend.

The previous KITTI failure had three interacting causes: tight motion gates replaced measured poses with predictions; rejected reprojection matches remained in the live map; and failed frames continued integrating velocity and seeding new landmarks. The photometric optimizer also compared the old cost after committing an update, retaining an error-increasing step. These paths are now covered by the synthetic tests.

Status names reported by `status_name()`:

| Status | Meaning |
| --- | --- |
| `INIT` | No frame has initialized the map yet. Monocular `semi_direct` stays here until the two-view initialization succeeds. |
| `TRACKING_GOOD` | Inliers above `num_features_tracking` |
| `TRACKING_BAD` | Accepted pose with inliers above `num_features_tracking_bad` only |
| `LOST` | Tracking could not be maintained |

## Settings

`from_config` accepts a nested `DIRECT_SLAM` dict, flat pyslam keys prefixed with `DIRECT_SLAM.`, or the block itself. For a monocular camera, `DirectSlam` also applies the `monocular` sub-block (`DIRECT_SLAM.monocular.<key>` or a nested `monocular` dict) over the base values. Unknown `mode` values throw. `semi_direct` and `semidirect` select the planar-patch mode. `direct` still maps to `hybrid`. The code default is `klt`. The checked-in camera settings use `hybrid`, except `settings/TUM3.yaml` and `settings/KITTI04-12.yaml`, which use `semi_direct`. A monocular camera also applies `DIRECT_SLAM.monocular.*`, which selects `semi_direct` and a larger seed budget in those settings. Defaults live in `direct_params.h`.

`settings/KITTI00-02.yaml` selects `hybrid` for stereo and `semi_direct` for monocular input. `settings/KITTI04-12.yaml` and `settings/TUM3.yaml` select `semi_direct` (TUM3 is undistorted RGBD). `settings/TUM_VI.yaml` selects `hybrid`. Rectified TUM-VI is an undistorted pinhole. Native TUM-VI keeps the distorted images: Lucas–Kanade finds the right-image pixel, and the photometric warp uses the Kannala–Brandt Jacobian. A distorted linear pinhole such as TUM1 must be remapped before calling the native frontend; the runner does this automatically for monocular/RGBD input.

Corners come from `detect_corners`. `grid_fast` (the default) splits the image into `grid_cols` by `grid_rows` cells, runs FAST on each cell padded by 4 px (so corners on inner cell edges are found exactly once, as with whole-image FAST), and keeps the strongest responses. The `num_features` budget is water-filled over the cells: a cell with little texture keeps all its corners and its unused share goes to the cells that have more, so the budget is met whenever the image has enough corners. Pass a cell list to `detect_grid_fast` to extract only those cells. `gftt` remains the single global Shi-Tomasi detector. Zero pixels in the occupancy mask are skipped, and `border` excludes the image edge. `num_features` is both the detection cap and the live-track target.

```yaml
# Tracking mode: klt uses LK + geometry; hybrid adds photometric refinement/fallback.
# semi_direct uses planar patches and inverse-depth seeds, and is the only monocular mode.
# direct is a deprecated alias of hybrid.
DIRECT_SLAM.mode: hybrid   # or klt, or semi_direct
# gftt or grid_fast. grid_fast keeps the strongest FAST corners in each cell.
DIRECT_SLAM.corner_detector: grid_fast
# Maximum corners per detection; also the target live-track count when replenishing.
DIRECT_SLAM.num_features: 200
DIRECT_SLAM.grid_cols: 5
DIRECT_SLAM.grid_rows: 5
DIRECT_SLAM.fast_threshold: 10
DIRECT_SLAM.fast_nonmax_suppression: true
DIRECT_SLAM.corner_subpix: true
DIRECT_SLAM.subpixel_window: 5
# LK search-window side length, in pixels at each pyramid level (left/right and temporal).
DIRECT_SLAM.lk_patch_size: 11
# Highest zero-based LK pyramid level; 3 means levels 0 through 3.
DIRECT_SLAM.lk_max_level: 3
# Photometric patch half-width; 1 samples a 3x3 patch at each pyramid level.
DIRECT_SLAM.half_patch_size: 1
# Maximum photometric Gauss-Newton iterations per pyramid level.
DIRECT_SLAM.num_iterations: 10
# Number of photometric pyramid levels, including the full-resolution image.
DIRECT_SLAM.num_pyramid_levels: 4
# Photometric image/intrinsic scale between adjacent levels (0 < value < 1).
DIRECT_SLAM.pyramid_scale: 0.5
# TRACKING_GOOD requires strictly more geometric/accepted inliers than this count.
DIRECT_SLAM.num_features_tracking: 50
# Accepted tracking requires strictly more inliers than this count; otherwise LOST.
DIRECT_SLAM.num_features_tracking_bad: 20
# Replenish tracks and insert a keyframe when the live-track count falls below this value.
DIRECT_SLAM.num_features_needed_for_keyframe: 80
# Run synchronous local BA on keyframes; manual bundle_adjustment() remains available.
DIRECT_SLAM.local_ba: true
# Maximum recent keyframes in the BA window (>= 2); the oldest window pose is fixed.
DIRECT_SLAM.num_active_keyframes: 7
# Initial robust LM iteration cap; refinement uses max(1, ba_iterations / 2) iterations.
DIRECT_SLAM.ba_iterations: 10
# Force a keyframe after this input-frame ID gap when automatic BA is enabled.
DIRECT_SLAM.max_keyframe_interval: 10
# RGBD depth sigma (m) = value * max(1, z*z); larger values weaken depth constraints.
DIRECT_SLAM.ba_depth_sigma: 0.02  # RGBD: sigma(z) = value * max(1, z*z) meters
```

## Build, run, test

From the repository root, with the pyslam environment active:

```bash
./build_cpp_core.sh
```

`main_slam_direct.py` uses `Config`, `dataset_factory`, and `camera_from_config`. The dataset must be stereo or RGBD, unless the resolved mode is `semi_direct`. That mode can come from `DIRECT_SLAM.monocular.mode` in the camera settings or from `--mode`. Monocular runs use Sim(3) loop closing and align ground truth with Sim(3).

```bash
python main_slam_direct.py -c config.yaml --headless
python main_slam_direct.py -c config.yaml --headless --max-frames 100
python main_slam_direct.py -c config.yaml                  # 3D map and tracked feature trails
python main_slam_direct.py -c config.yaml --headless --mode klt --max-frames 1000
python main_slam_direct.py -c config.yaml --headless --trajectory-file /tmp/direct-kitti.txt
python main_slam_direct.py -c config.yaml --headless --no-local-ba  # frontend-only comparison
python main_slam_direct.py -c config.yaml --headless --no-loop-closing
```

The viewer shows the current camera, keyframes, accumulated landmarks in black, the full accepted estimated trajectory as a blue line, feature trails, and ground truth when configured. With loop closing enabled it also draws covisibility, spanning-tree and loop edges from the descriptor keyframes, using each direct keyframe's camera center. The trajectory line incorporates local BA and loop corrections and is cleared on reset. Pause, step, reset, local BA and quit are supported; the shared viewer's save control is not connected to this frontend. Feature drawing uses `cpp/core/common/feature_trails.h`, the same helper as `Frame::draw_feature_trails` in `main_slam.py`: stable ID colors, filled 4-pixel markers, antialiased trails and the optional radius/track-age palette. Detected corners are drawn at octave 0. Non-headless playback is paced at the dataset frame rate; headless playback runs as fast as possible.

`--trajectory-file` saves the final corrected **accepted** `Twc` poses as 12-column KITTI rows, after local BA and loop corrections. Failed input frames have no row, so frame-index comparisons require a run without losses. `DirectSlam.trajectory()` retains the existing `Tcw` convention. `map_points()` returns a copied Nx3 array of all valid optimized landmarks. `draw_feature_trails()` renders the current tracks without changing its input. `tracked_features()` returns copied `(ids, pixels Nx2, world_points Nx3)` snapshots and `keyframes()` returns `(Twc_poses, timestamps)`.

Synthetic checks:

```bash
cmake --build cpp/core/slam/build --target test_direct_slam test_direct_components -j"$(nproc)"
./cpp/core/slam/build/tests_cpp/direct_slam/test_direct_slam
./cpp/core/slam/build/tests_cpp/direct_slam/test_direct_components
python -m unittest discover -s cpp/core/slam/direct/tests_py
```

Lucas-Kanade comes from OpenCV's `video` module. The SLAM `CMakeLists.txt` must list that component.

## KITTI validation

Validated locally on the configured color KITTI sequence 00. The baseline rows below have automatic local BA disabled. Translation RMSE below uses rigid SE(3) alignment, without scale correction; it is not KITTI's segment drift score.

| Mode | Frames | Lost frames | Translation RMSE |
| --- | ---: | ---: | ---: |
| `klt`, before local BA | 4,541 (complete sequence) | 0 | 10.73 m |
| `hybrid`, before local BA | 1,000 | 0 | 4.24 m |
| `klt`, with local BA | 4,541 (complete sequence) | 0 | 8.85 m |
| `hybrid`, with local BA | 1,000 | 0 | 3.79 m |

The C++ regression executable and Python binding test pass, including stereo/native-rig/RGBD BA recovery, outlier rejection, fixed gauges, bounded windows, unobservable graph rejection, corrected trajectories, frontend propagation, reset, and shared feature rendering. The shared viewer and feature window were smoke-tested for 30 frames with clean shutdown. These runs use 200 requested corners, a 7-keyframe window, and 10 initial BA iterations. Real RGBD sequences have not been evaluated in this change.
