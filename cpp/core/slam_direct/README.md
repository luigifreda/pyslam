# Direct and semi-direct SLAM

See the [direct-SLAM guide](../../../docs/direct_slam.md) for the runner, architecture diagrams, complete parameter reference, and comparison with `main_slam.py`.

A direct/semi-direct frontend and local bundle-adjustment backend compiled into `cpp_core`. It reuses the pyslam camera, settings, and datasets. It has no loop closure or global pose-graph optimization. `main_direct_slam.py` reuses the same `Viewer3D` and `CvImageViewer` as `main_slam.py`.

Two modes, selected by `DIRECT_SLAM.mode`:

- `semi_direct`: GFTT corners, pyramidal Lucas-Kanade, then a geometric reprojection pose.
- `direct`: sparse photometric alignment on patches with known depth (the optic-flow direct method), seeded and checked by geometric tracking. If corner flow fails, a photometric-only step requires support from at least 60% of the reference landmarks.

Both need an undistorted pinhole camera and a depth source. Monocular and distorted cameras are rejected at construction.

## Layout

| File | Role |
| --- | --- |
| `direct_frontend.h/.cpp` | `DirectSlam`: track, status, pose, keyframes, trajectory |
| `semi_direct_tracker.h/.cpp` | GFTT, pyramidal LK, reprojection Gauss-Newton |
| `direct_aligner.h/.cpp` | Single-level and coarse-to-fine photometric alignment |
| `direct_backend.h/.cpp` | Robust local g2o bundle adjustment |
| `direct_map.h` | Persistent landmarks, keyframe observations and corrected `Tcw` trajectory |
| `direct_params.h/.cpp` | `DIRECT_SLAM` settings |
| `se3.h` | Eigen left SE(3) increment (no Sophus) |
| `py_module/direct_slam_module.h` | pybind11 `DirectSlam` |
| `tests_cpp/test_direct_slam.cpp` | Synthetic photometric, LK, stereo, RGBD, outlier, loss/recovery and settings checks |
| `tests_python/test_direct_slam.py` | Binding snapshot ownership, shapes, metadata and reset |

The sources are listed in `cpp/core/slam/CMakeLists.txt` and linked into `cpp_core`. Python imports the class from `pyslam.slam.cpp`.

## Poses

`CameraPose` stores `Tcw` (world to camera), the same convention as `cpp/core/slam`. A live point keeps a stable feature ID, its pixel in the last accepted image and its position in the world. The current point is `X_c = Tcw * X_w`.

`se3.h` applies a left increment `T <- exp(xi) * T`. The two solvers use different `xi` orders and must not be mixed:

- Photometric alignment follows the Sophus order `xi = [tx, ty, tz, wx, wy, wz]` (`left_multiply_translation_rotation`).
- The geometric pose step uses `g2o_left_se3_jacobian`, so `xi = [wx, wy, wz, tx, ty, tz]` (`left_multiply_rotation_translation`).

The photometric Jacobian is the pinhole formula. Pyramid levels scale `fx`, `fy`, `cx`, `cy` and the pixels. Depth stays metric.

## Depth

Stereo with a right camera uses `Camera::triangulate_stereo_point`. A rectified stereo camera with no right camera (KITTI) uses `depth = bf / (u_l - u_r)` when the vertical mismatch is under 2 px and the disparity is positive. RGBD samples the depth image and multiplies by `Camera::depth_factor`.

A depth is kept when it is finite and in `(0.5, 80)` meters. `Camera::depth_threshold` is the sparse SLAM close/far split and is not used as a validity gate here.

## Tracking

Each frame is aligned to the last accepted image, not to the oldest keyframe. Reprojection pose estimation uses PnP RANSAC followed by robust Gauss-Newton and a per-observation chi-square inlier mask. Rejected matches are removed from the live set. A constant-velocity prediction initializes Lucas-Kanade or the photometric pyramid. New corners are added when the live set falls below `num_features_needed_for_keyframe`, and a keyframe is stored with that pose. Keyframes record observations of persistent landmark IDs and trigger local bundle adjustment. With local BA enabled, a keyframe is also inserted every `max_keyframe_interval` frames so that stable tracks continue to constrain the backend. A failed frame reports `LOST`, leaves the last measured pose and map intact, and clears the motion prediction. Subsequent frames can recover against that retained reference. Recovery does not change the world origin; persistent loss needs an explicit reset. Direct mode robustly weights photometric residuals and uses the optimized landmark depths. When automatic local BA is disabled, it retains the previous per-frame sensor depth refresh.

The backend follows `slamplay/full_slam/vslam`: poses and shared landmarks are optimized together from left/right observations using Levenberg-Marquardt and Huber kernels. This implementation runs synchronously on keyframe insertion, avoiding concurrent map mutation during tracking and Python snapshots. Its active set is the latest `num_active_keyframes` keyframes (default 7), with the oldest pose fixed to remove the gauge freedom. Compared with vslam's distance-based keyframe eviction, this is a chronological window.

Right pixels are measured again at every keyframe, including existing landmarks; they are not synthesized from the estimated depth. Calibrated rigs use their full right-camera extrinsics and projection model. RGBD adds a measured depth edge with standard deviation `ba_depth_sigma * max(1, z*z)` meters. Both modes keep metric scale constrained. Robust optimization is followed by outlier exclusion and refinement. Rejected left/right/depth observations are recorded separately, and failed, non-finite, disconnected or scale-unconstrained solves leave the map intact.

After a successful solve, corrected landmarks are copied into live tracking points, rejected current observations are removed, and poses/velocity are updated. Non-keyframe trajectory entries retain their transform relative to their reference keyframe and are reconstructed after BA. The viewer reads the complete optimized landmark map instead of caching stale coordinates. `bundle_adjustment()` also exposes a manual local solve and the viewer's BA button invokes it. `last_ba()` reports graph sizes, costs, outliers and whether the update was applied. Local BA reduces drift but does not provide loop closure.

The previous KITTI failure had three interacting causes: tight motion gates replaced measured poses with predictions; rejected reprojection matches remained in the live map; and failed frames continued integrating velocity and seeding new landmarks. The photometric optimizer also compared the old cost after committing an update, retaining an error-increasing step. These paths are now covered by the synthetic tests.

Status names reported by `status_name()`:

| Status | Meaning |
| --- | --- |
| `INIT` | No frame has been tracked yet |
| `TRACKING_GOOD` | Inliers above `num_features_tracking` |
| `TRACKING_BAD` | Accepted pose with inliers above `num_features_tracking_bad` only |
| `LOST` | Tracking could not be maintained |

## Settings

`from_config` accepts a nested `DIRECT_SLAM` dict, flat pyslam keys prefixed with `DIRECT_SLAM.`, or the block itself. Unknown `mode` values throw. Defaults live in `direct_params.h`.

`settings/KITTI00-02.yaml` selects `semi_direct`. `settings/TUM3.yaml` selects `direct` (undistorted RGBD). Do not point this frontend at a distorted calibration such as TUM1.

```yaml
# Tracking mode: semi_direct uses LK + geometry; direct adds photometric refinement/fallback.
DIRECT_SLAM.mode: semi_direct   # or direct
# Maximum GFTT corners per detection; also the target live-track count when replenishing.
DIRECT_SLAM.num_features: 200
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

`main_direct_slam.py` uses `Config`, `dataset_factory`, and `camera_from_config`. The dataset must be stereo or RGBD.

```bash
python main_direct_slam.py -c config.yaml --headless
python main_direct_slam.py -c config.yaml --headless --max-frames 100
python main_direct_slam.py -c config.yaml                  # 3D map and tracked feature trails
python main_direct_slam.py -c config.yaml --headless --mode direct --max-frames 1000
python main_direct_slam.py -c config.yaml --headless --trajectory-file /tmp/direct-kitti.txt
python main_direct_slam.py -c config.yaml --headless --no-local-ba  # frontend-only comparison
```

The viewer shows the current camera, keyframes, accumulated landmarks in black, the full accepted estimated trajectory as a blue line, feature trails, and ground truth when configured. The trajectory line incorporates local BA corrections and is cleared on reset. Pause, step, reset, local BA and quit are supported; the shared viewer's save control is not connected to this frontend. Feature drawing uses `cpp/core/slam/utils/feature_trails.h`, the same helper as `Frame::draw_feature_trails` in `main_slam.py`: stable ID colors, filled 4-pixel markers, antialiased trails and the optional radius/track-age palette. GFTT corners have octave 0. Non-headless playback is paced at the dataset frame rate; headless playback runs as fast as possible.

`--trajectory-file` saves the final BA-corrected **accepted** `Twc` poses as 12-column KITTI rows. Failed input frames have no row, so frame-index comparisons require a run without losses. `DirectSlam.trajectory()` retains the existing `Tcw` convention. `map_points()` returns a copied Nx3 array of all valid optimized landmarks. `draw_feature_trails()` renders the current tracks without changing its input. `tracked_features()` returns copied `(ids, pixels Nx2, world_points Nx3)` snapshots and `keyframes()` returns `(Twc_poses, timestamps)`.

Synthetic checks:

```bash
cmake --build cpp/core/slam/build --target test_direct_slam -j"$(nproc)"
./cpp/core/slam/build/tests_cpp/direct_slam/test_direct_slam
python -m unittest discover -s cpp/core/slam_direct/tests_python
```

Lucas-Kanade comes from OpenCV's `video` module. The SLAM `CMakeLists.txt` must list that component.

## KITTI validation

Validated locally on the configured color KITTI sequence 00. The baseline rows below have automatic local BA disabled. Translation RMSE below uses rigid SE(3) alignment, without scale correction; it is not KITTI's segment drift score.

| Mode | Frames | Lost frames | Translation RMSE |
| --- | ---: | ---: | ---: |
| `semi_direct`, before local BA | 4,541 (complete sequence) | 0 | 10.73 m |
| `direct`, before local BA | 1,000 | 0 | 4.24 m |
| `semi_direct`, with local BA | 4,541 (complete sequence) | 0 | 8.85 m |
| `direct`, with local BA | 1,000 | 0 | 3.79 m |

The C++ regression executable and Python binding test pass, including stereo/native-rig/RGBD BA recovery, outlier rejection, fixed gauges, bounded windows, unobservable graph rejection, corrected trajectories, frontend propagation, reset, and shared feature rendering. The shared viewer and feature window were smoke-tested for 30 frames with clean shutdown. These runs use 200 requested corners, a 7-keyframe window, and 10 initial BA iterations. Real RGBD sequences have not been evaluated in this change.
