# TUM-VI visual mono and stereo

pySLAM supports the 512×512 TUM-VI image sequences in mono and stereo, using the calibration and right-to-left transform from PLVS2's `Examples/Stereo/TUM-VI.yaml`. IMU measurements are not consumed.

Set `DATASET.type: TUM_VI_DATASET` in `config.yaml` and configure:

```yaml
TUM_VI_DATASET:
  type: tum_vi
  sensor_type: stereo
  base_path: <tum_vi_root>
  name: dataset-room1_512_16
  stereo_mode: rectified  # or native
  clahe: true
  settings: settings/TUM_VI.yaml
  groundtruth_file: auto
  start_frame_id: 0
```

The sequence layout is:

```text
dataset-room1_512_16/
  mav0/cam0/data.csv
  mav0/cam0/data/<timestamp>.png
  mav0/cam1/data.csv
  mav0/cam1/data/<timestamp>.png
  mav0/mocap0/data.csv
```

Camera CSV filenames are respected. Cameras are associated by exact integer nanosecond timestamps, so a missing exposure cannot shift subsequent pairs. If a CSV is absent, timestamp-named PNG files are indexed instead. An optional dataset `timestamps` field selects a PLVS2-style timestamp text file, relative to the sequence directory or absolute. Start indices are applied once by the shared dataset API.

For visual monocular operation set `sensor_type: mono` and `stereo_mode: native`.
All cam0 exposures are retained, including those without a matching cam1 image;
cam1 images are not required. The camera is KB4 without a stereo rig, and ATE uses
Sim(3) alignment to account for unknown monocular scale. `stereo_mode: rectified`
also supports mono using the left stereo-rectification map and its camera frame.

Like PLVS2's TUM-VI example, images are loaded as 8-bit grayscale and CLAHE with clip limit 3.0 and an 8×8 tile grid is applied before optional rectification. CLAHE objects are cached per camera; set `clahe: false` to disable this preprocessing.

## Rectified mode

`stereo_mode: rectified` uses `cv2.fisheye.stereoRectify`, following PLVS2's pre-rectification option. It inverts `Stereo.T_c1_c2` to obtain the left-to-right transform expected by OpenCV. `Camera.fovScale: 0.7` and a default rectification balance of `1.0` match PLVS2; `Camera.rectifyBalance` can override the latter.

The loader derives the pinhole intrinsics and `bf` from the rectified projection matrices. Distortion is zero in this tracking camera. Fixed-point remap tables are computed once, and each frame uses `cv2.remap`; rectification is not recomputed during playback. This mode uses the established pinhole stereo matcher and factors and is the mode to use with stereo depth estimators and dense reconstruction. `settings/TUM_VI.yaml` selects `DIRECT_SLAM.mode: hybrid` for the rectified pinhole.

## Native mode

`stereo_mode: native` keeps the original fisheye images. It constructs two KB4 cameras with their individual intrinsics and distortion, plus the calibrated `T_right_left` transform. KB4 is the four-distortion-coefficient model called `KannalaBrandt8` by PLVS2/ORB-SLAM3; the other four parameters are the intrinsics.

`main_slam_direct.py` tracks these raw images in `klt` and `hybrid` mode. Lucas–Kanade supplies the right-image pixel. The photometric warp uses the KB4 projection Jacobian and does not remap the frames. `settings/TUM_VI.yaml` selects `mode: hybrid`.

Stereo matching does not assume rectified rows. For each left keypoint, `search_stereo_rig` unprojects the left ray and keeps right keypoints whose ray lies on its epipolar plane, between the image of the point at infinity and the right camera center (positive depth on both rays). The band is the 1-DoF chi-squared bound of the right keypoint's octave, measured in pixels with the right camera's projection Jacobian. Candidates must be at most one octave apart, the best descriptor must pass a strict ratio test, and matches are one-to-one. Each accepted pair is triangulated from both rays and must reproject within the 2-DoF chi-squared bound in both images; its forward depth feeds the existing stereo initialization and backprojection paths. See [Epipolar geometry on raw pixels](../cpp/modules/camera_models/README.md#epipolar-geometry-on-raw-pixels).

Tracking and optimization use raw left pixels and the predicted horizontal coordinate in the actual right camera. The existing three-component stereo observation contract is `(u_left, v_left, u_right)`; the measured right vertical coordinate participates in triangulation/reprojection validation. This is not a four-component stereo bundle-adjustment residual. Both g2o and GTSAM use analytic camera and pose Jacobians in C++. Native images should not be sent directly into a depth estimator that assumes rectified stereo; select rectified mode for that pipeline.

The rig's right camera and extrinsics are preserved in camera/frame/map serialization and C++ pickling. Arbitrary fisheye stereo needs both calibrations and the relative pose; specifying only `Camera.type` still cannot define a stereo rig.

## Ground truth

The loader reads `mav0/mocap0/data.csv` as timestamp, body translation, and quaternion `(qw, qx, qy, qz)`. The [calibrated TUM-VI exports](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) provide ground-truth poses in the IMU frame. `TUMVI.T_body_left` converts these poses to the tracking camera; rectified mode additionally accounts for the rectification rotation. These values are supplied in the calibration file. Replace them when using another calibration.

Mocap can cover only the beginning and end of a sequence. Full trajectory access retains all available mocap samples. Per-frame queries reject samples farther than `groundtruth_max_gap` (default 0.02 seconds), rather than inventing ground truth across unobserved intervals.

Ground truth supports the standard `GroundTruth.save()` / `GroundTruth.load()`
system-state path. Simple-format export writes every available mocap sample in
the tracking-camera frame, preserving late segments after gaps.

## Tests and execution

With the pySLAM environment active and the C++ core rebuilt (see [tests/README.md](../tests/README.md)):

```bash
python tests/camera/test_tum_vi.py -v
python main_slam.py --headless
```

The synthetic tests exercise timestamp association, missing exposures, start indices, native and rectified camera construction, independent OpenCV fisheye projections, triangulation and its rejection checks, analytic Jacobians, Python/C++ frame matching, native g2o/GTSAM factors, camera lifetime, serialization, and sparse mocap handling.

Mono tests additionally cover absent cam1 images, preservation of unmatched left
exposures, both camera backends, and corresponding mocap/start indices. Calibration
loading rejects nonrigid stereo or body-to-camera transforms.

For a recorded replay, run each mode in a fresh process with the generic replay harness (`--frames 0`, the default, replays the whole sequence):

```bash
for mode in rectified native; do
  python benchmarking/camera/replay_sequence.py --dataset TUM_VI_DATASET \
    --base-path <tum_vi_root> --name dataset-room1_512_16 \
    --set stereo_mode=$mode --output /tmp/tum-vi-room1-$mode.json
done
```

The harness uses ORB2 with the configured feature count, includes synchronous local mapping in the per-frame timings, and disables loop closure (unless `--loop-closure`), semantics, and dense mapping. It writes JSON with tracking states, timings, keyframe/map-point counts, and translation ATE, plus an NPZ with the trajectory. ATE interpolates mocap between bracketing samples at most `--max-align-dt` (default 0.1 s) apart and uses SE(3) alignment, preserving stereo scale. `--start` and `--frames` select a segment. Set `PYSLAM_USE_CPP=false` before launch to exercise the Python core.

## Recorded-data smoke validation

On 2026-10-18 all five local TUM-VI sequences passed loading and timestamp association in both modes, and the first 300 room1 frames tracked with 299/300 frames OK in each mode (SE(3)-aligned translation RMSE 0.0061 m rectified, 0.0127 m native). The full table, timings and limits of that run are in [benchmarking/benchmarks/tum_vi/room1-smoke-2026-10-18.md](../benchmarking/benchmarks/tum_vi/room1-smoke-2026-10-18.md).

## Comparing with PLVS2 / ORB-SLAM3

Use visual-only examples (`mono_tum_vi` or `stereo_tum_vi`), matching calibration,
sequence timestamps, preprocessing, and feature count. Export EuRoC trajectories
(`f_*.txt`); these contain nanosecond timestamps, whereas pySLAM NPZ files contain
seconds. Evaluate both with the same camera-frame conversion and association policy:

```bash
python benchmarking/camera/evaluate_tum_vi.py \
  --trajectory results/reference/f_room1_mono.txt --sensor-type mono \
  --base-path <tum_vi_root> \
  --output results/reference/mono_ate.json
python benchmarking/camera/evaluate_tum_vi.py \
  --trajectory results/room1_native.npz --sensor-type stereo --mode native \
  --base-path <tum_vi_root> \
  --output results/room1_native_ate.json
```

`--mode` must describe the estimated tracking camera (`native` or `rectified`),
including its effect on the body-to-camera transform. Use `--timestamp-scale 1`
for text exports already in seconds. Compare trajectory coverage as well as RMSE;
an accurate short surviving map does not establish full-sequence accuracy.

Some local PLVS2 builds crash in `LoopClosing::Run` with `loopClosing: 0` because
the disabled detector returns without consuming its keyframe queue. For a visual
odometry reference with loop closure off, build local copies of its examples:

```bash
python benchmarking/camera/build_plvs2_reference.py --plvs2-root /path/to/plvs2 \
  --output results/plvs2-reference
```

This uses an existing PLVS2 CMake build and leaves that checkout untouched. The
copies call `System::DisableLoopClosing()` after construction and set OpenCV to
one thread. Keep `loopClosing: 1` in their YAML so the thread consumes the queue;
set `Viewer.on`, `Line.on`, and `PointCloudMapping.on` to zero. Their command-line
arguments match the original examples. Mapping remains asynchronous, so their
tracking-call timings and pySLAM's synchronous tracking-plus-mapping timings
measure different workloads.
