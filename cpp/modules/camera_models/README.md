# Camera models

pySLAM represents the camera as an object with `project`, `unproject_points`, and the analytic Jacobian of the projection. Implementations are available through `pyslam.slam` for the selected backend. The projection polynomials live in one header, and the Python and C++ cores do not load each other's camera classes. The layout is in [Where the code lives](#where-the-code-lives).

The default model is the linear pinhole used by the existing datasets. Kannala–Brandt, the OpenCV rational model, and the double-sphere model are selected explicitly with `Camera.type`.

## Integration status

Pinhole, KB4, rational pinhole, and double sphere share model-aware sparse tracking and optimization. [TUM-VI](../../../docs/tum-vi.md) has both native fisheye and rectified stereo loaders. The regression suite under [Tests](#tests) covers the original defect cases.

| Configuration | Implementation |
| --- | --- |
| Default pinhole, mono/stereo/RGB-D | Existing projection convention; native pinhole optimizer factors retained |
| New models, mono | Raw-pixel projection, search, map admission/culling, and native analytic optimizer factors |
| New models, RGB-D | Raw left projection plus virtual right coordinate derived from axial depth and `bf` |
| New models, stereo | Requires an explicit right camera and relative pose; TUM-VI supplies these in native mode |
| Rectified TUM-VI | Cached fisheye remapping and the existing pinhole stereo pipeline |

Stereo observations retain three components `(u_left, v_left, u_right)`. For a physical rig, the last coordinate comes from projecting through the actual right camera and its extrinsics. For RGB-D it is the virtual pinhole coordinate `fx * x/z + cx - bf/z`; it is not the raw left coordinate minus disparity. Raw-model optimization compares the first two components with detected pixels.

The camera math and map-admission checks preserve oriented backward rays for KB4 and double sphere. Stereo depth initialization/backprojection still uses forward axial depths, and legacy initialization/Sim(3) paths have not been validated for a full field of view beyond 180 degrees. Do not infer complete omnidirectional SLAM support from the projection model alone. Use rectified stereo with dense stereo depth estimators.

For raw models, map-admission scale consistency uses distance from each camera center, rather than axial depth. Axial depth changes under camera rotation and can reject a correct wide-angle correspondence even when its viewing distance and observed feature scale remain consistent. Pinhole keeps its existing check.

## Models

| `Camera.type` | Class | Parameters stored in `D` |
| --- | --- | --- |
| omitted, `pinhole`, `0`, `1` | `PinholeCamera` | OpenCV 5 coefficients, used only by undistortion |
| `kannala_brandt4`, `kb4`, `2` | `KannalaBrandt4Camera` | `k1, k2, k3, k4` |
| `pinhole_radtan8`, `radtan8`, `rational`, `3` | `PinholeRadtan8Camera` | `k1, k2, p1, p2, k3, k4, k5, k6` |
| `double_sphere`, `ds`, `4` | `DoubleSphereCamera` | `xi, alpha` |

`Camera.k1` … `Camera.k4` mean different things for the pinhole and Kannala–Brandt files. On a pinhole file, `k3` is the OpenCV radial coefficient. On a Kannala–Brandt file, `k3` and `k4` are the higher polynomial coefficients, not `p1` and `p2`.

### Pinhole

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

`project()` does not apply `D`. Detected points are undistorted first, and bundle adjustment compares the linear projection with those undistorted pixels. A missing `Camera.type` keeps this behavior, including when `Camera.k1` … `Camera.k3` are nonzero, so the current KITTI, TUM, EuRoC, and other calibration files stay on the linear model.

Unprojection returns the point on the plane `z = 1`, `(x, y, 1)`, not a unit bearing. A point is valid when `z` is positive.

### Kannala–Brandt4

```text
r = hypot(x, y)
theta = atan2(r, z)
theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
u = fx * theta_d * x / r + cx
v = fy * theta_d * y / r + cy
```

This is the same polynomial used by OpenCV `cv2.fisheye`. The field of view can pass 180 degrees, so a negative `z` can still be a valid projection. The exact backward optical axis is a singularity and is reported invalid.

### Rational pinhole

The normalized point `(x/z, y/z)` is distorted with the OpenCV 8-coefficient model (`k1, k2, p1, p2, k3, k4, k5, k6`) and then multiplied by `K`. This matches `cv2.projectPoints`. Validity requires `z > 0` and a finite distortion. Unprojection returns the undistorted point on `z = 1`.

### Double sphere

The model of Usenko, Demmel, and Cremers, with parameters `Camera.xi` and `Camera.alpha`. A ray is valid when it lies inside the model domain, which can include negative `z`. `alpha = 0.5` and `xi = 0` is not a pinhole camera away from the optical axis.

## Pixels used by tracking and optimization

`undistort_points()` always returns the pinhole image of the ray:

```text
bearing = unproject(detected pixel)
undistorted = (fx * bx / bz + cx, fy * by / bz + cy)
```

For every model, `kpsu` is that pinhole image. The fundamental matrix, rectified-stereo disparity, and `keypoint_undistorted()` keep using it.

`projects_raw_keypoints()` is false for the linear pinhole and true for the three new models. When it is true, `project()` predicts the detected pixel, so projection search and mono bundle adjustment measure `kps`. `unproject_points()` on those models also expects the detected pixel. Passing `kpsu` into it applies the model twice.

## Optimization

New-model mono bundle adjustment, local bundle adjustment, and pose optimization call `project_with_jacobian` in C++. Default pinhole observations retain the original native factors, including when using the Python core, so no per-observation Python callbacks enter optimization. The returned `2 x 3` block is `d(u, v) / d(Xc)` and is shared by the residual and the Jacobian.

- g2o stores `Tcw` and uses a left perturbation. Its error is `measurement - projection`, so the error Jacobians carry a minus sign.
- gtsam stores `Twc` and uses `Pose3.transformTo`. Its error is `projection - measurement`.

An invalid projection is not a usable residual. In the gtsam factors it becomes `(2 * fx, 2 * fx)` with zero Jacobians. In g2o, invalid model projections produce a finite penalty and zero Jacobians, and pose outlier classification checks validity as well as chi-squared error. For Kannala–Brandt and double sphere validity is not equivalent to `z > 0`.

Pinhole stereo and Sim(3) edges retain their established conventions. New-model stereo and RGB-D edges use native model-aware projection and analytic Jacobians. A non-pinhole stereo configuration requires `Stereo.right` and `Stereo.T_right_left`; an uncalibrated stereo rig is rejected.

## Epipolar geometry on raw pixels

Rectified-row stereo matching and the fundamental-matrix test assume a pinhole image. For cameras with `projects_raw_keypoints()` both checks are done on unit bearings instead and expressed in pixels of the image being searched:

- **Stereo rig matching** (`EpipolarMatcher.search_stereo_rig`, used by `Frame::compute_stereo_matches` in both cores). Each left bearing `f` defines the epipolar plane with normal `n = t × (R f)` in the right camera. A right keypoint is a candidate when its bearing `fr` lies between `R f` (infinite depth) and `t` (zero depth) on that plane, its octave differs by at most one, and its distance to the plane's image, `|n·fr| · ||J(fr) n||` with `J` the projection Jacobian, passes the 1-DoF χ² test (3.84 σ²) for its octave. The best descriptor in that band must pass the ratio test (`best < ratio · second`) and the distance threshold; matches are then made one-to-one and triangulated with `Camera::triangulate_stereo_point`, which checks reprojection in both cameras against the 2-DoF χ² bound.
- **Triangulation search between keyframes** (`search_frame_for_triangulation`). The epipole distance is measured as the angle to the baseline direction times `fx`, and the epipolar distance uses `Camera::bearing_to_plane_pixel_distance`, the same first-order distance as above. The Python core vectorizes the same test with a numerical directional derivative.

Pinhole cameras keep the original rectified-row matcher and fundamental-matrix test unchanged.

## Where the code lives

`USE_CPP_CORE` chooses the SLAM classes. It does not choose the projection formulas. Both backends include `cpp/core/slam/camera_model_math.h`, which depends only on Eigen. That header is the only copy of the pinhole, Kannala–Brandt4, rational-pinhole, and double-sphere polynomials and of `d(u, v) / d(Xc)`.

`USE_CPP_CORE = True` imports the `cpp_core` classes from `pyslam.slam`. Those cameras, their g2o edges, and their gtsam factors are the types in `cpp/core/slam/`. Entry points build them with `camera_from_config(config)`.

`USE_CPP_CORE = False` keeps the Python classes. Pinhole batch projection stays on the original Numba path, and pinhole observations keep the original native g2o and gtsam factors. Kannala–Brandt4, rational pinhole, and double sphere call the modules under `cpp/modules/camera_models/`. Importing `camera_models` does not load `cpp_core`, g2o, or gtsam.

| Module | Loaded when | Contents |
| --- | --- | --- |
| `camera_models` | A Python-core new-model camera is first used | Projection, unprojection, rig triangulation, `EpipolarMatcher.search_stereo_rig` |
| `camera_models_g2o` | The Python g2o optimizer builds a new-model edge | g2o edges only |
| `camera_models_gtsam` | The Python gtsam optimizer builds a new-model factor | gtsam factors only |
| `cpp_core` | `USE_CPP_CORE` is true | SLAM cameras, edges, and factors for that backend |

The camera object itself is `libcamera_model_core.so`. The two factor modules link that library, so an edge and a factor hold the same camera the projection module constructed. A camera that is already a `cpp_core` object keeps `cpp_core` edges and factors, because those types store a `cpp_core` camera pointer. `pyslam/slam/camera_factors.py` selects the module from the camera it is given.

Batch projection, undistortion, unprojection, and rig triangulation write into caller-owned NumPy buffers (`project_into` and the other `*_into` methods). The bindings take those arrays with `noconvert()`, so a C-contiguous `float64` buffer is used in place. Stereo-rig search views the `float32` keypoints, `int32` octaves, `float32` octave scales, and descriptor matrices the same way. The match-index lists are allocated after the search, once their length is known.

The rig geometry, the stereo-rig search, the g2o edges, and the gtsam factors are written once, as header-only templates over the camera type in `cpp/core/slam/`. `cpp_core` instantiates them for `pyslam::Camera` and `camera_models` for `camera_models::CameraModel`; each side only adds type aliases and bindings. The templates hold the camera through `CameraHandle`, which is non-owning for a raw pointer and owning for a `shared_ptr`, so both ownership styles share one edge type.

g2o and gtsam headers are never included in the same translation unit. The g2o templates and the gtsam templates are in separate headers, `cpp_core` binds its edges and its factors in two separate source files, and the C++ factor tests are split the same way.

| Piece | Path |
| --- | --- |
| Projection math and Jacobians | `cpp/core/slam/camera_model_math.h` |
| Rig projection, rig triangulation, bearing-to-plane distance, stereo-rig search (shared) | `cpp/core/slam/camera_rig_geometry.h` |
| g2o edges (shared, no gtsam) | `cpp/core/slam/camera_projection_edges_g2o.h` |
| gtsam factors and `python_noise_model` (shared, no g2o) | `cpp/core/slam/camera_projection_factors_gtsam.h` |
| Camera reference held by edges and factors | `cpp/core/slam/camera_handle.h` |
| Python-core camera, shared by the factor modules | `cpp/modules/camera_models/camera_model.h`, `libcamera_model_core.so` |
| Python-core projection bindings | `cpp/modules/camera_models/camera_models_module.cpp` |
| Python-core g2o edges (aliases) | `cpp/modules/camera_models/camera_model_g2o_factors.h`, `camera_models_g2o_module.cpp` |
| Python-core gtsam factors (aliases) | `cpp/modules/camera_models/camera_model_gtsam_factors.h`, `camera_models_gtsam_module.cpp` |
| `cpp_core` camera classes | `cpp/core/slam/camera_models.h` |
| `cpp_core` g2o edges (aliases) | `cpp/core/slam/camera_edges_g2o.h`, `py_module/camera_g2o_module.cpp` |
| `cpp_core` gtsam factors (aliases, stereo factor helpers) | `cpp/core/slam/camera_factors_gtsam.h`, `py_module/camera_gtsam_module.cpp` |
| Triangulation-search epipolar mask (Python core) | `pyslam/slam/geometry_matchers.py` |
| Python classes and `camera_from_config` | `pyslam/slam/camera.py` |
| Edge/factor selection for the Python optimizers | `pyslam/slam/camera_factors.py` |

## Example

```yaml
Camera.type: kannala_brandt4
Camera.fx: 300.0
Camera.fy: 310.0
Camera.cx: 320.0
Camera.cy: 240.0
Camera.k1: -0.02
Camera.k2: 0.005
Camera.k3: -0.001
Camera.k4: 0.0002
Camera.width: 640
Camera.height: 480
Camera.fps: 30
```

```yaml
Camera.type: pinhole_radtan8
Camera.k1: -0.1
Camera.k2: 0.02
Camera.p1: 0.001
Camera.p2: -0.0005
Camera.k3: 0.003
Camera.k4: 0.01
Camera.k5: -0.004
Camera.k6: 0.001
```

```yaml
Camera.type: double_sphere
Camera.xi: -0.2
Camera.alpha: 0.6
```

The rational and double-sphere snippets supply model-specific fields; include the dimensions, intrinsics, and frame rate shown in the first example. These examples describe calibration selection, subject to the integration limits above.

In Python, `pyslam.slam.camera.camera_from_json(payload)` selects the serialized subclass and preserves any right camera and extrinsics. `Frame.from_json` uses this factory. Constructors accept a `Config` object or its serialized dictionary.

## Tests

`cpp/core/slam/tests_cpp/test_camera_model_math.cpp` checks the analytic Jacobians against finite differences. `test_camera_models.cpp` checks round trips, undistortion, visibility, the bearing-to-plane pixel distance, and rig triangulation (including a ray beyond 90°). `test_camera_edges_g2o.cpp` checks that a pinhole edge matches the native g2o edge and that the Kannala–Brandt edge Jacobians match finite differences; `test_camera_factors_gtsam.cpp` does the same for the gtsam factors. The two files are separate so that neither includes both libraries.

`cpp/core/slam/tests_py/test_camera_models.py` checks the Python bindings, the OpenCV projection, and the calibrations in `settings/`. For every settings file the omitted camera type must stay a linear pinhole, including files whose distortion coefficients are nonzero. That test also times pinhole batch projection against the existing Numba path and compares a pinhole bundle adjustment against `EdgeSE3ProjectXYZ`. It covers `search_stereo_rig` on a synthetic KB4 rig (true matches and depths, off-curve and beyond-infinity distractors, octave gap, ambiguous duplicates, one-to-one) and the Python bearing epipolar mask.

`tests/camera/test_camera_models_modules.py` covers the Python-core path (`USE_CPP_CORE=False`): importing `camera_models`, `camera_models_g2o`, or `camera_models_gtsam` loads only its own optimizer library; a Python KB4 rig binds to a `camera_models` camera; `search_stereo_rig` recovers synthetic matches and depths; and the g2o pose edges and gtsam resectioning factors (mono and stereo) pull a perturbed pose back to the true one. It also checks that a gtsam graph of these factors can be cloned.

Additional integration and pinhole compatibility checks are in `tests/camera/test_camera_regressions.py`. They exercise 54 calibration/sensor combinations in both backends, stereo disparity, RGB-D backprojection, undistortion, raw-pixel KD trees, and Python camera serialization. The six original defect reproductions are ordinary passing regression tests; no expected-failure decorators remain. A pinhole control verifies the map-admission test fixture.

Run from the repository root using the pySLAM Python environment and a rebuilt C++ core:

```bash
python cpp/core/slam/tests_py/test_camera_models.py
python tests/camera/test_camera_regressions.py -v
python tests/camera/test_tum_vi.py -v
python tests/camera/test_camera_models_modules.py -v
# Run just the original defect reproductions:
python tests/camera/test_camera_regressions.py TestKnownCameraDefects
```

The opt-in paired benchmark loads the pinhole implementation from a supplied Git commit into a temporary module. It warms Numba, alternates measurement order, records repeated samples and medians, checks numerical equivalence, and writes machine-readable JSON. It also compares the new Python g2o/GTSAM factor paths with the previous native factors, including a pose-only solve with fixed 3D landmarks. The default regression budget is 20%, adjustable for a controlled benchmark machine. An exceeded budget returns exit status 1.

```bash
# Compare against a pinhole-only commit, for example:
python tests/camera/benchmark_pinhole.py \
  --baseline-ref 96019cfafcfc099ac9866884d7143a9ed1451a0d \
  --output /tmp/pinhole-benchmark.json
# Faster optimizer-only check:
python tests/camera/benchmark_pinhole.py \
  --baseline-ref 96019cfafcfc099ac9866884d7143a9ed1451a0d \
  --optimizers-only --output /tmp/pinhole-optimizers.json
# Native C++ edges/factors against the original g2o/GTSAM implementations:
cpp/core/slam/build/tests_cpp/test_camera_edges_g2o --benchmark
cpp/core/slam/build/tests_cpp/test_camera_factors_gtsam --benchmark
```

Timing ratios use the median of adjacent candidate/baseline sample ratios. The JSON also retains independent duration medians, their ratio, and all raw samples, so CPU-frequency/load drift can be inspected.

Python camera timings compare actual revisions. The C++ camera rows compare current C++ against the old Python/Numba implementation for context and are not gated as revision regressions. They cannot establish C++ performance relative to the old C++ build. The native factor benchmark separately compares the actual previous g2o/GTSAM factor implementations with the current factors. Component checks also cannot establish unchanged trajectory accuracy, tracking loss, or end-to-end throughput; use the end-to-end replay below for those.

### End-to-end replay and pinhole regression

`tests/camera/replay_sequence.py` replays any dataset section of `config.yaml` deterministically (ORB2, synchronous local mapping, no loop closure unless `--loop-closure`) and writes JSON with tracking states, per-frame track+map timings, keyframe/map-point counts, and ATE (Sim(3) for mono, SE(3) otherwise), plus an NPZ with the trajectory and raw timings. `--pyslam-root` imports another checkout, so a pinhole-only baseline exported from the last pinhole commit (with its own C++ core built) is replayed by the same harness:

```bash
git archive <pinhole-commit> | tar -x -C ../pyslam-baseline   # then build its C++ core
for root in ../pyslam-baseline .; do
  python tests/camera/replay_sequence.py --pyslam-root $root \
    --dataset KITTI_DATASET --base-path <kitti>/dataset --name 00 --sensor-type stereo \
    --settings settings/KITTI00-02.yaml --set is_color=false --frames 1500 \
    --output /tmp/kitti00_stereo_$(basename $root).json
done
```

Run comparisons serially on an otherwise idle machine. Concurrent SLAM processes compete for CPU and memory and can distort timing ratios. The harness fixes OpenCV to one thread and seeds OpenCV, NumPy, and Python RNGs. Its timings include synchronous local mapping, so they must not be compared directly with another system's asynchronous tracking-only timings.

For playback scheduling, pass `--local-mapping threaded --loop-closure` to either replay command. These runs include asynchronous mapping and loop-detection work, but exclude GUI rendering and real-time pacing; thread scheduling also makes map construction less repeatable. Compare matching modes and inspect image-loading time as well as tracking time. Initial frames can include Numba compilation, so use the per-frame NPZ timings to distinguish startup costs from steady operation. TUM ATE uses the original mocap samples rather than frame associations, which can repeat a ground-truth timestamp for consecutive images.

On Linux, the replay and component benchmark share an exclusive process lock. A watchdog interrupts a run above 8 GiB resident memory or below 8 GiB available host memory; replay options `--max-rss-gib` and `--min-available-gib` set these budgets. The replay checks memory at frame boundaries and records sampled peak RSS. These checks are safeguards, not operating-system memory limits. Run other tests, reference executables, and builds separately as well. Interrupted runs do not count as completed comparisons.

For memory investigations, `--trace-memory` prints Python allocation summaries at 200-frame intervals. Tracing adds overhead and its timings must be excluded from throughput comparisons.

For repeated comparisons, edit the dataset roots and sequences in `tests/camera/pinhole_replay_cases.example.json`, then run:

```bash
python tests/camera/run_pinhole_regression.py \
  --baseline-root ../pyslam-baseline-pinhole \
  --cases tests/camera/pinhole_replay_cases.example.json \
  --frames 300 --repeats 3 --output results/pinhole-regression
```

The example covers KITTI mono/stereo, EuRoC mono/stereo, and TUM mono/RGB-D. Additional datasets use the same manifest schema. This recorded-data sample is separate from the calibration tests; passing it does not validate every supported dataset or full-sequence loop closure. Increase the frame bound for release checks.

The runner alternates baseline/candidate order between pairs and saves each JSON, trajectory, and log. It fails if the median paired mean frame-time ratio exceeds 1.20, any pair loses more than two percentage points of tracking/trajectory/mocap coverage, or ATE increases by more than the larger of 20% and 0.02 m. Timing budget is configurable with `--max-ratio`. Baselines exported with `git archive` are supported: source and native binary hashes identify them when no `.git` is present. Compilation caches and replay logs stay in the output directory.

Replay ATE uses `tests/camera/trajectory_evaluation.py`, with Sim(3) alignment for mono and SE(3) for metric sensors. Interpolation requires the entire bracketing mocap interval to be at most `--max-align-dt`; ground truth is never extrapolated or filled across longer gaps. Fewer than three associations is an error. The report includes association coverage and alignment scale. Tests cover a known similarity transform, interpolation, sparse gaps, invalid/degenerate input, and the independent regression gates:

```bash
python -m unittest discover -s tests/camera -p 'test_*.py'
```
