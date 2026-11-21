# Camera models

A pySLAM camera is an object. The object has the functions `project` and `unproject_points`. The object also has the analytic Jacobian of the projection. You get the implementation from `pyslam.slam`. The selected backend supplies that implementation. One header contains the projection polynomials. The Python core does not load the C++ camera classes. The C++ core does not load the Python camera classes. [Where the code lives](#where-the-code-lives) shows the file layout.

The default model is the linear pinhole. The current datasets use this model. Set `Camera.type` to select Kannala–Brandt, the OpenCV rational model, or the double-sphere model.

## Notation

A vector symbol is a lowercase bold letter. A matrix symbol is an uppercase bold letter. A scalar symbol is an italic letter and is not bold.

Camera points are metric 3D column vectors. Pixels and normalized image points are written below as homogeneous 3-vectors with last coordinate 1; the code stores their first two coordinates. Bearings are 3D ray directions: multiplying a bearing by a positive scalar preserves the ray, but scaling a camera point changes its depth. The pinhole projection uses:

$$
\mathbf{x}_c = \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix},
\qquad
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}.
$$

\(\mathbf{x}_c\) is a point in the camera frame. \(\mathbf{K}\) is the camera calibration matrix. \(\mathbf{K}\) uses only the intrinsic parameters. The focal lengths are the scalars \(f_x\) and \(f_y\).

Two coordinates live on the image plane. The raw pixel \(\mathbf{p}_{\mathrm{raw}}\) is the detected point. The undistorted pixel \(\mathbf{p}_{\mathrm{u}}\) is the pinhole image of the same ray:

$$
\mathbf{p}_{\mathrm{raw}} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix},
\qquad
\mathbf{p}_{\mathrm{u}} = \begin{bmatrix} u_{\mathrm{u}} \\ v_{\mathrm{u}} \\ 1 \end{bmatrix}.
$$

In the code, `kps` stores the first two coordinates of \(\mathbf{p}_{\mathrm{raw}}\). `kpsu` stores the first two coordinates of \(\mathbf{p}_{\mathrm{u}}\). In a projection formula, \(\mathbf{p}\) is the pixel that `project()` returns. For the linear pinhole, that pixel is \(\mathbf{p}_{\mathrm{u}}\). For Kannala–Brandt, the rational model, and double sphere, that pixel is \(\mathbf{p}_{\mathrm{raw}}\).

The normalized coordinates \(\mathbf{x}\) are the intersection of the ray with the plane \(z = 1\). They are not pixels. Compute them from the undistorted pixel with the inverse calibration matrix:

$$
\mathbf{x} = \mathbf{K}^{-1} \mathbf{p}_{\mathrm{u}} = \begin{bmatrix} (u_{\mathrm{u}} - c_x)/f_x \\ (v_{\mathrm{u}} - c_y)/f_y \\ 1 \end{bmatrix}.
$$

The same relation forward is \(\mathbf{p}_{\mathrm{u}} = \mathbf{K} \, \mathbf{x}\).

A bearing \(\mathbf{d}\) is the direction of that ray in the camera frame. `unproject_bearing` computes it from an input pixel \(\mathbf{p}_{\mathrm{in}}\):

$$
\mathbf{d} = \operatorname{unproject\_bearing}(\mathbf{p}_{\mathrm{in}}) = \begin{bmatrix} d_x \\ d_y \\ d_z \end{bmatrix}.
$$

For Kannala–Brandt, the rational model, and double sphere, \(\mathbf{p}_{\mathrm{in}}\) is the raw pixel \(\mathbf{p}_{\mathrm{raw}}\). For the linear pinhole, \(\mathbf{p}_{\mathrm{in}}\) is the undistorted pixel \(\mathbf{p}_{\mathrm{u}}\). The linear bearing unprojection does not use `D`. `unproject_points()` instead returns the two normalized coordinates on the plane \(z = 1\).

Every model first removes the calibration matrix:

$$
\begin{bmatrix} m_x \\ m_y \\ 1 \end{bmatrix} = \mathbf{K}^{-1} \mathbf{p}_{\mathrm{in}} = \begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix}.
$$

The model then maps \((m_x, m_y)\) to \(\mathbf{d}\):

- Pinhole: \(\mathbf{d} = \begin{bmatrix} m_x & m_y & 1 \end{bmatrix}^{\top}\). This is the normalized point \(\mathbf{x}\).
- Rational pinhole: invert the distortion so that the distorted normalized point equals \((m_x, m_y)\). The solution \((x, y)\) gives \(\mathbf{d} = \begin{bmatrix} x & y & 1 \end{bmatrix}^{\top}\). This is also \(\mathbf{x}\).
- Kannala–Brandt: let \(\theta_d = \sqrt{m_x^2 + m_y^2}\), and solve \(\theta\) from \(\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)\). Then

$$
\mathbf{d} = \begin{bmatrix} m_x \sin\theta / \theta_d \\ m_y \sin\theta / \theta_d \\ \cos\theta \end{bmatrix}.
$$

At \(\theta_d = 0\), use the optical-axis limit \(\mathbf{d} = [0, 0, 1]^{\top}\). This \(\mathbf{d}\) has unit length. \(\theta > \pi/2\) gives \(d_z < 0\).

- Double sphere: apply the closed-form inverse of the model to \((m_x, m_y)\). The result is \(\mathbf{d}\). It has unit length, and \(d_z\) can be negative.

\(\mathbf{d}\) is not a focal length. The unit bearing is \(\mathbf{d}/\|\mathbf{d}\|\). When \(d_z \neq 0\), its pinhole normalized coordinates are

$$
\mathbf{x} = \frac{1}{d_z}\mathbf{d},
\qquad
d_z \, \mathbf{p}_{\mathrm{u}} = \mathbf{K} \, \mathbf{d}.
$$

If the model has distortion, the model applies that distortion before \(\mathbf{K}\). \(\mathbf{K}\) then maps the resulting homogeneous point to \(\mathbf{p}\).

## Integration status

The pinhole, KB4, rational pinhole, and double-sphere models use the same tracking. They use the same optimization. The tracking and the optimization use the selected camera model. [TUM-VI](../../../docs/tum-vi.md) has a native fisheye loader and a rectified stereo loader. The tests in [Tests](#tests) cover the original defect cases.

| Configuration | Implementation |
| --- | --- |
| Default pinhole, for mono, stereo, or RGB-D | The code keeps the current projection. The code keeps the native pinhole optimizer factors. |
| New models, mono | Projection, search, map admission, and map culling use raw pixels. The optimizer factors are native and analytic. |
| New models, RGB-D | The left projection uses raw pixels. The virtual right coordinate comes from the axial depth and `bf`. |
| New models, stereo | Supply an explicit right camera and a relative pose. Native TUM-VI supplies these data. |
| Rectified TUM-VI | The loader caches the fisheye remap. The pipeline is the current pinhole stereo pipeline. |

A stereo observation is the vector

$$
\begin{bmatrix} u_{\mathrm{left}} \\ v_{\mathrm{left}} \\ u_{\mathrm{right}} \end{bmatrix}.
$$

On a physical rig, the last component comes from the right camera and its extrinsics. For RGB-D, \(u_{\mathrm{right}}\) is the first component of

$$
z_c \begin{bmatrix} u_{\mathrm{right}} \\ v \\ 1 \end{bmatrix} = \mathbf{K} \, \mathbf{x}_c - \begin{bmatrix} b_f \\ 0 \\ 0 \end{bmatrix}.
$$

This component is not the raw left coordinate minus the disparity. Optimization of a raw model compares the first two components with the detected pixels.

The camera math keeps the direction of a backward ray for KB4 and for double sphere. The map-admission checks do the same. Stereo depth initialization uses forward axial depths. Stereo backprojection uses forward axial depths. The tests do not validate the legacy initialization paths for a field of view of more than 180 degrees. The tests do not validate the legacy Sim(3) paths for that field of view. Do not treat the projection model as full omnidirectional SLAM support. For that field of view, use rectified stereo and a dense stereo depth estimator.

For a raw model, the map-admission scale check uses the distance from each camera center. The check does not use axial depth. Axial depth changes when the camera rotates. The check can then reject a correct wide-angle match. The viewing distance and the feature scale of that match can still agree. The pinhole model keeps its current check.

## Models

| `Camera.type` | Class | Parameters stored in `D` |
| --- | --- | --- |
| omitted, `pinhole`, `0`, `1` | `PinholeCamera` | OpenCV 5 coefficients, used only by undistortion |
| `kannala_brandt4`, `kb4`, `2` | `KannalaBrandt4Camera` | `k1, k2, k3, k4` |
| `pinhole_radtan8`, `radtan8`, `rational`, `3` | `PinholeRadtan8Camera` | `k1, k2, p1, p2, k3, k4, k5, k6` |
| `double_sphere`, `ds`, `4` | `DoubleSphereCamera` | `xi, alpha` |

`Camera.k1` through `Camera.k4` do not have the same meaning in every file. In a pinhole file, `k3` is the OpenCV radial coefficient. In a Kannala–Brandt file, `k3` and `k4` are the higher polynomial coefficients. They are not `p1` and `p2`.

### Pinhole

$$
z_c \, \mathbf{p} = \mathbf{K} \, \mathbf{x}_c.
$$

`project()` does not use `D`. The tracking pipeline undistorts the detected points before comparing them with the projection. Bundle adjustment compares the linear projection with those undistorted pixels. If `Camera.type` is absent, the camera stays on this linear model. This is true when `Camera.k1`, `Camera.k2`, and `Camera.k3` are not zero. The KITTI, TUM, EuRoC, and other calibration files stay on the linear model.

Unprojection returns the normalized coordinates \(\mathbf{x}\) on the plane \(z_c = 1\). The result is not the unit bearing \(\mathbf{d}/\|\mathbf{d}\|\). A point is valid when \(z_c > 0\).

### Kannala–Brandt4

$$
\begin{aligned}
r &= \sqrt{x_c^2 + y_c^2}, \\
\theta &= \operatorname{atan2}(r, z_c), \\
\theta_d &= \theta \bigl(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8\bigr).
\end{aligned}
$$

$$
\mathbf{p} = \mathbf{K} \begin{bmatrix} \theta_d x_c / r \\ \theta_d y_c / r \\ 1 \end{bmatrix}.
$$

OpenCV `cv2.fisheye` uses this polynomial. The field of view can be more than 180 degrees. A negative \(z_c\) can still be a valid projection. The exact backward optical axis is a singularity. The function reports that point as not valid.

### Rational pinhole

The model distorts the normalized point \(\mathbf{x}\) with the eight OpenCV coefficients \((k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6)\). \(\mathbf{K}\) then maps the distorted point to the raw pixel:

$$
\mathbf{x} = \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} x_c / z_c \\ y_c / z_c \\ 1 \end{bmatrix}, \qquad r^2 = x'^2 + y'^2.
$$

$$
\begin{bmatrix} x'' \\ y'' \end{bmatrix} = \begin{bmatrix} x' \dfrac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2 \bigl(r^2 + 2 x'^2\bigr) \\ y' \dfrac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 \bigl(r^2 + 2 y'^2\bigr) + 2 p_2 x' y' \end{bmatrix},
$$

$$
\mathbf{p} = \mathbf{K} \begin{bmatrix} x'' \\ y'' \\ 1 \end{bmatrix}.
$$

This projection agrees with `cv2.projectPoints`. The point is valid when \(z_c > 0\) and the distortion is finite. Unprojection returns the normalized coordinates \(\mathbf{x}\) on the plane \(z_c = 1\).

### Double sphere

This is the model of Usenko, Demmel, and Cremers. The parameters are `Camera.xi` and `Camera.alpha`:

$$
\begin{aligned}
d_1 &= \lVert \mathbf{x}_c \rVert, \\
k &= \xi \, d_1 + z_c, \\
d_2 &= \sqrt{x_c^2 + y_c^2 + k^2}, \\
\gamma &= \alpha \, d_2 + (1 - \alpha) \, k.
\end{aligned}
$$

$$
\mathbf{p} = \mathbf{K} \begin{bmatrix} x_c / \gamma \\ y_c / \gamma \\ 1 \end{bmatrix}.
$$

A ray is valid when it is inside the model domain. The domain can include a negative \(z_c\). \(\alpha = 0.5\) and \(\xi = 0\) do not give a pinhole camera away from the optical axis.

## Pixels used by tracking and optimization

`undistort_points()` always returns the undistorted pixel \(\mathbf{p}_{\mathrm{u}}\). The function first unprojects the raw pixel to a bearing. It then makes the pinhole image of that bearing:

$$
\mathbf{d} = \operatorname{unproject\_bearing}(\mathbf{p}_{\mathrm{raw}}),
\qquad
d_z \, \mathbf{p}_{\mathrm{u}} = \mathbf{K} \, \mathbf{d}.
$$

\(\mathbf{d}\) is not a pair of normalized coordinates. The normalized coordinates are \(\mathbf{x} = \mathbf{d}/d_z\) when \(d_z \neq 0\). For every model, `kpsu` is \(\mathbf{p}_{\mathrm{u}}\). The fundamental matrix uses `kpsu`. Rectified-stereo disparity uses `kpsu`. `keypoint_undistorted()` uses `kpsu`.

`projects_raw_keypoints()` is false for the linear pinhole. `projects_raw_keypoints()` is true for the three new models. If the value is true, `project()` predicts the detected pixel. Projection search then measures `kps`. Mono bundle adjustment also measures `kps`. On those models, `unproject_points()` expects the detected pixel. A call with `kpsu` applies the model twice.

## Optimization

Mono bundle adjustment for a new model calls `project_with_jacobian` in C++. Local bundle adjustment for a new model calls the same function. Pose optimization for a new model calls the same function. Default pinhole observations keep the original native factors. This is also true for the Python core. Optimization does not call Python for each observation. The function returns the \(2 \times 3\) matrix

$$
\frac{\partial}{\partial \mathbf{x}_c}\begin{bmatrix} u \\ v \end{bmatrix}.
$$

The residual and the Jacobian use this matrix.

- g2o stores the matrix \(\mathbf{T}_{cw}\). g2o uses a left perturbation. The error is the measurement minus the projection. The error Jacobians have a minus sign.
- gtsam stores the matrix \(\mathbf{T}_{wc}\). gtsam uses `Pose3.transformTo`. The error is the projection minus the measurement.

An invalid projection is not a residual that you can use. In a gtsam factor, an invalid projection becomes

$$
\begin{bmatrix} 2 f_x \\ 2 f_x \end{bmatrix}.
$$

The Jacobians are zero. In g2o, an invalid model projection gives a finite penalty. The Jacobians are zero. Pose outlier classification checks the validity. It also checks the chi-squared error. For Kannala–Brandt and double sphere, validity is not the test \(z_c > 0\).

Pinhole stereo edges keep their current conventions. Pinhole Sim(3) edges keep their current conventions. New-model stereo edges use the native projection for that model. New-model RGB-D edges use the same projection. These edges use analytic Jacobians. A stereo setup that is not a pinhole needs `Stereo.right` and `Stereo.T_right_left`. The code rejects a stereo rig that has no calibration.

## Epipolar geometry on raw pixels

Rectified-row stereo matching assumes a pinhole image. The fundamental-matrix test assumes a pinhole image. If `projects_raw_keypoints()` is true, both checks use the unit bearing \(\mathbf{d}/\|\mathbf{d}\|\). The function takes \(\mathbf{d}\) from `unproject_bearing` of the raw pixel, then divides by \(\|\mathbf{d}\|\). \(\mathbf{d}\) is the ray direction from [Notation](#notation). It is not the focal length \(f_x\) or \(f_y\). The checks report the distance in pixels of the image under search.

- **Stereo rig matching** (`EpipolarMatcher.search_stereo_rig`). `Frame::compute_stereo_matches` uses this function in both cores. Each left raw pixel gives a unit bearing \(\mathbf{d}\). This vector sets an epipolar plane. The plane normal in the right camera is \(\mathbf{n} = \mathbf{t} \times (\mathbf{R} \mathbf{d})\). \(\mathbf{R}\) is the rotation from the left camera to the right camera. \(\mathbf{t}\) is the baseline translation. A right keypoint is a candidate when all of these conditions are true. Its unit bearing \(\mathbf{d}_r\) lies on that plane between \(\mathbf{R}\mathbf{d}\) and \(\mathbf{t}\). \(\mathbf{R}\mathbf{d}\) is infinite depth. \(\mathbf{t}\) is zero depth. Its octave differs by at most one. Its distance to the image of the plane passes the chi-squared test for its octave. The distance is \(|\mathbf{n} \cdot \mathbf{d}_r| \cdot \|\mathbf{J}(\mathbf{d}_r)\, \mathbf{n}\|\). \(\mathbf{J}\) is the projection Jacobian. The test has one degree of freedom. The limit is \(3.84\,\sigma^2\). The best descriptor in that band must pass the ratio test \(\mathrm{best} < \mathrm{ratio} \cdot \mathrm{second}\). It must also pass the distance threshold. The function then makes the matches one-to-one. `Camera::triangulate_stereo_point` triangulates the matches. The function checks the reprojection in both cameras against the chi-squared bound for two degrees of freedom.
- **Triangulation search between keyframes** (`search_frame_for_triangulation`). The epipole distance is the angle between the unit bearing and the baseline, multiplied by \(f_x\). The epipolar distance uses `Camera::bearing_to_plane_pixel_distance`. This is the same first-order distance as above. The Python core uses the same test with a numerical directional derivative. The test is vectorized.

Pinhole cameras keep the original rectified-row matcher. They keep the original fundamental-matrix test.

## Where the code lives

`USE_CPP_CORE` selects the SLAM classes. It does not select the projection formulas. Both backends include `cpp/core/camera/camera_model_math.h`. That header depends only on Eigen. The header contains the only copy of each projection polynomial. The polynomials are the pinhole, Kannala–Brandt4, rational pinhole, and double sphere. The header also contains the only copy of \(\partial \begin{bmatrix} u \\ v \end{bmatrix} / \partial \mathbf{x}_c\).

When `USE_CPP_CORE` is true, `pyslam.slam` imports the `cpp_core` classes. Those classes are the cameras, g2o edges, and gtsam factors in `cpp/core/camera/`. The entry points build them with `camera_from_config(config)`.

When `USE_CPP_CORE` is false, the code keeps the Python classes. Pinhole batch projection uses the original Numba path. Pinhole observations keep the original native g2o factors and the original native gtsam factors. Kannala–Brandt4, rational pinhole, and double sphere call the modules in `cpp/modules/camera_models/`. An import of `camera_models` does not load `cpp_core`, g2o, or gtsam.

| Module | Loaded when | Contents |
| --- | --- | --- |
| `camera_models` | A Python-core new-model camera is first used | Projection, unprojection, rig triangulation, `EpipolarMatcher.search_stereo_rig` |
| `camera_models_g2o` | The Python g2o optimizer builds a new-model edge | g2o edges only |
| `camera_models_gtsam` | The Python gtsam optimizer builds a new-model factor | gtsam factors only |
| `cpp_core` | `USE_CPP_CORE` is true | SLAM cameras, edges, and factors for that backend |

The camera object is `libcamera_model_core.so`. The two factor modules link that library. An edge and a factor then hold the same camera. The projection module built that camera. If the camera is already a `cpp_core` object, the edge and the factor stay in `cpp_core`. Those types store a `cpp_core` camera pointer. `pyslam/slam/camera_factors.py` selects the module from the given camera.

Batch projection, undistortion, unprojection, and rig triangulation write into NumPy buffers. The caller owns the buffers. The methods are `project_into` and the other `*_into` methods. The bindings take the arrays with `noconvert()`. A C-contiguous `float64` buffer is used in place. Stereo-rig search reads keypoints, octaves, octave scales, and descriptors in the same way. The keypoints are `float32`. The octaves are `int32`. The octave scales are `float32`. The function allocates the match-index lists after the search, when their length is known.

The rig geometry, the stereo-rig search, the g2o edges, and the gtsam factors are each written once. They are header-only templates on the camera type in `cpp/core/camera/`. `cpp_core` instantiates the templates for `pyslam::Camera`. `camera_models` instantiates the templates for `camera_models::CameraModel`. Each side adds only type aliases and bindings. The templates hold the camera with `CameraHandle`. A raw pointer does not own the camera. A `shared_ptr` owns the camera. Both ownership styles share one edge type.

The code does not include the g2o headers and the gtsam headers in the same translation unit. The g2o templates and the gtsam templates are in separate headers. `cpp_core` binds its edges and its factors in two source files. The C++ factor tests are split in the same way.

| Piece | Path |
| --- | --- |
| Projection math and Jacobians | `cpp/core/camera/camera_model_math.h` |
| Rig projection, rig triangulation, bearing-to-plane distance, stereo-rig search (shared) | `cpp/core/camera/camera_rig_geometry.h` |
| g2o edges (shared, no gtsam) | `cpp/core/camera/camera_projection_edges_g2o.h` |
| gtsam factors and `python_noise_model` (shared, no g2o) | `cpp/core/camera/camera_projection_factors_gtsam.h` |
| Camera reference held by edges and factors | `cpp/core/camera/camera_handle.h` |
| Python-core camera, shared by the factor modules | `cpp/modules/camera_models/camera_model.h`, `libcamera_model_core.so` |
| Python-core projection bindings | `cpp/modules/camera_models/camera_models_module.cpp` |
| Python-core g2o edges (aliases) | `cpp/modules/camera_models/camera_model_g2o_factors.h`, `camera_models_g2o_module.cpp` |
| Python-core gtsam factors (aliases) | `cpp/modules/camera_models/camera_model_gtsam_factors.h`, `camera_models_gtsam_module.cpp` |
| `cpp_core` camera classes | `cpp/core/camera/camera_models.h` |
| `cpp_core` g2o edges (aliases) | `cpp/core/camera/camera_edges_g2o.h`, `cpp/core/camera/py_module/camera_g2o_module.cpp` |
| `cpp_core` gtsam factors (aliases, stereo factor helpers) | `cpp/core/camera/camera_factors_gtsam.h`, `cpp/core/camera/py_module/camera_gtsam_module.cpp` |
| Triangulation-search epipolar mask (Python core) | `pyslam/slam/geometry_matchers.py` |
| Python classes and `camera_from_config` | `pyslam/slam/camera.py` |
| Edge and factor selection for the Python optimizers | `pyslam/slam/camera_factors.py` |

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

The rational example and the double-sphere example show only the fields of that model. Also set the dimensions, the intrinsics, and the frame rate from the first example. These examples show how you select a calibration. The integration limits above still apply.

In Python, `pyslam.slam.camera.camera_from_json(payload)` selects the serialized subclass. The function keeps a right camera and the extrinsics when the payload has them. `Frame.from_json` uses this factory. A constructor accepts a `Config` object. A constructor also accepts the serialized dictionary of that object.

## Tests

`cpp/core/slam/tests_cpp/test_camera_model_math.cpp` compares the analytic Jacobians with finite differences. `test_camera_models.cpp` checks round trips, undistortion, visibility, the bearing-to-plane pixel distance, and rig triangulation. The rig test includes a ray at more than 90 degrees. `test_camera_edges_g2o.cpp` checks that a pinhole edge agrees with the native g2o edge. It also checks the Kannala–Brandt edge Jacobians against finite differences. `test_camera_factors_gtsam.cpp` does the same checks for the gtsam factors. The two files are separate. Neither file includes both libraries.

`cpp/core/slam/tests_py/test_camera_models.py` checks the Python bindings, the OpenCV projection, and the calibrations in `settings/`. If a settings file omits the camera type, the camera must stay a linear pinhole. This rule applies when the distortion coefficients are not zero. The test also times pinhole batch projection against the Numba path. It compares a pinhole bundle adjustment with `EdgeSE3ProjectXYZ`. It checks `search_stereo_rig` on a synthetic KB4 rig. The rig has true matches and depths, off-curve distractors, and distractors beyond infinite depth. It also has an octave gap, ambiguous duplicates, and a one-to-one result. The test checks the Python bearing epipolar mask.

`tests/camera/test_camera_models_modules.py` covers the Python-core path (`USE_CPP_CORE=False`). An import of `camera_models`, `camera_models_g2o`, or `camera_models_gtsam` loads only the optimizer library of that module. A Python KB4 rig binds to a `camera_models` camera. `search_stereo_rig` recovers the synthetic matches and depths. The g2o pose edges and the gtsam resection factors move a perturbed pose back to the true pose. The check covers mono and stereo. The test also checks that you can clone a gtsam graph of these factors.

`tests/camera/test_camera_regressions.py` has more integration checks and pinhole compatibility checks. The checks cover 54 calibration and sensor combinations in both backends. They cover stereo disparity, RGB-D backprojection, undistortion, raw-pixel KD trees, and Python camera serialization. The six original defect reproductions are normal passing tests. No expected-failure decorator remains. A pinhole control checks the map-admission test fixture.

Run the commands from the repository root. Use the pySLAM Python environment. Use a rebuilt C++ core:

```bash
python cpp/core/slam/tests_py/test_camera_models.py
python tests/camera/test_camera_regressions.py -v
python tests/camera/test_tum_vi.py -v
python tests/camera/test_camera_models_modules.py -v
# Run just the original defect reproductions:
python tests/camera/test_camera_regressions.py TestKnownCameraDefects
```

The optional paired benchmark loads the pinhole implementation from a Git commit that you supply. It loads that implementation into a temporary module. The benchmark warms Numba. It alternates the measurement order. It records repeated samples and their medians. It checks numerical agreement. It writes JSON. It also compares the new Python g2o and GTSAM factor paths with the previous native factors. The comparison includes a pose-only solve with fixed 3D landmarks. The default regression budget is 20 percent. You can change the budget for a controlled benchmark machine. If the result is above the budget, the process returns exit status 1.

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

A timing ratio is the median of the ratios of adjacent candidate and baseline samples. The JSON also keeps the separate duration medians, their ratio, and all raw samples. You can use those values to inspect drift from CPU frequency or CPU load.

The Python camera timings compare real revisions. The C++ camera rows compare the current C++ code with the old Python and Numba implementation. Those rows give context. They are not gates for a revision regression. Do not use those rows to judge C++ speed against the old C++ build. The native factor benchmark compares the previous g2o and GTSAM factor implementations with the current factors. Component checks do not show that trajectory accuracy is unchanged. They do not show tracking loss. They do not show end-to-end throughput. Use the end-to-end replay below for those results.

### End-to-end replay and pinhole regression

`tests/camera/replay_sequence.py` replays a dataset section of `config.yaml` with a fixed procedure. The procedure uses ORB2 and synchronous local mapping. It does not run loop closure unless you pass `--loop-closure`. The script writes JSON with the tracking states, the track and map time of each frame, the keyframe count, and the map-point count. The JSON also has the ATE. Mono ATE uses a Sim(3) alignment. Every other sensor uses an SE(3) alignment. The script also writes an NPZ file with the trajectory and the raw timings. `--pyslam-root` imports another checkout. You can export a pinhole-only baseline from the last pinhole commit and build its C++ core. The same harness then replays that baseline:

```bash
git archive <pinhole-commit> | tar -x -C ../pyslam-baseline   # then build its C++ core
for root in ../pyslam-baseline .; do
  python tests/camera/replay_sequence.py --pyslam-root $root \
    --dataset KITTI_DATASET --base-path <kitti>/dataset --name 00 --sensor-type stereo \
    --settings settings/KITTI00-02.yaml --set is_color=false --frames 1500 \
    --output /tmp/kitti00_stereo_$(basename $root).json
done
```

Run the comparisons one after the other. Use a machine that does no other heavy work. Two SLAM processes at the same time compete for CPU and memory. That competition can change the timing ratios. The harness sets OpenCV to one thread. It sets the seeds of OpenCV, NumPy, and the Python random generators. The timings include synchronous local mapping. Do not compare these timings with tracking-only timings from a system that maps in another thread.

For a playback schedule, pass `--local-mapping threaded --loop-closure` to either replay command. These runs include asynchronous mapping and loop detection. They do not include GUI rendering or real-time pacing. Thread timing also makes the map less repeatable. Compare the same modes. Inspect the image-load time and the tracking time. The first frames can include Numba compilation. Use the per-frame NPZ timings to separate the startup cost from steady operation. TUM ATE uses the original motion-capture samples. It does not use frame associations. A ground-truth timestamp can repeat for consecutive images.

On Linux, the replay and the component benchmark share one process lock. A watchdog stops a run in two cases. It stops the run when the resident memory is above 8 GiB. It stops the run when the available host memory is below 8 GiB. Set these limits with `--max-rss-gib` and `--min-available-gib`. The replay checks memory at frame boundaries. It records sampled peak resident memory. These checks are safeguards. They are not operating-system memory limits. Run other tests, reference executables, and builds as separate jobs. An interrupted run is not a completed comparison.

For a memory investigation, `--trace-memory` prints Python allocation summaries every 200 frames. Tracing adds overhead. Do not use those timings in a throughput comparison.

For repeated comparisons, edit the dataset roots and the sequences in `tests/camera/pinhole_replay_cases.example.json`. Then run:

```bash
python tests/camera/run_pinhole_regression.py \
  --baseline-root ../pyslam-baseline-pinhole \
  --cases tests/camera/pinhole_replay_cases.example.json \
  --frames 300 --repeats 3 --output results/pinhole-regression
```

The example covers KITTI mono and stereo, EuRoC mono and stereo, and TUM mono and RGB-D. More datasets use the same manifest schema. This recorded-data sample is separate from the calibration tests. A pass does not validate every supported dataset. A pass does not validate loop closure on a full sequence. For a release check, increase the frame limit.

The runner alternates the baseline and candidate order between pairs. It saves each JSON file, each trajectory, and each log. The run fails in three cases. The median of the paired mean frame-time ratios is above 1.20. A pair loses more than two percentage points of tracking, trajectory, or motion-capture coverage. The ATE increases by more than the larger of 20 percent and 0.02 m. Set the timing budget with `--max-ratio`. A baseline from `git archive` is supported. If `.git` is absent, the source hash and the native binary hash identify the baseline. Compilation caches and replay logs stay in the output directory.

Replay ATE uses `tests/camera/trajectory_evaluation.py`. Mono alignment is a Sim(3). A metric sensor uses an SE(3) alignment. Interpolation needs the full motion-capture interval around the sample. That interval must be at most `--max-align-dt`. The tool does not extrapolate ground truth. The tool does not fill a longer gap. Fewer than three associations is an error. The report includes the association coverage and the alignment scale. The tests cover a known similarity transform, interpolation, sparse gaps, and input that is not valid or is degenerate. They also cover the separate regression gates:

```bash
python -m unittest discover -s tests/camera -p 'test_*.py'
```
