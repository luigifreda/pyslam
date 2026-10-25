"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*
* Edge/factor selection for the Python optimizers (USE_CPP_CORE=False).
*
* Pinhole observations keep the original Python g2o edges and gtsam factors.
* Those computations stay here and do not import cpp_core. USE_CPP_CORE=True
* selects cpp_core.OptimizerG2o / OptimizerGTSAM instead of this module.
*
* KB4, RadTan8, and double sphere use edges from camera_models_g2o and factors
* from camera_models_gtsam. Those modules share the camera_models camera and
* are loaded only by the optimizer that needs them. A camera that is already a
* cpp_core object keeps cpp_core factors, because those edges store a cpp_core
* camera pointer.
"""

import importlib

import numpy as np
import sys

import gtsam


def projects_raw_keypoints(camera):
    query = getattr(camera, "projects_raw_keypoints", None)
    if query is None:
        return False
    return bool(query() if callable(query) else query)


def reprojection_measurement(frame, idx, stereo_observation):
    """Pixel compared with project(): raw for nonlinear models, undistorted for pinhole."""
    if not projects_raw_keypoints(frame.camera):
        uv = frame.kpsu[idx]
    else:
        uv = frame.projection_keypoints()[idx]
    return np.asarray([uv[0], uv[1]], dtype=np.float64)


def bound_camera(camera):
    """Camera object whose module owns the projection factors.

    Python nonlinear cameras return their camera_models instance. An object
    that is already a cpp_core camera stays a cpp_core camera.
    """
    if type(camera).__module__ in ("cpp_core", "camera_models"):
        return camera
    if hasattr(camera, "_cpp_camera"):
        return camera._cpp_camera()
    raise TypeError(
        "Nonlinear projection factors require a camera_models camera. "
        "Pinhole observations use the original g2o and gtsam factors."
    )


def projection_module(camera):
    native = bound_camera(camera)
    return importlib.import_module(type(native).__module__)


def _factor_module(camera, module_name):
    native = bound_camera(camera)
    if type(native).__module__ == "cpp_core":
        return importlib.import_module("cpp_core")
    return importlib.import_module(module_name)


def cpp_camera_of(camera):
    """Camera passed to a projection edge. The caller must keep it alive."""
    return bound_camera(camera)


def g2o_camera_project_edge(camera):
    if not projects_raw_keypoints(camera):
        # Original Python pinhole edge. cpp_core supplies its own when USE_CPP_CORE is True.
        import g2o
        edge = g2o.EdgeSE3ProjectXYZ()
        edge.fx, edge.fy, edge.cx, edge.cy = camera.fx, camera.fy, camera.cx, camera.cy
        return edge
    cpp_camera = cpp_camera_of(camera)
    edge = _factor_module(cpp_camera, "camera_models_g2o").EdgeCameraProjectXYZ(cpp_camera)
    edge._cpp_camera = cpp_camera
    return edge


def g2o_camera_project_pose_edge(camera):
    if not projects_raw_keypoints(camera):
        # Original Python pose edge. cpp_core.OptimizerG2o replaces it when USE_CPP_CORE is True.
        import g2o
        edge = g2o.EdgeSE3ProjectXYZOnlyPose()
        edge.fx, edge.fy, edge.cx, edge.cy = camera.fx, camera.fy, camera.cx, camera.cy
        return edge
    cpp_camera = cpp_camera_of(camera)
    edge = _factor_module(cpp_camera, "camera_models_g2o").EdgeCameraProjectXYZOnlyPose(cpp_camera)
    edge._cpp_camera = cpp_camera
    return edge


def camera_resectioning_factor(noise_model, pose_key, camera, measured, point, calibration=None):
    if not projects_raw_keypoints(camera):
        import gtsam_factors
        calib = calibration if calibration is not None else gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
        if sys.platform == "darwin":
            from .optimizer_gtsam import resectioning_mono_factor_py
            return resectioning_mono_factor_py(noise_model, pose_key, calib, measured, point)
        factor = gtsam_factors.ResectioningFactor(noise_model, pose_key, calib, measured, point)
        return factor, factor
    cpp_camera = cpp_camera_of(camera)
    factor = _factor_module(cpp_camera, "camera_models_gtsam").CameraResectioningFactor(
        noise_model, pose_key, cpp_camera, measured, point
    )
    return factor, factor


def camera_projection_factor(measured, noise_model, pose_key, point_key, camera):
    """Original Python GTSAM factor for pinhole. cpp_core is a separate optimizer."""
    if not projects_raw_keypoints(camera):
        import gtsam_factors
        calib = gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
        return gtsam_factors.WeightedGenericProjectionFactorCal3_S2(
            measured, noise_model, pose_key, point_key, calib
        )
    cpp_camera = cpp_camera_of(camera)
    return _factor_module(cpp_camera, "camera_models_gtsam").CameraProjectionFactor(
        measured, noise_model, pose_key, point_key, cpp_camera
    )


def g2o_stereo_camera_edge(camera, pose_only=False):
    import g2o
    if projects_raw_keypoints(camera):
        cpp_camera = cpp_camera_of(camera)
        module = _factor_module(cpp_camera, "camera_models_g2o")
        cls = module.EdgeStereoCameraProjectXYZOnlyPose if pose_only else module.EdgeStereoCameraProjectXYZ
        edge = cls(cpp_camera)
    else:
        # Original Python stereo edge for the rectified pinhole.
        cls = g2o.EdgeStereoSE3ProjectXYZOnlyPose if pose_only else g2o.EdgeStereoSE3ProjectXYZ
        edge = cls()
    return edge


def stereo_camera_projection_factor(measured, noise, pose_key, point_key, camera):
    if projects_raw_keypoints(camera):
        cpp_camera = cpp_camera_of(camera)
        return _factor_module(cpp_camera, "camera_models_gtsam").StereoCameraProjectionFactor(
            measured, noise, pose_key, point_key, cpp_camera
        )
    import gtsam_factors
    K = gtsam.Cal3_S2Stereo(camera.fx, camera.fy, 0.0, camera.cx, camera.cy, camera.b)
    return gtsam_factors.WeightedGenericStereoProjectionFactor3D(measured, noise, pose_key, point_key, K)


def stereo_camera_resectioning_factor(noise, pose_key, camera, measured, point, calibration=None):
    if projects_raw_keypoints(camera):
        cpp_camera = cpp_camera_of(camera)
        factor = _factor_module(cpp_camera, "camera_models_gtsam").StereoCameraResectioningFactor(
            noise, pose_key, cpp_camera, measured, point
        )
    else:
        import gtsam_factors
        K = calibration if calibration is not None else gtsam.Cal3_S2Stereo(camera.fx, camera.fy, 0.0, camera.cx, camera.cy, camera.b)
        if sys.platform == "darwin":
            from .optimizer_gtsam import resectioning_stereo_factor_py
            return resectioning_stereo_factor_py(noise, pose_key, K, measured, point)
        factor = gtsam_factors.ResectioningFactorStereo(noise, pose_key, K, measured, point)
    return factor, factor
