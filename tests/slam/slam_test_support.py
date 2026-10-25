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
* Headless SLAM on the synthetic sequence of compare_cpp_python_trajectory.py.
*
* A textured plane at 2 m in front of the camera; frame i translates the camera by
* STEP_M * i along x. Depth is exact. Whatever core PYSLAM_USE_CPP selects is used.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp" / "core" / "slam" / "tests_py"))

import pyslam.config  # noqa: E402,F401  (native library paths)
import numpy as np  # noqa: E402
from compare_cpp_python_trajectory import _frame, _make_camera, _texture  # noqa: E402
from pyslam.config_parameters import Parameters  # noqa: E402
from pyslam.io.dataset_types import SensorType  # noqa: E402
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs  # noqa: E402
from pyslam.slam import USE_CPP, Slam  # noqa: E402

STEP_M = 0.02

HEADLESS_PARAMETERS = {
    "kUseLoopClosing": False,
    "kDoVolumetricIntegration": False,
    "kDoSparseSemanticMappingAndSegmentation": False,
    "kUseLargeWindowBA": False,
    "kUseParallelProcessLBA": False,
    "kLocalMappingParallelKpsMatching": False,
    "kUseMotionBlurDection": False,
}


def make_slam(test_case, sensor_type=SensorType.RGBD, threaded=False):
    """Build a headless Slam and restore Parameters when the test (or class) ends."""
    overrides = dict(HEADLESS_PARAMETERS, kLocalMappingOnSeparateThread=threaded)
    saved = {name: getattr(Parameters, name) for name in overrides}
    for name, value in overrides.items():
        setattr(Parameters, name, value)
    tracker = dict(FeatureTrackerConfigs.ORB2, deterministic=True, num_features=800)
    slam = Slam(_make_camera(USE_CPP), tracker, None, None, sensor_type, headless=True)

    def cleanup():
        slam.quit()
        for name, value in saved.items():
            setattr(Parameters, name, value)

    if isinstance(test_case, type):
        test_case.addClassCleanup(cleanup)
    else:
        test_case.addCleanup(cleanup)
    return slam


def init_feature_tracker(num_features=200):
    """MapPoint and the matchers read scale levels and descriptor norms from FeatureTrackerShared."""
    from pyslam.local_features.feature_tracker import feature_tracker_factory
    from pyslam.slam.feature_tracker_shared import FeatureTrackerShared

    config = dict(FeatureTrackerConfigs.ORB2, deterministic=True, num_features=num_features)
    FeatureTrackerShared.set_feature_tracker(feature_tracker_factory(**config), force=True)


def track_sequence(slam, frame_ids, sensor_type=SensorType.RGBD, texture=None):
    """Track one image per entry of frame_ids; return a record per tracked image."""
    texture = _texture() if texture is None else texture
    records = []
    for img_id, frame_id in enumerate(frame_ids):
        image, depth = _frame(texture, frame_id)
        if sensor_type == SensorType.MONOCULAR:
            depth = None
        slam.track(image, None, depth, img_id, float(img_id))
        tracking = slam.tracking
        records.append({
            "state": tracking.state.name,
            "num_keyframes": int(slam.map.num_keyframes()),
            "num_points": int(slam.map.num_points()),
            "position": None if tracking.cur_t is None else np.asarray(tracking.cur_t, float).ravel(),
            "true_x": STEP_M * frame_id,
        })
    return records
