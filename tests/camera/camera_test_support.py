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
* Dataset calibration fixtures; no images, downloads, or user config changes.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyslam.config  # Configure native module search paths.
import numpy as np
from pyslam.io.dataset_types import SensorType


def read_settings(path):
    settings = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if not key.startswith("Camera.") and key not in ("ThDepth", "DepthMapFactor"):
            continue
        try:
            settings[key] = float(value.strip())
        except ValueError:
            settings[key] = value.strip()
    return settings


class SettingsConfig:
    def __init__(self, settings, sensor="mono"):
        self.cam_settings = dict(settings)
        self.dataset_settings = {"sensor_type": sensor}
        self.sensor_type = {
            "mono": SensorType.MONOCULAR,
            "stereo": SensorType.STEREO,
            "rgbd": SensorType.RGBD,
        }[sensor]

    @property
    def DistCoef(self):
        if self.sensor_type == SensorType.STEREO:
            return np.zeros(5)
        return np.array([
            self.cam_settings.get("Camera." + key, 0.0)
            for key in ("k1", "k2", "p1", "p2", "k3")
        ])

    def to_json(self):
        return {"cam_settings": self.cam_settings, "dataset_settings": self.dataset_settings}


def calibration_cases():
    for path in sorted((ROOT / "settings").glob("*.yaml")):
        settings = read_settings(path)
        if "Camera.fx" not in settings:
            continue
        # A virtual stereo observation requires a calibrated bf, also in RGB-D.
        sensors = ("mono", "stereo", "rgbd") if "Camera.bf" in settings else ("mono",)
        for sensor in sensors:
            yield path.name, sensor, SettingsConfig(settings, sensor)


def sample_points(count=512):
    rng = np.random.default_rng(413)
    return np.ascontiguousarray(np.column_stack((
        rng.uniform(-1.0, 1.0, count),
        rng.uniform(-0.5, 0.5, count),
        rng.uniform(2.0, 20.0, count),
    )))


def model_config(model="kannala_brandt4", sensor="mono"):
    return SettingsConfig({
        "Camera.type": model,
        "Camera.width": 640, "Camera.height": 480,
        "Camera.fx": 300.0, "Camera.fy": 310.0,
        "Camera.cx": 320.0, "Camera.cy": 240.0,
        "Camera.fps": 30, "Camera.bf": 40.0,
        "ThDepth": 40.0, "DepthMapFactor": 1000.0,
        "Camera.k1": -0.02, "Camera.k2": 0.005,
        "Camera.k3": -0.001, "Camera.k4": 0.0002,
        "Camera.xi": -0.2, "Camera.alpha": 0.6,
    }, sensor)
