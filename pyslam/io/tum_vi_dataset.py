"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
* Copyright (C) 2024 Anathonic <anathonic@protonmail.com>
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
* TUM-VI visual mono/stereo, with native KB4 or cached fisheye rectification.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from .dataset import Dataset
from .dataset_types import DatasetType, SensorType, DatasetEnvironmentType
from .image_prefetcher import make_stream


def camera_records(sequence, camera):
    """Keep integer nanoseconds until association; never pair CSV row numbers."""
    folder = Path(sequence) / "mav0" / camera
    manifest = folder / "data.csv"
    records = {}
    if manifest.exists():
        with manifest.open() as handle:
            for row in csv.reader(handle):
                if not row or row[0].strip().startswith("#"):
                    continue
                timestamp = int(row[0])
                filename = row[1].strip() if len(row) > 1 else f"{timestamp}.png"
                if timestamp in records:
                    raise ValueError(f"Duplicate camera timestamp in {manifest}: {timestamp}")
                records[timestamp] = folder / "data" / filename
    else:
        records = {int(path.stem): path for path in (folder / "data").glob("*.png")}
    return records


def image_records(sequence, selected=None, stereo=True):
    left = camera_records(sequence, "cam0")
    right = camera_records(sequence, "cam1") if stereo else {}
    timestamps = set(left).intersection(right) if stereo else set(left)
    if selected:
        file = Path(selected).expanduser()
        if not file.is_absolute():
            file = Path(sequence) / file
        timestamps.intersection_update(
            int(line.split()[0])
            for line in file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return left, right, sorted(timestamps)


def stereo_records(sequence, selected=None):
    return image_records(sequence, selected, stereo=True)


def prepare_calibration(config):
    if hasattr(config, "_tum_vi_maps"):
        return config._tum_vi_maps
    settings = config.cam_settings
    stereo = str(config.dataset_settings.get("sensor_type", "stereo")).lower() == "stereo"
    mode = str(config.dataset_settings.get("stereo_mode", "rectified")).lower()
    if mode not in ("native", "rectified"):
        raise ValueError("TUM-VI stereo_mode must be native or rectified")
    size = (int(settings["Camera.width"]), int(settings["Camera.height"]))

    def calibration(index):
        prefix = f"Camera{index}."
        fx, fy, cx, cy = [float(settings[prefix + key]) for key in ("fx", "fy", "cx", "cy")]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        D = np.array([settings[prefix + f"k{i}"] for i in range(1, 5)], dtype=np.float64)
        return K, D

    Kl, Dl = calibration(1)
    Kr, Dr = calibration(2)
    # PLVS2 supplies T_c1_c2 (right -> left); OpenCV and this rig use its inverse.
    Tlr = np.asarray(settings["Stereo.T_c1_c2"], dtype=np.float64)
    if Tlr.shape != (4, 4) or not np.isfinite(Tlr).all() or not np.allclose(Tlr[3], [0, 0, 0, 1]):
        raise ValueError("Stereo.T_c1_c2 must be a finite homogeneous 4x4 transform")
    rotation = Tlr[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0:
        raise ValueError("Stereo.T_c1_c2 must contain a proper rotation")
    Trl = np.linalg.inv(Tlr)
    baseline = float(np.linalg.norm(Trl[:3, 3]))
    if baseline <= 0:
        raise ValueError("TUM-VI stereo calibration has a zero baseline")
    maps = None
    Rl = np.eye(3)
    if mode == "rectified":
        fov_scale = float(settings.get("Camera.fovScale", 0.7))
        if not np.isfinite(fov_scale) or fov_scale <= 0:
            raise ValueError("Camera.fovScale must be positive")
        Rl, Rr, Pl, Pr, _Q = cv2.fisheye.stereoRectify(
            Kl,
            Dl,
            Kr,
            Dr,
            size,
            Trl[:3, :3],
            Trl[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=size,
            balance=float(settings.get("Camera.rectifyBalance", 1.0)),
            fov_scale=fov_scale,
        )
        maps = (
            cv2.fisheye.initUndistortRectifyMap(Kl, Dl, Rl, Pl, size, cv2.CV_16SC2),
            cv2.fisheye.initUndistortRectifyMap(Kr, Dr, Rr, Pr, size, cv2.CV_16SC2),
        )
        K = Pl[:3, :3]
        config._tum_vi_rectification = (Rl, Rr, Pl, Pr)
        bf = -float(Pr[0, 3])
        if bf <= 0:
            raise ValueError("TUM-VI calibration must place cam1 to the right of cam0")
        settings["Camera.type"] = "pinhole"
        distortion = np.zeros(4)
        settings.pop("Stereo.right", None)
    else:
        K, distortion, bf = Kl, Dl, float(Kl[0, 0] * baseline)
        settings["Camera.type"] = "kannala_brandt4"
        right = {
            "Camera.type": "kannala_brandt4",
            "Camera.width": size[0],
            "Camera.height": size[1],
            "Camera.fx": float(Kr[0, 0]),
            "Camera.fy": float(Kr[1, 1]),
            "Camera.cx": float(Kr[0, 2]),
            "Camera.cy": float(Kr[1, 2]),
            "Camera.fps": settings.get("Camera.fps", 20),
        }
        right.update({f"Camera.k{i + 1}": float(value) for i, value in enumerate(Dr)})
        if stereo:
            settings["Stereo.right"] = right
            settings["Stereo.T_right_left"] = Trl.tolist()
        else:
            settings.pop("Stereo.right", None)
            settings.pop("Stereo.T_right_left", None)
    for name, value in zip(("fx", "fy", "cx", "cy"), (K[0, 0], K[1, 1], K[0, 2], K[1, 2])):
        settings["Camera." + name] = float(value)
    for i, value in enumerate(distortion):
        settings[f"Camera.k{i + 1}"] = float(value)
    settings["Camera.p1"] = settings["Camera.p2"] = 0.0
    settings["Camera.bf"] = bf
    settings["ThDepth"] = float(settings.get("Stereo.ThDepth", 40.0))
    for cached in ("_K", "_Kinv", "_DistCoef", "_bf", "_cam_stereo_settings"):
        config.__dict__.pop(cached, None)
    if "TUMVI.T_body_left" in settings:
        T = np.asarray(settings["TUMVI.T_body_left"], dtype=np.float64).copy()
        if (
            T.shape != (4, 4)
            or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(T[:3, :3]) <= 0
        ):
            raise ValueError("TUMVI.T_body_left must be a finite rigid 4x4 transform")
        T[:3, :3] = T[:3, :3] @ Rl.T
        config.dataset_settings["T_body_camera"] = T.tolist()
    config._tum_vi_maps = maps
    return maps


class TumViDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        sensor_type=SensorType.STEREO,
        associations=None,
        start_frame_id=0,
        config=None,
    ):
        if sensor_type not in (SensorType.MONOCULAR, SensorType.STEREO):
            raise ValueError("TUM-VI loader requires mono or stereo sensor_type")
        super().__init__(
            path,
            name,
            sensor_type,
            config.cam_settings.get("Camera.fps", 20),
            associations,
            start_frame_id,
            DatasetType.TUM_VI,
            DatasetEnvironmentType.INDOOR,
        )
        # Monocular init forces median depth to 1, so the default 0.1 frustum covers the
        # local cloud. 0.02 also places the follow camera near Viewer.ViewpointZ (-1.8).
        self.scale_viewer_3d = 0.02 if sensor_type == SensorType.MONOCULAR else 0.1
        self.maps = prepare_calibration(config)
        self.clahe = (
            tuple(cv2.createCLAHE(3.0, (8, 8)) for _ in range(2))
            if config.dataset_settings.get("clahe", True)
            else None
        )
        self.sequence = Path(path) / name
        stereo = sensor_type == SensorType.STEREO
        left, right, self.timestamps_ns = image_records(
            self.sequence, config.dataset_settings.get("timestamps"), stereo=stereo
        )
        if not self.timestamps_ns:
            raise ValueError(
                f"No TUM-VI {'stereo pairs' if stereo else 'images'} in {self.sequence}"
            )
        self.timestamps = np.asarray(self.timestamps_ns, dtype=np.float64) * 1e-9
        self.left = [left[t] for t in self.timestamps_ns]
        self.right = [right[t] for t in self.timestamps_ns] if stereo else []
        self.num_frames = self.max_frame_id = len(self.timestamps)
        self.size = (
            int(config.cam_settings["Camera.width"]),
            int(config.cam_settings["Camera.height"]),
        )
        streams = [
            make_stream(
                "image",
                self.left,
                imread_flag=cv2.IMREAD_GRAYSCALE,
                ops=self._prefetch_ops(0),
                expected_size=self.size,
            )
        ]
        if stereo:
            streams.append(
                make_stream(
                    "image_right",
                    self.right,
                    imread_flag=cv2.IMREAD_GRAYSCALE,
                    ops=self._prefetch_ops(1),
                    expected_size=self.size,
                )
            )
        self.start_prefetch(streams)

    def _prefetch_ops(self, side):
        ops = []
        if self.clahe is not None:
            ops.append({"type": "clahe", "clip": 3.0, "tile": 8})
        if self.maps is not None:
            map1, map2 = self.maps[side]
            ops.append({"type": "remap", "map1": map1, "map2": map2})
        ops.append({"type": "gray_to_rgb"})
        return ops

    def _image(self, frame_id, side):
        if frame_id < 0 or frame_id >= self.num_frames:
            return None
        image = cv2.imread(str((self.left, self.right)[side][frame_id]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            self.is_ok = False
            return None
        if image.shape[::-1] != self.size:
            raise ValueError(
                f"TUM-VI image size {image.shape[::-1]} does not match calibration {self.size}"
            )
        if self.clahe is not None:
            image = self.clahe[side].apply(image)
        if self.maps is not None:
            image = cv2.remap(image, *self.maps[side], cv2.INTER_LINEAR)
        if side == 0:
            self._timestamp = self.timestamps[frame_id]
            self._next_timestamp = (
                self.timestamps[frame_id + 1]
                if frame_id + 1 < self.num_frames
                else self._timestamp + self.Ts
            )
        return np.ascontiguousarray(image)

    def getImage(self, frame_id):
        return self._image(frame_id, 0)

    def getImageRight(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None
        return self._image(frame_id, 1)
