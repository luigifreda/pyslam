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
"""

import argparse
from enum import Enum
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from pyslam.config import Config
from pyslam.io.dataset_factory import dataset_factory


kRootDir = Path(__file__).parent.parent.parent
kDataDir = kRootDir / "data"
kModelsDir = kDataDir / "models"


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from exc


class YoloSegModel(Enum):
    YOLO26N = ("v8.4.0", "yolo26n-seg.pt")
    YOLO11N = ("v8.4.0", "yolo11n-seg.pt")
    YOLOV8N = ("v8.4.0", "yolov8n-seg.pt")

    @property
    def release_tag(self) -> str:
        return self.value[0]

    @property
    def filename(self) -> str:
        return self.value[1]


def _get_model_path(model: YoloSegModel) -> Path:
    kModelsDir.mkdir(parents=True, exist_ok=True)
    model_path = kModelsDir / model.filename
    if model_path.exists():
        return model_path

    import urllib.request

    url = (
        "https://github.com/ultralytics/assets/releases/"
        f"download/{model.release_tag}/{model.filename}"
    )
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _build_maps(result):
    if result.masks is None or result.boxes is None:
        return None, None

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    instance_map = np.zeros(masks.shape[1:], dtype=np.uint16)
    class_map = np.zeros(masks.shape[1:], dtype=np.uint8)
    for idx, (mask, cls_id) in enumerate(zip(masks, classes), start=1):
        instance_map[mask > 0.5] = np.uint16(idx)
        class_map[mask > 0.5] = np.uint8(cls_id + 1)

    return class_map, instance_map


def _colorize_map(gray_map, colormap):
    if gray_map is None:
        return None
    vis = gray_map
    if vis.max() > 0:
        vis = (vis.astype(np.float32) / vis.max() * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, colormap)


if __name__ == "__main__":
    _ensure_ultralytics()
    from ultralytics import YOLO  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(description="YOLO dataset segmentation demo")
    parser.add_argument(
        "--yolo-version",
        type=int,
        default=26,
        choices=[11, 26, 8],
        help="YOLO version number (11, 26, 8). Default: 26.",
    )
    args = parser.parse_args()

    version_map = {
        26: YoloSegModel.YOLO26N,
        11: YoloSegModel.YOLO11N,
        8: YoloSegModel.YOLOV8N,
    }

    config = Config()
    dataset = dataset_factory(config)

    model_path = _get_model_path(version_map[args.yolo_version])
    model = YOLO(str(model_path))

    window_tag = f"yolo{args.yolo_version}"
    cv2.namedWindow("image")
    cv2.namedWindow(f"{window_tag} Instance Segmentation Viz")
    cv2.namedWindow(f"{window_tag} class map")
    cv2.namedWindow(f"{window_tag} instance map")

    img_id = 0
    while True:
        img = None
        if dataset.is_ok:
            img = dataset.getImageColor(img_id)

        if img is None:
            cv2.waitKey(100)
            img_id += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(source=img_rgb, task="segment")
        first = results[0]

        class_map, instance_map = _build_maps(first)
        class_vis = _colorize_map(class_map, cv2.COLORMAP_VIRIDIS)
        inst_vis = _colorize_map(instance_map, cv2.COLORMAP_TURBO)

        instance_viz = first.plot()

        cv2.imshow("image", img)
        cv2.imshow(f"{window_tag} Instance Segmentation Viz", instance_viz)
        if class_vis is not None:
            cv2.imshow(f"{window_tag} class map", class_vis)
        if inst_vis is not None:
            cv2.imshow(f"{window_tag} instance map", inst_vis)

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

        img_id += 1
