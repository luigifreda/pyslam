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


kRootDir = Path(__file__).parent.parent.parent
kDataDir = kRootDir / "data"
kModelsDir = kDataDir / "models"
kOutputsDir = kDataDir / "outputs" / "yolo26"


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from exc


class YoloSegmentationModel(Enum):
    YOLO26N = ("v8.4.0", "yolo26n-seg.pt")
    YOLO11N = ("v8.4.0", "yolo11n-seg.pt")
    YOLOV8N = ("v8.4.0", "yolov8n-seg.pt")

    @property
    def release_tag(self) -> str:
        return self.value[0]

    @property
    def filename(self) -> str:
        return self.value[1]


def _get_model_path(model: YoloSegmentationModel) -> Path:
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


def run_instance_and_semantic_segmentation(
    image_source: str,
    model: YoloSegmentationModel = YoloSegmentationModel.YOLO26N,
):
    """Run one segmentation model and visualize instance + class id maps."""
    _ensure_ultralytics()
    from ultralytics import YOLO  # type: ignore[import-not-found]
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    model_path = _get_model_path(model)
    model = YOLO(str(model_path))
    results = model(source=image_source, task="segment")

    first = results[0]
    if first.masks is None or first.boxes is None:
        return None, None

    masks = first.masks.data.cpu().numpy()
    classes = first.boxes.cls.cpu().numpy().astype(np.int32)

    instance_map = np.zeros(masks.shape[1:], dtype=np.uint16)
    class_map = np.zeros(masks.shape[1:], dtype=np.uint8)
    for idx, (mask, cls_id) in enumerate(zip(masks, classes), start=1):
        instance_map[mask > 0.5] = np.uint16(idx)
        class_map[mask > 0.5] = np.uint8(cls_id + 1)

    instance_viz = first.plot()
    window_tag = f"YOLO{model.name[4:]}"
    cv2.imshow(f"{window_tag} Instance Segmentation Viz", instance_viz)

    class_vis = class_map
    if class_vis.max() > 0:
        class_vis = (class_vis.astype(np.float32) / class_vis.max() * 255).astype(np.uint8)
    class_vis = cv2.applyColorMap(class_vis, cv2.COLORMAP_VIRIDIS)

    inst_vis = instance_map
    if inst_vis.max() > 0:
        inst_vis = (inst_vis.astype(np.float32) / inst_vis.max() * 255).astype(np.uint8)
    inst_vis = cv2.applyColorMap(inst_vis, cv2.COLORMAP_TURBO)

    cv2.imshow(f"{window_tag} Semantic Class Map", class_vis)
    cv2.imshow(f"{window_tag} Instance ID Map", inst_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return class_map, instance_map


if __name__ == "__main__":
    # Example from the Ultralytics README, using a YOLO26 model on an image.
    # https://github.com/ultralytics/ultralytics/blob/main/README.md
    parser = argparse.ArgumentParser(description="YOLO image segmentation demo")
    parser.add_argument(
        "--yolo-version",
        type=int,
        default=26,
        choices=[11, 26, 8],
        help="YOLO version number (11, 26, 8). Default: 26.",
    )
    args = parser.parse_args()

    version_map = {
        26: YoloSegmentationModel.YOLO26N,
        11: YoloSegmentationModel.YOLO11N,
        8: YoloSegmentationModel.YOLOV8N,
    }

    image = "https://ultralytics.com/images/bus.jpg"
    run_instance_and_semantic_segmentation(image, model=version_map[args.yolo_version])
