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

import cv2
import supervision as sv
from rfdetr import RFDETRSegMedium
from rfdetr.main import HOSTED_MODELS
from rfdetr.util.files import download_file
from pathlib import Path

from pyslam.config import Config
from pyslam.io.dataset_factory import dataset_factory


kRootDir = Path(__file__).parent.parent.parent
kDataDir = kRootDir / "data"
kModelsDir = kDataDir / "models"

if __name__ == "__main__":
    config = Config()
    dataset = dataset_factory(config)

    rf_detr_threshold = 0.3

    models_dir = Path(kModelsDir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "rf-detr-seg-medium.pt"
    weights_path = models_dir / weights_name
    if not weights_path.exists():
        download_file(HOSTED_MODELS[weights_name], str(weights_path))

    model = RFDETRSegMedium(pretrain_weights=str(weights_path))

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    cv2.namedWindow("image")
    cv2.namedWindow("rfdetr masks")

    img_id = 0
    while True:
        img = None
        if dataset.is_ok:
            img = dataset.getImageColor(img_id)

        if img is None:
            cv2.waitKey(100)
            img_id += 1
            continue

        # RF-DETR expects RGB input.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = model.predict(img_rgb, threshold=rf_detr_threshold)
        labels = [
            str(model.class_names.get(class_id, class_id)) for class_id in detections.class_id
        ]

        annotated = mask_annotator.annotate(img_rgb, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        cv2.imshow("image", img)
        cv2.imshow("rfdetr masks", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:
            break

        img_id += 1
