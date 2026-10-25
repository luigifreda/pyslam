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

from pathlib import Path

import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegMedium
from rfdetr.main import HOSTED_MODELS
from rfdetr.util.files import download_file
from rfdetr.util.coco_classes import COCO_CLASSES


kRootDir = Path(__file__).parent.parent.parent
kDataDir = kRootDir / "data"
kModelsDir = kDataDir / "models"

if __name__ == "__main__":
    models_dir = Path(kModelsDir)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "rf-detr-seg-medium.pt"
    weights_path = models_dir / weights_name
    if not weights_path.exists():
        download_file(HOSTED_MODELS[weights_name], str(weights_path))

    model = RFDETRSegMedium(pretrain_weights=str(weights_path))

    image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
    # image = Image.open(requests.get("https://ultralytics.com/images/zidane.jpg", stream=True).raw)
    detections = model.predict(image, threshold=0.4)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

    annotated_image = sv.MaskAnnotator().annotate(image, detections)
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    sv.plot_image(annotated_image)
