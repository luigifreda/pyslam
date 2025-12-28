"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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

from .semantic_mapping_base import SemanticMappingType
from .semantic_types import SemanticFeatureType
from .semantic_mapping_configs import SemanticMappingConfigs
from .semantic_mapping_shared import SemanticMappingShared
from .semantic_types import SemanticDatasetType

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Printer


def evaluate_semantic_mapping(slam, dataset, metrics_save_dir):
    if (
        Parameters.kDoSparseSemanticMappingAndSegmentation
        and slam.semantic_mapping.semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY
        and dataset.has_gt_semantics
        and slam.semantic_mapping.semantic_mapping_type == SemanticMappingType.DENSE
    ):
        Printer.green("Evaluating semantic mapping...")
        # Get all the KFs
        keyframes = slam.map.get_keyframes()
        Printer.green(f"Number of keyframes: {len(keyframes)}")

        labels_2d = []
        labels_3d = []
        gt_labels = []
        total_mps = 0
        # Get all the final MPs that project on it
        for kf in keyframes:
            if kf.kps_sem is None:
                Printer.yellow(f"Keyframe {kf.id} has no semantics!")
                continue
            if kf.points is None:
                Printer.yellow(f"Keyframe {kf.id} has no points!")
                continue

            semantic_gt = dataset.getSemanticGroundTruth(kf.id)

            # Get the semantic_des of projected points
            points = kf.get_points()
            total_mps += len(points)
            # Get the per-frame gt semantic label for projected MPs
            for idx, kp in enumerate(kf.kps):
                if (
                    points[idx] is not None
                    and points[idx].semantic_des is not None
                    and kf.kps_sem[idx] is not None
                ):
                    gt_kf_label = semantic_gt[int(kp[1]), int(kp[0])]
                    # Filter out ignore-labels
                    if dataset.ignore_label != None and gt_kf_label == dataset.ignore_label:
                        continue
                    gt_labels.append(gt_kf_label)
                    if SemanticMappingShared.semantic_feature_type == SemanticFeatureType.LABEL:
                        labels_2d.append(kf.kps_sem[idx])
                        labels_3d.append(points[idx].semantic_des)
                    elif (
                        SemanticMappingShared.semantic_feature_type
                        == SemanticFeatureType.PROBABILITY_VECTOR
                    ):
                        labels_2d.append(np.argmax(kf.kps_sem[idx]))
                        labels_3d.append(np.argmax(points[idx].semantic_des))

            # For debugging:
            # Recover image
            # rgb_img = dataset.getImageColor(kf.id)
            # cv2.imshow('rgb', rgb_img)
            # semantic_gt_color = SemanticMappingShared.sem_img_to_rgb(semantic_gt, bgr=True)
            # cv2.imshow('semantic_gt', semantic_gt_color)
            # Get the predicted semantic label for the MP projection (baseline)
            # predicted_semantics = slam.semantic_mapping.semantic_segmentation.infer(rgb_img)
            # print(f"Predicted labels: {np.unique(predicted_semantics)}")
            # predicted_semantics_color = SemanticMappingShared.sem_img_to_rgb(predicted_semantics, bgr=True)
            # cv2.imshow('predicted_semantics', predicted_semantics_color)
            # cv2.waitKey(0)
        Printer.orange(f"Number of projected MPs: {len(labels_2d)}")
        Printer.orange(f"Number of projected MPs (3D): {len(labels_3d)}")
        Printer.orange(f"Number of GT MPs: {len(gt_labels)}")
        Printer.orange(f"Number of evaluated MPs: {total_mps}")
        Printer.orange(f"Number of evaluated KFs: {len(keyframes)}")
        from sklearn.metrics import (
            classification_report,
            accuracy_score,
            confusion_matrix,
            ConfusionMatrixDisplay,
            precision_recall_fscore_support,
        )

        # Class labels and names
        num_classes = dataset.num_labels
        labels_range = range(num_classes)
        labels_names = [str(i) for i in labels_range]

        # Determine which labels are actually present in the GT
        present_labels = sorted(set(gt_labels))  # list of int
        Printer.blue(f"Evaluating only on present GT labels: {present_labels}")

        # --- Baseline (2D) ---
        confusion_matrix_base = confusion_matrix(gt_labels, labels_2d, labels=labels_range)
        overall_accuracy_2d = accuracy_score(gt_labels, labels_2d)
        Printer.green(f"Overall Accuracy 2D: {overall_accuracy_2d:.4f}")

        # Macro average (only on present labels)
        report_2d = classification_report(
            gt_labels, labels_2d, labels=present_labels, zero_division=0, output_dict=True
        )
        macro_avg_2d = report_2d["macro avg"]
        Printer.green(
            f"2D Macro Avg: precision={macro_avg_2d['precision']:.4f}, recall={macro_avg_2d['recall']:.4f}, f1-score={macro_avg_2d['f1-score']:.4f}"
        )

        # Micro average
        precision_2d, recall_2d, f1_2d, _ = precision_recall_fscore_support(
            gt_labels, labels_2d, average="micro", zero_division=0
        )
        Printer.green(
            f"2D Micro Avg: precision={precision_2d:.4f}, recall={recall_2d:.4f}, f1-score={f1_2d:.4f}"
        )

        # Confusion matrix - 2D
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_base, display_labels=labels_names
        )
        fig, ax = plt.subplots(figsize=(24, 18))
        cm_display.plot(ax=ax, xticks_rotation=90)
        plt.savefig(os.path.join(metrics_save_dir, "confusion_matrix_est2d.png"), dpi=300)

        # --- 3D Projection ---
        confusion_matrix_proj = confusion_matrix(gt_labels, labels_3d, labels=labels_range)
        overall_accuracy_3d = accuracy_score(gt_labels, labels_3d)
        Printer.green(f"Overall Accuracy 3D: {overall_accuracy_3d:.4f}")

        # Macro average (only on present labels)
        report_3d = classification_report(
            gt_labels, labels_3d, labels=present_labels, zero_division=0, output_dict=True
        )
        macro_avg_3d = report_3d["macro avg"]
        Printer.green(
            f"3D Macro Avg: precision={macro_avg_3d['precision']:.4f}, recall={macro_avg_3d['recall']:.4f}, f1-score={macro_avg_3d['f1-score']:.4f}"
        )

        # Micro average
        precision_3d, recall_3d, f1_3d, _ = precision_recall_fscore_support(
            gt_labels, labels_3d, average="micro", zero_division=0
        )
        Printer.green(
            f"3D Micro Avg: precision={precision_3d:.4f}, recall={recall_3d:.4f}, f1-score={f1_3d:.4f}"
        )

        # Confusion matrix - 3D
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix_proj, display_labels=labels_names
        )
        fig, ax = plt.subplots(figsize=(24, 18))
        cm_display.plot(ax=ax, xticks_rotation=90)
        plt.savefig(os.path.join(metrics_save_dir, "confusion_matrix_est3d.png"), dpi=300)

        semantic_metrics_file_path = os.path.join(metrics_save_dir, "semantic_metrics_info.txt")
        with open(semantic_metrics_file_path, "w") as f:
            f.write("Evaluated labels: " + str(present_labels) + "\n")
            f.write(f"Feature type: {slam.semantic_mapping.semantic_feature_type}\n")
            f.write(f"Number of KFs: {len(keyframes)}\n")
            f.write(f"Number of MPs: {total_mps}\n")
            f.write(f"Number of GT labels {len(gt_labels)}\n")
            f.write(f"Number of estimated labels 2D: {len(labels_2d)}\n")
            f.write(f"Number of estimated labels 3D: {len(labels_3d)}\n")
            # --- 2D Metrics ---
            f.write("=== 2D Semantic Evaluation ===\n")
            f.write(f"Accuracy: {overall_accuracy_2d:.4f}\n")
            f.write(f"Micro Precision: {precision_2d:.4f}\n")
            f.write(f"Micro Recall:    {recall_2d:.4f}\n")
            f.write(f"Micro F1-score:  {f1_2d:.4f}\n")
            f.write(f"Macro Precision: {macro_avg_2d['precision']:.4f}\n")
            f.write(f"Macro Recall:    {macro_avg_2d['recall']:.4f}\n")
            f.write(f"Macro F1-score:  {macro_avg_2d['f1-score']:.4f}\n\n")

            # --- 3D Metrics ---
            f.write("=== 3D Semantic Evaluation ===\n")
            f.write(f"Accuracy: {overall_accuracy_3d:.4f}\n")
            f.write(f"Micro Precision: {precision_3d:.4f}\n")
            f.write(f"Micro Recall:    {recall_3d:.4f}\n")
            f.write(f"Micro F1-score:  {f1_3d:.4f}\n")
            f.write(f"Macro Precision: {macro_avg_3d['precision']:.4f}\n")
            f.write(f"Macro Recall:    {macro_avg_3d['recall']:.4f}\n")
            f.write(f"Macro F1-score:  {macro_avg_3d['f1-score']:.4f}\n")
