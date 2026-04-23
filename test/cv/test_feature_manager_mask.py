#!/usr/bin/env -S python3 -O
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import cv2

from pyslam.config import Config
from pyslam.viz.mplot_figure import MPlotFigure

from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_manager_configs import FeatureManagerConfigs
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


def make_test_image(height=480, width=640):
    img = np.zeros((height, width), dtype=np.uint8)

    # Build a feature-rich synthetic image with corners, edges, and repeated structure.
    for y in range(40, height - 40, 80):
        for x in range(40, width - 40, 80):
            cv2.rectangle(img, (x - 18, y - 18), (x + 18, y + 18), 255, 2)
            cv2.circle(img, (x, y), 9, 180, -1)

    cv2.line(img, (0, 0), (width - 1, height - 1), 200, 2)
    cv2.line(img, (width - 1, 0), (0, height - 1), 200, 2)
    cv2.putText(img, "PYSLAM", (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 220, 3)

    return img


def make_left_half_mask(image):
    mask = np.zeros(image.shape[:2], dtype=np.int32)
    mask[:, : image.shape[1] // 2] = 1
    return mask


def assert_keypoints_inside_mask(kps, mask):
    assert len(kps) > 0, "Expected at least one keypoint"
    for kp in kps:
        assert isinstance(kp, cv2.KeyPoint), f"Expected cv2.KeyPoint, got {type(kp)}"
        x = int(np.floor(kp.pt[0]))
        y = int(np.floor(kp.pt[1]))
        assert 0 <= x < mask.shape[1], f"Keypoint x out of bounds: {kp.pt}"
        assert 0 <= y < mask.shape[0], f"Keypoint y out of bounds: {kp.pt}"
        assert mask[y, x] != 0, f"Keypoint escaped mask at {kp.pt}"


def visualize_results(img, mask, kps_unmasked, kps_detect, kps_masked):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

    img_unmasked = cv2.drawKeypoints(
        img_bgr.copy(),
        kps_unmasked,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Darken the masked-out region to make the allowed area clear.
    masked_overlay = img_bgr.copy()
    masked_overlay[mask == 0] = masked_overlay[mask == 0] // 4

    img_detect = cv2.drawKeypoints(
        masked_overlay.copy(),
        kps_detect,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    img_detect_and_compute = cv2.drawKeypoints(
        masked_overlay.copy(),
        kps_masked,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    MPlotFigure(img_unmasked[:, :, [2, 1, 0]], title="unmasked detectAndCompute keypoints")
    MPlotFigure(mask_vis[:, :, [2, 1, 0]], title="mask")
    MPlotFigure(img_detect[:, :, [2, 1, 0]], title="masked detect keypoints")
    MPlotFigure(
        img_detect_and_compute[:, :, [2, 1, 0]], title="masked detectAndCompute keypoints"
    )
    MPlotFigure.show()


def run_timed_iterations(func, num_iterations=20, num_warmup_iterations=3):
    out = None

    for _ in range(num_warmup_iterations):
        out = func()

    time_start = time.perf_counter()
    for _ in range(num_iterations):
        out = func()
    elapsed = time.perf_counter() - time_start

    avg_time_ms = 1e3 * elapsed / num_iterations
    return out, avg_time_ms


def print_summary_table(rows):
    headers = ("variant", "keypoints", "avg time [ms]")
    widths = [
        max(len(headers[0]), max(len(row[0]) for row in rows)),
        max(len(headers[1]), max(len(str(row[1])) for row in rows)),
        max(len(headers[2]), max(len(f"{row[2]:.3f}") for row in rows)),
    ]

    def format_row(values):
        return (
            f"| {str(values[0]).ljust(widths[0])} "
            f"| {str(values[1]).rjust(widths[1])} "
            f"| {str(values[2]).rjust(widths[2])} |"
        )

    separator = (
        f"+-{'-' * widths[0]}-+-{'-' * widths[1]}-+-{'-' * widths[2]}-+"
    )

    print(separator)
    print(format_row(headers))
    print(separator)
    for name, keypoints, avg_time_ms in rows:
        print(format_row((name, keypoints, f"{avg_time_ms:.3f}")))
    print(separator)


def main():
    img = make_test_image()
    mask = make_left_half_mask(img)
    num_iterations = 20
    num_warmup_iterations = 3

    num_features = 500
    feature_tracker_config = dict(FeatureTrackerConfigs.ORB2)
    feature_tracker_config["num_features"] = num_features
    feature_tracker_config["deterministic"] = True

    feature_manager_config = FeatureManagerConfigs.extract_from(feature_tracker_config)
    feature_manager_config["deterministic"] = True
    feature_manager = feature_manager_factory(**feature_manager_config)

    assert feature_manager.need_mask_management, "Expected ORB2 to require mask post-filtering"

    kps_detect_unmasked, unmasked_detect_time_ms = run_timed_iterations(
        lambda: feature_manager.detect(img),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    kps_detect, masked_detect_time_ms = run_timed_iterations(
        lambda: feature_manager.detect(img, mask=mask),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    (kps_unmasked, des_unmasked), unmasked_detect_and_compute_time_ms = run_timed_iterations(
        lambda: feature_manager.detectAndCompute(img),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    (kps_masked, des_masked), masked_detect_and_compute_time_ms = run_timed_iterations(
        lambda: feature_manager.detectAndCompute(img, mask=mask),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )

    assert len(kps_unmasked) > 0, "Unmasked detection returned no keypoints"
    assert len(kps_detect_unmasked) > 0, "Unmasked detect returned no keypoints"
    assert des_unmasked is not None, "Unmasked descriptors are unexpectedly None"
    assert len(kps_unmasked) == len(des_unmasked), "Unmasked keypoints/descriptors are misaligned"

    assert_keypoints_inside_mask(kps_detect, mask)
    assert_keypoints_inside_mask(kps_masked, mask)

    assert des_masked is not None, "Masked descriptors are unexpectedly None"
    assert len(kps_masked) == len(des_masked), "Masked keypoints/descriptors are misaligned"
    assert len(kps_masked) <= len(kps_unmasked), "Masking should not increase keypoint count"
    assert len(kps_detect) <= len(
        kps_unmasked
    ), "Masked detect() should not increase keypoint count"

    summary_rows = [
        ("unmasked detect", len(kps_detect_unmasked), unmasked_detect_time_ms),
        ("unmasked detectAndCompute", len(kps_unmasked), unmasked_detect_and_compute_time_ms),
        ("masked detect", len(kps_detect), masked_detect_time_ms),
        ("masked detectAndCompute", len(kps_masked), masked_detect_and_compute_time_ms),
    ]

    print("Mask test passed")
    print(
        f"timing setup: warmup={num_warmup_iterations}, iterations={num_iterations}, num_features={num_features}"
    )
    print_summary_table(summary_rows)
    visualize_results(img, mask, kps_unmasked, kps_detect, kps_masked)


if __name__ == "__main__":
    main()
