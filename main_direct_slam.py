#!/usr/bin/env -S python3 -O
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
import time

import cv2
import numpy as np

from pyslam.config import Config
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.slam import USE_CPP, camera_from_config
from pyslam.utilities.logging import Printer


class DirectSlamDisplay:
    """Adapt direct landmarks to the same viewer input used by main_slam.py."""

    def __init__(self, scale):
        from pyslam.viz.viewer3D import Viewer3D
        from pyslam.viz.cvimage_thread import CvImageViewer

        self.viewer = Viewer3D(scale=scale)
        try:
            self.viewer.wait_for_ready()
            self.images = CvImageViewer()
        except Exception:
            self.viewer.quit()
            raise

    def draw(self, slam, image, img_id, timestamp):
        from types import SimpleNamespace
        from pyslam.viz.viewer3D import Viewer3DMapInput
        from pyslam.config_parameters import Parameters

        if self.viewer.is_closed():
            return

        # Draw in C++ using the same point colors and trail renderer as main_slam.py.
        shown = slam.draw_feature_trails(
            image, with_level_radius=self.viewer.is_draw_features_with_radius(),
            trail_max_length=Parameters.kMaxFeatureTrailLength)
        cv2.putText(shown, f"id: {img_id} {slam.status_name()} inliers: {slam.last_inliers()}",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        self.images.draw(shown, "Camera")

        state = Viewer3DMapInput()
        state.cur_frame_data.cur_frame_id = img_id
        state.cur_frame_data.cur_pose = slam.pose().get_inverse_matrix()
        state.cur_frame_data.cur_pose_timestamp = timestamp
        poses, timestamps = slam.keyframes()
        # Read optimized map coordinates, including landmarks no longer tracked live.
        points = slam.map_points()
        limit = Parameters.kMaxSparseMapPointsToVisualize
        if len(points) > limit:
            points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
        colors = np.zeros((len(points), 3), dtype=np.float64)
        # Re-read every accepted pose so local BA corrections update the entire line.
        # Tcw camera center is -Rcw.T @ tcw; this also handles an empty/reset trajectory.
        trajectory = np.asarray(slam.trajectory(), dtype=np.float64).reshape(-1, 4, 4)
        state.estimated_trajectory = -np.einsum(
            "nji,nj->ni", trajectory[:, :3, :3], trajectory[:, :3, 3])
        state.map_data = SimpleNamespace(
            poses=np.asarray(poses), pose_timestamps=np.asarray(timestamps), points=points,
            colors=colors, semantic_colors=colors, fov_centers=[], fov_centers_colors=[],
            covisibility_graph=[], spanning_tree=[], loops=[])
        self.viewer.draw_map(state)

    def quit(self):
        self.images.quit()
        self.viewer.quit()


def main():
    parser = argparse.ArgumentParser(description="Direct and semi-direct SLAM frontend")
    parser.add_argument("-c", "--config_path", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--headless", action="store_true", help="Disable both viewers")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0: all)")
    parser.add_argument("--mode", choices=("direct", "semi_direct"), help="Override DIRECT_SLAM.mode")
    parser.add_argument("--no-local-ba", action="store_true", help="Disable automatic local bundle adjustment")
    parser.add_argument("--trajectory-file", help="Save accepted camera poses in KITTI format (Twc, 12 columns)")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("max-frames must be nonnegative")
    if not USE_CPP:
        raise SystemExit("main_direct_slam.py requires the C++ core (Parameters.USE_CPP_CORE)")

    from pyslam.slam.cpp import DirectSlam

    config = Config(args.config_path) if args.config_path else Config()
    dataset = dataset_factory(config)
    if dataset.sensor_type == SensorType.MONOCULAR:
        raise SystemExit("Direct SLAM needs stereo or RGBD depth. The selected dataset is monocular.")

    # Keep dataset calibration/sensor setup shared with the standard SLAM entry point.
    camera = camera_from_config(config)
    settings = dict(config.cam_settings)
    # Override both settings forms because a nested DIRECT_SLAM block takes precedence.
    if args.mode:
        # Camera settings normally use flattened OpenCV YAML keys.
        settings["DIRECT_SLAM.mode"] = args.mode
        if "DIRECT_SLAM" in settings:
            settings["DIRECT_SLAM"] = dict(settings["DIRECT_SLAM"], mode=args.mode)
    if args.no_local_ba:
        settings["DIRECT_SLAM.local_ba"] = False
        if "DIRECT_SLAM" in settings:
            settings["DIRECT_SLAM"] = dict(settings["DIRECT_SLAM"], local_ba=False)
    slam = DirectSlam(camera, settings)
    Printer.green(f"Direct SLAM mode: {slam.mode}")
    display = None
    processed = 0
    timer = time.perf_counter()
    try:
        if not args.headless:
            display = DirectSlamDisplay(dataset.scale_viewer_3d)
            from pyslam.io.ground_truth import groundtruth_factory, is_valid_groundtruth

            groundtruth = groundtruth_factory(config.dataset_settings, cam_settings=config.cam_settings)
            if is_valid_groundtruth(groundtruth):
                positions, _, timestamps = groundtruth.getFull6dTrajectory()
                display.viewer.set_gt_trajectory(positions, timestamps, align_with_scale=False)
        img_id = 0
        image = None
        timestamp = None
        while dataset.is_ok:
            frame_start = time.perf_counter()
            if display:
                if display.viewer.is_closed() or display.images.get_key() == "q":
                    break
                if display.viewer.reset():
                    slam.reset()

                # Viewer controls are consumed here; local BA runs on this tracking thread.
                if display.viewer.is_bundle_adjust():
                    result = slam.bundle_adjustment()
                    print(f"Local BA: applied={result.applied}, keyframes={result.keyframes}, "
                          f"landmarks={result.landmarks}, outliers={result.outliers}")
                    if image is not None:
                        display.draw(slam, image, img_id - 1, timestamp)
                step = display.viewer.do_step()
                if display.viewer.is_paused() and not step:
                    time.sleep(0.01)
                    continue
            if args.max_frames and processed >= args.max_frames:
                break
            image = dataset.getImageColor(img_id)
            if image is None:
                break
            right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            depth = dataset.getDepth(img_id)
            timestamp = dataset.getTimestamp()
            # Native tracking includes any keyframe-triggered BA before returning its pose.
            slam.track(image, right, depth, img_id, timestamp)
            position = np.asarray(slam.pose().Ow, dtype=np.float64).reshape(-1)
            print(
                f"frame {img_id:6d}  {slam.status_name():14s}  inliers {slam.last_inliers():4d}  "
                f"keyframes {slam.num_keyframes():4d}  Ow [{position[0]:8.3f} {position[1]:8.3f} {position[2]:8.3f}]"
            )
            if display:
                display.draw(slam, image, img_id, timestamp)
            img_id += 1
            processed += 1
            # Pace interactive playback; headless benchmarks run without dataset-rate sleeps.
            if display and dataset.fps:
                time.sleep(max(0.0, 1.0 / dataset.fps - (time.perf_counter() - frame_start)))
    finally:
        if display:
            display.quit()
        dataset.close()
        if args.trajectory_file:
            # Export the final corrected trajectory, not poses cached before later BA updates.
            # Only accepted frames are stored; reset starts a new trajectory/world frame.
            poses = [np.linalg.inv(Tcw)[:3].reshape(12) for Tcw in slam.trajectory()]
            np.savetxt(args.trajectory_file, np.asarray(poses).reshape(-1, 12))
    elapsed = time.perf_counter() - timer
    print(f"processed {processed} frames in {elapsed:.2f}s, keyframes {slam.num_keyframes()}")


if __name__ == "__main__":
    main()
