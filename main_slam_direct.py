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
from pyslam.config_parameters import Parameters
from pyslam.slam import USE_CPP, camera_from_config
from pyslam.utilities.logging import Printer
from pyslam.viz.direct_slam_display import DirectSlamDisplay

# A session lost with at most this many keyframes is reset, as main_slam.py does.
kMaxKeyframesForResetOnLoss = 5


def main():
    parser = argparse.ArgumentParser(description="Direct SLAM frontend (klt, hybrid, or semi-direct tracking)")
    parser.add_argument("-c", "--config_path", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--headless", action="store_true", help="Disable both viewers")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0: all)")
    parser.add_argument(
        "--mode",
        choices=("klt", "hybrid", "semi_direct", "direct"),
        help="Override DIRECT_SLAM.mode (klt, hybrid, or semi_direct; direct remains an alias of hybrid)",
    )
    parser.add_argument("--no-local-ba", action="store_true", help="Disable automatic local bundle adjustment")
    parser.add_argument("--no-loop-closing", action="store_true", help="Disable the loop-closing thread")
    parser.add_argument("--trajectory-file", help="Save accepted camera poses in KITTI format (Twc, 12 columns)")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("max-frames must be nonnegative")
    if not USE_CPP:
        raise SystemExit("main_slam_direct.py requires the C++ core (Parameters.USE_CPP_CORE)")

    from pyslam.slam.cpp import DirectSlam

    config = Config(args.config_path) if args.config_path else Config()
    dataset = dataset_factory(config)
    settings = dict(config.cam_settings)
    # Override both settings forms because a nested DIRECT_SLAM block takes precedence.
    is_monocular = dataset.sensor_type == SensorType.MONOCULAR
    if args.mode:
        # Camera settings normally use flattened OpenCV YAML keys.
        settings["DIRECT_SLAM.mode"] = args.mode
        # The command line also wins over a DIRECT_SLAM.monocular.mode override.
        settings.pop("DIRECT_SLAM.monocular.mode", None)
        if "DIRECT_SLAM" in settings:
            nested = dict(settings["DIRECT_SLAM"], mode=args.mode)
            nested.pop("monocular.mode", None)
            if isinstance(nested.get("monocular"), dict):
                nested["monocular"] = {k: v for k, v in nested["monocular"].items() if k != "mode"}
            settings["DIRECT_SLAM"] = nested
    requested_mode = str(args.mode or "")
    if not requested_mode:
        # Same precedence as DirectSlamParams::from_config: a nested block wins over flat keys,
        # and monocular input applies the DIRECT_SLAM.monocular overrides.
        nested = settings.get("DIRECT_SLAM")
        if isinstance(nested, dict):
            mono = nested.get("monocular") if isinstance(nested.get("monocular"), dict) else {}
            mono_mode = mono.get("mode") or nested.get("monocular.mode")
            requested_mode = str((is_monocular and mono_mode) or nested.get("mode") or "klt")
        else:
            mono_mode = settings.get("DIRECT_SLAM.monocular.mode")
            requested_mode = str((is_monocular and mono_mode) or settings.get("DIRECT_SLAM.mode", "klt"))
    requested_mode = requested_mode.lower()
    if is_monocular and requested_mode not in ("semi_direct", "semidirect"):
        raise SystemExit(
            "Monocular direct SLAM requires DIRECT_SLAM.mode: semi_direct. "
            "klt and hybrid need stereo or RGBD depth."
        )

    # Keep dataset calibration/sensor setup shared with the standard SLAM entry point.
    camera = camera_from_config(config)
    if str(config.dataset_settings.get("type", "")).lower() in ("tum_vi", "tumvi", "tum-vi"):
        if dataset.sensor_type != SensorType.STEREO:
            raise SystemExit(
                "Direct SLAM on TUM-VI needs sensor_type: stereo. "
                "Native and rectified images are both tracked; native images stay distorted."
            )
    undistorter = ImageUndistorter.for_camera(camera, dataset.sensor_type)
    if args.no_local_ba:
        settings["DIRECT_SLAM.local_ba"] = False
        if "DIRECT_SLAM" in settings:
            settings["DIRECT_SLAM"] = dict(settings["DIRECT_SLAM"], local_ba=False)
    slam = DirectSlam(camera, settings)
    Printer.green(f"Direct SLAM mode: {slam.mode}")
    loop_closing = None
    # Monocular loop closing optimizes Sim(3); DirectLoopClosing publishes the per-keyframe
    # scale with the poses, and the direct map rescales landmarks, priors and seeds with it.
    if Parameters.kUseLoopClosing and not args.no_loop_closing:
        from pyslam.slam.direct_loop_closing import DirectLoopClosing

        loop_closing = DirectLoopClosing(
            slam, camera, dataset.sensor_type, config, headless=args.headless
        )
    from pyslam.io.ground_truth import groundtruth_factory, is_valid_groundtruth, need_sim3_alignment

    groundtruth = groundtruth_factory(config.dataset_settings, cam_settings=config.cam_settings)
    has_groundtruth = is_valid_groundtruth(groundtruth)
    # Stereo/RGBD are metric. Monocular scale follows the depth prior, so ground truth is Sim(3)-aligned.
    eval_ate_correct_scale = has_groundtruth and (
        need_sim3_alignment(groundtruth) or dataset.sensor_type == SensorType.MONOCULAR
    )
    display = None
    processed = 0
    timer = time.perf_counter()
    try:
        if not args.headless:
            display = DirectSlamDisplay(viewer_scale(dataset, slam))
            if has_groundtruth:
                positions, _, timestamps = groundtruth.getFull6dTrajectory()
                display.viewer.set_gt_trajectory(
                    positions, timestamps, align_with_scale=eval_ate_correct_scale)
            display.timer.start()
        img_id = 0
        image = None
        timestamp = None
        last_frame = None  # (image, img_id, timestamp) of the last tracked frame, for redraws.
        user_quit = False
        while dataset.is_ok:
            frame_start = time.perf_counter()
            if display:
                if display_quit_requested(display):
                    user_quit = True
                    break
                if display.viewer.reset():
                    if loop_closing:
                        # Resets DirectSlam too, inside the tracking gate.
                        if not loop_closing.reset():
                            break
                    else:
                        slam.reset()

                # Viewer controls are consumed here; local BA runs on this tracking thread.
                if display.viewer.is_bundle_adjust():
                    if loop_closing and not loop_closing.acquire_tracking():
                        break
                    try:
                        result = slam.bundle_adjustment()
                        snapshot = (display.capture(slam, image, img_id - 1, timestamp, loop_closing)
                                    if image is not None else None)
                    finally:
                        if loop_closing:
                            loop_closing.finish_tracking()
                    print(f"Local BA: applied={result.applied}, keyframes={result.keyframes}, "
                          f"landmarks={result.landmarks}, outliers={result.outliers}")
                    if snapshot is not None:
                        display.draw(snapshot, img_id - 1, refresh_fps=False)
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
            if undistorter:
                image, depth = undistorter.image(image), undistorter.depth(depth)
            timestamp = dataset.getTimestamp()
            # Native tracking includes any keyframe-triggered BA before returning its pose.
            if loop_closing and not loop_closing.acquire_tracking():
                break
            try:
                slam.track(image, right, depth, img_id, timestamp)
                if loop_closing and slam.status_name() == "LOST":
                    # As in main_slam.py: relocalize on the loop-closing keyframes.
                    loop_closing.relocalize(image, right, depth, img_id, timestamp)
                if loop_closing:
                    loop_closing.note_tracked_frame(image, right, depth, img_id, timestamp)
                snapshot = (display.capture(slam, image, img_id, timestamp, loop_closing)
                            if display else None)
                # Read before releasing the gate; a loop correction may move poses next.
                position = np.asarray(slam.pose().Ow, dtype=np.float64).reshape(-1)
                status = (f"frame {img_id:6d}  {slam.status_name():14s}  inliers {slam.last_inliers():4d}  "
                          f"keyframes {slam.num_keyframes():4d}  ")
            finally:
                if loop_closing:
                    loop_closing.finish_tracking()
            print(f"{status}Ow [{position[0]:8.3f} {position[1]:8.3f} {position[2]:8.3f}]")
            if slam.status_name() == "LOST" and slam.num_keyframes() <= kMaxKeyframesForResetOnLoss:
                # As in main_slam.py: a session lost this early is restarted (re-initialized).
                Printer.yellow(f"Direct SLAM: lost with {slam.num_keyframes()} keyframes, resetting")
                if loop_closing:
                    if not loop_closing.reset():
                        break
                else:
                    slam.reset()
            if display:
                display.draw(snapshot, img_id)
            last_frame = (image, img_id, timestamp)
            img_id += 1
            processed += 1
            # Pace interactive playback; headless benchmarks run without dataset-rate sleeps.
            if display and dataset.fps:
                time.sleep(max(0.0, 1.0 / dataset.fps - (time.perf_counter() - frame_start)))
        elapsed = time.perf_counter() - timer
        if display and not user_quit:
            # As in main_slam.py, keep the viewers open after the last frame.
            Printer.green("Done. Close the 3D viewer (or press ESC in it), or press 'q' in the image window, to exit.")
            wait_for_viewer_close(display, slam, loop_closing, last_frame)
    except KeyboardInterrupt:
        # Ctrl+C also reaches the viewer and loop-detection processes, so an orderly quit()
        # can wait forever on their queues. Save what was asked for, then exit as main_slam.py does.
        from pyslam.utilities.system import force_kill_all_and_exit

        Printer.yellow("\nCTRL+C detected. Shutting down ...\n")
        if args.trajectory_file:
            save_trajectory(slam, args.trajectory_file)
        force_kill_all_and_exit(verbose=False)
    finally:
        if loop_closing:
            loop_closing.quit()
        if display:
            display.quit()
        dataset.close()
        if args.trajectory_file:
            save_trajectory(slam, args.trajectory_file)
    # quit() joins loop closing and its final correction. Evaluate the same settled map
    # that was exported above, without racing a pose update or counting viewer wait time.
    report(slam, processed, elapsed, groundtruth if has_groundtruth else None, eval_ate_correct_scale)


class ImageUndistorter:
    """Undistort monocular/RGBD input for a distorted pinhole camera.

    Direct tracking works on raw pixels and only models distortion for raw-pixel cameras
    (e.g. Kannala-Brandt); main_slam.py instead undistorts keypoints. The images are remapped
    once per frame to the same intrinsics, and the camera is switched to zero distortion, so
    DirectSlam and loop closing both see an undistorted pinhole.
    """

    def __init__(self, camera):
        size = (int(camera.width), int(camera.height))
        K = np.asarray(camera.K, dtype=np.float64)
        D = np.asarray(camera.D, dtype=np.float64)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_32FC1)
        camera.D = np.zeros_like(D)
        camera.is_distorted = False
        camera.u_min, camera.u_max = 0.0, float(camera.width)
        camera.v_min, camera.v_max = 0.0, float(camera.height)

    @staticmethod
    def for_camera(camera, sensor_type):
        """An undistorter when the camera needs one, else None."""
        if not camera.is_distorted or camera.projects_raw_keypoints():
            return None
        if sensor_type == SensorType.STEREO:
            raise SystemExit("Direct SLAM: a distorted stereo pinhole needs rectified images")
        Printer.yellow("Direct SLAM: undistorting the input images (distorted pinhole camera)")
        return ImageUndistorter(camera)

    def image(self, image):
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def depth(self, depth):
        # Nearest neighbor: interpolating would blend depths across object boundaries.
        return None if depth is None else cv2.remap(depth, self.map1, self.map2, cv2.INTER_NEAREST)


def viewer_scale(dataset, slam):
    """3D viewer scale (camera size, grid and initial viewpoint distance).

    Datasets tune scale_viewer_3d for main_slam.py. Its monocular initializer puts the median
    scene depth at Parameters.kInitializerDesiredMedianDepth, while the direct one puts it at
    sqrt(seed_min_depth * seed_max_depth) (6.3 by default), so a monocular direct map is drawn
    with the scale grown by the same ratio. Stereo and RGBD maps are metric in both pipelines.
    """
    scale = dataset.scale_viewer_3d
    if dataset.sensor_type == SensorType.MONOCULAR:
        scale *= slam.monocular_init_median_depth / Parameters.kInitializerDesiredMedianDepth
    return scale


def save_trajectory(slam, path):
    # Export the final corrected trajectory, not poses cached before later BA updates.
    # Only accepted frames are stored; reset starts a new trajectory/world frame.
    poses = [np.linalg.inv(Tcw)[:3].reshape(12) for Tcw in slam.trajectory()]
    np.savetxt(path, np.asarray(poses).reshape(-1, 12))


def report(slam, processed, elapsed, groundtruth, correct_scale):
    print(f"processed {processed} frames in {elapsed:.2f}s, keyframes {slam.num_keyframes()}")
    evaluate_final_ate(slam, groundtruth, correct_scale)


def display_quit_requested(display):
    """True when the 3D viewer was closed (window or ESC) or 'q' was pressed in the image window."""
    return display.viewer.is_closed() or display.images.get_key() == "q"


def wait_for_viewer_close(display, slam, loop_closing, last_frame, period=0.05, redraw_every=0.5):
    """Keep the viewers alive after the last frame until the user closes them.

    The map is redrawn periodically, so a loop correction or global BA that lands after
    the last frame is still shown. Snapshots are captured inside the tracking gate.
    """
    last_redraw = time.perf_counter()
    while not display_quit_requested(display):
        if last_frame is not None and time.perf_counter() - last_redraw >= redraw_every:
            image, img_id, timestamp = last_frame
            if loop_closing and not loop_closing.acquire_tracking():
                break
            try:
                snapshot = display.capture(slam, image, img_id, timestamp, loop_closing)
            finally:
                if loop_closing:
                    loop_closing.finish_tracking()
            display.draw(snapshot, img_id, refresh_fps=False)
            last_redraw = time.perf_counter()
        time.sleep(period)


def evaluate_final_ate(slam, groundtruth, correct_scale):
    """Print the ATE of the final (BA/loop-corrected) trajectory against ground truth."""
    if groundtruth is None:
        Printer.yellow("Ground truth not available: skipping ATE evaluation")
        return
    try:
        from pyslam.utilities.evaluation import eval_ate
        from pyslam.utilities.geom_trajectory import find_poses_associations

        # Twc poses, as expected by the evaluation (same convention as GT poses).
        est_poses = [np.linalg.inv(Tcw) for Tcw in slam.trajectory()]
        est_timestamps = np.asarray(slam.trajectory_timestamps(), dtype=np.float64)
        if len(est_poses) < 3:
            Printer.yellow(f"ATE: only {len(est_poses)} estimated poses, skipping evaluation")
            return
        _, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        _, assoc_est_poses, assoc_gt_poses = find_poses_associations(
            est_timestamps, est_poses, np.asarray(gt_timestamps), gt_poses
        )
        if len(assoc_est_poses) < 3:
            Printer.yellow(f"ATE: only {len(assoc_est_poses)} poses associated with GT, skipping")
            return
        ape_stats, _ = eval_ate(
            poses_est=assoc_est_poses,
            poses_gt=assoc_gt_poses,
            frame_ids=list(range(len(assoc_est_poses))),
            curr_frame_id=len(est_poses),
            is_final=True,
            is_monocular=correct_scale,
            save_dir=None,
        )
        if ape_stats is None:
            Printer.red("ATE evaluation failed")
            return
        alignment = "Sim(3)" if correct_scale else "SE(3)"
        Printer.green(
            f"Final ATE ({alignment} aligned, {len(assoc_est_poses)}/{len(est_poses)} poses): "
            f"rmse {ape_stats['rmse']:.4f} m, mean {ape_stats['mean']:.4f} m, "
            f"median {ape_stats['median']:.4f} m, max {ape_stats['max']:.4f} m"
        )
    except Exception as e:
        Printer.red(f"ATE evaluation failed: {e}")


if __name__ == "__main__":
    main()
