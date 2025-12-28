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


from pyslam.utilities.logging import Printer

import platform
import queue
import threading
import traceback
import numpy as np
import matplotlib.colors as mcolors
from collections import namedtuple

from pyslam.slam.slam import Slam
from pyslam.io.dataset_types import SensorType

from .viewer3D import Viewer3D
from .mplot_thread import Mplot2d
from .qplot_thread import Qplot2d

from pyslam.slam import Sim3Pose
from pyslam.utilities.geom_trajectory import TrajectoryAlignementData
from pyslam.utilities.data_management import get_last_item_from_queue


kUseQtplot2d = False
if platform.system() == "Darwin":
    kUseQtplot2d = True  # Under mac force the usage of Qtplot2d: It is smoother


def factory_plot2d(*args, **kwargs):
    if kUseQtplot2d:
        return Qplot2d(*args, **kwargs)
    else:
        return Mplot2d(*args, **kwargs)


# Data structure to pass SLAM state snapshot to drawing thread
SlamStateSnapshot = namedtuple(
    "SlamStateSnapshot",
    [
        "img_id",
        # Tracking data
        "num_matched_kps",
        "num_inliers",
        "num_matched_map_points",
        "num_kf_ref_tracked_points",
        "descriptor_distance_sigma",
        "mean_pose_opt_chi2_error",
        "time_track",
        "last_num_static_stereo_map_points",
        "total_num_static_stereo_map_points",
        # Local mapping data
        "last_processed_kf_img_id",
        "last_num_triangulated_points",
        "total_num_triangulated_points",
        "last_num_fused_points",
        "total_num_fused_points",
        "last_num_culled_points",
        "total_num_culled_points",
        "mean_ba_chi2_error",
        "time_local_mapping",
        "time_local_opt_last_elapsed",
        "timer_triangulation_last_elapsed",
        "timer_pts_culling_last_elapsed",
        "timer_kf_culling_last_elapsed",
        "timer_pts_fusion_last_elapsed",
        "last_num_culled_keyframes",
        "total_num_culled_keyframes",
        # Loop closing data
        "has_loop_closing",
        "mean_graph_chi2_error",
        "loop_closing_timer_last_elapsed",
        "time_loop_detection_value",
        # GBA data
        "gba_mean_squared_error",
        # Volumetric integrator data
        "has_volumetric_integrator",
        "time_volumetric_integration_value",
        # Map data
        "num_keyframes",
    ],
)


# ============================== SlamPlotDrawerThread ==============================


class SlamPlotDrawerThread:
    """
    Threaded SlamPlotDrawer that moves drawing operations to a separate thread.
    This allows the main SLAM loop to continue without waiting for plot data preparation.
    """

    def __init__(self, slam: Slam, viewer3D: Viewer3D = None):
        self.slam = slam
        self.viewer3D = viewer3D

        self.matched_points_plt = None
        self.info_3dpoints_plt = None
        self.info_keyframes_plt = None
        self.chi2_error_plt = None
        self.timing_plt = None
        self.traj_error_plt = None

        self.last_alignment_timestamp = None
        self.last_alignment_gt_data = TrajectoryAlignementData()

        # Initialize plots (same as SlamPlotDrawer)
        self.matched_points_plt = factory_plot2d(
            xlabel="img id", ylabel="# matches", title="# matches"
        )
        if False:
            self.info_3dpoints_plt = factory_plot2d(
                xlabel="img id", ylabel="# points", title="info 3d points"
            )
            self.info_keyframes_plt = factory_plot2d(
                xlabel="img id", ylabel="# keyframes", title="# keyframes"
            )
        self.chi2_error_plt = factory_plot2d(
            xlabel="img id", ylabel="error", title="mean chi2 error"
        )
        self.timing_plt = factory_plot2d(xlabel="img id", ylabel="s", title="timing")
        self.traj_error_plt = factory_plot2d(
            xlabel="time [s]", ylabel="error", title="trajectories: gt vs (aligned)estimated"
        )

        self.plt_list = [
            self.matched_points_plt,
            self.info_3dpoints_plt,
            self.chi2_error_plt,
            self.timing_plt,
            self.traj_error_plt,
            self.info_keyframes_plt,
        ]

        self.last_processed_kf_img_id = -1

        # Thread-safe queue for plot drawing requests
        # Use maxsize=1 to keep only the latest snapshot (drop old ones)
        self.draw_queue = queue.Queue(maxsize=10)
        self.is_running = threading.Event()
        self.is_running.set()

        # Start drawing thread
        self.draw_thread = threading.Thread(
            target=self.run, daemon=True, name="SlamPlotDrawerThread"
        )
        self.draw_thread.start()

    def _create_snapshot(self, img_id):
        """Create a snapshot of SLAM state for thread-safe access"""
        try:
            # Access SLAM attributes (should be fast, mostly reading scalars)
            tracking = self.slam.tracking
            local_mapping = self.slam.local_mapping
            loop_closing = self.slam.loop_closing if hasattr(self.slam, "loop_closing") else None
            gba = self.slam.GBA if hasattr(self.slam, "GBA") else None
            volumetric_integrator = (
                self.slam.volumetric_integrator
                if hasattr(self.slam, "volumetric_integrator")
                else None
            )

            snapshot = SlamStateSnapshot(
                img_id=img_id,
                # Tracking
                num_matched_kps=getattr(tracking, "num_matched_kps", None),
                num_inliers=getattr(tracking, "num_inliers", None),
                num_matched_map_points=getattr(tracking, "num_matched_map_points", None),
                num_kf_ref_tracked_points=getattr(tracking, "num_kf_ref_tracked_points", None),
                descriptor_distance_sigma=getattr(tracking, "descriptor_distance_sigma", None),
                mean_pose_opt_chi2_error=getattr(tracking, "mean_pose_opt_chi2_error", None),
                time_track=getattr(tracking, "time_track", None),
                last_num_static_stereo_map_points=getattr(
                    tracking, "last_num_static_stereo_map_points", None
                ),
                total_num_static_stereo_map_points=getattr(
                    tracking, "total_num_static_stereo_map_points", None
                ),
                # Local mapping
                last_processed_kf_img_id=getattr(local_mapping, "last_processed_kf_img_id", None),
                last_num_triangulated_points=getattr(
                    local_mapping, "last_num_triangulated_points", None
                ),
                total_num_triangulated_points=getattr(
                    local_mapping, "total_num_triangulated_points", None
                ),
                last_num_fused_points=getattr(local_mapping, "last_num_fused_points", None),
                total_num_fused_points=getattr(local_mapping, "total_num_fused_points", None),
                last_num_culled_points=getattr(local_mapping, "last_num_culled_points", None),
                total_num_culled_points=getattr(local_mapping, "total_num_culled_points", None),
                mean_ba_chi2_error=getattr(local_mapping, "mean_ba_chi2_error", None),
                time_local_mapping=getattr(local_mapping, "time_local_mapping", None),
                time_local_opt_last_elapsed=(
                    local_mapping.time_local_opt.last_elapsed
                    if hasattr(local_mapping, "time_local_opt")
                    and hasattr(local_mapping.time_local_opt, "last_elapsed")
                    else None
                ),
                timer_triangulation_last_elapsed=(
                    local_mapping.timer_triangulation.last_elapsed
                    if hasattr(local_mapping, "timer_triangulation")
                    and hasattr(local_mapping.timer_triangulation, "last_elapsed")
                    else None
                ),
                timer_pts_culling_last_elapsed=(
                    local_mapping.timer_pts_culling.last_elapsed
                    if hasattr(local_mapping, "timer_pts_culling")
                    and hasattr(local_mapping.timer_pts_culling, "last_elapsed")
                    else None
                ),
                timer_kf_culling_last_elapsed=(
                    local_mapping.timer_kf_culling.last_elapsed
                    if hasattr(local_mapping, "timer_kf_culling")
                    and hasattr(local_mapping.timer_kf_culling, "last_elapsed")
                    else None
                ),
                timer_pts_fusion_last_elapsed=(
                    local_mapping.timer_pts_fusion.last_elapsed
                    if hasattr(local_mapping, "timer_pts_fusion")
                    and hasattr(local_mapping.timer_pts_fusion, "last_elapsed")
                    else None
                ),
                last_num_culled_keyframes=getattr(local_mapping, "last_num_culled_keyframes", None),
                total_num_culled_keyframes=getattr(
                    local_mapping, "total_num_culled_keyframes", None
                ),
                # Loop closing
                has_loop_closing=(loop_closing is not None),
                mean_graph_chi2_error=(
                    getattr(loop_closing, "mean_graph_chi2_error", None) if loop_closing else None
                ),
                loop_closing_timer_last_elapsed=(
                    loop_closing.timer.last_elapsed
                    if loop_closing
                    and hasattr(loop_closing, "timer")
                    and hasattr(loop_closing.timer, "last_elapsed")
                    else None
                ),
                time_loop_detection_value=(
                    loop_closing.time_loop_detection.value
                    if loop_closing
                    and hasattr(loop_closing, "time_loop_detection")
                    and hasattr(loop_closing.time_loop_detection, "value")
                    else None
                ),
                # GBA
                gba_mean_squared_error=(
                    gba.mean_squared_error.value
                    if gba
                    and hasattr(gba, "mean_squared_error")
                    and hasattr(gba.mean_squared_error, "value")
                    else None
                ),
                # Volumetric integrator
                has_volumetric_integrator=(volumetric_integrator is not None),
                time_volumetric_integration_value=(
                    volumetric_integrator.time_volumetric_integration.value
                    if volumetric_integrator
                    and hasattr(volumetric_integrator, "time_volumetric_integration")
                    and hasattr(volumetric_integrator.time_volumetric_integration, "value")
                    else None
                ),
                # Map
                num_keyframes=self.slam.map.num_keyframes() if hasattr(self.slam, "map") else 0,
            )
            return snapshot
        except Exception as e:
            Printer.red(f"SlamPlotDrawerThread: _create_snapshot: encountered exception: {e}")
            traceback_details = traceback.format_exc()
            print(f"\t traceback details: {traceback_details}")
            return None

    def run(self):
        """Worker thread that processes drawing requests"""
        while self.is_running.is_set():
            try:
                # Get snapshot with timeout to allow checking is_running
                snapshot = self.draw_queue.get(timeout=0.1)
                if snapshot is None:
                    continue
                self._draw_from_snapshot(snapshot)
            except queue.Empty:
                continue
            except Exception as e:
                Printer.red(f"SlamPlotDrawerThread: run: encountered exception: {e}")
                traceback_details = traceback.format_exc()
                print(f"\t traceback details: {traceback_details}")
        print(f"SlamPlotDrawerThread: run: closed")

    def _draw_from_snapshot(self, snapshot):
        """Actual drawing logic using snapshot data (same as original draw method)"""
        try:
            img_id = snapshot.img_id

            # draw matching info
            if self.matched_points_plt is not None:
                if snapshot.num_matched_kps is not None:
                    matched_kps_signal = [img_id, snapshot.num_matched_kps]
                    self.matched_points_plt.draw(
                        matched_kps_signal, "# keypoint matches", color="r"
                    )
                if snapshot.num_inliers is not None:
                    inliers_signal = [img_id, snapshot.num_inliers]
                    self.matched_points_plt.draw(inliers_signal, "# inliers", color="g")
                if snapshot.num_matched_map_points is not None:
                    valid_matched_map_points_signal = [
                        img_id,
                        snapshot.num_matched_map_points,
                    ]
                    self.matched_points_plt.draw(
                        valid_matched_map_points_signal, "# matched map pts", color="b"
                    )
                if snapshot.num_kf_ref_tracked_points is not None:
                    kf_ref_tracked_points_signal = [
                        img_id,
                        snapshot.num_kf_ref_tracked_points,
                    ]
                    self.matched_points_plt.draw(
                        kf_ref_tracked_points_signal, "# $KF_{ref}$ tracked pts", color="c"
                    )
                if snapshot.descriptor_distance_sigma is not None:
                    descriptor_sigma_signal = [img_id, snapshot.descriptor_distance_sigma]
                    self.matched_points_plt.draw(
                        descriptor_sigma_signal, "descriptor distance $\sigma_{th}$", color="k"
                    )

            # draw info about 3D points management by local mapping
            if self.info_3dpoints_plt is not None:
                if (
                    snapshot.last_processed_kf_img_id is not None
                    and self.last_processed_kf_img_id != snapshot.last_processed_kf_img_id
                ):
                    self.last_processed_kf_img_id = snapshot.last_processed_kf_img_id
                    print(f"last_processed_kf_img_id: {self.last_processed_kf_img_id}")
                    if snapshot.last_num_triangulated_points is not None:
                        num_triangulated_points_signal = [
                            img_id,
                            snapshot.last_num_triangulated_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            num_triangulated_points_signal,
                            "# curr temporal triangulated pts",
                            color="r",
                        )
                    if snapshot.total_num_triangulated_points is not None:
                        total_num_triangulated_points_signal = [
                            img_id,
                            snapshot.total_num_triangulated_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            total_num_triangulated_points_signal,
                            "# total triangulated pts",
                            color="r",
                            marker="*",
                        )
                    if snapshot.last_num_fused_points is not None:
                        num_fused_points_signal = [
                            img_id,
                            snapshot.last_num_fused_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            num_fused_points_signal, "# currfused pts", color="g"
                        )
                    if snapshot.total_num_fused_points is not None:
                        total_num_fused_points_signal = [
                            img_id,
                            snapshot.total_num_fused_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            total_num_fused_points_signal,
                            "# total fused pts",
                            color="g",
                            marker="*",
                        )
                    if snapshot.last_num_culled_points is not None:
                        num_culled_points_signal = [
                            img_id,
                            snapshot.last_num_culled_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            num_culled_points_signal, "# curr culled pts", color="c"
                        )
                    if snapshot.total_num_culled_points is not None:
                        total_num_culled_points_signal = [
                            img_id,
                            snapshot.total_num_culled_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            total_num_culled_points_signal,
                            "# total culled pts",
                            color="c",
                            marker="*",
                        )
                    if snapshot.last_num_static_stereo_map_points is not None:
                        num_static_stereo_map_points_signal = [
                            img_id,
                            snapshot.last_num_static_stereo_map_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            num_static_stereo_map_points_signal,
                            "# curr static triangulated pts",
                            color="k",
                        )
                    if snapshot.total_num_static_stereo_map_points is not None:
                        total_num_static_stereo_map_points_signal = [
                            img_id,
                            snapshot.total_num_static_stereo_map_points,
                        ]
                        self.info_3dpoints_plt.draw(
                            total_num_static_stereo_map_points_signal,
                            "# total static triangulated pts",
                            color="k",
                            marker="*",
                        )

            # draw chi2 error
            if self.chi2_error_plt is not None:
                if snapshot.mean_pose_opt_chi2_error is not None:
                    mean_pose_opt_chi2_error_signal = [
                        img_id,
                        snapshot.mean_pose_opt_chi2_error,
                    ]
                    self.chi2_error_plt.draw(
                        mean_pose_opt_chi2_error_signal, "pose opt chi2 error", color="r"
                    )
                if snapshot.mean_ba_chi2_error is not None:
                    mean_squared_reproj_error_signal = [
                        img_id,
                        snapshot.mean_ba_chi2_error,
                    ]
                    self.chi2_error_plt.draw(
                        mean_squared_reproj_error_signal, "LBA chi2 error", color="g"
                    )
                if snapshot.has_loop_closing:
                    if snapshot.mean_graph_chi2_error is not None:
                        mean_graph_chi2_error_signal = [
                            img_id,
                            snapshot.mean_graph_chi2_error,
                        ]
                        self.chi2_error_plt.draw(
                            mean_graph_chi2_error_signal, "graph chi2 error", color="b"
                        )
                    if (
                        snapshot.gba_mean_squared_error is not None
                        and snapshot.gba_mean_squared_error > 0
                    ):
                        mean_BA_chi2_error_signal = [
                            img_id,
                            snapshot.gba_mean_squared_error,
                        ]
                        self.chi2_error_plt.draw(
                            mean_BA_chi2_error_signal, "GBA chi2 error", color="k"
                        )

            # draw timings
            if self.timing_plt is not None:
                if snapshot.time_track is not None:
                    time_track_signal = [img_id, snapshot.time_track]
                    self.timing_plt.draw(time_track_signal, "tracking", color="r")
                if snapshot.time_local_mapping is not None:
                    time_local_mapping_signal = [img_id, snapshot.time_local_mapping]
                    self.timing_plt.draw(time_local_mapping_signal, "local mapping", color="g")
                if snapshot.time_local_opt_last_elapsed:
                    time_LBA_signal = [img_id, snapshot.time_local_opt_last_elapsed]
                    self.timing_plt.draw(time_LBA_signal, "LBA", color="b")
                if snapshot.timer_triangulation_last_elapsed:
                    time_local_mapping_triangulation_signal = [
                        img_id,
                        snapshot.timer_triangulation_last_elapsed,
                    ]
                    self.timing_plt.draw(
                        time_local_mapping_triangulation_signal,
                        "local mapping triangulation",
                        color="k",
                    )
                if snapshot.timer_pts_culling_last_elapsed:
                    time_local_mapping_pts_culling_signal = [
                        img_id,
                        snapshot.timer_pts_culling_last_elapsed,
                    ]
                    self.timing_plt.draw(
                        time_local_mapping_pts_culling_signal,
                        "local mapping pts culling",
                        color="c",
                    )
                if snapshot.timer_kf_culling_last_elapsed:
                    time_local_mapping_kf_culling_signal = [
                        img_id,
                        snapshot.timer_kf_culling_last_elapsed,
                    ]
                    self.timing_plt.draw(
                        time_local_mapping_kf_culling_signal, "local mapping kf culling", color="m"
                    )
                if snapshot.timer_pts_fusion_last_elapsed:
                    time_local_mapping_pts_fusion_signal = [
                        img_id,
                        snapshot.timer_pts_fusion_last_elapsed,
                    ]
                    self.timing_plt.draw(
                        time_local_mapping_pts_fusion_signal, "local mapping pts fusion", color="y"
                    )
                if snapshot.has_loop_closing:
                    if snapshot.loop_closing_timer_last_elapsed:
                        time_loop_closing_signal = [
                            img_id,
                            snapshot.loop_closing_timer_last_elapsed,
                        ]
                        self.timing_plt.draw(
                            time_loop_closing_signal,
                            "loop closing",
                            color=mcolors.CSS4_COLORS["darkgoldenrod"],
                        )
                    if snapshot.time_loop_detection_value:
                        time_loop_detection_signal = [
                            img_id,
                            snapshot.time_loop_detection_value,
                        ]
                        self.timing_plt.draw(
                            time_loop_detection_signal,
                            "loop detection",
                            color=mcolors.CSS4_COLORS["slategrey"],
                        )
                if snapshot.has_volumetric_integrator:
                    if snapshot.time_volumetric_integration_value:
                        time_volumetric_integration_signal = [
                            img_id,
                            snapshot.time_volumetric_integration_value,
                        ]
                        self.timing_plt.draw(
                            time_volumetric_integration_signal,
                            "volumetric integration",
                            color=mcolors.CSS4_COLORS["darkviolet"],
                            marker="+",
                        )

            # draw number of keyframes
            if self.info_keyframes_plt is not None:
                if snapshot.num_keyframes > 0:
                    num_keyframes_signal = [img_id, snapshot.num_keyframes]
                    self.info_keyframes_plt.draw(num_keyframes_signal, "# keyframes", color="m")
                if snapshot.last_num_culled_keyframes is not None:
                    last_num_culled_keyframes_signal = [
                        img_id,
                        snapshot.last_num_culled_keyframes,
                    ]
                    self.info_keyframes_plt.draw(
                        last_num_culled_keyframes_signal,
                        "# curr culled keyframes",
                        color="b",
                    )
                if snapshot.total_num_culled_keyframes is not None:
                    total_num_culled_keyframes_signal = [
                        img_id,
                        snapshot.total_num_culled_keyframes,
                    ]
                    self.info_keyframes_plt.draw(
                        total_num_culled_keyframes_signal,
                        "# total culled keyframes",
                        color="b",
                        marker="*",
                    )

            # draw trajectory and alignment error
            # NOTE: we must empty the alignment queue in any case
            # This is done in the worker thread to avoid blocking the main thread
            new_alignment_data = None
            if self.viewer3D is not None:
                new_alignment_data = get_last_item_from_queue(self.viewer3D.alignment_gt_data_queue)
            if self.traj_error_plt is not None and new_alignment_data is not None:
                num_samples = len(new_alignment_data.timestamps_associations)
                new_alignment_timestamp = (
                    new_alignment_data.timestamps_associations[-1] if num_samples > 20 else None
                )
                print(
                    f"SlamPlotDrawerThread: new gt alignment timestamp: {new_alignment_timestamp}, rms_error: {new_alignment_data.rms_error}, max_error: {new_alignment_data.max_error}, is_est_aligned: {new_alignment_data.is_est_aligned}"
                )
                if (
                    new_alignment_data.rms_error > 0
                    and self.last_alignment_timestamp != new_alignment_timestamp
                ):
                    self.last_alignment_timestamp = new_alignment_timestamp

                    self.last_alignment_gt_data = new_alignment_data
                    traj_errors = np.asarray(self.last_alignment_gt_data.errors)
                    rms_error = self.last_alignment_gt_data.rms_error

                    gt_traj = np.asarray(self.last_alignment_gt_data.gt_t_wi)
                    aligned_estimated_traj = np.asarray(
                        self.last_alignment_gt_data.estimated_trajectory_aligned
                    )
                    filter_timestamps = np.asarray(
                        self.last_alignment_gt_data.timestamps_associations
                    )
                    if False:
                        time_gt_x_signal = [filter_timestamps, gt_traj[:, 0]]
                        time_gt_y_signal = [filter_timestamps, gt_traj[:, 1]]
                        time_gt_z_signal = [filter_timestamps, gt_traj[:, 2]]
                        time_filter_x_signal = [filter_timestamps, aligned_estimated_traj[:, 0]]
                        time_filter_y_signal = [filter_timestamps, aligned_estimated_traj[:, 1]]
                        time_filter_z_signal = [filter_timestamps, aligned_estimated_traj[:, 2]]
                        self.traj_error_plt.draw(
                            time_filter_x_signal, "filter_x", color="r", append=False
                        )
                        self.traj_error_plt.draw(
                            time_filter_y_signal, "filter_y", color="g", append=False
                        )
                        self.traj_error_plt.draw(
                            time_filter_z_signal, "filter_z", color="b", append=False
                        )
                        self.traj_error_plt.draw(
                            time_gt_x_signal, "gt_x", color="r", linestyle=":", append=False
                        )
                        self.traj_error_plt.draw(
                            time_gt_y_signal, "gt_y", color="g", linestyle=":", append=False
                        )
                        self.traj_error_plt.draw(
                            time_gt_z_signal, "gt_z", color="b", linestyle=":", append=False
                        )
                    else:
                        if rms_error == 0:
                            print(f"SlamPlotDrawerThread: rms_error is 0, computing traj_errors")
                            traj_errors = gt_traj - aligned_estimated_traj
                            traj_dists = np.linalg.norm(traj_errors, axis=1)
                            rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))
                        print(f"SlamPlotDrawerThread: traj_errors: {traj_errors.shape}")
                        if False:
                            err_x_max = np.max(np.abs(traj_errors[:, 0]))
                            err_y_max = np.max(np.abs(traj_errors[:, 1]))
                            err_z_max = np.max(np.abs(traj_errors[:, 2]))
                            print(
                                f"SlamPlotDrawerThread: err_x_max: {err_x_max}, err_y_max: {err_y_max}, err_z_max: {err_z_max}"
                            )
                        time_errx_signal = [filter_timestamps, traj_errors[:, 0]]
                        time_erry_signal = [filter_timestamps, traj_errors[:, 1]]
                        time_errz_signal = [filter_timestamps, traj_errors[:, 2]]
                        time_rms_error_signal = [filter_timestamps[-1], rms_error]
                        self.traj_error_plt.draw(time_errx_signal, "err_x", color="r", append=False)
                        self.traj_error_plt.draw(time_erry_signal, "err_y", color="g", append=False)
                        self.traj_error_plt.draw(time_errz_signal, "err_z", color="b", append=False)
                        self.traj_error_plt.draw(
                            time_rms_error_signal, "RMS error (ATE)", color="c", append=True
                        )

        except Exception as e:
            Printer.red(f"SlamPlotDrawerThread: _draw_from_snapshot: encountered exception: {e}")
            traceback_details = traceback.format_exc()
            print(f"\t traceback details: {traceback_details}")

    def draw(self, img_id):
        """
        Fast, non-blocking method that creates snapshot and queues it.
        Returns immediately without waiting for drawing operations.
        """
        try:
            # Create snapshot of SLAM state (fast, just reading attributes)
            snapshot = self._create_snapshot(img_id)
            if snapshot is None:
                return

            # Non-blocking queue put (discard old if queue full)
            try:
                self.draw_queue.put_nowait(snapshot)
            except queue.Full:
                # Discard old snapshot, keep latest
                try:
                    self.draw_queue.get_nowait()
                    self.draw_queue.put_nowait(snapshot)
                except queue.Empty:
                    pass
        except Exception as e:
            Printer.red(f"SlamPlotDrawerThread: draw: encountered exception: {e}")
            traceback_details = traceback.format_exc()
            print(f"\t traceback details: {traceback_details}")

    def quit(self):
        """Clean shutdown"""
        self.is_running.clear()
        # Wait for thread to finish (with timeout)
        self.draw_thread.join(timeout=5.0)
        if self.draw_thread.is_alive():
            Printer.yellow("SlamPlotDrawerThread: draw thread did not terminate in time")
        # Empty queue
        while not self.draw_queue.empty():
            try:
                self.draw_queue.get_nowait()
            except queue.Empty:
                break
        # Quit all plots
        for plt in self.plt_list:
            if plt is not None:
                plt.quit()
        print(f"SlamPlotDrawerThread: closed")

    def get_key(self):
        """Get key from plot windows (same as original)"""
        for plt in self.plt_list:
            if plt is not None:
                key = plt.get_key()
                if key != "":
                    return key
        return ""
