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


class LocalizationPlotDrawer:
    def __init__(self, viewer3D: Viewer3D = None):
        self.viewer3D = viewer3D

        self.matched_points_plt = None
        self.info_3dpoints_plt = None
        self.chi2_error_plt = None
        self.timing_plt = None
        self.traj_error_plt = None

        self.last_alignment_timestamp = None
        self.last_alignment_gt_data = TrajectoryAlignementData()

        # To disable one of them just comment it out
        self.matched_points_plt = factory_plot2d(
            xlabel="img id", ylabel="# matches", title="# matches"
        )
        # self.info_3dpoints_plt = factory_plot2d(xlabel='img id', ylabel='# points',title='info 3d points')
        # self.chi2_error_plt = factory_plot2d(xlabel='img id', ylabel='error',title='mean chi2 error')
        # self.timing_plt = factory_plot2d(xlabel='img id', ylabel='s',title='timing')
        self.traj_error_plt = factory_plot2d(
            xlabel="time [s]", ylabel="error", title="trajectories: gt vs (aligned)estimated"
        )

        self.plt_list = [
            self.matched_points_plt,
            self.info_3dpoints_plt,
            self.chi2_error_plt,
            self.timing_plt,
            self.traj_error_plt,
        ]
        self.last_processed_kf_img_id = -1

    def quit(self):
        for plt in self.plt_list:
            if plt is not None:
                plt.quit()

    def get_key(self):
        for plt in self.plt_list:
            if plt is not None:
                key = plt.get_key()
                if key != "":
                    return key

    def draw(self, img_id, data_dict: dict):
        try:
            # draw matching info
            if self.matched_points_plt is not None:
                if "num_matched_kps" in data_dict:
                    self.matched_points_plt.draw(
                        [img_id, data_dict["num_matched_kps"]], "# keypoint matches", color="r"
                    )
                if "num_inliers" in data_dict:
                    self.matched_points_plt.draw(
                        [img_id, data_dict["num_inliers"]], "# inliers", color="g"
                    )
                if "num_matched_map_points" in data_dict:
                    self.matched_points_plt.draw(
                        [img_id, data_dict["num_matched_map_points"]],
                        "# matched map pts",
                        color="b",
                    )
                if "num_kf_ref_tracked_points" in data_dict:
                    self.matched_points_plt.draw(
                        [img_id, data_dict["num_kf_ref_tracked_points"]],
                        "# $KF_{ref}$ tracked pts",
                        color="c",
                    )
                if "descriptor_distance_sigma" in data_dict:
                    self.matched_points_plt.draw(
                        [img_id, data_dict["descriptor_distance_sigma"]],
                        "descriptor distance $\sigma_{th}$",
                        color="k",
                    )

            # draw info about 3D points management by local mapping
            if self.info_3dpoints_plt is not None:
                if (
                    "last_processed_kf_img_id" in data_dict
                    and self.last_processed_kf_img_id != data_dict["last_processed_kf_img_id"]
                ):
                    self.last_processed_kf_img_id = data_dict["last_processed_kf_img_id"]
                    print(f"last_processed_kf_img_id: {self.last_processed_kf_img_id}")
                    if "last_num_triangulated_points" in data_dict:
                        self.info_3dpoints_plt.draw(
                            [img_id, data_dict["last_num_triangulated_points"]],
                            "# temporal triangulated pts",
                            color="r",
                        )
                    if "last_num_fused_points" in data_dict:
                        self.info_3dpoints_plt.draw(
                            [img_id, data_dict["last_num_fused_points"]], "# fused pts", color="g"
                        )
                    if "last_num_culled_keyframes" in data_dict:
                        self.info_3dpoints_plt.draw(
                            [img_id, data_dict["last_num_culled_keyframes"]],
                            "# culled keyframes",
                            color="b",
                        )
                    if "last_num_culled_points" in data_dict:
                        self.info_3dpoints_plt.draw(
                            [img_id, data_dict["last_num_culled_points"]], "# culled pts", color="c"
                        )
                    if "last_num_static_stereo_map_points" in data_dict:
                        self.info_3dpoints_plt.draw(
                            [img_id, data_dict["last_num_static_stereo_map_points"]],
                            "# static triangulated pts",
                            color="k",
                        )

            # draw chi2 error
            if self.chi2_error_plt is not None:
                if "mean_pose_opt_chi2_error" in data_dict:
                    self.chi2_error_plt.draw(
                        [img_id, data_dict["mean_pose_opt_chi2_error"]],
                        "pose opt chi2 error",
                        color="r",
                    )
                if "mean_ba_chi2_error" in data_dict:
                    self.chi2_error_plt.draw(
                        [img_id, data_dict["mean_ba_chi2_error"]], "LBA chi2 error", color="g"
                    )
                if "mean_graph_chi2_error" in data_dict:
                    self.chi2_error_plt.draw(
                        [img_id, data_dict["mean_graph_chi2_error"]], "graph chi2 error", color="b"
                    )
                if "mean_BA_chi2_error" in data_dict:
                    self.chi2_error_plt.draw(
                        [img_id, data_dict["mean_BA_chi2_error"]], "GBA chi2 error", color="k"
                    )

            # draw timings
            if self.timing_plt is not None:
                if "time_track" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_track"]], "tracking", color="r")
                if "time_local_mapping" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_local_mapping"]], "local mapping", color="g"
                    )
                if "time_LBA" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_LBA"]], "LBA", color="b")
                if "time_local_mapping_triangulation" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_local_mapping_triangulation"]],
                        "local mapping triangulation",
                        color="k",
                    )
                if "time_local_mapping_pts_culling" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_local_mapping_pts_culling"]],
                        "local mapping pts culling",
                        color="c",
                    )
                if "time_local_mapping_kf_culling" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_local_mapping_kf_culling"]],
                        "local mapping kf culling",
                        color="m",
                    )
                if "time_local_mapping_pts_fusion" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_local_mapping_pts_fusion"]],
                        "local mapping pts fusion",
                        color="y",
                    )
                if "time_loop_closing" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_loop_closing"]],
                        "loop closing",
                        color="darkgoldenrod",
                    )
                if "time_loop_detection" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_loop_detection"]],
                        "loop detection",
                        color="slategrey",
                    )
                if "time_volumetric_integration" in data_dict:
                    self.timing_plt.draw(
                        [img_id, data_dict["time_volumetric_integration"]],
                        "volumetric integration",
                        color="darkviolet",
                        marker="+",
                    )

            # draw trajectory and alignment error
            # NOTE: we must empty the alignment queue in any case
            new_alignment_data = None
            if self.viewer3D:
                new_alignment_data = get_last_item_from_queue(self.viewer3D.alignment_gt_data_queue)
            if self.traj_error_plt is not None and new_alignment_data is None:
                print(f"LocalizationPlotDrawer: no new alignment data")
            if self.traj_error_plt is not None and new_alignment_data is not None:
                num_samples = len(new_alignment_data.timestamps_associations)
                new_alignment_timestamp = (
                    new_alignment_data.timestamps_associations[-1] if num_samples > 10 else None
                )
                print(
                    f"LocalizationPlotDrawer: new gt alignment timestamp: {new_alignment_timestamp}, rms_error: {new_alignment_data.rms_error}, max_error: {new_alignment_data.max_error}, is_est_aligned: {new_alignment_data.is_est_aligned}"
                )
                if (
                    new_alignment_data.rms_error > 0
                    and self.last_alignment_timestamp != new_alignment_timestamp
                ):
                    self.last_alignment_timestamp = new_alignment_timestamp
                    # new_alignment_data.copyTo(self.last_alignment_gt_data)
                    self.last_alignment_gt_data = new_alignment_data
                    gt_traj = np.array(self.last_alignment_gt_data.gt_t_wi, dtype=float)
                    # estimated_traj = np.array(
                    #     self.last_alignment_gt_data.estimated_t_wi, dtype=float
                    # )
                    aligned_estimated_traj = np.array(
                        self.last_alignment_gt_data.estimated_trajectory_aligned, dtype=float
                    )
                    filter_timestamps = np.array(
                        self.last_alignment_gt_data.timestamps_associations, dtype=float
                    )
                    # print(f'LocalizationPlotDrawer: gt_traj: {gt_traj.shape}, estimated_traj: {estimated_traj.shape}, filter_timestamps: {filter_timestamps.shape}')
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
                        traj_error = gt_traj - aligned_estimated_traj
                        traj_dists = np.linalg.norm(traj_error, axis=1)
                        rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))
                        print(f"LocalizationPlotDrawer: traj_error: {traj_error.shape}")
                        if False:
                            err_x_max = np.max(np.abs(traj_error[:, 0]))
                            err_y_max = np.max(np.abs(traj_error[:, 1]))
                            err_z_max = np.max(np.abs(traj_error[:, 2]))
                            print(
                                f"LocalizationPlotDrawer: err_x_max: {err_x_max}, err_y_max: {err_y_max}, err_z_max: {err_z_max}"
                            )
                        time_errx_signal = [filter_timestamps, traj_error[:, 0]]
                        time_erry_signal = [filter_timestamps, traj_error[:, 1]]
                        time_errz_signal = [filter_timestamps, traj_error[:, 2]]
                        time_rms_error_signal = [filter_timestamps[-1], rms_error]
                        self.traj_error_plt.draw(time_errx_signal, "err_x", color="r", append=False)
                        self.traj_error_plt.draw(time_erry_signal, "err_y", color="g", append=False)
                        self.traj_error_plt.draw(time_errz_signal, "err_z", color="b", append=False)
                        self.traj_error_plt.draw(
                            time_rms_error_signal, "RMS error (ATE)", color="c", append=True
                        )

        except Exception as e:
            Printer.red(f"LocalizationPlotDrawer: draw: encountered exception: {e}")
            traceback_details = traceback.format_exc()
            print(f"\t traceback details: {traceback_details}")
