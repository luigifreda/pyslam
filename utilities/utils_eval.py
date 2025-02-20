import json
import os
import evo
import numpy as np

from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt

from errno import EEXIST

from utils_sys import Printer
from utils_geom import poseRt


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        os.makedirs(folder_path)
    except OSError as exc:  
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise


# Evaluate the estimated poses against the ground truth poses and save/plot the results.
# Returns the RMSE ATE and the EVO stats
# Inputs: poses_est: List of estimated poses, each on a [4x4] transformation matrix
#         poses_gt: List of ground truth poses, each on a [4x4] transformation matrix
#         is_monocular: True if the camera is monocular
def evaluate_evo(poses_est, poses_gt, is_monocular, plot_dir, label, save_metrics=True, save_plot=True):

    traj_est = PosePath3D(poses_se3=poses_est)
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est_aligned = traj_est #PosePath3D(poses_se3=poses_est)
    R_a, t_a, s_a = traj_est_aligned.align(traj_ref=traj_ref, correct_scale=is_monocular)
    
    #poses_est_aligned = traj_est_aligned.poses_se3

    # Compute metrics
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    #Printer.green(f"RMSE ATE {ape_stat} [m]")
    #Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")

    if save_metrics and plot_dir is not None:
        if label is None:
            label = ""
        target_metrics_filename = os.path.join(plot_dir, f"stats_{label}.json")
        with open(target_metrics_filename, "w", encoding="utf-8") as f:
            json.dump(ape_stats, f, indent=4)

        Printer.green(f'Saved metrics to {target_metrics_filename}')

    if save_plot:
        # Use the Agg backend for non-interactive matplotlib
        plt.switch_backend('Agg') # This backend does not require a display and is suitable for saving figures to files without displaying them.
        
        plot_modes = [PlotMode.xy, PlotMode.yz, PlotMode.xyz]
        for plot_mode in plot_modes:        
            fig = plt.figure()        
            ax = evo.tools.plot.prepare_axis(fig, plot_mode)
            ax.set_title(f"ATE RMSE: {ape_stat}")
            evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
            evo.tools.plot.traj_colormap(
                ax,
                traj_est_aligned,
                ape_metric.error,
                plot_mode,
                min_map=ape_stats["min"],
                max_map=ape_stats["max"],
            )
            ax.legend()
            
            target_plot_filename = os.path.join(plot_dir, f"evo_2dplot_{label}_{plot_mode.name}.png")
            plt.savefig(target_plot_filename, dpi=90)
            plt.close(fig)  # Close the figure to free up memory
        
        Printer.green(f'Saved EVO plot to {target_plot_filename}')

    T_gt_est = poseRt(s_a*R_a, t_a)
    return ape_stats, T_gt_est


# Evaluate the estimated poses against the ground truth poses and save/plot the results.
# Returns the RMSE ATE and the EVO stats
# Inputs: poses_est: List of estimated poses, each on a [4x4] transformation matrix
#         poses_gt: List of ground truth poses, each on a [4x4] transformation matrix
#         frame_ids: List of frame ids
#         is_monocular: True if the camera is monocular
def eval_ate(poses_est, poses_gt, frame_ids, curr_frame_id, is_final=False, is_monocular=False, save_dir=None, save_metrics=True, save_plot=True):
    if save_dir is None:
        save_metrics = False 
        save_plot = False
    
    trj_data = dict()
    trj_id, trj_est, trj_gt = [], [], []

    for id, pose_est, pose_gt in zip(frame_ids, poses_est, poses_gt):
        trj_id.append(id)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

    trj_data["trajectory_id"] = trj_id
    trj_data["trajectory_est"] = trj_est
    trj_data["trajectory_gt"] = trj_gt

    if save_dir is not None:
        plot_dir = os.path.join(save_dir, "plot")
        mkdir_p(plot_dir)
    else: 
        plot_dir = None

    label_evo = "final" if is_final else "{:04}".format(curr_frame_id)

    ape_stats, T_gt_est = evaluate_evo(
        poses_est=poses_est,
        poses_gt=poses_gt,        
        is_monocular=is_monocular,
        plot_dir=plot_dir,
        label=label_evo,        
        save_metrics=save_metrics,
        save_plot=save_plot
    )
    trj_data["ate"] = ape_stats["rmse"]
    
    if save_dir is not None:    
        target_data_file = os.path.join(plot_dir, f"trajectory_{label_evo}.json")
        with open(target_data_file, "w", encoding="utf-8") as f:
            json.dump(trj_data, f, indent=4)
        
    return ape_stats, T_gt_est