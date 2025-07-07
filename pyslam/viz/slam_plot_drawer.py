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


from pyslam.utilities.utils_sys import getchar, Printer 

import platform
import traceback
import numpy as np 
import matplotlib.colors as mcolors

from pyslam.slam.slam import Slam
from pyslam.io.dataset_types import SensorType

from .viewer3D import Viewer3D
from .mplot_thread import Mplot2d
from .qplot_thread import Qplot2d

from pyslam.utilities.utils_geom import Sim3Pose
from pyslam.utilities.utils_geom_trajectory import TrajectoryAlignementData


kUseQtplot2d = False
if platform.system() == 'Darwin':
    kUseQtplot2d = True # Under mac force the usage of Qtplot2d: It is smoother 


def factory_plot2d(*args,**kwargs):
    if kUseQtplot2d:
        return Qplot2d(*args,**kwargs)
    else:
        return Mplot2d(*args,**kwargs)
    
    
class SlamPlotDrawer:
    def __init__(self, slam: Slam, viewer3D: Viewer3D = None):
        self.slam = slam
        self.viewer3D = viewer3D
        
        self.matched_points_plt = None
        self.info_3dpoints_plt = None
        self.chi2_error_plt = None
        self.timing_plt = None
        self.traj_error_plt = None
        
        self.last_alignment_timestamp = None
        self.last_alignment_gt_data = TrajectoryAlignementData()
        
        # To disable one of them just comment it out
        self.matched_points_plt = factory_plot2d(xlabel='img id', ylabel='# matches',title='# matches')  
        #self.info_3dpoints_plt = factory_plot2d(xlabel='img id', ylabel='# points',title='info 3d points')      
        self.chi2_error_plt = factory_plot2d(xlabel='img id', ylabel='error',title='mean chi2 error')
        self.timing_plt = factory_plot2d(xlabel='img id', ylabel='s',title='timing')        
        self.traj_error_plt = factory_plot2d(xlabel='time [s]', ylabel='error',title='trajectories: gt vs (aligned)estimated')
        
        
        self.plt_list = [self.matched_points_plt, self.info_3dpoints_plt, self.chi2_error_plt, self.timing_plt, self.traj_error_plt]
        
        self.last_processed_kf_img_id = -1
        
    def quit(self):
        for plt in self.plt_list:
            if plt is not None:
                plt.quit()        
            
    def get_key(self):
        for plt in self.plt_list:
            if plt is not None:
                key = plt.get_key()
                if key != '':
                    return key    

    def draw(self, img_id):
        try:
            # draw matching info
            if self.matched_points_plt is not None: 
                if self.slam.tracking.num_matched_kps is not None: 
                    matched_kps_signal = [img_id, self.slam.tracking.num_matched_kps]     
                    self.matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')                         
                if self.slam.tracking.num_inliers is not None: 
                    inliers_signal = [img_id, self.slam.tracking.num_inliers]                    
                    self.matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                if self.slam.tracking.num_matched_map_points is not None: 
                    valid_matched_map_points_signal = [img_id, self.slam.tracking.num_matched_map_points]   # valid matched map points (in current pose optimization)                                       
                    self.matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')  
                if self.slam.tracking.num_kf_ref_tracked_points is not None: 
                    kf_ref_tracked_points_signal = [img_id, self.slam.tracking.num_kf_ref_tracked_points]                    
                    self.matched_points_plt.draw(kf_ref_tracked_points_signal,'# $KF_{ref}$ tracked pts',color='c')   
                if self.slam.tracking.descriptor_distance_sigma is not None: 
                    descriptor_sigma_signal = [img_id, self.slam.tracking.descriptor_distance_sigma]                    
                    self.matched_points_plt.draw(descriptor_sigma_signal,'descriptor distance $\sigma_{th}$',color='k')                                                                  
                
            # draw info about 3D points management by local mapping 
            if self.info_3dpoints_plt is not None:
                if self.slam.local_mapping.last_processed_kf_img_id is not None and self.last_processed_kf_img_id != self.slam.local_mapping.last_processed_kf_img_id:
                    self.last_processed_kf_img_id = self.slam.local_mapping.last_processed_kf_img_id
                    print(f'last_processed_kf_img_id: {self.last_processed_kf_img_id}')                             
                    if self.slam.local_mapping.last_num_triangulated_points is not None:
                        num_triangulated_points_signal = [img_id, self.slam.local_mapping.last_num_triangulated_points]     
                        self.info_3dpoints_plt.draw(num_triangulated_points_signal,'# temporal triangulated pts',color='r') 
                    if self.slam.local_mapping.last_num_fused_points is not None:
                        num_fused_points_signal = [img_id, self.slam.local_mapping.last_num_fused_points]
                        self.info_3dpoints_plt.draw(num_fused_points_signal,'# fused pts',color='g')
                    if self.slam.local_mapping.last_num_culled_keyframes is not None:
                        num_culled_keyframes_signal = [img_id, self.slam.local_mapping.last_num_culled_keyframes]
                        self.info_3dpoints_plt.draw(num_culled_keyframes_signal,'# culled keyframes',color='b')
                    if self.slam.local_mapping.last_num_culled_points is not None:
                        num_culled_points_signal = [img_id, self.slam.local_mapping.last_num_culled_points]
                        self.info_3dpoints_plt.draw(num_culled_points_signal,'# culled pts',color='c')
                    if self.slam.tracking.last_num_static_stereo_map_points is not None: 
                        num_static_stereo_map_points_signal = [img_id, self.slam.tracking.last_num_static_stereo_map_points]
                        self.info_3dpoints_plt.draw(num_static_stereo_map_points_signal,'# static triangulated pts',color='k') 
                
            #  draw chi2 error
            if self.chi2_error_plt is not None:
                if self.slam.tracking.mean_pose_opt_chi2_error is not None:
                    mean_pose_opt_chi2_error_signal = [img_id, self.slam.tracking.mean_pose_opt_chi2_error]
                    self.chi2_error_plt.draw(mean_pose_opt_chi2_error_signal,'pose opt chi2 error',color='r')
                if self.slam.local_mapping.mean_ba_chi2_error is not None:
                    mean_squared_reproj_error_signal = [img_id, self.slam.local_mapping.mean_ba_chi2_error]
                    self.chi2_error_plt.draw(mean_squared_reproj_error_signal,'LBA chi2 error',color='g')
                if self.slam.loop_closing is not None: 
                    if self.slam.loop_closing.mean_graph_chi2_error is not None:
                        mean_graph_chi2_error_signal = [img_id, self.slam.loop_closing.mean_graph_chi2_error]
                        self.chi2_error_plt.draw(mean_graph_chi2_error_signal,'graph chi2 error',color='b')
                    if self.slam.GBA.mean_squared_error.value >0:
                        mean_BA_chi2_error_signal = [img_id, self.slam.GBA.mean_squared_error.value]
                        self.chi2_error_plt.draw(mean_BA_chi2_error_signal,'GBA chi2 error',color='k')
                                        
            # draw timings
            if self.timing_plt is not None:
                if self.slam.tracking.time_track is not None:
                    time_track_signal = [img_id, self.slam.tracking.time_track]
                    self.timing_plt.draw(time_track_signal,'tracking',color='r')
                if self.slam.local_mapping.time_local_mapping is not None:
                    time_local_mapping_signal = [img_id, self.slam.local_mapping.time_local_mapping]
                    self.timing_plt.draw(time_local_mapping_signal,'local mapping',color='g')  
                if self.slam.local_mapping.time_local_opt.last_elapsed:
                    time_LBA_signal = [img_id, self.slam.local_mapping.time_local_opt.last_elapsed]
                    self.timing_plt.draw(time_LBA_signal,'LBA',color='b')
                if self.slam.local_mapping.timer_triangulation.last_elapsed:
                    time_local_mapping_triangulation_signal = [img_id, self.slam.local_mapping.timer_triangulation.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_triangulation_signal,'local mapping triangulation',color='k')
                if self.slam.local_mapping.timer_pts_culling.last_elapsed:
                    time_local_mapping_pts_culling_signal = [img_id, self.slam.local_mapping.timer_pts_culling.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_pts_culling_signal,'local mapping pts culling',color='c')
                if self.slam.local_mapping.timer_kf_culling.last_elapsed:
                    time_local_mapping_kf_culling_signal = [img_id, self.slam.local_mapping.timer_kf_culling.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_kf_culling_signal,'local mapping kf culling',color='m')
                if self.slam.local_mapping.timer_pts_fusion.last_elapsed:
                    time_local_mapping_pts_fusion_signal = [img_id, self.slam.local_mapping.timer_pts_fusion.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_pts_fusion_signal,'local mapping pts fusion',color='y')
                if self.slam.loop_closing is not None:
                    if self.slam.loop_closing.timer.last_elapsed:
                        time_loop_closing_signal = [img_id, self.slam.loop_closing.timer.last_elapsed]
                        self.timing_plt.draw(time_loop_closing_signal,'loop closing',color=mcolors.CSS4_COLORS['darkgoldenrod'])
                    if self.slam.loop_closing.time_loop_detection.value:
                        time_loop_detection_signal = [img_id, self.slam.loop_closing.time_loop_detection.value]
                        self.timing_plt.draw(time_loop_detection_signal,'loop detection',color=mcolors.CSS4_COLORS['slategrey'])
                if self.slam.volumetric_integrator is not None:
                    if self.slam.volumetric_integrator.time_volumetric_integration.value:
                        time_volumetric_integration_signal = [img_id, self.slam.volumetric_integrator.time_volumetric_integration.value]
                        self.timing_plt.draw(time_volumetric_integration_signal,'volumetric integration',color=mcolors.CSS4_COLORS['darkviolet'], marker='+')
                        
                    
            # draw trajectory and alignment error    
            # NOTE: we must empty the alignment queue in any case
            new_alignment_data = None
            if self.viewer3D is not None:
                while not self.viewer3D.alignment_gt_data_queue.empty():
                    new_alignment_data = self.viewer3D.alignment_gt_data_queue.get_nowait()
            if self.traj_error_plt is not None and new_alignment_data is not None:
                num_samples = len(new_alignment_data.timestamps_associations)                
                new_alignment_timestamp = new_alignment_data.timestamps_associations[-1] if num_samples > 20 else None
                print(f'SlamPlotDrawer: new gt alignment timestamp: {new_alignment_timestamp}, rms_error: {new_alignment_data.rms_error}, max_error: {new_alignment_data.max_error}, is_est_aligned: {new_alignment_data.is_est_aligned}')
                if new_alignment_data.rms_error > 0 and self.last_alignment_timestamp != new_alignment_timestamp:
                    self.last_alignment_timestamp = new_alignment_timestamp
                    #new_alignment_data.copyTo(self.last_alignment_gt_data)
                    self.last_alignment_gt_data = new_alignment_data
                    # if not self.last_alignment_gt_data.is_est_aligned:
                    #     print(f'SlamPlotDrawer: realigning estimated and gt trajectories')
                    #     if self.slam.sensor_type != SensorType.MONOCULAR:
                    #         # align the gt to estimated trajectory (T_gt_est is estimated in SE(3))
                    #         for i in range(len(self.last_alignment_gt_data.gt_t_wi)):
                    #             self.last_alignment_gt_data.gt_t_wi[i] = np.dot(self.last_alignment_gt_data.T_gt_est[:3, :3], self.last_alignment_gt_data.gt_t_wi[i]) + self.last_alignment_gt_data.T_gt_est[:3, 3]
                    #     else:
                    #         # align the estimated trajectory to the gt (T_gt_est is estimated in Sim(3))
                    #         #T_est_gt = np.linalg.inv(self.last_alignment_gt_data.T_gt_est)
                    #         T_est_gt = Sim3Pose().from_matrix(self.last_alignment_gt_data.T_gt_est).inverse_matrix()
                    #         for i in range(len(self.last_alignment_gt_data.estimated_t_wi)):
                    #             self.last_alignment_gt_data.estimated_t_wi[i] = np.dot(T_est_gt[:3, :3], self.last_alignment_gt_data.estimated_t_wi[i]) + T_est_gt[:3, 3]
                                                
                    gt_traj = np.array(self.last_alignment_gt_data.gt_t_wi, dtype=float)
                    estimated_traj = np.array(self.last_alignment_gt_data.estimated_t_wi, dtype=float)
                    aligned_estimated_traj = (self.last_alignment_gt_data.T_gt_est[:3,:3] @ estimated_traj.T + self.last_alignment_gt_data.T_gt_est[:3, 3].reshape(3,1)).T                  
                    filter_timestamps = np.array(self.last_alignment_gt_data.timestamps_associations, dtype=float)
                    #print(f'SlamPlotDrawer: gt_traj: {gt_traj.shape}, estimated_traj: {estimated_traj.shape}, filter_timestamps: {filter_timestamps.shape}')
                    if False:
                        time_gt_x_signal = [filter_timestamps, gt_traj[:, 0]]
                        time_gt_y_signal = [filter_timestamps, gt_traj[:, 1]]
                        time_gt_z_signal = [filter_timestamps, gt_traj[:, 2]]
                        time_filter_x_signal = [filter_timestamps, aligned_estimated_traj[:, 0]]
                        time_filter_y_signal = [filter_timestamps, aligned_estimated_traj[:, 1]]
                        time_filter_z_signal = [filter_timestamps, aligned_estimated_traj[:, 2]]
                        self.traj_error_plt.draw(time_filter_x_signal, 'filter_x', color='r', append=False)
                        self.traj_error_plt.draw(time_filter_y_signal, 'filter_y', color='g', append=False)
                        self.traj_error_plt.draw(time_filter_z_signal, 'filter_z', color='b', append=False)
                        self.traj_error_plt.draw(time_gt_x_signal, 'gt_x', color='r', linestyle=':', append=False)
                        self.traj_error_plt.draw(time_gt_y_signal, 'gt_y', color='g', linestyle=':', append=False)
                        self.traj_error_plt.draw(time_gt_z_signal, 'gt_z', color='b', linestyle=':', append=False)
                    else:
                        traj_error = gt_traj - aligned_estimated_traj
                        traj_dists = np.linalg.norm(traj_error, axis=1)
                        rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))
                        print(f'SlamPlotDrawer: traj_error: {traj_error.shape}')
                        if False:
                            err_x_max = np.max(np.abs(traj_error[:, 0]))
                            err_y_max = np.max(np.abs(traj_error[:, 1]))                        
                            err_z_max = np.max(np.abs(traj_error[:, 2]))
                            print(f'SlamPlotDrawer: err_x_max: {err_x_max}, err_y_max: {err_y_max}, err_z_max: {err_z_max}')
                        time_errx_signal = [filter_timestamps, traj_error[:, 0]]
                        time_erry_signal = [filter_timestamps, traj_error[:, 1]]
                        time_errz_signal = [filter_timestamps, traj_error[:, 2]]
                        time_rms_error_signal = [filter_timestamps[-1], rms_error]
                        self.traj_error_plt.draw(time_errx_signal, 'err_x', color='r', append=False)
                        self.traj_error_plt.draw(time_erry_signal, 'err_y', color='g', append=False) 
                        self.traj_error_plt.draw(time_errz_signal, 'err_z', color='b', append=False)
                        self.traj_error_plt.draw(time_rms_error_signal, 'RMS error (ATE)', color='c', append=True)                                  
                
        except Exception as e:
            Printer.red(f'SlamPlotDrawer: draw: encountered exception: {e}')
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')
    


    
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
        self.matched_points_plt = factory_plot2d(xlabel='img id', ylabel='# matches',title='# matches')  
        #self.info_3dpoints_plt = factory_plot2d(xlabel='img id', ylabel='# points',title='info 3d points')      
        #self.chi2_error_plt = factory_plot2d(xlabel='img id', ylabel='error',title='mean chi2 error')
        #self.timing_plt = factory_plot2d(xlabel='img id', ylabel='s',title='timing')        
        self.traj_error_plt = factory_plot2d(xlabel='time [s]', ylabel='error',title='trajectories: gt vs (aligned)estimated')
        
        self.plt_list = [self.matched_points_plt, self.info_3dpoints_plt, self.chi2_error_plt, self.timing_plt, self.traj_error_plt]
        self.last_processed_kf_img_id = -1
        
    def quit(self):
        for plt in self.plt_list:
            if plt is not None:
                plt.quit()        
            
    def get_key(self):
        for plt in self.plt_list:
            if plt is not None:
                key = plt.get_key()
                if key != '':
                    return key    

    def draw(self, img_id, data_dict: dict):
        try:
            # draw matching info
            if self.matched_points_plt is not None:
                if "num_matched_kps" in data_dict:
                    self.matched_points_plt.draw([img_id, data_dict["num_matched_kps"]], '# keypoint matches', color='r')
                if "num_inliers" in data_dict:
                    self.matched_points_plt.draw([img_id, data_dict["num_inliers"]], '# inliers', color='g')
                if "num_matched_map_points" in data_dict:
                    self.matched_points_plt.draw([img_id, data_dict["num_matched_map_points"]], '# matched map pts', color='b')
                if "num_kf_ref_tracked_points" in data_dict:
                    self.matched_points_plt.draw([img_id, data_dict["num_kf_ref_tracked_points"]], '# $KF_{ref}$ tracked pts', color='c')
                if "descriptor_distance_sigma" in data_dict:
                    self.matched_points_plt.draw([img_id, data_dict["descriptor_distance_sigma"]], 'descriptor distance $\sigma_{th}$', color='k')

            # draw info about 3D points management by local mapping
            if self.info_3dpoints_plt is not None:
                if "last_processed_kf_img_id" in data_dict and self.last_processed_kf_img_id != data_dict["last_processed_kf_img_id"]:
                    self.last_processed_kf_img_id = data_dict["last_processed_kf_img_id"]
                    print(f'last_processed_kf_img_id: {self.last_processed_kf_img_id}')
                    if "last_num_triangulated_points" in data_dict:
                        self.info_3dpoints_plt.draw([img_id, data_dict["last_num_triangulated_points"]], '# temporal triangulated pts', color='r')
                    if "last_num_fused_points" in data_dict:
                        self.info_3dpoints_plt.draw([img_id, data_dict["last_num_fused_points"]], '# fused pts', color='g')
                    if "last_num_culled_keyframes" in data_dict:
                        self.info_3dpoints_plt.draw([img_id, data_dict["last_num_culled_keyframes"]], '# culled keyframes', color='b')
                    if "last_num_culled_points" in data_dict:
                        self.info_3dpoints_plt.draw([img_id, data_dict["last_num_culled_points"]], '# culled pts', color='c')
                    if "last_num_static_stereo_map_points" in data_dict:
                        self.info_3dpoints_plt.draw([img_id, data_dict["last_num_static_stereo_map_points"]], '# static triangulated pts', color='k')

            # draw chi2 error
            if self.chi2_error_plt is not None:
                if "mean_pose_opt_chi2_error" in data_dict:
                    self.chi2_error_plt.draw([img_id, data_dict["mean_pose_opt_chi2_error"]], 'pose opt chi2 error', color='r')
                if "mean_ba_chi2_error" in data_dict:
                    self.chi2_error_plt.draw([img_id, data_dict["mean_ba_chi2_error"]], 'LBA chi2 error', color='g')
                if "mean_graph_chi2_error" in data_dict:
                    self.chi2_error_plt.draw([img_id, data_dict["mean_graph_chi2_error"]], 'graph chi2 error', color='b')
                if "mean_BA_chi2_error" in data_dict:
                    self.chi2_error_plt.draw([img_id, data_dict["mean_BA_chi2_error"]], 'GBA chi2 error', color='k')

            # draw timings
            if self.timing_plt is not None:
                if "time_track" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_track"]], 'tracking', color='r')
                if "time_local_mapping" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_local_mapping"]], 'local mapping', color='g')
                if "time_LBA" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_LBA"]], 'LBA', color='b')
                if "time_local_mapping_triangulation" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_local_mapping_triangulation"]], 'local mapping triangulation', color='k')
                if "time_local_mapping_pts_culling" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_local_mapping_pts_culling"]], 'local mapping pts culling', color='c')
                if "time_local_mapping_kf_culling" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_local_mapping_kf_culling"]], 'local mapping kf culling', color='m')
                if "time_local_mapping_pts_fusion" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_local_mapping_pts_fusion"]], 'local mapping pts fusion', color='y')
                if "time_loop_closing" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_loop_closing"]], 'loop closing', color='darkgoldenrod')
                if "time_loop_detection" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_loop_detection"]], 'loop detection', color='slategrey')
                if "time_volumetric_integration" in data_dict:
                    self.timing_plt.draw([img_id, data_dict["time_volumetric_integration"]], 'volumetric integration', color='darkviolet', marker='+')
            
            # draw trajectory and alignment error    
            # NOTE: we must empty the alignment queue in any case
            new_alignment_data = None
            if self.viewer3D:
                while not self.viewer3D.alignment_gt_data_queue.empty():
                    new_alignment_data = self.viewer3D.alignment_gt_data_queue.get_nowait()
            if self.traj_error_plt is not None and new_alignment_data is not None:
                num_samples = len(new_alignment_data.timestamps_associations)                
                new_alignment_timestamp = new_alignment_data.timestamps_associations[-1] if num_samples > 20 else None
                print(f'SlamPlotDrawer: new gt alignment timestamp: {new_alignment_timestamp}, rms_error: {new_alignment_data.rms_error}, max_error: {new_alignment_data.max_error}, is_est_aligned: {new_alignment_data.is_est_aligned}')
                if new_alignment_data.rms_error > 0 and self.last_alignment_timestamp != new_alignment_timestamp:
                    self.last_alignment_timestamp = new_alignment_timestamp
                    #new_alignment_data.copyTo(self.last_alignment_gt_data)
                    self.last_alignment_gt_data = new_alignment_data
                    # if not self.last_alignment_gt_data.is_est_aligned:
                    #     print(f'SlamPlotDrawer: realigning estimated and gt trajectories')
                    #     if self.slam.sensor_type != SensorType.MONOCULAR:
                    #         # align the gt to estimated trajectory (T_gt_est is estimated in SE(3))
                    #         for i in range(len(self.last_alignment_gt_data.gt_t_wi)):
                    #             self.last_alignment_gt_data.gt_t_wi[i] = np.dot(self.last_alignment_gt_data.T_gt_est[:3, :3], self.last_alignment_gt_data.gt_t_wi[i]) + self.last_alignment_gt_data.T_gt_est[:3, 3]
                    #     else:
                    #         # align the estimated trajectory to the gt (T_gt_est is estimated in Sim(3))
                    #         #T_est_gt = np.linalg.inv(self.last_alignment_gt_data.T_gt_est)
                    #         T_est_gt = Sim3Pose().from_matrix(self.last_alignment_gt_data.T_gt_est).inverse_matrix()
                    #         for i in range(len(self.last_alignment_gt_data.estimated_t_wi)):
                    #             self.last_alignment_gt_data.estimated_t_wi[i] = np.dot(T_est_gt[:3, :3], self.last_alignment_gt_data.estimated_t_wi[i]) + T_est_gt[:3, 3]
                                                
                    gt_traj = np.array(self.last_alignment_gt_data.gt_t_wi, dtype=float)
                    estimated_traj = np.array(self.last_alignment_gt_data.estimated_t_wi, dtype=float)
                    aligned_estimated_traj = (self.last_alignment_gt_data.T_gt_est[:3,:3] @ estimated_traj.T + self.last_alignment_gt_data.T_gt_est[:3, 3].reshape(3,1)).T                  
                    filter_timestamps = np.array(self.last_alignment_gt_data.timestamps_associations, dtype=float)
                    #print(f'SlamPlotDrawer: gt_traj: {gt_traj.shape}, estimated_traj: {estimated_traj.shape}, filter_timestamps: {filter_timestamps.shape}')
                    if False:
                        time_gt_x_signal = [filter_timestamps, gt_traj[:, 0]]
                        time_gt_y_signal = [filter_timestamps, gt_traj[:, 1]]
                        time_gt_z_signal = [filter_timestamps, gt_traj[:, 2]]
                        time_filter_x_signal = [filter_timestamps, aligned_estimated_traj[:, 0]]
                        time_filter_y_signal = [filter_timestamps, aligned_estimated_traj[:, 1]]
                        time_filter_z_signal = [filter_timestamps, aligned_estimated_traj[:, 2]]
                        self.traj_error_plt.draw(time_filter_x_signal, 'filter_x', color='r', append=False)
                        self.traj_error_plt.draw(time_filter_y_signal, 'filter_y', color='g', append=False)
                        self.traj_error_plt.draw(time_filter_z_signal, 'filter_z', color='b', append=False)
                        self.traj_error_plt.draw(time_gt_x_signal, 'gt_x', color='r', linestyle=':', append=False)
                        self.traj_error_plt.draw(time_gt_y_signal, 'gt_y', color='g', linestyle=':', append=False)
                        self.traj_error_plt.draw(time_gt_z_signal, 'gt_z', color='b', linestyle=':', append=False)
                    else:
                        traj_error = gt_traj - aligned_estimated_traj
                        traj_dists = np.linalg.norm(traj_error, axis=1)
                        rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))
                        print(f'SlamPlotDrawer: traj_error: {traj_error.shape}')
                        if False:
                            err_x_max = np.max(np.abs(traj_error[:, 0]))
                            err_y_max = np.max(np.abs(traj_error[:, 1]))                        
                            err_z_max = np.max(np.abs(traj_error[:, 2]))
                            print(f'SlamPlotDrawer: err_x_max: {err_x_max}, err_y_max: {err_y_max}, err_z_max: {err_z_max}')
                        time_errx_signal = [filter_timestamps, traj_error[:, 0]]
                        time_erry_signal = [filter_timestamps, traj_error[:, 1]]
                        time_errz_signal = [filter_timestamps, traj_error[:, 2]]
                        time_rms_error_signal = [filter_timestamps[-1], rms_error]
                        self.traj_error_plt.draw(time_errx_signal, 'err_x', color='r', append=False)
                        self.traj_error_plt.draw(time_erry_signal, 'err_y', color='g', append=False) 
                        self.traj_error_plt.draw(time_errz_signal, 'err_z', color='b', append=False)
                        self.traj_error_plt.draw(time_rms_error_signal, 'RMS error (ATE)', color='c', append=True)                                  
                
        except Exception as e:
            Printer.red(f'SlamPlotDrawer: draw: encountered exception: {e}')
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')
    