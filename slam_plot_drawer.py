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


from utils_sys import getchar, Printer 

import platform
import traceback

from slam import Slam

from mplot_thread import Mplot2d
from qtplot_thread import Qplot2d
import matplotlib.colors as mcolors

kUseQtplot2d = False
if platform.system() == 'Darwin':
    kUseQtplot2d = True # Under mac force the usage of Qtplot2d: It is smoother 


def factory_plot2d(*args,**kwargs):
    if kUseQtplot2d:
        return Qplot2d(*args,**kwargs)
    else:
        return Mplot2d(*args,**kwargs)
    
    
class SlamPlotDrawer:
    def __init__(self, slam: Slam):
        self.slam = slam
        
        self.matched_points_plt = None
        self.info_3dpoints_plt = None
        self.chi2_error_plt = None
        self.timing_plt = None
        
        # To disable one of them just comment it out
        self.matched_points_plt = factory_plot2d(xlabel='img id', ylabel='# matches',title='# matches')  
        #self.info_3dpoints_plt = factory_plot2d(xlabel='img id', ylabel='# points',title='info 3d points')      
        self.chi2_error_plt = factory_plot2d(xlabel='img id', ylabel='error',title='mean chi2 error')
        self.timing_plt = factory_plot2d(xlabel='img id', ylabel='s',title='timing')        
        
        self.last_processed_kf_img_id = -1
        
    def quit(self):
        if self.matched_points_plt is not None:
            self.matched_points_plt.quit()
        if self.info_3dpoints_plt is not None:
            self.info_3dpoints_plt.quit()    
        if self.chi2_error_plt is not None:
            self.chi2_error_plt.quit()
        if self.timing_plt is not None:
            self.timing_plt.quit()            
            
    def get_key(self):
        key = self.matched_points_plt.get_key() if self.matched_points_plt is not None else None
        if key == '' or key is None:
            key = self.info_3dpoints_plt.get_key() if self.info_3dpoints_plt is not None else None
        if key == '' or key is None:
            key = self.chi2_error_plt.get_key() if self.chi2_error_plt is not None else None
        if key == '' or key is None:
            key = self.timing_plt.get_key() if self.timing_plt is not None else None
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
                
        except Exception as e:
            Printer.red(f'SlamPlotDrawer: draw: encountered exception: {e}')
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')
    
