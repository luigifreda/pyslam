import sys
import numpy as np
import math

import time

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


from pyslam.config import Config

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.utils_geom import yaw_matrix, roll_matrix, pitch_matrix, poseRt
from pyslam.utilities.utils_geom_trajectory import align_trajectories_with_svd, set_rotations_from_translations, find_poses_associations
from pyslam.utilities.utils_eval import evaluate_evo


class TestAlignSVD():
    def __init__(self):
        self.decimation_factor = 2      # between gt and estimated trajectories
        self.spatial_noise_mean = 1e-3  # [m] added to gt 3D positions
        self.spatial_noise_std = 0.01   # [m] added to gt 3D positions
        self.time_s_noise_mean = 1e-5   # [s] added to gt timestamps
        self.time_s_noise_std = 1e-5    # [s] added to gt timestamps
    
    
    def generate_gt_poses(self):
        Ts = 0.1 # 0.015  # sample time 
        t_start_s = 0
        t_stop_s = 100
        vel = 0.2 # m/s
        
        Tperiod_s = 20 # s
        omega = 2*math.pi/Tperiod_s # rad/s
        num_samples = math.ceil((t_stop_s - t_start_s)/Ts)
        
        self.t_s = np.linspace(t_start_s, t_stop_s, num_samples) 
        self.t_ns = self.t_s * 1e9
        
        # generate the ground truth data 
        self.gt_x = self.t_s * vel
        self.gt_y = 2*np.sin(omega*self.t_s)
        self.gt_z = np.zeros_like(self.t_s)
        
        self.gt_t_wi = np.stack((self.gt_x, self.gt_y, self.gt_z), axis=1)
        self.gt_t_ns = self.t_ns
        self.gt_t_s = self.t_ns * 1e-9
        self.num_gt_samples = len(self.gt_t_ns)
        print(f'num gt samples: {self.num_gt_samples}')
        
        self.gt_R_wi = set_rotations_from_translations(self.gt_t_wi)
        self.gt_T_wi = [poseRt(self.gt_R_wi[i],self.gt_t_wi[i]) for i in range(self.num_gt_samples)]
        
        
    def generate_estimated_poses(self):
        # To get the estimated trajectory, first, subsample and add noise to gt          
        self.filter_t_ns =  self.gt_t_ns[::self.decimation_factor].copy() # select every decimation_factor-th timestamp    
        self.filter_t_wi = self.gt_t_wi[::self.decimation_factor].copy()
        self.num_filter_samples = len(self.filter_t_ns)
        print(f'num filter samples: {self.num_filter_samples}')
                
        self.filter_t_s = self.filter_t_ns * 1e-9
        time_noise = np.random.normal(self.time_s_noise_mean, self.time_s_noise_std, self.filter_t_s.shape)
        self.filter_t_s = self.filter_t_s + time_noise
        
        spatial_noise = np.random.normal(self.spatial_noise_mean, self.spatial_noise_std, self.filter_t_wi.shape)
        self.filter_t_wi = self.filter_t_wi + spatial_noise
        
        self.filter_R_wi = set_rotations_from_translations(self.filter_t_wi)
        self.filter_T_wi = [poseRt(self.filter_R_wi[i],self.filter_t_wi[i]) for i in range(self.num_filter_samples)]    
                
        # Transform the first version of estimated traj by using a know Sim(3) transform
        scale = 6.4
        yaw = math.pi/4
        pitch = -math.pi/8
        roll = math.pi/3
        sRwc = scale*(yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll))
        twc = np.array([0.4, 0.1, -0.1])
        self.Twc = poseRt(sRwc, twc)  # known Sim(3) transform
        
        self.filter_t_wi = (sRwc @ self.filter_t_wi.T + twc.reshape(3,1)).T
        self.filter_R_wi = [sRwc @ self.filter_R_wi[i] for i in range(self.num_filter_samples)]
        self.filter_T_wi = [self.Twc @ self.filter_T_wi[i] for i in range(self.num_filter_samples)]
        
        self.find_scale = scale != 1.0
        print("find_scale:", self.find_scale)        
                
    def set_up(self):
        
        #to remove randomness 
        np.random.seed(0)
                
        self.generate_gt_poses()
        
        self.generate_estimated_poses()

    def test_align_svd(self):
        
        align_to_gt = True

        assoc_timestamps, assoc_filter_poses, assoc_gt_poses = find_poses_associations(self.filter_t_s , self.filter_T_wi, self.gt_t_s, self.gt_T_wi)
        num_assoc_samples = len(assoc_timestamps)
        assoc_t_wi = np.array([assoc_filter_poses[i][0:3,3] for i in range(num_assoc_samples)]).reshape(num_assoc_samples, 3)
        assoc_gt_t_wi = np.array([assoc_gt_poses[i][0:3,3] for i in range(num_assoc_samples)]).reshape(num_assoc_samples, 3)
        
        # T_gt_est, rms_error, alignment_data = align_trajectories_with_svd(self.filter_t_s, self.filter_t_wi, self.gt_t_s, self.gt_t_wi, \
        #     compute_align_error=True, find_scale=self.find_scale)
        T_gt_est, rms_error, alignment_data = align_trajectories_with_svd(assoc_timestamps,  assoc_t_wi, assoc_timestamps, assoc_gt_t_wi, \
            compute_align_error=True, find_scale=self.find_scale)        
        
        ape_stats, T_evo_gt_est = evaluate_evo(poses_est=assoc_filter_poses, poses_gt=assoc_gt_poses, 
                                                is_monocular=self.find_scale, plot_dir=None, label=None,  
                                                save_metrics=False, save_plot=False)
        
        if align_to_gt: 
            self.aligned_t_wi = (T_gt_est[:3,:3] @ self.filter_t_wi.T + T_gt_est[:3,3].reshape(3,1)).T
            self.aligned_T_wi = [T_gt_est @ self.filter_T_wi[i] for i in range(self.num_filter_samples)]
        else:
            self.aligned_t_wi = self.filter_t_wi # no need to align
            self.aligned_T_wi = self.filter_T_wi
     
        print(f'time noise mean: {self.time_s_noise_mean}, time noise std: {self.time_s_noise_std}')
        print(f'spatial noise mean: {self.spatial_noise_mean}, spatial noise std: {self.spatial_noise_std}')
                    
        print(f'num associations: {alignment_data.num_associations}')
        print("alignment rms error:", rms_error)

        print("T_gt:\n", np.linalg.inv(self.Twc))                    
        print("T_align:\n", T_gt_est)

        T_err = T_gt_est @ self.Twc - np.eye(4)
        print(f"T_err norm: {np.linalg.norm(T_err)}")
        
        print(f'T_evo_align:\n {T_evo_gt_est}')        
        T_err_evo = T_evo_gt_est @ self.Twc  - np.eye(4)
        print(f"T_err_evo norm: {np.linalg.norm(T_err_evo)}")
        
        assoc_filter_poses_aligned = [T_gt_est @ assoc_filter_poses[i] for i in range(num_assoc_samples)]
        errors = [assoc_filter_poses_aligned[i][:3,3] - assoc_gt_poses[i][:3,3] for i in range(num_assoc_samples)]   
        distances = np.linalg.norm(errors, axis=1)
        squared_errors = np.power(distances, 2) 
        #print(f'align_trajectories: squared_errors: {squared_errors}')        
        rms_error = np.sqrt(np.mean(squared_errors))  
        print(f"errors rms: {rms_error}")
        
        poses_est_evo_aligned = [T_evo_gt_est @ assoc_filter_poses[i] for i in range(num_assoc_samples)]
        evo_errors = [poses_est_evo_aligned[i][:3,3] - assoc_gt_poses[i][:3,3] for i in range(num_assoc_samples)]
        evo_distances = np.linalg.norm(evo_errors, axis=1)
        evo_squared_errors = np.power(evo_distances,2)
        evo_rms_error = np.sqrt(np.mean(evo_squared_errors))
        print(f"evo errors rms: {evo_rms_error}")
        
        print(f'evo ATE: {ape_stats["rmse"]}')
                
        
        viewer3d = Viewer3D()
        viewer3d.draw_cameras([self.gt_T_wi, self.aligned_T_wi, poses_est_evo_aligned], [[1,0,0], [0,1,0], [0,0,1]])
        
        while not viewer3d.is_closed():
            time.sleep(0.1)
            

if __name__ == '__main__':
    test = TestAlignSVD()
    test.set_up()
    test.test_align_svd()

    