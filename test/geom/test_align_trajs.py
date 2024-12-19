import sys
import numpy as np
import math

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

sys.path.append("../../")

from utils_geom import align_trajs_with_svd, yaw_matrix, roll_matrix, pitch_matrix, poseRt

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # The center of the data ranges
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # Maximum range across all axes
    max_range = max(x_range, y_range, z_range)

    # Set the limits to be symmetric about the center
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

class TestAlignSVD():
    def __init__(self):
        pass

    def setUp(self):
        
        
        Ts = 0.015  # sample time 
        t_start_s = 0
        t_stop_s = 100
        vel = 0.2 # m/s
        
        Tperiod_s = 20 # s
        omega = 2*math.pi/20 # rad/s
        
        num = math.ceil((t_stop_s - t_start_s)/Ts)
        # generate the ground truth data for testing
        self.t_s = np.linspace(t_start_s, t_stop_s, num) 
        self.t_ns = self.t_s * 1e9
        
        self.gt_x = self.t_s * vel
        self.gt_y = 2*np.sin(omega*self.t_s)
        self.gt_z = np.zeros_like(self.t_s)
        
        self.gt_t_w_i = np.stack((self.gt_x, self.gt_y, self.gt_z), axis=1)
        self.gt_t_ns = self.t_ns
        
        # subsample and add noise 
        self.decimation_factor = 2
        self.noise_mean = 1e-3
        self.noise_std = 0.01
        
        self.filter_t_ns =  self.gt_t_ns[::self.decimation_factor].copy() # select every decimation_factor-th timestamp    
        self.filter_t_w_i = self.gt_t_w_i[::self.decimation_factor].copy()
        
        noise = np.random.normal(self.noise_mean, self.noise_std, self.filter_t_w_i.shape)
        self.filter_t_w_i = self.filter_t_w_i + noise
        
        # transform the estimated traj
        scale = 6.4
        yaw = math.pi/4
        pitch = -math.pi/8
        roll = math.pi/3
        Rwc = scale*(yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll))
        twc = np.array([0.4, 0.1, -0.1])
        self.Twc = poseRt(Rwc, twc)
        
        self.filter_t_w_i = (Rwc @ self.filter_t_w_i.T + twc.reshape(3,1)).T
        
        self.find_scale = scale != 1.0
        print("find_scale:", self.find_scale)
    
    def test_align_svd(self):
        
        align_to_gt = True

        T_gt_est, error, _ = align_trajs_with_svd(self.filter_t_ns, self.filter_t_w_i, self.gt_t_ns, self.gt_t_w_i, \
            align_gt=False, compute_align_error=True, find_scale=self.find_scale)
        
        if align_to_gt: 
            self.aligned_t_w_i = (T_gt_est[:3,:3] @ self.filter_t_w_i.T + T_gt_est[:3,3].reshape(3,1)).T
        else:
            self.aligned_t_w_i = self.filter_t_w_i # no need to align
            
        print("alignment error:", error)
                    
        print("T_align:\n", T_gt_est)
        print("T_gt:\n", np.linalg.inv(self.Twc))
        
        T_err = T_gt_est @ self.Twc - np.eye(4)
        print(f"T_err norm: {np.linalg.norm(T_err)}")
        print(f'noise mean: {self.noise_mean}, noise std: {self.noise_std}')
        
        # plot ground truth and estimated trajectories
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='3d')
        
        #ax.plot(self.filter_t_w_i[:,0], self.filter_t_w_i[:,1], self.filter_t_w_i[:,2], label='estimated traj')     
        ax.plot(self.aligned_t_w_i[:,0], self.aligned_t_w_i[:,1], self.aligned_t_w_i[:,2], label='aligned traj')   
        ax.plot(self.gt_t_w_i[:,0], self.gt_t_w_i[:,1], self.gt_t_w_i[:,2], label='gt traj')
                
        set_axes_equal(ax)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
            

if __name__ == '__main__':
    test = TestAlignSVD()
    test.setUp()
    test.test_align_svd()

    