import sys
import numpy as np
import math

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


from pyslam.config import Config


from pyslam.utilities.geometry import yaw_matrix, roll_matrix, pitch_matrix, poseRt
from pyslam.utilities.geom_trajectory import (
    align_trajectories_with_svd,
    set_rotations_from_translations,
)


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


class TestAlignSVD:
    def __init__(self):
        self.decimation_factor = 2  # between gt and estimated trajectories
        self.spatial_noise_mean = 1e-3  # [m] added to gt 3D positions
        self.spatial_noise_std = 0.01  # [m] added to gt 3D positions
        self.time_s_noise_mean = 1e-5  # [s] added to gt timestamps
        self.time_s_noise_std = 1e-5  # [s] added to gt timestamps

    def generate_gt_poses(self):
        Ts = 0.1  # 0.015  # sample time
        t_start_s = 0
        t_stop_s = 100
        vel = 0.2  # m/s

        Tperiod_s = 20  # s
        omega = 2 * math.pi / Tperiod_s  # rad/s
        num_samples = math.ceil((t_stop_s - t_start_s) / Ts)

        self.t_s = np.linspace(t_start_s, t_stop_s, num_samples)
        self.t_ns = self.t_s * 1e9

        # generate the ground truth data
        self.gt_x = self.t_s * vel
        self.gt_y = 2 * np.sin(omega * self.t_s)
        self.gt_z = np.zeros_like(self.t_s)

        self.gt_t_wi = np.stack((self.gt_x, self.gt_y, self.gt_z), axis=1)
        self.gt_t_ns = self.t_ns
        self.gt_t_s = self.t_ns * 1e-9
        self.num_gt_samples = len(self.gt_t_ns)
        print(f"num gt samples: {self.num_gt_samples}")

        self.gt_R_wi = set_rotations_from_translations(self.gt_t_wi)
        self.gt_T_wi = [
            poseRt(self.gt_R_wi[i], self.gt_t_wi[i]) for i in range(self.num_gt_samples)
        ]

    def generate_estimated_poses(self):
        # To get the estimated trajectory, first, subsample and add noise to gt
        self.filter_t_ns = self.gt_t_ns[
            :: self.decimation_factor
        ].copy()  # select every decimation_factor-th timestamp
        self.filter_t_wi = self.gt_t_wi[:: self.decimation_factor].copy()
        self.num_filter_samples = len(self.filter_t_ns)
        print(f"num filter samples: {self.num_filter_samples}")

        self.filter_t_s = self.filter_t_ns * 1e-9
        time_noise = np.random.normal(
            self.time_s_noise_mean, self.time_s_noise_std, self.filter_t_s.shape
        )
        self.filter_t_s = self.filter_t_s + time_noise

        spatial_noise = np.random.normal(
            self.spatial_noise_mean, self.spatial_noise_std, self.filter_t_wi.shape
        )
        self.filter_t_wi = self.filter_t_wi + spatial_noise

        self.filter_R_wi = set_rotations_from_translations(self.filter_t_wi)
        self.filter_T_wi = [
            poseRt(self.filter_R_wi[i], self.filter_t_wi[i]) for i in range(self.num_filter_samples)
        ]

        # Transform the first version of estimated traj by using a know Sim(3) transform
        scale = 6.4
        yaw = math.pi / 4
        pitch = -math.pi / 8
        roll = math.pi / 3
        sRwc = scale * (yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll))
        twc = np.array([0.4, 0.1, -0.1])
        self.Twc = poseRt(sRwc, twc)  # known Sim(3) transform

        self.filter_t_wi = (sRwc @ self.filter_t_wi.T + twc.reshape(3, 1)).T
        self.filter_R_wi = [sRwc @ self.filter_R_wi[i] for i in range(self.num_filter_samples)]
        self.filter_T_wi = [self.Twc @ self.filter_T_wi[i] for i in range(self.num_filter_samples)]

        self.find_scale = scale != 1.0
        print("find_scale:", self.find_scale)

    def set_up(self):

        # to remove randomness
        np.random.seed(0)

        self.generate_gt_poses()

        self.generate_estimated_poses()

    def test_align_svd(self):

        align_to_gt = True

        T_gt_est, error, alignment_data = align_trajectories_with_svd(
            self.filter_t_s,
            self.filter_t_wi,
            self.gt_t_s,
            self.gt_t_wi,
            compute_align_error=True,
            find_scale=self.find_scale,
        )

        if align_to_gt:
            self.aligned_t_wi = (
                T_gt_est[:3, :3] @ self.filter_t_wi.T + T_gt_est[:3, 3].reshape(3, 1)
            ).T
        else:
            self.aligned_t_wi = self.filter_t_wi  # no need to align

        print(f"time noise mean: {self.time_s_noise_mean}, time noise std: {self.time_s_noise_std}")
        print(
            f"spatial noise mean: {self.spatial_noise_mean}, spatail noise std: {self.spatial_noise_std}"
        )

        print(f"num associations: {alignment_data.num_associations}")
        print("alignment error:", error)

        print("T_align:\n", T_gt_est)
        print("T_gt:\n", np.linalg.inv(self.Twc))

        T_err = T_gt_est @ self.Twc - np.eye(4)
        print(f"T_err norm: {np.linalg.norm(T_err)}")

        # plot ground truth and estimated trajectories
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        # ax.plot(self.filter_t_wi[:,0], self.filter_t_wi[:,1], self.filter_t_wi[:,2], label='estimated traj')
        ax.plot(
            self.aligned_t_wi[:, 0],
            self.aligned_t_wi[:, 1],
            self.aligned_t_wi[:, 2],
            label="aligned 3D traj",
        )
        ax.plot(self.gt_t_wi[:, 0], self.gt_t_wi[:, 1], self.gt_t_wi[:, 2], label="gt 3D traj")

        set_axes_equal(ax)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()


if __name__ == "__main__":
    test = TestAlignSVD()
    test.set_up()
    test.test_align_svd()
