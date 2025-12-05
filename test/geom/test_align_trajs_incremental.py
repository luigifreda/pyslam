import sys
import numpy as np
import math

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


from pyslam.config import Config

import trajectory_tools


from pyslam.utilities.geometry import yaw_matrix, roll_matrix, pitch_matrix, poseRt
from pyslam.utilities.geom_trajectory import (
    align_trajectories_with_svd,
    set_rotations_from_translations,
    compute_alignment_errors,
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
        self.decimation_factor = 4  # between gt and estimated trajectories
        self.spatial_noise_mean = 1e-3  # [m] added to gt 3D positions
        self.spatial_noise_std = 0.01  # [m] added to gt 3D positions
        self.time_s_noise_mean = 1e-5  # [s] added to gt timestamps
        self.time_s_noise_std = 1e-5  # [s] added to gt timestamps
        # Additional noise for stress testing incremental aligners
        # This noise is added to filter positions at each time step to simulate LBA updates
        # At each step, noise is added to both current and previous positions
        self.incremental_noise_std = (
            0.001  # [m] std dev of additional noise per sample per time step
        )
        self.incremental_noise_enabled = True  # Enable/disable incremental noise

    def generate_gt_poses(self):
        Ts = 0.05  # 0.015  # sample time
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

        # Final batch alignment on all data (for comparison)
        T_gt_est_final, error_final, alignment_data_final = align_trajectories_with_svd(
            self.filter_t_s,
            self.filter_t_wi,
            self.gt_t_s,
            self.gt_t_wi,
            compute_align_error=True,
            find_scale=self.find_scale,
        )

        if align_to_gt:
            self.aligned_t_wi = (
                T_gt_est_final[:3, :3] @ self.filter_t_wi.T + T_gt_est_final[:3, 3].reshape(3, 1)
            ).T
        else:
            self.aligned_t_wi = self.filter_t_wi  # no need to align

        print(f"time noise mean: {self.time_s_noise_mean}, time noise std: {self.time_s_noise_std}")
        print(
            f"spatial noise mean: {self.spatial_noise_mean}, spatail noise std: {self.spatial_noise_std}"
        )

        print(f"Final batch alignment - num associations: {alignment_data_final.num_associations}")
        print("Final batch alignment error:", error_final)

        print("Final T_align:\n", T_gt_est_final)
        print("T_gt:\n", np.linalg.inv(self.Twc))

        T_err_final = T_gt_est_final @ self.Twc - np.eye(4)
        print(f"Final T_err norm: {np.linalg.norm(T_err_final)}")

        # plot ground truth and estimated trajectories
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        # ax.plot(self.filter_t_wi[:,0], self.filter_t_wi[:,1], self.filter_t_wi[:,2], label='estimated traj')
        ax.plot(
            self.aligned_t_wi[:, 0],
            self.aligned_t_wi[:, 1],
            self.aligned_t_wi[:, 2],
            label="Aligned 3D trajectory (batch)",
        )
        ax.plot(
            self.gt_t_wi[:, 0], self.gt_t_wi[:, 1], self.gt_t_wi[:, 2], label="GT 3D trajectory"
        )

        set_axes_equal(ax)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # plt.show()

    def test_align_incremental_vs_batch(self):
        """Test incremental alignment vs batch alignment at each step"""

        # Store original filter trajectory (before incremental noise)
        filter_t_wi_original = self.filter_t_wi.copy()
        filter_t_s_original = self.filter_t_s.copy()

        # Pre-generate drift-like incremental noise as a single vector of delta position samples
        # The drift is generated by integrating random noise starting from the first timestamp
        # At each time step i, a drift increment is added to all positions j <= i
        # This creates a cumulative drift that accumulates over time steps
        # This simulates LBA where all positions get updated at each step with accumulating drift
        if self.incremental_noise_enabled:
            # Use a separate random seed for incremental noise to ensure reproducibility
            # but different from the initial setup noise
            rng_incremental = np.random.RandomState(42)
            # Generate delta position samples (random noise increments per sample)
            # delta_positions[sample, dim] is the noise increment for sample 'sample'
            delta_positions = rng_incremental.normal(
                0.0, self.incremental_noise_std, (self.num_filter_samples, 3)
            )
            # Integrate (cumulative sum) starting from sample 0 to get base drift per sample
            # base_drift[sample, dim] = sum of delta_positions[0:sample+1, dim]
            base_drift = np.cumsum(delta_positions, axis=0)
            print(
                f"\nIncremental drift-like noise enabled: std={self.incremental_noise_std} [m] per sample"
            )
            print(f"  Drift is generated by integrating random noise starting from first timestamp")
            print(f"  At each time step i, drift increment is added to all positions j <= i")
            print(f"  This simulates LBA with systematic bias that accumulates over time steps")
        else:
            base_drift = np.zeros((self.num_filter_samples, 3))
            print("\nIncremental noise disabled")

        # Initialize incremental aligners with GT data
        opts = trajectory_tools.AlignmentOptions()
        opts.max_align_dt = 1e-1
        opts.find_scale = self.find_scale
        opts.verbose = False

        # Convert GT data to lists for C++ module
        gt_timestamps_list = self.gt_t_s.tolist()
        gt_t_wi_list = [self.gt_t_wi[i] for i in range(self.num_gt_samples)]

        incremental_aligner_no_lba = trajectory_tools.IncrementalTrajectoryAlignerNoLBA(
            gt_timestamps_list, gt_t_wi_list, opts
        )

        incremental_aligner = trajectory_tools.IncrementalTrajectoryAligner(
            gt_timestamps_list, gt_t_wi_list, opts
        )

        # Storage for results at each step
        batch_T_gt_est_list = []
        incremental_no_lba_T_gt_est_list = []
        incremental_T_gt_est_list = []
        batch_errors = []
        incremental_no_lba_errors = []
        incremental_errors = []
        batch_n_pairs_list = []
        incremental_no_lba_n_pairs = []
        incremental_n_pairs = []
        T_err_batch_list = []
        T_err_incremental_no_lba_list = []
        T_err_incremental_list = []

        # Process each estimated sample incrementally
        for i in range(self.num_filter_samples):
            # Compute corrected positions for all samples up to current step i
            # All three aligners should see the same data at each step
            # At each step i, incremental noise is added to all positions j <= i to simulate LBA
            filter_t_s_subset_list = filter_t_s_original[: i + 1].tolist()
            filter_t_wi_subset_list = []
            for j in range(i + 1):
                # Start with original position
                corrected_position = filter_t_wi_original[j].copy()
                # Add drift that decays over time
                # At step i, sample j gets drift = base_drift[j] * exp(-(i - j) / deltaT_decay)
                # This means: the longer a sample has been around, the less drift it accumulates
                # Sample j first appears at step j, so at step i it has been around for (i - j) steps
                # deltaT_decay is the time constant for the decay of the drift
                deltaT_decay = 5.0  # [s] time constant for the decay of the drift
                if self.incremental_noise_enabled:
                    # Number of steps since sample j was first added
                    steps_since_added = i - j
                    factor = math.exp(-steps_since_added / deltaT_decay)
                    # Drift accumulates: each step adds base_drift[j] to sample j
                    drift_for_sample = base_drift[j, :] * factor
                    corrected_position = corrected_position + drift_for_sample
                filter_t_wi_subset_list.append(corrected_position)

            # All three aligners see the same corrected positions at step i
            # The difference is:
            # - NoLBA: can only add sample i with its current corrected position
            #   It can't update statistics for samples 0..i-1 that were already added
            # - Full and Batch: recompute from scratch with all corrected positions up to i

            # Add current sample i to NoLBA aligner with its current corrected position
            # Note: NoLBA can't go back and update samples 0..i-1 that were already added
            current_corrected_position = filter_t_wi_subset_list[i]
            incremental_aligner_no_lba.add_estimate(
                filter_t_s_original[i], current_corrected_position
            )

            # Get incremental alignment result (NoLBA version)
            inc_no_lba_result = incremental_aligner_no_lba.result()

            # Update incremental aligner (full trajectory version) with all corrected positions
            # This recomputes from scratch, so it can use updated positions for all samples
            incremental_aligner.update_trajectory(filter_t_s_subset_list, filter_t_wi_subset_list)

            # Get incremental alignment result (full trajectory version)
            inc_result = incremental_aligner.result()

            # Compute batch alignment using all corrected positions up to current step
            # This also recomputes from scratch with all current positions
            # Ensure arrays are contiguous for C++ module
            filter_t_s_subset = np.ascontiguousarray(filter_t_s_original[: i + 1])
            filter_t_wi_subset = np.ascontiguousarray(filter_t_wi_subset_list)

            # Initialize batch associations (will be set if batch alignment succeeds)
            batch_est_associations = None
            batch_gt_associations = None

            if len(filter_t_s_subset) >= 2:  # Need at least 2 points for alignment
                batch_T_gt_est, batch_error, batch_alignment_data = align_trajectories_with_svd(
                    filter_t_s_subset,
                    filter_t_wi_subset,
                    self.gt_t_s,
                    self.gt_t_wi,
                    compute_align_error=True,
                    find_scale=self.find_scale,
                    verbose=False,
                )

                # Store batch results
                batch_T_gt_est_list.append(batch_T_gt_est.copy())
                batch_n_pairs_list.append(batch_alignment_data.num_associations)

                # Store batch associations for fair error comparison
                batch_est_associations = np.array(batch_alignment_data.estimated_t_wi)
                batch_gt_associations = np.array(batch_alignment_data.gt_t_wi)

                # Compute alignment RMS error using current estimated trajectory and current transformation
                # Use the current estimated trajectory (updated positions at step i) and GT interpolated positions
                gt_interp_positions = np.array(
                    [
                        [
                            np.interp(filter_t_s_subset[j], self.gt_t_s, self.gt_t_wi[:, dim])
                            for dim in range(3)
                        ]
                        for j in range(len(filter_t_s_subset))
                    ]
                )
                # Compute RMS error: apply batch's current transformation to current estimated trajectory
                # and compare with GT
                batch_rms_error, _, _, _ = compute_alignment_errors(
                    batch_T_gt_est, filter_t_wi_subset, gt_interp_positions
                )
                batch_errors.append(batch_rms_error)

                # Compute error w.r.t. known transform
                T_err_batch = batch_T_gt_est @ self.Twc - np.eye(4)
                T_err_batch_list.append(np.linalg.norm(T_err_batch))
            else:
                # Not enough points yet
                batch_T_gt_est_list.append(None)
                batch_errors.append(-1.0)
                batch_n_pairs_list.append(0)
                T_err_batch_list.append(-1.0)

            # Initialize transformation variables for debugging
            T_no_lba = None
            T_inc = None

            # Store incremental results (NoLBA version)
            if inc_no_lba_result.valid:
                incremental_no_lba_T_gt_est_list.append(inc_no_lba_result.T_gt_est.copy())
                incremental_no_lba_n_pairs.append(inc_no_lba_result.n_pairs)

                # Compute error w.r.t. known transform
                T_err_incremental_no_lba = inc_no_lba_result.T_gt_est @ self.Twc - np.eye(4)
                T_err_incremental_no_lba_list.append(np.linalg.norm(T_err_incremental_no_lba))

                # Compute alignment RMS error using current estimated trajectory and current transformation
                # Use the current estimated trajectory (updated positions at step i) and GT interpolated positions
                if len(filter_t_wi_subset) >= 2:
                    # Store transformation for debugging
                    T_no_lba = inc_no_lba_result.T_gt_est.copy()
                    # Interpolate GT positions for the current estimated trajectory timestamps
                    gt_interp_positions = np.array(
                        [
                            [
                                np.interp(filter_t_s_subset[j], self.gt_t_s, self.gt_t_wi[:, dim])
                                for dim in range(3)
                            ]
                            for j in range(len(filter_t_s_subset))
                        ]
                    )
                    # Compute RMS error: apply NoLBA's current transformation to current estimated trajectory
                    # and compare with GT
                    rms_error, _, _, _ = compute_alignment_errors(
                        T_no_lba, filter_t_wi_subset, gt_interp_positions
                    )
                    incremental_no_lba_errors.append(rms_error)
                else:
                    incremental_no_lba_errors.append(-1.0)
            else:
                incremental_no_lba_T_gt_est_list.append(None)
                incremental_no_lba_n_pairs.append(0)
                T_err_incremental_no_lba_list.append(-1.0)
                incremental_no_lba_errors.append(-1.0)

            # Store incremental results (full trajectory version)
            if inc_result.valid:
                incremental_T_gt_est_list.append(inc_result.T_gt_est.copy())
                incremental_n_pairs.append(inc_result.n_pairs)

                # Compute error w.r.t. known transform
                T_err_incremental = inc_result.T_gt_est @ self.Twc - np.eye(4)
                T_err_incremental_list.append(np.linalg.norm(T_err_incremental))

                # Compute alignment RMS error using current estimated trajectory and current transformation
                # Use the current estimated trajectory (updated positions at step i) and GT interpolated positions
                if len(filter_t_wi_subset) >= 2:
                    # Store transformation for debugging
                    T_inc = inc_result.T_gt_est.copy()
                    # Interpolate GT positions for the current estimated trajectory timestamps
                    gt_interp_positions = np.array(
                        [
                            [
                                np.interp(filter_t_s_subset[j], self.gt_t_s, self.gt_t_wi[:, dim])
                                for dim in range(3)
                            ]
                            for j in range(len(filter_t_s_subset))
                        ]
                    )
                    # Compute RMS error: apply Incremental's current transformation to current estimated trajectory
                    # and compare with GT
                    rms_error, _, _, _ = compute_alignment_errors(
                        T_inc, filter_t_wi_subset, gt_interp_positions
                    )
                    incremental_errors.append(rms_error)
                else:
                    incremental_errors.append(-1.0)

                # Debug: Compare associations between batch and incremental
                # Check both early samples (where differences are large) and late samples
                debug_early = (i < 10) and batch_est_associations is not None
                debug_late = (
                    i >= self.num_filter_samples - 3
                ) and batch_est_associations is not None
                if debug_early or debug_late:
                    # Compare number of valid associations (not the full stored trajectory)
                    if len(batch_est_associations) != inc_result.n_pairs:
                        print(f"WARNING at step {i}: Different number of valid associations!")
                        print(
                            f"  Batch valid associations: {len(batch_est_associations)}, "
                            f"Incremental valid associations (n_pairs): {inc_result.n_pairs}"
                        )
                        print(
                            f"  Incremental stored samples: {len(incremental_aligner.get_estimated_positions())}"
                        )
                    else:
                        # Get incremental aligner's stored data (full trajectory)
                        inc_est_pos_full = np.array(incremental_aligner.get_estimated_positions())
                        inc_gt_pos_full = np.array(
                            incremental_aligner.get_gt_interpolated_positions()
                        )

                        # Filter to only valid associations (we can't directly access has_association_,
                        # but we can compare the input data)
                        # The issue is that we need to compare only the associated samples,
                        # but we can't easily filter them without knowing which ones have associations

                        # Compare input estimated positions (should match what we passed in)
                        if len(filter_t_wi_subset) == len(inc_est_pos_full):
                            est_input_diff = np.linalg.norm(
                                filter_t_wi_subset - inc_est_pos_full, axis=1
                            )
                            max_input_diff = np.max(est_input_diff)
                            if max_input_diff > 1e-10:
                                print(f"WARNING at step {i}: Input estimated positions differ!")
                                print(f"  Max difference: {max_input_diff:.6e}")

                        # Compare batch vs incremental transformations
                        if batch_T_gt_est_list[-1] is not None:
                            T_batch_inc_diff = np.linalg.norm(batch_T_gt_est_list[-1] - T_inc)
                            if T_batch_inc_diff > 1e-6 or debug_early:
                                print(f"DEBUG at step {i}: Batch and Incremental transformations")
                                print(f"  ||T_batch - T_incremental||: {T_batch_inc_diff:.6e}")
                                print(
                                    f"  Batch n_pairs: {len(batch_est_associations)}, Incremental n_pairs: {inc_result.n_pairs}"
                                )

                                # Try to compare the actual associations used
                                # We need to filter incremental's stored data to only valid associations
                                # Since we can't access has_association_ directly, we'll compare what we can
                                if len(batch_est_associations) == inc_result.n_pairs:
                                    # Compare GT interpolated positions (recompute from batch timestamps for comparison)
                                    batch_gt_recomputed = np.array(
                                        [
                                            [
                                                np.interp(
                                                    filter_t_s_subset[j],
                                                    self.gt_t_s,
                                                    self.gt_t_wi[:, dim],
                                                )
                                                for dim in range(3)
                                            ]
                                            for j in range(len(filter_t_s_subset))
                                        ]
                                    )

                                    # Compare batch GT associations with recomputed GT
                                    if len(batch_gt_associations) == len(batch_gt_recomputed):
                                        gt_assoc_diff = np.linalg.norm(
                                            batch_gt_associations - batch_gt_recomputed, axis=1
                                        )
                                        max_gt_assoc_diff = np.max(gt_assoc_diff)
                                        if max_gt_assoc_diff > 1e-10:
                                            print(
                                                f"  WARNING: Batch GT associations differ from recomputed GT!"
                                            )
                                            print(f"    Max difference: {max_gt_assoc_diff:.6e}")
                                        else:
                                            print(
                                                f"  Batch GT associations match recomputed GT (max diff: {max_gt_assoc_diff:.6e})"
                                            )

                                    # Compare estimated positions used in alignment
                                    # Batch uses batch_est_associations, incremental uses filtered est_pos_
                                    # We need to get the incremental aligner's valid estimated positions
                                    inc_est_pos_full = np.array(
                                        incremental_aligner.get_estimated_positions()
                                    )
                                    inc_gt_pos_full = np.array(
                                        incremental_aligner.get_gt_interpolated_positions()
                                    )

                                    # Filter to only valid associations (those that match batch)
                                    # Since we can't access has_association_ directly, we'll compare by matching timestamps
                                    if len(batch_est_associations) == len(batch_gt_associations):
                                        # For each batch association, find the corresponding incremental position
                                        # by matching timestamps
                                        batch_timestamps = filter_t_s_subset
                                        inc_timestamps = np.array(
                                            incremental_aligner.get_estimated_timestamps()
                                        )

                                        # Find matching indices
                                        est_pos_diffs = []
                                        for b_idx, b_t in enumerate(batch_timestamps):
                                            # Find corresponding index in incremental
                                            inc_idx = np.where(
                                                np.abs(inc_timestamps - b_t) < 1e-10
                                            )[0]
                                            if len(inc_idx) > 0:
                                                inc_idx = inc_idx[0]
                                                est_diff = np.linalg.norm(
                                                    batch_est_associations[b_idx]
                                                    - inc_est_pos_full[inc_idx]
                                                )
                                                est_pos_diffs.append(est_diff)

                                        if est_pos_diffs:
                                            max_est_diff = np.max(est_pos_diffs)
                                            mean_est_diff = np.mean(est_pos_diffs)
                                            if max_est_diff > 1e-10:
                                                print(
                                                    f"  WARNING: Estimated positions differ between batch and incremental!"
                                                )
                                                print(
                                                    f"    Max difference: {max_est_diff:.6e}, Mean: {mean_est_diff:.6e}"
                                                )
                                            else:
                                                print(
                                                    f"  Estimated positions match (max diff: {max_est_diff:.6e}, mean: {mean_est_diff:.6e})"
                                                )

                # Debug: Check if transformations are identical (only for last few samples to avoid spam)
                if i >= self.num_filter_samples - 5 and T_no_lba is not None:
                    T_diff = T_inc - T_no_lba
                    T_diff_norm = np.linalg.norm(T_diff)
                    if T_diff_norm < 1e-10:
                        print(
                            f"WARNING at step {i}: IncrementalNoLBA and Incremental transformations are identical!"
                        )
                    elif i == self.num_filter_samples - 1:
                        print(
                            f"Final step {i}: ||T_incremental - T_incrementalNoLBA|| = {T_diff_norm:.6e}"
                        )
            else:
                incremental_T_gt_est_list.append(None)
                incremental_n_pairs.append(0)
                T_err_incremental_list.append(-1.0)
                incremental_errors.append(-1.0)

            # Print progress every 10% of samples
            if (i + 1) % max(1, self.num_filter_samples // 10) == 0:
                print(
                    f"Step {i+1}/{self.num_filter_samples}: "
                    f"Batch pairs={batch_n_pairs_list[i]}, "
                    f"IncrementalNoLBA pairs={inc_no_lba_result.n_pairs if inc_no_lba_result.valid else 0}, "
                    f"Incremental pairs={inc_result.n_pairs if inc_result.valid else 0}"
                )

        # Print summary statistics
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY: Batch vs IncrementalNoLBA vs Incremental Alignment")
        print("=" * 80)
        if self.incremental_noise_enabled:
            print("NOTE: Stress test configuration:")
            print(
                f"  - Incremental drift-like noise (std={self.incremental_noise_std} [m] per sample) applied to filter positions"
            )
            print(
                "    * Drift is generated by integrating random noise starting from first timestamp"
            )
            print("    * At each time step i, drift is applied to all positions j <= i")
            print(
                "    * Drift decays exponentially over time: newer samples have more drift, older samples have less"
            )
            print("    * This simulates LBA where recent position updates have more impact")
            print("    * All three aligners see the same updated positions at each step")
            print(
                "    * IncrementalTrajectoryAlignerNoLBA: adds samples incrementally, can't update"
            )
            print("      statistics for already-processed samples when their positions change")
            print(
                "    * IncrementalTrajectoryAligner: recomputes from scratch with all updated positions"
            )
            print("    * Batch: recomputes from scratch with all updated positions (reference)")
            print(
                "  Expected: IncrementalTrajectoryAligner should match batch, IncrementalTrajectoryAlignerNoLBA should diverge."
            )
            print("=" * 80)

        # Find valid comparisons (where all methods have results)
        valid_indices = [
            i
            for i in range(self.num_filter_samples)
            if batch_T_gt_est_list[i] is not None
            and incremental_no_lba_T_gt_est_list[i] is not None
            and incremental_T_gt_est_list[i] is not None
        ]

        # Compare transformation matrices
        T_diff_norms = []
        if len(valid_indices) > 0:
            print(f"\nValid comparisons: {len(valid_indices)}/{self.num_filter_samples}")

            # Print a few sample comparisons for debugging
            sample_indices = valid_indices[:: max(1, len(valid_indices) // 5)][:3]  # 3 samples
            print("\nSample comparisons (first, middle, last valid):")
            for idx in sample_indices:
                print(f"\n  Sample index {idx}:")
                print(f"    Batch T_gt_est:\n{batch_T_gt_est_list[idx]}")
                print(f"    IncrementalNoLBA T_gt_est:\n{incremental_no_lba_T_gt_est_list[idx]}")
                print(f"    Incremental T_gt_est:\n{incremental_T_gt_est_list[idx]}")
                print(f"    Expected (Twc^-1):\n{np.linalg.inv(self.Twc)}")
                T_diff_batch_inc_no_lba = (
                    batch_T_gt_est_list[idx] - incremental_no_lba_T_gt_est_list[idx]
                )
                T_diff_batch_inc = batch_T_gt_est_list[idx] - incremental_T_gt_est_list[idx]
                T_diff_inc_no_lba_inc = (
                    incremental_no_lba_T_gt_est_list[idx] - incremental_T_gt_est_list[idx]
                )
                print(
                    f"    ||T_batch - T_incrementalNoLBA||: {np.linalg.norm(T_diff_batch_inc_no_lba):.6e}"
                )
                print(f"    ||T_batch - T_incremental||: {np.linalg.norm(T_diff_batch_inc):.6e}")
                print(
                    f"    ||T_incrementalNoLBA - T_incremental||: {np.linalg.norm(T_diff_inc_no_lba_inc):.6e}"
                )
                print(
                    f"    Batch pairs: {batch_n_pairs_list[idx]}, "
                    f"IncrementalNoLBA pairs: {incremental_no_lba_n_pairs[idx]}, "
                    f"Incremental pairs: {incremental_n_pairs[idx]}"
                )

                # Check if transformations are close to expected
                T_batch_err = batch_T_gt_est_list[idx] @ self.Twc - np.eye(4)
                T_inc_no_lba_err = incremental_no_lba_T_gt_est_list[idx] @ self.Twc - np.eye(4)
                T_inc_err = incremental_T_gt_est_list[idx] @ self.Twc - np.eye(4)
                print(f"    ||T_batch @ Twc - I||: {np.linalg.norm(T_batch_err):.6e}")
                print(
                    f"    ||T_incrementalNoLBA @ Twc - I||: {np.linalg.norm(T_inc_no_lba_err):.6e}"
                )
                print(f"    ||T_incremental @ Twc - I||: {np.linalg.norm(T_inc_err):.6e}")

                # Check scale factors (extract scale from Sim(3) transform)
                # For T = [sR t; 0 1], scale s can be extracted as mean column norm of sR
                batch_R_scaled = batch_T_gt_est_list[idx][:3, :3]
                inc_no_lba_R_scaled = incremental_no_lba_T_gt_est_list[idx][:3, :3]
                inc_R_scaled = incremental_T_gt_est_list[idx][:3, :3]
                batch_scale = np.mean([np.linalg.norm(batch_R_scaled[:, i]) for i in range(3)])
                inc_no_lba_scale = np.mean(
                    [np.linalg.norm(inc_no_lba_R_scaled[:, i]) for i in range(3)]
                )
                inc_scale = np.mean([np.linalg.norm(inc_R_scaled[:, i]) for i in range(3)])
                expected_scale = 1.0 / 6.4  # inverse of applied scale
                print(
                    f"    Batch scale: {batch_scale:.6f}, "
                    f"IncrementalNoLBA scale: {inc_no_lba_scale:.6f}, "
                    f"Incremental scale: {inc_scale:.6f}, "
                    f"Expected: {expected_scale:.6f}"
                )

            for idx in valid_indices:
                T_diff_batch_inc_no_lba = (
                    batch_T_gt_est_list[idx] - incremental_no_lba_T_gt_est_list[idx]
                )
                T_diff_batch_inc = batch_T_gt_est_list[idx] - incremental_T_gt_est_list[idx]
                T_diff_inc_no_lba_inc = (
                    incremental_no_lba_T_gt_est_list[idx] - incremental_T_gt_est_list[idx]
                )
                T_diff_norms.append(
                    {
                        "batch_vs_inc_no_lba": np.linalg.norm(T_diff_batch_inc_no_lba),
                        "batch_vs_inc": np.linalg.norm(T_diff_batch_inc),
                        "inc_no_lba_vs_inc": np.linalg.norm(T_diff_inc_no_lba_inc),
                    }
                )

            if len(T_diff_norms) > 0:
                batch_vs_inc_no_lba = [d["batch_vs_inc_no_lba"] for d in T_diff_norms]
                batch_vs_inc = [d["batch_vs_inc"] for d in T_diff_norms]
                inc_no_lba_vs_inc = [d["inc_no_lba_vs_inc"] for d in T_diff_norms]

                print(f"\nTransformation matrix differences:")
                print(f"  ||T_batch - T_incrementalNoLBA||:")
                print(
                    f"    Mean: {np.mean(batch_vs_inc_no_lba):.6e}, Std: {np.std(batch_vs_inc_no_lba):.6e}, Max: {np.max(batch_vs_inc_no_lba):.6e}, Min: {np.min(batch_vs_inc_no_lba):.6e}"
                )
                print(f"  ||T_batch - T_incremental||:")
                print(
                    f"    Mean: {np.mean(batch_vs_inc):.6e}, Std: {np.std(batch_vs_inc):.6e}, Max: {np.max(batch_vs_inc):.6e}, Min: {np.min(batch_vs_inc):.6e}"
                )
                print(f"  ||T_incrementalNoLBA - T_incremental||:")
                print(
                    f"    Mean: {np.mean(inc_no_lba_vs_inc):.6e}, Std: {np.std(inc_no_lba_vs_inc):.6e}, Max: {np.max(inc_no_lba_vs_inc):.6e}, Min: {np.min(inc_no_lba_vs_inc):.6e}"
                )

            # Compare errors w.r.t. ground truth transform
            batch_errs_valid = [T_err_batch_list[i] for i in valid_indices]
            inc_no_lba_errs_valid = [T_err_incremental_no_lba_list[i] for i in valid_indices]
            inc_errs_valid = [T_err_incremental_list[i] for i in valid_indices]

            if len(batch_errs_valid) > 0:
                print(f"\nError w.r.t. known transform (||T_align @ T_known - I||):")
                print(f"  Batch alignment:")
                print(
                    f"    Mean: {np.mean(batch_errs_valid):.6e}, Std: {np.std(batch_errs_valid):.6e}"
                )
                print(f"  IncrementalNoLBA alignment:")
                print(
                    f"    Mean: {np.mean(inc_no_lba_errs_valid):.6e}, Std: {np.std(inc_no_lba_errs_valid):.6e}"
                )
                print(f"  Incremental alignment:")
                print(f"    Mean: {np.mean(inc_errs_valid):.6e}, Std: {np.std(inc_errs_valid):.6e}")

            # Compare RMS alignment errors
            batch_rms_valid = [batch_errors[i] for i in valid_indices if batch_errors[i] >= 0]
            inc_no_lba_rms_valid = [
                incremental_no_lba_errors[i]
                for i in valid_indices
                if incremental_no_lba_errors[i] >= 0
            ]
            inc_rms_valid = [
                incremental_errors[i] for i in valid_indices if incremental_errors[i] >= 0
            ]

            if len(batch_rms_valid) > 0:
                print(
                    f"\nRMS Alignment Error (computed on current estimated trajectory using current transformation):"
                )
                print(f"  Batch alignment:")
                print(
                    f"    Mean: {np.mean(batch_rms_valid):.6e}, Std: {np.std(batch_rms_valid):.6e}"
                )
                print(f"  IncrementalNoLBA alignment:")
                print(
                    f"    Mean: {np.mean(inc_no_lba_rms_valid):.6e}, Std: {np.std(inc_no_lba_rms_valid):.6e}"
                )
                print(f"  Incremental alignment:")
                print(f"    Mean: {np.mean(inc_rms_valid):.6e}, Std: {np.std(inc_rms_valid):.6e}")

                # Check if incremental errors are identical (which would indicate a bug)
                if len(inc_no_lba_rms_valid) == len(inc_rms_valid):
                    rms_diffs = [
                        abs(inc_no_lba_rms_valid[i] - inc_rms_valid[i])
                        for i in range(len(inc_rms_valid))
                    ]
                    max_rms_diff = max(rms_diffs) if rms_diffs else 0.0
                    print(
                        f"\n  Max difference between IncrementalNoLBA and Incremental RMS errors: {max_rms_diff:.6e}"
                    )
                    if max_rms_diff < 1e-10:
                        print(
                            "  WARNING: IncrementalNoLBA and Incremental RMS errors are identical!"
                        )
                        print(
                            "  This suggests they might be using the same transformation or there's a bug."
                        )
        else:
            print("\nNo valid comparisons found between batch and incremental alignment methods.")

        # Plot comparison over time
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Number of pairs used
        ax1 = axes[0, 0]
        ax1.plot(
            range(self.num_filter_samples), batch_n_pairs_list, "b-", label="Batch", linewidth=2
        )
        ax1.plot(
            range(self.num_filter_samples),
            incremental_no_lba_n_pairs,
            "r--",
            label="IncrementalNoLBA",
            linewidth=2,
        )
        ax1.plot(
            range(self.num_filter_samples),
            incremental_n_pairs,
            "g-o",
            label="Incremental",
            linewidth=3,
            markersize=4,
            markevery=max(1, self.num_filter_samples // 50),
        )
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Number of Associated Pairs")
        ax1.set_title("Number of Associated Pairs vs Sample Index")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Transformation error norm
        ax2 = axes[0, 1]
        valid_batch_errs = [
            T_err_batch_list[i] if T_err_batch_list[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        valid_inc_no_lba_errs = [
            T_err_incremental_no_lba_list[i] if T_err_incremental_no_lba_list[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        valid_inc_errs = [
            T_err_incremental_list[i] if T_err_incremental_list[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        ax2.plot(range(self.num_filter_samples), valid_batch_errs, "b-", label="Batch", linewidth=2)
        ax2.plot(
            range(self.num_filter_samples),
            valid_inc_no_lba_errs,
            "r--",
            label="IncrementalNoLBA",
            linewidth=2,
        )
        ax2.plot(
            range(self.num_filter_samples),
            valid_inc_errs,
            "g-o",
            label="Incremental",
            linewidth=2,
            markersize=4,
            markevery=max(1, self.num_filter_samples // 50),
        )
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("||T_align @ T_known - I||")
        ax2.set_title("Transformation Error vs Sample Index")
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale("log")

        # Plot 3: Difference between batch and incremental transformations
        ax3 = axes[1, 0]
        if len(valid_indices) > 0:
            T_diff_batch_inc_no_lba_plot = [
                (
                    T_diff_norms[valid_indices.index(i)]["batch_vs_inc_no_lba"]
                    if i in valid_indices
                    else np.nan
                )
                for i in range(self.num_filter_samples)
            ]
            T_diff_batch_inc_plot = [
                (
                    T_diff_norms[valid_indices.index(i)]["batch_vs_inc"]
                    if i in valid_indices
                    else np.nan
                )
                for i in range(self.num_filter_samples)
            ]
            T_diff_inc_no_lba_inc_plot = [
                (
                    T_diff_norms[valid_indices.index(i)]["inc_no_lba_vs_inc"]
                    if i in valid_indices
                    else np.nan
                )
                for i in range(self.num_filter_samples)
            ]
            ax3.plot(
                range(self.num_filter_samples),
                T_diff_batch_inc_no_lba_plot,
                "b-",
                label="Batch vs IncrementalNoLBA",
                linewidth=2,
            )
            ax3.plot(
                range(self.num_filter_samples),
                T_diff_batch_inc_plot,
                "g-o",
                label="Batch vs Incremental",
                linewidth=2,
                markersize=4,
                markevery=max(1, self.num_filter_samples // 50),
            )
            ax3.plot(
                range(self.num_filter_samples),
                T_diff_inc_no_lba_inc_plot,
                "m-s",
                label="IncrementalNoLBA vs Incremental",
                linewidth=2,
                markersize=3,
                markevery=max(1, self.num_filter_samples // 50),
            )
        else:
            ax3.text(
                0.5,
                0.5,
                "No valid comparisons",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax3.transAxes,
            )
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("||T_diff||")
        ax3.set_title("Difference Between Transformations")
        ax3.legend()
        ax3.grid(True)
        if len(valid_indices) > 0:
            ax3.set_yscale("log")

        # Plot 4: Alignment errors (RMS) and differences
        ax4 = axes[1, 1]
        valid_batch_alignment_errors = [
            batch_errors[i] if batch_errors[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        valid_incremental_no_lba_alignment_errors = [
            incremental_no_lba_errors[i] if incremental_no_lba_errors[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        valid_incremental_alignment_errors = [
            incremental_errors[i] if incremental_errors[i] >= 0 else np.nan
            for i in range(self.num_filter_samples)
        ]
        ax4.plot(
            range(self.num_filter_samples),
            valid_batch_alignment_errors,
            "b-",
            label="Batch",
            linewidth=2,
        )
        ax4.plot(
            range(self.num_filter_samples),
            valid_incremental_no_lba_alignment_errors,
            "r--",
            label="IncrementalNoLBA",
            linewidth=2,
        )
        ax4.plot(
            range(self.num_filter_samples),
            valid_incremental_alignment_errors,
            "g-o",
            label="Incremental",
            linewidth=2,
            markersize=4,
            markevery=max(1, self.num_filter_samples // 50),
        )
        ax4.set_xlabel("Sample Index")
        ax4.set_ylabel("RMS Alignment Error")
        ax4.set_title(
            "Alignment Error (RMS) vs Sample Index\n(computed on current estimated trajectory using current transformation)"
        )
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale("log")

        # Add text annotation explaining why errors might be similar
        if len(valid_indices) > 0:
            # Compute RMS error differences
            rms_diff_no_lba_inc = [
                (
                    abs(
                        valid_incremental_no_lba_alignment_errors[i]
                        - valid_incremental_alignment_errors[i]
                    )
                    if not (
                        np.isnan(valid_incremental_no_lba_alignment_errors[i])
                        or np.isnan(valid_incremental_alignment_errors[i])
                    )
                    else np.nan
                )
                for i in range(self.num_filter_samples)
            ]
            max_rms_diff = max([d for d in rms_diff_no_lba_inc if not np.isnan(d)], default=0.0)
            if max_rms_diff < 1e-3:
                ax4.text(
                    0.02,
                    0.98,
                    f"Note: RMS errors computed on current\nestimated trajectory using current\ntransformation for each aligner.\n"
                    f"Max diff: {max_rms_diff:.2e}",
                    transform=ax4.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        plt.tight_layout()
        # plt.show()

        # Plot 3D trajectories: final aligned trajectories for all three methods
        figure_3d = plt.figure(figsize=(12, 9))
        ax_3d = figure_3d.add_subplot(111, projection="3d")

        # Get the final corrected positions (with drift applied at final step)
        filter_t_s_final = filter_t_s_original
        filter_t_wi_final = []
        final_step = self.num_filter_samples - 1
        deltaT_decay = 5.0  # [s] time constant for the decay of the drift (same as in main loop)
        for j in range(self.num_filter_samples):
            corrected_position = filter_t_wi_original[j].copy()
            # Add drift that decays over time (same logic as in main loop)
            if self.incremental_noise_enabled:
                steps_since_added = final_step - j
                factor = math.exp(-steps_since_added / deltaT_decay)
                drift_for_sample = base_drift[j, :] * factor
                corrected_position = corrected_position + drift_for_sample
            filter_t_wi_final.append(corrected_position)
        filter_t_wi_final = np.array(filter_t_wi_final)

        # Get final transformation matrices
        final_batch_T = batch_T_gt_est_list[-1] if batch_T_gt_est_list[-1] is not None else None
        final_inc_no_lba_T = (
            incremental_no_lba_T_gt_est_list[-1]
            if incremental_no_lba_T_gt_est_list[-1] is not None
            else None
        )
        final_inc_T = (
            incremental_T_gt_est_list[-1] if incremental_T_gt_est_list[-1] is not None else None
        )

        # Plot GT trajectory
        ax_3d.plot(
            self.gt_t_wi[:, 0],
            self.gt_t_wi[:, 1],
            self.gt_t_wi[:, 2],
            "k-",
            label="GT trajectory",
            linewidth=2,
            alpha=0.7,
        )

        # Plot batch aligned trajectory
        if final_batch_T is not None:
            batch_aligned = (
                final_batch_T[:3, :3] @ filter_t_wi_final.T + final_batch_T[:3, 3].reshape(3, 1)
            ).T
            ax_3d.plot(
                batch_aligned[:, 0],
                batch_aligned[:, 1],
                batch_aligned[:, 2],
                "b-",
                label="Batch aligned",
                linewidth=2,
                alpha=0.8,
            )

        # Plot IncrementalTrajectoryAlignerNoLBA aligned trajectory
        if final_inc_no_lba_T is not None:
            inc_no_lba_aligned = (
                final_inc_no_lba_T[:3, :3] @ filter_t_wi_final.T
                + final_inc_no_lba_T[:3, 3].reshape(3, 1)
            ).T
            ax_3d.plot(
                inc_no_lba_aligned[:, 0],
                inc_no_lba_aligned[:, 1],
                inc_no_lba_aligned[:, 2],
                "r--",
                label="IncrementalNoLBA aligned",
                linewidth=2,
                alpha=0.8,
            )

        # Plot IncrementalTrajectoryAligner aligned trajectory
        if final_inc_T is not None:
            inc_aligned = (
                final_inc_T[:3, :3] @ filter_t_wi_final.T + final_inc_T[:3, 3].reshape(3, 1)
            ).T
            ax_3d.plot(
                inc_aligned[:, 0],
                inc_aligned[:, 1],
                inc_aligned[:, 2],
                "g-o",
                label="Incremental aligned",
                linewidth=2,
                markersize=5,
                markevery=max(1, len(inc_aligned) // 30),
                alpha=0.9,
            )

        set_axes_equal(ax_3d)
        ax_3d.legend()
        ax_3d.set_xlabel("x [m]")
        ax_3d.set_ylabel("y [m]")
        ax_3d.set_zlabel("z [m]")
        ax_3d.set_title("3D Trajectories: GT vs Aligned (Final)")
        plt.tight_layout()
        # plt.show()

        return {
            "batch_T_gt_est_list": batch_T_gt_est_list,
            "incremental_no_lba_T_gt_est_list": incremental_no_lba_T_gt_est_list,
            "incremental_T_gt_est_list": incremental_T_gt_est_list,
            "batch_errors": batch_errors,
            "incremental_no_lba_errors": incremental_no_lba_errors,
            "incremental_errors": incremental_errors,
            "incremental_no_lba_n_pairs": incremental_no_lba_n_pairs,
            "incremental_n_pairs": incremental_n_pairs,
            "T_err_batch_list": T_err_batch_list,
            "T_err_incremental_no_lba_list": T_err_incremental_no_lba_list,
            "T_err_incremental_list": T_err_incremental_list,
        }


if __name__ == "__main__":
    """
    The same ground truth trajectory is used for all three aligners and kept fixed.
    At each time step:
    - The same estimated trajectory is used to feed all three aligners.
    - An incremental noise is added to the estimated trajectory at each time step to change both its current and previous positions,
        so as to simulate LBA.
    - IncrementalTrajectoryAlignerNoLBA: processes old stored samples (without updating them) + the last received sample incrementally
        (previous samples data remain unchanged, so errors accumulate and the estimated alignment transform diverges from batch).
    - IncrementalTrajectoryAligner: uses all the updated positions up to the current step, updates the previous samples data and
        re-optimizes the full trajectory at each step, so its estimated alignment transform should match batch alignment.
    - Batch alignment: uses all updated positions up to the current step, providing the reference alignment transform.
    - At each sample, for each aligner, RMS errors are computed by using the estimated alignment transform on the top of the latest estimated trajectory
        and the GT trajectory.
    The test should show that IncrementalTrajectoryAligner stays close to batch, while IncrementalTrajectoryAlignerNoLBA diverges as noise accumulates.
    Adjust incremental_noise_std to control the stress level.
    """

    test = TestAlignSVD()
    test.set_up()

    # # Run original batch alignment test
    # print("=" * 80)
    # print("Running original batch alignment test...")
    # print("=" * 80)
    # test.test_align_svd()

    # Run incremental vs batch comparison
    print("\n" + "=" * 80)
    print("Running incremental vs batch alignment comparison...")
    print("=" * 80)
    results = test.test_align_incremental_vs_batch()

    plt.show()
