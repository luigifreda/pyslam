"""Aligned ground truth must be packed xyz, which is what the 3D viewer draws."""
import unittest

import numpy as np

import pyslam.config  # noqa: F401  Configure paths for the compiled dependencies.

from pyslam.utilities.geom_trajectory import (
    align_trajectories_with_svd,
    packed_xyz,
    store_aligned_trajectories,
)


def _gl_points(points):
    """Reproduce DrawTrajectory/DrawLines2: consecutive triples, strides ignored."""
    raw = np.ravel(np.asarray(points), order="K")
    return raw.reshape(-1, 3)


class GroundTruthAlignmentLayoutTests(unittest.TestCase):
    def test_aligned_ground_truth_is_packed_for_opengl(self):
        count = 40
        angle = np.linspace(0.0, 2.0 * np.pi, count)
        estimated = np.stack((np.cos(angle), np.sin(angle), 0.05 * angle), axis=1)
        yaw = np.deg2rad(35.0)
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )
        ground_truth = (rotation @ estimated.T).T + np.array([4.0, -2.0, 0.5])
        timestamps = np.linspace(1_341_847_980.0, 1_341_847_982.0, count)

        _transform, _error, alignment = align_trajectories_with_svd(
            timestamps, estimated, timestamps, ground_truth, compute_align_error=True, find_scale=False
        )
        # The SVD pose itself is column-major. Drawing it without packing scrambles the curve.
        self.assertFalse(np.asarray(alignment.T_est_gt).flags.c_contiguous)
        store_aligned_trajectories(alignment, ground_truth)

        for points in (
            alignment.gt_trajectory_aligned,
            alignment.gt_trajectory_aligned_associated,
            alignment.estimated_t_wi,
            alignment.gt_t_wi,
        ):
            self.assertTrue(points.flags.c_contiguous)
            self.assertEqual(points.shape[1], 3)
            np.testing.assert_allclose(_gl_points(points), points)

        np.testing.assert_allclose(alignment.gt_trajectory_aligned, estimated, atol=1e-8)
        self.assertEqual(
            len(alignment.gt_trajectory_aligned_associated), len(alignment.estimated_t_wi)
        )
        np.testing.assert_allclose(
            alignment.gt_trajectory_aligned_associated, alignment.estimated_t_wi, atol=1e-8
        )

    def test_packed_xyz_repairs_fortran_points(self):
        logical = np.arange(12, dtype=np.float64).reshape(4, 3)
        fortran = np.asfortranarray(logical)
        self.assertFalse(np.allclose(_gl_points(fortran), logical))
        packed = packed_xyz(fortran)
        self.assertTrue(packed.flags.c_contiguous)
        np.testing.assert_allclose(_gl_points(packed), logical)


if __name__ == "__main__":
    unittest.main()
