"""GT association segments pair index i of two arrays; mismatched updates must be skipped."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

import pyslam.config  # noqa: F401  Configure paths for the compiled dependencies.
from pyslam.viz import viewer3D
from pyslam.viz.viewer3D import Viewer3D


def make_viewer(estimated, ground_truth):
    viewer = Viewer3D.__new__(Viewer3D)
    viewer.draw_gt = True
    viewer.draw_gt_associations = True
    viewer.thread_gt_aligned = True
    viewer.thread_est_associated_traj = estimated
    viewer.thread_gt_trajectory_aligned_associated = ground_truth
    return viewer


class GtAssociationDrawTests(unittest.TestCase):
    def draw(self, estimated, ground_truth):
        gl, glutils = Mock(), Mock()
        with patch.object(viewer3D, "gl", gl, create=True), \
             patch.object(viewer3D, "glutils", glutils, create=True):
            make_viewer(estimated, ground_truth)._draw_gt_associations()
        return glutils.DrawLines2

    def test_matching_arrays_draw_one_segment_per_sample(self):
        estimated = np.arange(12, dtype=np.float64).reshape(4, 3)
        ground_truth = estimated + 1.0
        draw = self.draw(estimated, ground_truth)
        draw.assert_called_once()
        gt_arg, est_arg, _width = draw.call_args.args
        np.testing.assert_array_equal(gt_arg, ground_truth)
        np.testing.assert_array_equal(est_arg, estimated)

    def test_mismatched_arrays_from_different_updates_are_skipped(self):
        estimated = np.zeros((5, 3))
        ground_truth = np.ones((3, 3))
        self.draw(estimated, ground_truth).assert_not_called()

    def test_empty_arrays_are_skipped(self):
        self.draw(np.zeros((0, 3)), np.zeros((0, 3))).assert_not_called()


if __name__ == "__main__":
    unittest.main()
