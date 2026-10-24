import unittest
from unittest.mock import patch

import numpy as np

from pyslam.utilities.geom_triangulation import triangulate_normalized_points


class TriangulationMaskTests(unittest.TestCase):
    def test_mask_keeps_first_point_and_preserves_invalid_positions(self):
        homogeneous = np.array([[2., 1., 6.], [4., 1., 8.], [6., 1., 10.], [2., 0., 2.]])
        with patch("pyslam.utilities.geom_triangulation.cv2.triangulatePoints", return_value=homogeneous):
            points, mask = triangulate_normalized_points(
                np.eye(4), np.eye(4), np.zeros((3, 2)), np.zeros((3, 2))
            )
        self.assertEqual(mask.dtype, np.dtype(bool))
        np.testing.assert_array_equal(mask, [True, False, True])
        np.testing.assert_allclose(points[mask], [[1., 2., 3.], [3., 4., 5.]])
        self.assertTrue(np.isfinite(points).all())

    def test_stereo_geometry_keeps_every_valid_point(self):
        world = np.array([[0., 0., 3.], [1., 1., 4.], [-1., 0.5, 5.]])
        left, right = np.eye(4), np.eye(4)
        right[0, 3] = -0.1
        kpn_left = world[:, :2] / world[:, 2:3]
        world_right = world + right[:3, 3]
        kpn_right = world_right[:, :2] / world_right[:, 2:3]
        points, mask = triangulate_normalized_points(left, right, kpn_left, kpn_right)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(points, world, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
