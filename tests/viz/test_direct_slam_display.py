"""Direct-viewer snapshots must follow BA corrections and use world camera centers."""
import unittest
from types import SimpleNamespace

import numpy as np

from main_direct_slam import DirectSlamDisplay


class SnapshotViewer:
    def __init__(self):
        self.state = None
        self.closed = False

    def is_closed(self):
        return self.closed

    def is_draw_features_with_radius(self):
        return False

    def draw_map(self, state):
        self.state = state


class SnapshotSlam:
    def __init__(self):
        self.world_poses = np.repeat(np.eye(4)[None], 3, axis=0)
        self.world_poses[1, :3, 3] = [4, -2, 1]
        self.world_poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.world_poses[2, :3, 3] = [5, -1, 2]
        self.points = np.array([[1., 2., 3.], [4., 5., 6.]])

    def pose(self):
        matrix = self.world_poses[-1] if len(self.world_poses) else np.eye(4)
        return SimpleNamespace(get_inverse_matrix=lambda: matrix)

    def trajectory(self):
        return np.linalg.inv(self.world_poses)

    def keyframes(self):
        return self.world_poses[::2], [0.0, 0.2] if len(self.world_poses) else []

    def map_points(self):
        return self.points

    def draw_feature_trails(self, image, **kwargs):
        return image.copy()

    def status_name(self):
        return "TRACKING_GOOD"

    def last_inliers(self):
        return 25


class DirectDisplayTests(unittest.TestCase):
    def test_black_map_and_corrected_full_trajectory(self):
        display = DirectSlamDisplay.__new__(DirectSlamDisplay)
        display.viewer = SnapshotViewer()
        display.images = SimpleNamespace(draw=lambda *args: None)
        slam = SnapshotSlam()
        image = np.zeros((40, 100, 3), dtype=np.uint8)
        display.draw(slam, image, 2, 0.2)
        first = display.viewer.state
        np.testing.assert_array_equal(first.map_data.colors, np.zeros((2, 3)))
        np.testing.assert_array_equal(first.map_data.semantic_colors, np.zeros((2, 3)))
        np.testing.assert_allclose(first.estimated_trajectory, slam.world_poses[:, :3, 3])
        self.assertEqual(len(first.estimated_trajectory), 3)  # Includes the non-keyframe.
        self.assertEqual(len(first.map_data.poses), 2)

        # A subsequent BA can correct an old pose, rather than only append a new one.
        slam.world_poses[1, :3, 3] = [-2, 6, 4]
        display.draw(slam, image, 2, 0.2)
        np.testing.assert_allclose(display.viewer.state.estimated_trajectory, slam.world_poses[:, :3, 3])
        np.testing.assert_allclose(first.estimated_trajectory[1], [4, -2, 1])

        # Reset must replace the entire path, without retaining old vertices.
        slam.world_poses = np.empty((0, 4, 4))
        slam.points = np.empty((0, 3))
        display.draw(slam, image, 3, 0.3)
        self.assertEqual(display.viewer.state.estimated_trajectory.shape, (0, 3))
        self.assertEqual(display.viewer.state.map_data.colors.shape, (0, 3))

    def test_no_publish_after_viewer_close(self):
        display = DirectSlamDisplay.__new__(DirectSlamDisplay)
        display.viewer = SnapshotViewer()
        display.viewer.closed = True
        display.draw(object(), None, 0, 0.0)
        self.assertIsNone(display.viewer.state)


if __name__ == "__main__":
    unittest.main()
