"""Direct-viewer snapshots must follow BA corrections and use world camera centers."""
import unittest
from threading import RLock
from types import SimpleNamespace

import numpy as np

import pyslam.config  # noqa: F401  Configure paths for the compiled dependencies.

from pyslam.viz.direct_slam_display import DirectSlamDisplay
from pyslam.slam.direct_loop_closing import DirectLoopClosing


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
        slam = SnapshotSlam()
        first = display.capture_map(slam, 2, 0.2)
        np.testing.assert_array_equal(first.map_data.colors, np.zeros((2, 3)))
        np.testing.assert_array_equal(first.map_data.semantic_colors, np.zeros((2, 3)))
        np.testing.assert_allclose(first.estimated_trajectory, slam.world_poses[:, :3, 3])
        self.assertEqual(len(first.estimated_trajectory), 3)  # Includes the non-keyframe.
        self.assertEqual(len(first.map_data.poses), 2)

        # A subsequent BA can correct an old pose, rather than only append a new one.
        slam.world_poses[1, :3, 3] = [-2, 6, 4]
        updated = display.capture_map(slam, 2, 0.2)
        np.testing.assert_allclose(updated.estimated_trajectory, slam.world_poses[:, :3, 3])
        np.testing.assert_allclose(first.estimated_trajectory[1], [4, -2, 1])

        # Reset must replace the entire path, without retaining old vertices.
        slam.world_poses = np.empty((0, 4, 4))
        slam.points = np.empty((0, 3))
        updated = display.capture_map(slam, 3, 0.3)
        self.assertEqual(updated.estimated_trajectory.shape, (0, 3))
        self.assertEqual(updated.map_data.colors.shape, (0, 3))

    def test_graph_uses_direct_centers_even_when_descriptor_poses_disagree(self):
        adapter = DirectLoopClosing.__new__(DirectLoopClosing)
        adapter.map = SimpleNamespace(update_lock=RLock())
        a = SimpleNamespace(id=41, kid=7)
        b = SimpleNamespace(id=83, kid=12)
        stale = SimpleNamespace(id=99, kid=13)
        a.get_parent = lambda: None
        b.get_parent = lambda: a
        a.get_loop_edges = lambda: [b, stale]
        b.get_loop_edges = lambda: [a]
        a.get_covisible_by_weight = lambda weight: [b]
        b.get_covisible_by_weight = lambda weight: [a]
        # Accessing descriptor camera centers would be a regression.
        a.Ow = b.Ow = lambda: self.fail("descriptor pose used for drawing")
        adapter._graph_revision = 0
        adapter._clear_state()
        adapter._keyframes = [a, b]
        adapter._keyframe_index = {a.id: 0, b.id: 1}
        adapter._graph_revision += 1  # Build the cached graph from these keyframes.
        slam = SnapshotSlam()
        state = DirectSlamDisplay.capture_map(slam, 2, 0.2, adapter)
        poses, _ = slam.keyframes()
        edge = np.r_[poses[0, :3, 3], poses[1, :3, 3]]
        np.testing.assert_allclose(state.map_data.loops, [edge])
        np.testing.assert_allclose(state.map_data.covisibility_graph, [edge])
        np.testing.assert_allclose(state.map_data.spanning_tree, [np.r_[edge[3:], edge[:3]]])

        # A later correction must replace endpoints without mutating an old snapshot.
        slam.world_poses[2, :3, 3] += 10
        updated = DirectSlamDisplay.capture_map(slam, 2, 0.2, adapter)
        np.testing.assert_allclose(updated.map_data.loops[0, 3:], slam.world_poses[2, :3, 3])
        np.testing.assert_allclose(state.map_data.loops, [edge])
        adapter._clear_state()
        self.assertEqual(adapter.graph_segments([])["loops"].shape, (0, 6))

    def test_no_publish_after_viewer_close(self):
        display = DirectSlamDisplay.__new__(DirectSlamDisplay)
        display.viewer = SnapshotViewer()
        display.viewer.closed = True
        display.draw(object(), 0)
        self.assertIsNone(display.viewer.state)

    def test_draw_uses_the_snapshot_captured_under_the_gate(self):
        display = DirectSlamDisplay.__new__(DirectSlamDisplay)
        display.viewer = SnapshotViewer()
        display.images = SimpleNamespace(draw=lambda image, name: setattr(self, "shown", image))
        display.img_writer = SimpleNamespace(
            kFont=0, font_scale=0.5, font_thickness=1, write=lambda image, text, pos: None)
        display.timer = SimpleNamespace(refresh=lambda: None, get_fps=lambda: 10.0)
        slam = SnapshotSlam()
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        snapshot = display.capture(slam, image, 2, 0.2)
        self.assertEqual(snapshot.tracking, "TRACKING_GOOD  inliers: 25")

        # After the gate is released a loop correction may rewrite the native
        # state, so draw() takes no slam handle and renders only the snapshot.
        self.assertNotIn("slam", DirectSlamDisplay.draw.__code__.co_varnames)
        slam.draw_feature_trails = slam.status_name = slam.last_inliers = None
        display.draw(snapshot, 2)
        self.assertIs(self.shown, snapshot.image)
        self.assertIs(display.viewer.state, snapshot.map_state)


if __name__ == "__main__":
    unittest.main()
