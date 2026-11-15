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

from types import SimpleNamespace

import cv2
import numpy as np

from pyslam.config_parameters import Parameters


class DirectSlamDisplay:
    """Adapt direct landmarks to the same viewer input used by main_slam.py."""

    def __init__(self, scale):
        from pyslam.viz.viewer3D import Viewer3D
        from pyslam.viz.cvimage_thread import CvImageViewer

        from pyslam.utilities.img_management import ImgWriter
        from pyslam.utilities.timer import TimerFps

        self.viewer = Viewer3D(scale=scale)
        try:
            self.viewer.wait_for_ready()
            self.images = CvImageViewer()
        except Exception:
            self.viewer.quit()
            raise
        # Same outlined simplex font used by main_slam.py.
        self.img_writer = ImgWriter(font_scale=0.5)
        self.timer = TimerFps("DirectSlam", is_verbose=False)

    def capture(self, slam, image, img_id, timestamp, loop_closing=None):
        """Copy everything draw() needs while the caller owns the tracking gate.

        A loop correction may rewrite live points and poses on the LoopClosing
        thread as soon as the gate is released, so draw() never reads `slam`.
        """
        # Draw in C++ using the same point colors and trail renderer as main_slam.py.
        shown = slam.draw_feature_trails(
            image, with_level_radius=self.viewer.is_draw_features_with_radius(),
            trail_max_length=Parameters.kMaxFeatureTrailLength)
        return SimpleNamespace(
            image=shown,
            tracking=f"{slam.status_name()}  inliers: {slam.last_inliers()}",
            map_state=self.capture_map(slam, img_id, timestamp, loop_closing))

    def draw(self, snapshot, img_id, refresh_fps=True):
        if self.viewer.is_closed():
            return

        shown = snapshot.image
        if refresh_fps:
            self.timer.refresh()
        self.img_writer.write(shown, f"id: {img_id}  fps: {self.timer.get_fps():.1f}", (20, 20))
        tracking = snapshot.tracking
        text_size, baseline = cv2.getTextSize(
            tracking, self.img_writer.kFont, self.img_writer.font_scale, self.img_writer.font_thickness)
        bottom = shown.shape[0] - baseline - 8
        self.img_writer.write(shown, tracking, (20, max(bottom, text_size[1] + 4)))
        self.images.draw(shown, "Camera")
        self.viewer.draw_map(snapshot.map_state)

    @staticmethod
    def capture_map(slam, img_id, timestamp, loop_closing=None):
        """Copy one map revision while the caller still owns the tracking gate."""
        from pyslam.viz.viewer3D import Viewer3DMapInput

        state = Viewer3DMapInput()
        state.cur_frame_data.cur_frame_id = img_id
        state.cur_frame_data.cur_pose = slam.pose().get_inverse_matrix()
        state.cur_frame_data.cur_pose_timestamp = timestamp
        poses, timestamps = slam.keyframes()
        # Read optimized map coordinates, including landmarks no longer tracked live.
        points = slam.map_points()
        limit = Parameters.kMaxSparseMapPointsToVisualize
        if len(points) > limit:
            points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
        colors = np.zeros((len(points), 3), dtype=np.float64)
        # Re-read every accepted pose so local BA corrections update the entire line.
        # Tcw camera center is -Rcw.T @ tcw; this also handles an empty/reset trajectory.
        trajectory = np.asarray(slam.trajectory(), dtype=np.float64).reshape(-1, 4, 4)
        state.estimated_trajectory = -np.einsum(
            "nji,nj->ni", trajectory[:, :3, :3], trajectory[:, :3, 3])
        graph = loop_closing.graph_segments(poses) if loop_closing else dict(
            covisibility_graph=[], spanning_tree=[], loops=[])
        state.map_data = SimpleNamespace(
            poses=np.asarray(poses), pose_timestamps=np.asarray(timestamps), points=points,
            colors=colors, semantic_colors=colors, fov_centers=[], fov_centers_colors=[],
            **graph)
        return state

    def quit(self):
        self.images.quit()
        self.viewer.quit()
