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

import numpy as np

# Import MapPoint locally to avoid circular imports
from pyslam.config_parameters import Parameters
from pyslam.slam.map_point import MapPoint
from pyslam.slam.frame import Frame
from pyslam.slam.keyframe import KeyFrame
from pyslam.slam.map import Map
from pyslam.io.dataset_types import SensorType
from pyslam.utilities.system import Printer


class TrackingUtils:

    # propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to lock)
    @staticmethod
    def propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur, max_descriptor_distance=None):
        # check_orientation=False):
        if max_descriptor_distance is None:
            max_descriptor_distance = Parameters.kMaxDescriptorDistance

        idx_ref_out = []
        idx_cur_out = []

        # rot_histo = RotationHistogram()
        # check_orientation = check_orientation and kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features

        # populate f_cur with map points by propagating map point matches of f_ref;
        # to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur
        num_matched_map_pts = 0
        for i, idx_ref in enumerate(idxs_ref):  # iterate over keypoint matches
            p_ref = f_ref.points[idx_ref]
            if (
                p_ref is None or f_ref.outliers[idx_ref] or p_ref.is_bad()
            ):  # do not consider pose optimization outliers or bad points
                continue
            idx_cur = idxs_cur[i]
            p_cur = f_cur.points[idx_cur]
            if (
                p_cur is not None
            ):  # and p_cur.num_observations() > 0: # if we already matched p_cur => no need to propagate anything
                continue
            des_distance = p_ref.min_des_distance(f_cur.des[idx_cur])
            if des_distance > max_descriptor_distance:
                continue
            if p_ref.add_frame_view(
                f_cur, idx_cur
            ):  # => P is matched to the i-th matched keypoint in f_cur
                num_matched_map_pts += 1
                idx_ref_out.append(idx_ref)
                idx_cur_out.append(idx_cur)

                # if check_orientation:
                #     index_match = len(idx_cur_out)-1
                #     rot = f_ref.angles[idx]-f_cur.angles[idx_cur]
                #     rot_histo.push(rot, index_match)

        # if check_orientation:
        #     valid_match_idxs = rot_histo.get_valid_idxs()
        #     print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #     #print('rotation histogram: ', rot_histo)
        #     idx_ref_out = np.array(idx_ref_out)[valid_match_idxs]
        #     idx_cur_out = np.array(idx_cur_out)[valid_match_idxs]
        #     num_matched_map_pts = len(valid_match_idxs)

        return num_matched_map_pts, idx_ref_out, idx_cur_out

    @staticmethod
    def create_vo_points(
        frame: Frame, max_num_points=Parameters.kMaxNumVisualOdometryPoints, color=(0, 0, 255)
    ):
        """
        Create VO (Visual Odometry) points on this frame using depth information.

        This method selects keypoints with valid depths, filters them based on depth thresholds
        and observation counts, then creates MapPoint objects for tracking purposes.

        Args:
            max_num_points: Maximum number of VO points to create
            depth_threshold: Depth threshold for point selection (uses camera.depth_threshold if None)
            color: Color for VO points (BGR format)

        Returns:
            list: List of created MapPoint objects
        """

        if frame.depths is None:
            return []

        depth_threshold = frame.camera.depth_threshold

        # Filter points with valid depths
        valid_mask = frame.depths > Parameters.kMinDepth
        if not np.any(valid_mask):
            return []

        valid_depths = frame.depths[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        # Sort by depth (increasing order)
        sort_indices = np.argsort(valid_depths)
        sorted_depths = valid_depths[sort_indices]
        sorted_indices = valid_indices[sort_indices]

        # Apply depth threshold and max points selection
        mask_depths_smaller_than_th = sorted_depths < depth_threshold
        mask_first_N_points = np.arange(len(sorted_depths)) < max_num_points
        mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)

        # Apply first selection mask
        selected_indices = sorted_indices[mask_first_selection]

        # Filter out points that already have observations
        vector_num_mp_observations = np.array(
            [p.num_observations() if p is not None else 0 for p in frame.points[selected_indices]],
            dtype=np.int32,
        )
        mask_where_to_create_new_map_points = vector_num_mp_observations < 1

        # Apply final selection mask
        final_indices = selected_indices[mask_where_to_create_new_map_points]

        if len(final_indices) == 0:
            return []

        # Create 3D points
        pts3d, pts3d_mask = frame.unproject_points_3d(final_indices, transform_in_world=True)
        if pts3d is None or pts3d_mask is None:
            return []

        # Create MapPoint objects
        created_points = []
        for i, (p, is_valid) in enumerate(zip(pts3d, pts3d_mask)):
            if not is_valid:
                continue

            # Create new map point (VO point)
            mp = MapPoint(p[0:3], color, frame, final_indices[i])
            frame.points[final_indices[i]] = mp
            created_points.append(mp)

        return created_points

    @staticmethod
    def create_and_add_stereo_map_points_on_new_kf(f: Frame, kf: KeyFrame, map: Map, img):
        valid_depths_and_idxs = [
            (z, i) for i, z in enumerate(kf.depths) if z > Parameters.kMinDepth
        ]
        valid_depths_and_idxs.sort()  # increasing-depth order

        if len(valid_depths_and_idxs) == 0:
            Printer.yellow(
                "[create_and_add_stereo_map_points_on_new_kf] no valid depths and idxs found, returning"
            )
            return 0

        sorted_z_values, sorted_idx_values = zip(
            *valid_depths_and_idxs
        )  # unpack the sorted z values and i values into separate lists
        sorted_z_values = np.array(sorted_z_values, dtype=np.float32)
        sorted_idx_values = np.array(sorted_idx_values, dtype=np.int32)

        N = Parameters.kMaxNumStereoPointsOnNewKeyframe
        # create new map points where the depth is smaller than the prefixed depth threshold
        #        or at least N new points with the closest depths
        mask_depths_smaller_than_th = sorted_z_values < kf.camera.depth_threshold
        mask_first_N_points = np.zeros(len(sorted_z_values), dtype=bool)
        mask_first_N_points[: min(N, len(sorted_z_values))] = (
            True  # set True for the first N points otherwise set all True if len(sorted_z_values) < N
        )
        mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)

        sorted_z_values = sorted_z_values[mask_first_selection]
        sorted_idx_values = sorted_idx_values[mask_first_selection]
        sorted_points = kf.points[sorted_idx_values]

        # get the points that are None or where the num of observations is smaller than 1
        vector_num_mp_observations = np.array(
            [p.num_observations() if p is not None else 0 for p in sorted_points],
            dtype=np.int32,
        )
        mask_where_to_create_new_map_points = vector_num_mp_observations < 1

        sorted_z_values = sorted_z_values[mask_where_to_create_new_map_points]
        sorted_idx_values = sorted_idx_values[mask_where_to_create_new_map_points]

        pts3d, pts3d_mask = f.unproject_points_3d(sorted_idx_values, transform_in_world=True)
        num_added_points = map.add_stereo_points(pts3d, pts3d_mask, f, kf, sorted_idx_values, img)
        return num_added_points
