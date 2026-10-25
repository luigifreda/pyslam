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

import sys

import pyslam.config as config

import numpy as np
import random

import unittest
from unittest import TestCase

import numpy as np

from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Printer

Parameters.USE_CPP_CORE = False
Printer.info("Using Python core - see the tests in cpp/core/slam/tests_py for C++ core tests")

from pyslam.slam import (
    Frame,
    KeyFrame,
    Camera,
    PinholeCamera,
    Sim3Pose,
    MapPoint,
    Map,
    optimizer_gtsam,
    optimizer_g2o,
)
from pyslam.utilities.geometry import (
    rotation_matrix_from_yaw_pitch_roll,
    pitch_matrix,
    poseRt,
    inv_T,
    rpy_from_rotation_matrix,
)
from pyslam.utilities.synthetic_data import (
    generate_random_points_2d,
    backproject_points,
    project_points,
)

from collections import defaultdict
import time


OPTIMIZE_ESSENTIAL_GRAPH = optimizer_gtsam.optimize_essential_graph


class DataGenerator:
    def __init__(self, n=100, radius=10.0, sigma_noise_xyz=0.001, sigma_noise_theta_deg=0.001):
        self.n = n
        self.radius = radius
        self.sigma_noise_xyz = sigma_noise_xyz
        self.sigma_noise_theta_rad = np.deg2rad(sigma_noise_theta_deg)
        self.scale_noise_sigma = 1e-2

        self.map_obj = Map()
        self.keyframes = []
        self.loop_connections = defaultdict(set)
        self.non_corrected_sim3_map = {}
        self.corrected_sim3_map = {}
        self.gt_poses = []

        self.loop_keyframe = None
        self.fix_scale = False

        seed = 0
        np.random.seed(seed)  # make it deterministic
        random.seed(seed)

    def generate_loop_data(self):
        n = self.n
        radius = self.radius
        sigma_noise_xyz = self.sigma_noise_xyz
        sigma_noise_theta_rad = self.sigma_noise_theta_rad

        delta_angle = 2 * np.pi / n
        omega = delta_angle
        velocity = omega * radius

        x2d, y2d, theta = 0, 0, 0
        x2d_n, y2d_n, theta_n = x2d, y2d, theta

        current_scale = 1.0

        for i in range(n):

            # in 3D with computer vision xyz coordinates (z along optical axis, x right, y down)
            Rwc = pitch_matrix(-theta_n + np.pi / 2)
            twc = np.array([x2d_n, 0, y2d_n])
            Rcw = Rwc.T
            tcw = -Rcw @ twc

            gt_Rwc = pitch_matrix(-theta + np.pi / 2)
            gt_twc = np.array([x2d, 0, y2d])
            gt_Twc = poseRt(gt_Rwc, gt_twc)
            self.gt_poses.append(gt_Twc)

            f = Frame(camera=None, img=None)
            f.update_pose(poseRt(Rcw, tcw))

            kf = KeyFrame(frame=f)
            self.keyframes.append(kf)
            if i > 0:
                kf_prev = self.keyframes[i - 1]
                kf.add_connection(kf_prev, 1000)  # fake weight
                kf_prev.add_connection(kf, 1000)  # fake weight
                kf.set_parent(kf_prev)
                d = 2
                while (i - d) >= 0 and d < 10:
                    kf_prev2 = self.keyframes[i - d]
                    kf.add_connection(kf_prev2, 1000)  # fake weight
                    kf_prev2.add_connection(kf, 1000)  # fake weight
                    d += 1

            self.map_obj.add_keyframe(kf)

            # update status

            # 2D classic x,y,theta no noise
            theta += delta_angle
            x2d += velocity * np.cos(theta)
            y2d += velocity * np.sin(theta)

            # 2D classic x,y,theta with noise
            theta_n += delta_angle + random.gauss(0, sigma_noise_theta_rad)
            x2d_n += current_scale * velocity * np.cos(theta_n) + abs(
                random.gauss(0, sigma_noise_xyz)
            )
            y2d_n += current_scale * velocity * np.sin(theta_n) + abs(
                random.gauss(0, sigma_noise_xyz)
            )

            if not self.fix_scale:
                current_scale = current_scale * (1.0 + random.gauss(0, self.scale_noise_sigma))

        self.current_keyframe = self.keyframes[-1]

    def roto_translate_all_keyframes(self, Rw2w1, tw2w1):
        for i, kf in enumerate(self.keyframes):
            Twc = kf.Twc()
            Rwci = Twc[:3, :3]
            twci = Twc[:3, 3]
            Rw2ci = Rw2w1 @ Rwci
            tw2ci = Rw2w1 @ twci + tw2w1
            Rciw2 = Rw2ci.T
            tciw2 = -Rciw2 @ tw2ci
            kf.update_pose(poseRt(Rciw2, tciw2))
            gt_Twci = self.gt_poses[i]
            gt_Rwci = gt_Twci[:3, :3]
            gt_twci = gt_Twci[:3, 3]
            gt_Rw2ci = Rw2w1 @ gt_Rwci
            gt_tw2ci = Rw2w1 @ gt_twci + tw2w1
            self.gt_poses[i] = poseRt(gt_Rw2ci, gt_tw2ci)

    def add_loop_closure(self):

        assert len(self.keyframes) > 1
        assert len(self.keyframes) == len(self.gt_poses)

        print(f"adding loop closure ...")
        # let's add a loop closure for the loop: last_keyframe -> first_keyframe
        first_kf = self.keyframes[0]
        last_kf = self.keyframes[-1]
        current_keyframe = last_kf
        self.loop_keyframe = first_kf

        # retrieve keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        current_keyframe.update_connections()

        current_connected_keyframes = current_keyframe.get_connected_keyframes()
        # print(f'current keyframe: {current_keyframe.id}, connected keyframes: {[k.id for k in current_connected_keyframes]}')
        current_connected_keyframes.append(current_keyframe)

        Twc = current_keyframe.Twc()
        # let's use the ground truth as loop closure correction
        last_gt_Twc = self.gt_poses[-1].copy()
        last_gt_Tcw = inv_T(last_gt_Twc)
        Scw = Sim3Pose(last_gt_Tcw[:3, :3], last_gt_Tcw[:3, 3], 1.0)

        self.corrected_sim3_map[current_keyframe] = Scw

        # Iterate over all current connected keyframes and propagate the sim3 correction obtained on current keyframe
        for connected_kfi in current_connected_keyframes:
            Tiw = connected_kfi.Tcw()
            if connected_kfi != current_keyframe:
                Tic = Tiw @ Twc
                Ric = Tic[:3, :3]
                tic = Tic[:3, 3]
                Sic = Sim3Pose(Ric, tic, 1.0)
                corrected_Siw = Sic @ Scw
                corrected_Swi = corrected_Siw.inverse()
                # Pose corrected with the Sim3 of the loop closure
                self.corrected_sim3_map[connected_kfi] = corrected_Siw
                # print(f'connected_kfi: {connected_kfi.id}, corrected_twi: {corrected_Swi.t.flatten()}, current_Rwi_rpy: {rpy_from_rotation_matrix(corrected_Swi.R)}, corrected_swi: {corrected_Swi.s}')

            Riw = Tiw[:3, :3]
            tiw = Tiw[:3, 3]
            Siw = Sim3Pose(Riw, tiw, 1.0)
            # Pose without correction
            self.non_corrected_sim3_map[connected_kfi] = Siw

        # add connections among last_keyframe and [first_keyframe + its connected keyframes]
        if True:
            first_connected_keyframes = first_kf.get_connected_keyframes()
            print(
                f"first keyframe: {first_kf.id}, connected keyframes: {[k.id for k in first_connected_keyframes]}"
            )
            for kf in first_connected_keyframes + [first_kf]:
                current_keyframe.add_connection(kf, 1000)  # fake weight
                kf.add_connection(current_keyframe, 1000)  # fake weight

        # Correct all map points observed by current keyframe and its neighbors,
        # so that they align with the other side of the loop
        for connected_kfi, corrected_Siw in self.corrected_sim3_map.items():
            corrected_Swi = corrected_Siw.inverse()
            Siw = self.non_corrected_sim3_map[connected_kfi]

            if False:
                correction_Sw = corrected_Swi @ Siw
                correction_sRw = correction_Sw.R * correction_Sw.s
                correction_tw = correction_Sw.t

                # Correct MapPoints
                map_points = connected_kfi.get_points()
                for i, map_point in enumerate(map_points):
                    if (
                        not map_point
                        or map_point.is_bad()
                        or map_point.corrected_by_kf == current_keyframe.kid
                    ):  # use kid here
                        continue

                    # Project with non-corrected pose and project back with corrected pose
                    p3dw = map_point.pt()
                    # corrected_p3dw = corrected_Swi @ Siw @ p3dw
                    corrected_p3dw = correction_sRw @ p3dw.reshape(3, 1) + correction_tw
                    map_point.update_position(corrected_p3dw.squeeze())
                    map_point.update_normal_and_depth()
                    map_point.corrected_by_kf = current_keyframe.kid  # use kid here
                    map_point.corrected_reference = connected_kfi.kid  # use kid here

            # Update keyframe pose with corrected Sim3
            corrected_Tiw = corrected_Siw.to_se3_matrix()  # [R t/s;0 1]
            print(
                f"correcting keyframe: {connected_kfi.id}, current_Rwi_rpy: {rpy_from_rotation_matrix(corrected_Swi.R)}, corrected_tiw: {corrected_Tiw[:3,3].flatten()}"
            )
            connected_kfi.update_pose(corrected_Tiw)
            connected_kfi.update_connections()

        # Create a dictionary where each key is a KeyFrame and the value is a set of connected KeyFrames
        loop_connections = defaultdict(set)

        loop_connections[first_kf] = set(
            [current_keyframe] + current_keyframe.get_connected_keyframes()
        )
        loop_connections[last_kf] = set([first_kf] + first_kf.get_connected_keyframes())
        # for kfi in current_connected_keyframes:
        #     # Get previous neighbors (covisible keyframes)
        #     previous_neighbors = kfi.get_covisible_keyframes()

        #     # Update connections and get the new ones
        #     kfi.update_connections()
        #     loop_connections[kfi] = set(kfi.get_connected_keyframes())

        #     # # Remove previous neighbors from connections
        #     # for previous_neighbor in previous_neighbors:
        #     #     try:
        #     #         loop_connections[kfi].remove(previous_neighbor)
        #     #     except:
        #     #         pass # not found

        #     # # Remove the current connected keyframes from the connection set
        #     # for other_kf in current_connected_keyframes:
        #     #     try:
        #     #         loop_connections[kfi].remove(other_kf)
        #     #     except:
        #     #         pass # not found

        self.loop_connections = loop_connections
        self.current_keyframe = current_keyframe

    def compute_ATE(self):
        error = 0
        for i, kf in enumerate(self.keyframes):
            gt_Twc = self.gt_poses[i]
            Twc = kf.Twc()
            error += np.linalg.norm(gt_Twc[:3, 3] - Twc[:3, 3]) ** 2
        return np.sqrt(error / len(self.keyframes))


def run_loop_closure(optimize_essential_graph, verbose=False):
    data_generator = DataGenerator(
        n=50, radius=5.0, sigma_noise_xyz=0.02, sigma_noise_theta_deg=0.1
    )
    data_generator.generate_loop_data()
    est_poses_before = [kf.Twc() for kf in data_generator.keyframes]
    ate_before = data_generator.compute_ATE()

    data_generator.add_loop_closure()
    mse = optimize_essential_graph(
        data_generator.map_obj,
        data_generator.loop_keyframe,
        data_generator.current_keyframe,
        data_generator.non_corrected_sim3_map,
        data_generator.corrected_sim3_map,
        data_generator.loop_connections,
        data_generator.fix_scale,
        verbose=verbose,
    )
    est_poses = [kf.Twc() for kf in data_generator.keyframes]
    ate_after = data_generator.compute_ATE()
    print(f"ATE before: {ate_before}, ATE after: {ate_after}, optimization MSE: {mse}")
    return data_generator, est_poses_before, est_poses, ate_before, ate_after


class TestOptimizeEssentialGraph(TestCase):
    def test_loop_closure_reduces_trajectory_error(self):
        _, _, est_poses, ate_before, ate_after = run_loop_closure(OPTIMIZE_ESSENTIAL_GRAPH)
        self.assertTrue(all(np.isfinite(T).all() for T in est_poses))
        self.assertLess(ate_after, 0.8 * ate_before)


def main():
    from pyslam.viz.viewer3D import Viewer3D

    data_generator, est_poses_before, est_poses, _, _ = run_loop_closure(
        OPTIMIZE_ESSENTIAL_GRAPH, verbose=True
    )
    viewer3d = Viewer3D()
    viewer3d.draw_cameras(
        [data_generator.gt_poses, est_poses, est_poses_before],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        show_trajectory_line=True,
    )
    while not viewer3d.is_closed():
        time.sleep(0.1)


if __name__ == "__main__":
    if "--view" in sys.argv:
        main()
    else:
        unittest.main()
