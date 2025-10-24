import numpy as np

import sys

from pyslam.config import Config

from pyslam.slam import Sim3Pose
from pyslam.utilities.geometry import rotation_matrix_from_yaw_pitch_roll

if __name__ == "__main__":

    rand_yaw, rand_pitch, rand_roll = np.random.uniform(-180, 180, 3)
    rand_tx, rand_ty, rand_tz = np.random.uniform(-10, 10, 3)
    R = rotation_matrix_from_yaw_pitch_roll(rand_yaw, rand_pitch, rand_roll)
    t = np.array([rand_tx, rand_ty, rand_tz]).reshape(3, 1)
    s = np.random.uniform(0.5, 1.5)
    sim3 = Sim3Pose(R, t, s)
    print(sim3)
    print(sim3.matrix())
    print(sim3.inverse())
    print(sim3.inverse_matrix())
    print(sim3.to_se3_matrix())

    sim3_inv = sim3.inverse()
    print(sim3_inv)
    print(sim3_inv.matrix())
    print(sim3_inv.inverse())
    print(sim3_inv.inverse_matrix())
    print(sim3_inv.to_se3_matrix())

    sim3_I = sim3 @ sim3_inv
    print(sim3_I)
    print(sim3_I.matrix())
    print(sim3_I.inverse())
    print(sim3_I.inverse_matrix())
    print(sim3_I.to_se3_matrix())
    diff_R = np.linalg.norm(sim3_I.R - np.eye(3))
    diff_t = np.linalg.norm(sim3_I.t)
    diff_s = abs(sim3_I.s - 1.0)
    print(f"diff_R: {diff_R}, diff_t: {diff_t}, diff_s: {diff_s}")
