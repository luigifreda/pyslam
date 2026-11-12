"""Regression coverage for documented trajectory export formats."""

import csv

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyslam.io.trajectory_writer import TrajectoryWriter


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("full_trajectory", [False, True])
def test_euroc_pose_round_trip(tmp_path, dtype, full_trajectory):
    path = tmp_path / "trajectory.csv"
    timestamps = [1.25, 1403636579.75]
    poses = np.repeat(np.eye(4, dtype=dtype)[None, ...], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    poses[:, :3, 3] = [[1.5, -2.25, 3.75], [-0.5, 0.25, 2.0]]
    writer = TrajectoryWriter("euroc", str(path))
    try:
        if full_trajectory:
            writer.write_full_trajectory(poses, timestamps)
        else:
            for pose, timestamp in zip(poses, timestamps):
                writer.write_trajectory(pose[:3, :3], pose[:3, 3], timestamp)
    finally:
        writer.close_file()

    with path.open() as file:
        rows = list(csv.reader(file))
    assert len(rows) == len(poses)
    for row, pose, timestamp in zip(rows, poses, timestamps):
        # EuRoC CSV stores integer nanoseconds and scalar-first quaternions.
        assert len(row) == 8
        assert int(row[0]) == round(timestamp * 1e9)
        assert int(row[0]) * 1e-9 == pytest.approx(timestamp)
        np.testing.assert_allclose(np.asarray(row[1:4], dtype=float), pose[:3, 3])
        qw, qx, qy, qz = map(float, row[4:8])
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        np.testing.assert_allclose(rotation, pose[:3, :3], atol=1e-6)


def test_tum_retains_seconds_and_scalar_last_quaternion(tmp_path):
    path = tmp_path / "trajectory.txt"
    writer = TrajectoryWriter("tum", str(path))
    try:
        writer.write_trajectory(np.eye(3), np.array([1.0, 2.0, 3.0]), 1.25)
    finally:
        writer.close_file()
    np.testing.assert_array_equal(np.loadtxt(path), [1.25, 1, 2, 3, 0, 0, 0, 1])
