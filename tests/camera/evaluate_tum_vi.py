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
*
* Evaluate pySLAM NPZ or ORB-SLAM3/PLVS2 EuRoC trajectories on TUM-VI.
*
* EuRoC exports use nanosecond timestamps; NPZ timestamps are seconds. Ground
* truth is converted to the left tracking camera before evaluating translation.
"""

import argparse
import hashlib
import json
from pathlib import Path

from camera_test_support import ROOT, SettingsConfig
import numpy as np
import yaml

from pyslam.io.tum_vi_dataset import prepare_calibration
from pyslam.io.tum_vi_ground_truth import TumViGroundTruth
from trajectory_evaluation import translation_ate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--name', default='dataset-room1_512_16')
    parser.add_argument('--sensor-type', choices=['mono', 'stereo'], required=True)
    parser.add_argument('--mode', choices=['native', 'rectified'], default='native')
    parser.add_argument('--timestamp-scale', type=float, default=1e-9,
                        help='Text export timestamps to seconds (NPZ always uses seconds)')
    parser.add_argument('--max-mocap-gap', type=float, default=0.1)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = SettingsConfig(yaml.safe_load((ROOT / 'settings/TUM_VI.yaml').read_text()), args.sensor_type)
    config.dataset_settings.update(type='tum_vi', stereo_mode=args.mode, base_path=args.base_path,
                                   name=args.name, groundtruth_file='auto')
    prepare_calibration(config)
    gt = TumViGroundTruth(config.dataset_settings)
    if args.trajectory.suffix == '.npz':
        with np.load(args.trajectory) as trajectory:
            timestamps, positions = trajectory['timestamps'], trajectory['poses'][:, :3, 3]
    else:
        trajectory = np.loadtxt(args.trajectory, ndmin=2)
        timestamps, positions = trajectory[:, 0] * args.timestamp_scale, trajectory[:, 1:4]
    result = translation_ate(timestamps, positions, gt.timestamps, gt.trajectory,
                             find_scale=args.sensor_type == 'mono', max_gap=args.max_mocap_gap)
    result.update(trajectory=str(args.trajectory), sensor=args.sensor_type, mode=args.mode,
                  sequence=args.name, dataset_frames=len(gt.image_timestamps),
                  trajectory_fraction=len(timestamps) / len(gt.image_timestamps),
                  trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
