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
* Bounded, deterministic replay of a recorded sequence; writes timings and trajectory error.
*
* Works for any dataset section in config.yaml (KITTI, EuRoC, TUM, TUM-VI, ...). Local mapping
* runs synchronously by default, so its cost is included in each tracking call;
* --local-mapping threaded mirrors playback scheduling. Loop closure, semantics
* and dense mapping are off unless requested. Use --pyslam-root to replay with another pySLAM
* checkout (e.g. a pinhole-only baseline) for regression comparisons. Run each configuration in
* a fresh process.
*
* Examples:
*   python tests/camera/replay_sequence.py --dataset TUM_VI_DATASET --base-path <tum_vi> \\
*       --name dataset-room1_512_16 --set stereo_mode=native --output /tmp/room1_native.json
*   python tests/camera/replay_sequence.py --dataset KITTI_DATASET --base-path <kitti>/dataset \\
*       --name 00 --sensor-type stereo --set is_color=false --frames 1000 --output /tmp/k00.json
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import subprocess
import os
import platform
import random
import hashlib

import numpy as np
import yaml

from trajectory_evaluation import translation_ate
from benchmark_guard import BenchmarkGuard


def parse_value(text):
    return yaml.safe_load(text)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pyslam-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", required=True, help="config.yaml section, e.g. EUROC_DATASET")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sensor-type", choices=("mono", "stereo", "rgbd"))
    parser.add_argument("--settings", help="camera settings file (relative to --pyslam-root)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra dataset-section override (repeatable)")
    parser.add_argument("--frames", type=int, default=0, help="0 means the whole sequence")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--loop-closure", action="store_true", help="enable DBoW3 loop closing")
    parser.add_argument("--frontend", choices=("python", "cpp"), default="python",
                        help="orchestrator implementation; both may use the native core")
    parser.add_argument("--local-mapping", choices=("synchronous", "threaded"),
                        default="synchronous",
                        help="threaded mirrors playback; synchronous keeps comparisons deterministic")
    parser.add_argument("--max-align-dt", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-gib", type=float, default=8.)
    parser.add_argument("--min-available-gib", type=float, default=8.)
    parser.add_argument("--trace-memory", action="store_true",
                        help="record Python allocation diagnostics (adds overhead)")
    args = parser.parse_args()
    if args.frames < 0 or args.start < 0:
        parser.error("frames and start must be nonnegative")
    if not np.isfinite(args.max_align_dt) or args.max_align_dt <= 0:
        parser.error("max-align-dt must be positive and finite")
    return args


def load_config(args, Config):
    settings = yaml.safe_load((args.pyslam_root / "config.yaml").read_text())
    section = settings[args.dataset]
    settings["DATASET"]["type"] = args.dataset
    section.update(base_path=str(args.base_path.resolve()), name=args.name,
                   start_frame_id=args.start)
    if args.sensor_type:
        section["sensor_type"] = args.sensor_type
    if args.settings:
        section["settings"] = args.settings
        section.pop("settings_stereo", None)
    for item in args.set:
        key, _, value = item.partition("=")
        section[key] = parse_value(value)
    log_folder = args.output.resolve().with_suffix('.logs')
    log_folder.mkdir(parents=True, exist_ok=True)
    settings["GLOBAL_PARAMETERS"] = dict(settings.get("GLOBAL_PARAMETERS") or {},
        kLocalMappingOnSeparateThread=args.local_mapping == "threaded",
        kUseLoopClosing=args.loop_closure,
        kLogsFolder=str(log_folder),
        kDoSparseSemanticMappingAndSegmentation=False, kDoVolumetricIntegration=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
        yaml.safe_dump(settings, handle)
        handle.flush()
        return Config(handle.name)


def make_camera(config):
    try:
        from pyslam.slam import camera_from_config
    except ImportError:  # pinhole-only checkouts
        from pyslam.slam import PinholeCamera
        return PinholeCamera(config)
    return camera_from_config(config)


def summarize(samples):
    return dict(median=1000 * float(np.median(samples)),
                p95=1000 * float(np.percentile(samples, 95)),
                mean=1000 * float(np.mean(samples)))


def groundtruth_samples(gt):
    # TUM's frame associations can reuse a mocap sample for adjacent images.
    # Evaluate against the raw mocap stream so interpolation sees unique times.
    if gt.type.name in ("TUM", "ICL_NUIM"):
        samples = np.loadtxt(gt.filename, dtype=np.float64, ndmin=2)
        return samples[:, 1:4] * gt.scale, samples[:, 0]
    return gt.getFull3dTrajectory()


def run(args, guard):
    import cv2
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    np.random.seed(0)
    random.seed(0)
    args.pyslam_root = args.pyslam_root.resolve()
    sys.path.insert(0, str(args.pyslam_root))
    from pyslam.config import Config
    from pyslam.config_parameters import Parameters
    from pyslam.io.dataset_factory import dataset_factory
    from pyslam.io.dataset_types import SensorType
    from pyslam.io.ground_truth import groundtruth_factory
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    if args.frontend == "cpp":
        if not Parameters.USE_CPP_CORE:
            raise ValueError("The C++ frontend requires PYSLAM_USE_CPP=1")
        from pyslam.slam import Slam
    else:
        from pyslam.slam.slam import Slam

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args, Config)
    dataset = dataset_factory(config)
    sensor = dataset.sensorType()
    count = dataset.num_frames - args.start
    if args.frames:
        count = min(args.frames, count)
    if count < 1:
        raise ValueError("Start index is beyond the sequence")
    report = dict(pyslam_root=str(args.pyslam_root), dataset=args.dataset, name=args.name,
                  sensor=sensor.name, overrides=args.set, cpp_core=Parameters.USE_CPP_CORE,
                  start=args.start, processed_frames=count, total_frames=dataset.num_frames,
                  local_mapping=args.local_mapping, loop_closure=args.loop_closure,
                  frontend=args.frontend)
    report['environment'] = dict(python=platform.python_version(), platform=platform.platform(),
                                 numpy=np.__version__, opencv=cv2.__version__,
                                 opencv_threads=cv2.getNumThreads(),
                                 omp_threads=os.environ.get('OMP_NUM_THREADS'),
                                 openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    try:
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.pyslam_root,
            stderr=subprocess.DEVNULL, text=True).strip()
        report["worktree_diff_stat"] = subprocess.check_output(
            ["git", "diff", "--stat"], cwd=args.pyslam_root, text=True)
    except subprocess.CalledProcessError:  # git-archive exports are valid baselines
        report["commit"], report["worktree_diff_stat"] = None, None
    report['source_sha256'] = {
        str(path.relative_to(args.pyslam_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for root, patterns in (
            (args.pyslam_root / 'pyslam/slam', ('*.py',)),
            (args.pyslam_root / 'cpp/core/slam', ('*.cpp', '*.h', 'lib/cpp_core*.so')),
            (args.pyslam_root / 'pyslam/slam/cpp', ('*.cpp', '*.h', 'lib/cpp_core*.so')),
        )
        for pattern in patterns
        for path in sorted(root.glob(pattern))}

    tracker = copy.deepcopy(FeatureTrackerConfigs.ORB2)
    tracker["num_features"] = config.num_features_to_extract
    tracker["deterministic"] = True
    camera = make_camera(config)
    report["camera"] = type(camera).__name__
    slam = Slam(camera, tracker,
                loop_detector_config=LoopDetectorConfigs.DBOW3 if args.loop_closure else None,
                sensor_type=sensor, environment_type=dataset.environmentType(), config=config,
                headless=True)
    tracking, loading, states = [], [], Counter()
    if args.trace_memory:
        import tracemalloc
        tracemalloc.start(3)
    started = time.perf_counter()
    try:
        for i in range(count):
            guard.check()
            load_start = time.perf_counter()
            image = dataset.getImageColor(i)
            timestamp = dataset.getTimestamp()
            right = dataset.getImageColorRight(i) if sensor == SensorType.STEREO else None
            depth = dataset.getDepth(i) if sensor == SensorType.RGBD else None
            loading.append(time.perf_counter() - load_start)
            if (image is None or (sensor == SensorType.STEREO and right is None)
                    or (sensor == SensorType.RGBD and depth is None)):
                raise RuntimeError(f"Cannot load frame {i + args.start}")
            begin = time.perf_counter()
            slam.track(image, right, depth, i, timestamp)
            guard.check()
            tracking.append(time.perf_counter() - begin)
            states[slam.tracking.state.name] += 1
            if (i + 1) % 200 == 0:
                print(f"REPLAY {args.name}: {i + 1}/{count} {dict(states)} "
                      f"peak_rss_gib={guard.peak_rss / 1024**3:.3f}", flush=True)
                if args.trace_memory:
                    print('MEMORY_TRACE ' + json.dumps({
                        'frame': i + 1,
                        'python_mib': tracemalloc.get_traced_memory()[0] / 1024**2,
                        'top': [str(row) for row in
                                tracemalloc.take_snapshot().statistics('lineno')[:10]]}), flush=True)
        report.update(elapsed_s=time.perf_counter() - started, states=dict(states),
                      track_and_map_ms=summarize(tracking),
                      load_ms=summarize(loading),
                      keyframes=slam.map.num_keyframes(), map_points=slam.map.num_points())
        poses, timestamps, ids = slam.get_final_trajectory()
        poses = np.asarray(poses)
        report["trajectory_poses"] = len(poses)
        if len(poses) and not np.isfinite(poses).all():
            raise RuntimeError("Nonfinite estimated trajectory")
        np.savez(args.output.with_suffix(".npz"), poses=poses, timestamps=timestamps, ids=ids,
                 track_and_map_s=np.asarray(tracking))
        gt = groundtruth_factory(config.dataset_settings)
        if gt is not None and len(poses) >= 3:
            gt_positions, gt_timestamps = groundtruth_samples(gt)
            sim3 = sensor == SensorType.MONOCULAR
            report.update(translation_ate(
                np.asarray(timestamps, dtype=np.float64), poses[:, :3, 3].astype(np.float64),
                np.asarray(gt_timestamps, dtype=np.float64),
                np.asarray(gt_positions, dtype=np.float64),
                find_scale=sim3, max_gap=args.max_align_dt))
    finally:
        slam.quit()
    report['memory'] = dict(peak_rss_gib=guard.peak_rss / 1024**3,
                          max_rss_gib=args.max_rss_gib,
                          min_available_gib=args.min_available_gib,
                          python_allocation_tracing=args.trace_memory)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("REPLAY_RESULT " + json.dumps(report), flush=True)
    if states.get("OK", 0) == 0:
        raise RuntimeError("Replay never reached tracking state OK")


def main():
    args = parse_args()
    with BenchmarkGuard(args.max_rss_gib, args.min_available_gib) as guard:
        run(args, guard)


if __name__ == "__main__":
    main()
