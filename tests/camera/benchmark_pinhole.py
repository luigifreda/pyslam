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
* Opt-in paired pinhole benchmark; exits 1 on regressions.
*
* Uses a git revision's Python camera implementation and the original native
* optimizer factors as baselines. No checkout, build, dataset images, or writes
* to config.yaml are needed. This is a component benchmark, not a SLAM replay.
"""

import argparse
import contextlib
import hashlib
import io
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time

from camera_test_support import ROOT, calibration_cases, sample_points
from benchmark_guard import BenchmarkGuard
import numpy as np
import cv2
import g2o
import gtsam
import gtsam_factors
from pyslam.slam.camera import camera_from_config
from pyslam.slam.camera_factors import camera_projection_factor, g2o_camera_project_pose_edge
from pyslam.slam.cpp import cpp_module

_BASELINE_TEMP = tempfile.TemporaryDirectory(prefix="pyslam-camera-baseline-")


def baseline_camera(revision):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", commit + ":pyslam/slam/camera.py"], cwd=ROOT, text=True
    )
    # Numba's cache needs an actual source file and an importable module name.
    from pathlib import Path
    path = Path(_BASELINE_TEMP.name) / "camera_benchmark_baseline.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("camera_benchmark_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return commit, module.PinholeCamera


def measure_pair(reference, candidate, samples, iterations):
    # Warm up both implementations, including Numba compilation.
    for _ in range(3):
        reference()
        candidate()
    timings = [[], []]
    functions = [reference, candidate]
    # Alternate order to reduce thermal / frequency drift bias.
    for sample in range(samples):
        for index in (0, 1) if sample % 2 == 0 else (1, 0):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                functions[index]()
            timings[index].append((time.perf_counter_ns() - start) / iterations / 1e6)
    ref, cur = (statistics.median(values) for values in timings)
    # Compare each adjacent pair before taking its median. Independent medians
    # can select samples from different CPU-frequency/load regimes.
    paired_ratio = statistics.median(current / baseline for baseline, current in zip(*timings))
    return {"baseline_ms": ref, "candidate_ms": cur, "ratio": paired_ratio,
            "median_duration_ratio": cur / ref,
            "baseline_samples_ms": timings[0], "candidate_samples_ms": timings[1]}


def gtsam_graph(camera, points, modern):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    pose_key = gtsam.symbol("x", 0)
    values.insert(pose_key, gtsam.Pose3())
    calibration = gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)
    noise = gtsam.noiseModel.Unit.Create(2)
    uv, _ = camera.project(points)
    hosts = []
    for index, point in enumerate(points):
        point_key = gtsam.symbol("l", index)
        values.insert(point_key, gtsam.Point3(*point))
        measurement = uv[index] + np.array([0.2, -0.1])
        if modern:
            host = camera_projection_factor(measurement, noise, pose_key, point_key, camera)
            hosts.append(host)
            graph.add(host.get_factor() if hasattr(host, "get_factor") else host)
        else:
            graph.add(gtsam_factors.WeightedGenericProjectionFactorCal3_S2(
                measurement, noise, pose_key, point_key, calibration
            ))
    return graph, values, hosts


def g2o_pose_graph(camera, points, modern):
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
        g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    ))
    pose = g2o.VertexSE3Expmap()
    pose.set_id(0)
    pose.set_estimate(g2o.SE3Quat(np.eye(3), np.array([0.04, -0.02, 0.03])))
    optimizer.add_vertex(pose)
    measurements, _ = camera.project(points)
    edges = []
    for point, measurement in zip(points, measurements):
        if modern:
            edge = g2o_camera_project_pose_edge(camera)
        else:
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()
            edge.fx, edge.fy, edge.cx, edge.cy = camera.fx, camera.fy, camera.cx, camera.cy
        edge.Xw = point.copy()
        edge.set_vertex(0, pose)
        edge.set_measurement(measurement)
        edge.set_information(np.eye(2))
        optimizer.add_edge(edge)
        # The optimizer owns the C++ edge; retain its Python wrapper as well so
        # the camera keep-alive policy remains effective for the whole solve.
        edges.append(edge)
    optimizer.initialize_optimization()
    return optimizer, pose, edges


def run(args):
    commit, reference_class = baseline_camera(args.baseline_ref)
    points = sample_points(args.points)
    rows = []

    def record(calibration, sensor, backend, operation, reference, candidate):
        row = {"calibration": calibration, "sensor": sensor, "backend": backend,
               "operation": operation}
        row.update(measure_pair(reference, candidate, args.samples, args.iterations))
        rows.append(row)

    cases = list(calibration_cases())
    for name, sensor, config in ([] if args.optimizers_only else cases):
        with contextlib.redirect_stdout(io.StringIO()):
            reference = reference_class(config)
            python = camera_from_config(config)
            native = camera_from_config(config, cpp_module.classes)
        pixels, depths = reference.project(points)
        pixels = np.ascontiguousarray(pixels)
        depths = np.ascontiguousarray(depths)
        for backend, candidate in (("python", python), ("cpp", native)):
            operations = [
                ("project", (points,)),
                ("unproject_points", (pixels,)),
                ("undistort_points", (pixels,)),
            ]
            if sensor != "mono":
                operations.extend([
                    ("project_stereo", (points,)),
                    ("unproject_points_3d", (pixels, depths)),
                ])
            for operation, params in operations:
                ref_call = lambda op=operation, p=params: getattr(reference, op)(*p)
                cur_call = lambda op=operation, p=params, cam=candidate: getattr(cam, op)(*p)
                expected, actual = ref_call(), cur_call()
                if isinstance(expected, tuple):
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)
                # Native vs Numba is useful context, but is not a revision comparison.
                record(name, sensor, backend, operation, ref_call, cur_call)

    # Mono observations also exist in stereo/RGB-D frames wherever depth is missing.
    # Test the changed Python optimizers independently from camera batch projection.
    config = next(config for name, sensor, config in cases if name == "KITTI00-02.yaml" and sensor == "mono")
    with contextlib.redirect_stdout(io.StringIO()):
        camera = camera_from_config(config)
    graph_points = sample_points(64)
    legacy_graph, legacy_values, _ = gtsam_graph(camera, graph_points, False)
    modern_graph, modern_values, hosts = gtsam_graph(camera, graph_points, True)
    np.testing.assert_allclose(modern_graph.error(modern_values), legacy_graph.error(legacy_values), atol=1e-8)
    for actual, expected in zip(modern_graph.linearize(modern_values).hessian(), legacy_graph.linearize(legacy_values).hessian()):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
    record("KITTI00-02.yaml", "mono-observations", "python-gtsam", "graph_error",
           lambda: legacy_graph.error(legacy_values), lambda: modern_graph.error(modern_values))
    record("KITTI00-02.yaml", "mono-observations", "python-gtsam", "graph_linearize",
           lambda: legacy_graph.linearize(legacy_values), lambda: modern_graph.linearize(modern_values))

    # Only the pose is optimized: fixed 3D landmarks remove the gauge freedom of
    # the existing all-free bundle-adjustment timing test.
    legacy_optimizer, legacy_pose, legacy_edges = g2o_pose_graph(camera, graph_points, False)
    modern_optimizer, modern_pose, modern_edges = g2o_pose_graph(camera, graph_points, True)
    legacy_optimizer.optimize(8)
    modern_optimizer.optimize(8)
    np.testing.assert_allclose(modern_pose.estimate().matrix(), legacy_pose.estimate().matrix(), atol=1e-7)
    np.testing.assert_allclose(modern_pose.estimate().translation(), np.zeros(3), atol=1e-7)
    record("KITTI00-02.yaml", "mono-observations", "python-g2o", "pose_graph_build",
           lambda: g2o_pose_graph(camera, graph_points, False),
           lambda: g2o_pose_graph(camera, graph_points, True))
    record("KITTI00-02.yaml", "mono-observations", "python-g2o", "graph_error",
           legacy_optimizer.compute_active_errors, modern_optimizer.compute_active_errors)

    for row in rows:
        # Only Python camera vs the selected git revision, and like-for-like
        # optimizer factors, support a temporal regression claim.
        row["gated"] = row["backend"] != "cpp"
        row["regression"] = row["gated"] and row["ratio"] > args.max_ratio
    report = {
        "baseline_commit": commit,
        "candidate_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_worktree_diff_stat": subprocess.check_output(
            ["git", "diff", "--stat", "--", "pyslam/slam", "cpp/core/slam"], cwd=ROOT, text=True),
        "candidate_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for root, patterns in (
                (ROOT / "pyslam/slam", ("camera*.py", "optimizer*.py")),
                (ROOT / "cpp/core/slam", ("camera*.cpp", "camera*.h", "optimizer*.cpp")),
            )
            for pattern in patterns
            for path in sorted(root.glob(pattern))
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "opencv": cv2.__version__},
        "points": args.points, "samples": args.samples, "iterations": args.iterations,
        "max_ratio": args.max_ratio, "calibration_sensor_cases": 0 if args.optimizers_only else len(cases),
        "ratio_method": "median of paired candidate/baseline durations",
        "scope": "synthetic components; cpp vs Numba rows are informational; no dataset replay",
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    failures = [row for row in rows if row["regression"]]
    print(f"{report['calibration_sensor_cases']} calibration/sensor cases; {len(rows)} measurements; {len(failures)} exceeded {args.max_ratio:.2f}x")
    for row in rows:
        if row["backend"].startswith("python-"):
            print(f"{row['backend']} {row['operation']}: {row['ratio']:.2f}x ({row['baseline_ms']:.4f} -> {row['candidate_ms']:.4f} ms)")
    print(f"Report: {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Commit before camera model support")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--max-ratio", type=float, default=1.20)
    parser.add_argument("--optimizers-only", action="store_true", help="Skip the calibration batch benchmarks")
    args = parser.parse_args()
    if min(args.samples, args.iterations, args.points) < 1 or args.max_ratio <= 1.0:
        parser.error("samples, iterations, points must be positive; max-ratio must exceed 1")
    with BenchmarkGuard():
        raise SystemExit(run(args))
