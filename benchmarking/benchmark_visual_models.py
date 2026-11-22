#!/usr/bin/env python3
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
* Benchmark existing tracker presets in isolated processes without downloading models.
*
* Example: python benchmarking/benchmark_visual_models.py --presets ORB2 SIFT \
*     --sizes tum --cores cpp python --device cpu --output /tmp/visual-baseline.json
*
* Uninstrumented internal stages are explicitly null. End-to-end SLAM/ATE measurements
* are separate experiments; synthetic pair inliers must not be presented as sequence ATE.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

from benchmark_common import ROOT, machine_info, provenance, sha256, write_json

SIZES = {"kitti": (1241, 376), "tum": (640, 480), "euroc": (752, 480)}
INTERNAL_STAGES = ("preprocess", "host_to_device", "inference", "postprocess",
                   "device_to_host", "keypoint_conversion")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--presets", nargs="+", default=["ORB2", "LIGHTGLUE", "XFEAT", "LIGHTGLUE_ALIKED", "LOFTR"])
    result.add_argument("--sizes", nargs="+", choices=tuple(SIZES), default=list(SIZES))
    result.add_argument("--cores", nargs="+", choices=("cpp", "python"), default=["cpp", "python"])
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--num-features", type=int, default=2000)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--image", type=Path, help="Use a real image instead of the deterministic synthetic texture")
    result.add_argument("--golden-dir", type=Path, help="Save keypoints, descriptors and match indices as NPZ")
    result.add_argument("--profile-dir", type=Path, help="Save cProfile output per case (changes timing; compare like-for-like)")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--strict", action="store_true", help="Fail if any requested case cannot be measured")
    result.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return result


def block_network():
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("Offline benchmark: network access is disabled; prefetch weights separately")
    sys.addaudithook(audit)


def run_case(args, preset_name, size_name, core):
    # The child process gets CUDA_VISIBLE_DEVICES before any import of torch/config.
    block_network()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import cv2
    started = time.perf_counter()
    import pyslam.config  # Set compiled dependency paths before importing the presets.
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
    # pyslam-only import cost: excludes the benchmark's own explicit torch import, so
    # lazy-import improvements are visible. Informational (not gated); import_ms keeps
    # its original scope (including torch) for comparability with existing baselines.
    pyslam_import_ms = (time.perf_counter() - started) * 1000
    torch_preloaded = "torch" in sys.modules
    import torch
    import_ms = (time.perf_counter() - started) * 1000
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    preset = getattr(FeatureTrackerConfigs, preset_name, None)
    if not isinstance(preset, dict):
        raise ValueError(f"Unknown tracker preset: {preset_name}")
    config = dict(preset, num_features=args.num_features)
    started = time.perf_counter()
    tracker = feature_tracker_factory(**config)
    sync()
    load_ms = (time.perf_counter() - started) * 1000
    width, height = SIZES[size_name]
    if args.image:
        image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image {args.image}")
        image = cv2.resize(image, (width, height))
    else:
        rng = np.random.default_rng(args.seed)
        image = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (5, 5), 0.8)
    affine = np.float32([[1, 0, 4], [0, 1, 2]])
    other = cv2.warpAffine(image, affine, (width, height), borderMode=cv2.BORDER_REFLECT)
    dense = bool(getattr(tracker.matcher, "detector_free", False))  # LoFTR, MASt3R, RoMa, ...
    timings = {"extraction": [], "matching": [], "total": []}
    signatures = set()  # per-iteration output fingerprints: detects run-to-run non-determinism
    kps1 = kps2 = des1 = des2 = None
    coords1 = coords2 = np.empty((0, 2), dtype=np.float32)
    idxs1 = idxs2 = np.empty(0, dtype=np.int32)
    profiler = None
    if args.profile_dir:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    try:
        for iteration in range(args.warmup + args.iterations):
            sync()
            started = time.perf_counter()
            if dense:
                split = started
                result = tracker.track(image, other)
                coords1, coords2 = result.kps_ref, result.kps_cur
                idxs1, idxs2 = result.idxs_ref, result.idxs_cur
            else:
                # Both frame extractions are reported: this is a pair microbenchmark.
                kps1, des1 = tracker.detectAndCompute(image)
                kps2, des2 = tracker.detectAndCompute(other)
                coords1 = np.asarray([kp.pt for kp in kps1], np.float32).reshape(-1, 2)
                coords2 = np.asarray([kp.pt for kp in kps2], np.float32).reshape(-1, 2)
                sync()
                split = time.perf_counter()
                result = tracker.matcher.match(image, other, des1=des1, des2=des2, kps1=coords1, kps2=coords2)
                idxs1, idxs2 = result.idxs1, result.idxs2
            sync()
            stopped = time.perf_counter()
            signatures.add(output_signature(np, coords1, coords2, idxs1, idxs2))
            if iteration >= args.warmup:
                timings["extraction"].append((split - started) * 1000)
                timings["matching"].append((stopped - split) * 1000)
                timings["total"].append((stopped - started) * 1000)
    finally:
        if profiler:
            profiler.disable()
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.profile_dir / f"{preset_name}-{size_name}-{core}-{args.device}.prof"))
    def summary(values):
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    stages = {name: None for name in INTERNAL_STAGES}
    stages.update({name: summary(values) for name, values in timings.items()})
    idxs1, idxs2 = np.asarray(idxs1, dtype=np.int32), np.asarray(idxs2, dtype=np.int32)
    # Some detector-free trackers return shape (0,) for a valid empty result.
    coords1, coords2 = np.asarray(coords1).reshape(-1, 2), np.asarray(coords2).reshape(-1, 2)
    residuals = coords2[idxs2] - coords1[idxs1] - np.array([4, 2])
    inliers = int(np.count_nonzero(np.linalg.norm(residuals, axis=1) < 2.0))
    golden = None
    if args.golden_dir:
        args.golden_dir.mkdir(parents=True, exist_ok=True)
        golden_path = args.golden_dir / f"{preset_name}-{size_name}-{core}-{args.device}.npz"
        data = dict(kps1=coords1, kps2=coords2, idxs1=idxs1, idxs2=idxs2)
        for key, keypoints in (("kp_attributes1", kps1), ("kp_attributes2", kps2)):
            if keypoints is not None:
                # Preserve scale/orientation/response/octave/class_id as well as pixels.
                data[key] = np.asarray([[kp.size, kp.angle, kp.response, kp.octave, kp.class_id]
                                        for kp in keypoints], dtype=np.float64).reshape(-1, 5)
        for key, value in (("des1", des1), ("des2", des2)):
            if value is not None:
                data[key] = np.asarray(value)
        np.savez_compressed(golden_path, **data)
        golden = {"path": str(golden_path), "sha256": sha256(golden_path)}
    import resource
    # ru_maxrss is bytes on macOS and KiB on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status": "ok", "import_ms": import_ms, "pyslam_import_ms": pyslam_import_ms,
        "torch_imported_by_pyslam": torch_preloaded, "model_load_ms": load_ms,
        "stages_ms": stages, "stage_note": "Internal stages uninstrumented; extraction covers two images and coordinate conversion",
        "fps": 1000.0 / stages["total"]["median"], "keypoints": [len(coords1), len(coords2)],
        "matches": len(idxs1), "synthetic_inliers": inliers, "ate": None,
        # False when repeated calls on the same input gave different outputs (e.g. ORB2's
        # default ORBextractor); compare_benchmarks then uses a count tolerance.
        "deterministic": len(signatures) == 1,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        "torch_cuda": torch.version.cuda, "opencv": cv2.__version__, "golden": golden,
    }


def output_signature(np, coords1, coords2, idxs1, idxs2):
    import hashlib
    digest = hashlib.sha1()
    for value in (coords1, coords2, idxs1, idxs2):
        array = np.ascontiguousarray(np.asarray(value.cpu() if hasattr(value, "cpu") else value))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    args = parser().parse_args()
    if (args.iterations < 1 or args.warmup < 0 or args.threads < 1 or args.num_features < 1
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser().error("iterations, threads, num-features and timeout must be positive; warmup must be nonnegative")
    if args.worker:
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                result = run_case(args, args.presets[0], args.sizes[0], args.cores[0])
            except Exception as exc:
                result = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}
        result["log"] = log.getvalue()[-12000:]
        write_json(args.worker, result)
        return 0
    experiment = {
        "schema_version": 1, "kind": "synthetic_pair_microbenchmark" if not args.image else "image_pair_microbenchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(), "machine": machine_info(),
        "provenance": provenance(), "settings": {
            "iterations": args.iterations, "warmup": args.warmup, "threads": args.threads,
            "num_features": args.num_features, "seed": args.seed, "offline": True,
            "image_sha256": sha256(args.image) if args.image else None,
            "profile": bool(args.profile_dir),
        }, "cases": [], "sequence_validation": "not run; pair benchmarks do not certify SLAM accuracy",
    }
    for preset in args.presets:
        for size in args.sizes:
            for core in args.cores:
                case = {"preset": preset, "resolution": list(SIZES[size]), "size": size,
                        "core": core, "device": args.device}
                with tempfile.TemporaryDirectory(prefix="pyslam-visual-") as directory:
                    result_path = Path(directory) / "result.json"
                    child = [sys.executable, str(Path(__file__).resolve()), "--worker", str(result_path),
                             "--output", str(args.output), "--presets", preset, "--sizes", size,
                             "--cores", core, "--device", args.device, "--iterations", str(args.iterations),
                             "--warmup", str(args.warmup), "--threads", str(args.threads),
                             "--num-features", str(args.num_features), "--seed", str(args.seed)]
                    for option in ("image", "golden_dir", "profile_dir"):
                        if getattr(args, option):
                            child.extend(["--" + option.replace("_", "-"), str(getattr(args, option).resolve())])
                    env = dict(os.environ, PYSLAM_USE_CPP="true" if core == "cpp" else "false",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS=str(args.threads),
                               OPENBLAS_NUM_THREADS=str(args.threads), MPLCONFIGDIR=str(Path(directory) / "mpl"))
                    if args.device == "cpu":
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    try:
                        process = subprocess.run(child, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
                        if result_path.exists():
                            case.update(json.loads(result_path.read_text()))
                        else:
                            case.update(status="error", reason=f"Worker exited {process.returncode}", log=process.stdout[-12000:])
                    except subprocess.TimeoutExpired:
                        case.update(status="timeout", reason=f"Exceeded {args.timeout}s")
                experiment["cases"].append(case)
                write_json(args.output, experiment)
                print(f"{preset}/{size}/{core}/{args.device}: {case['status']}", flush=True)
    return int(args.strict and any(case["status"] != "ok" for case in experiment["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
