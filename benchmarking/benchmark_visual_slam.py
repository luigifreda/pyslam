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
* Run bounded headless main_slam cases using dataset paths from config.yaml.
*
* This complements pair benchmarks. Missing stage/accuracy measurements remain null;
* short smoke runs do not certify the full visual-model roadmap performance gate.
* Configuration and camera-settings overrides are temporary; the originals are untouched.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from benchmark_common import ROOT, machine_info, provenance, write_json


def override_setting(contents, name, value):
    line = f"{name}: {value}"
    expression = re.compile(r"^" + re.escape(name) + r"\s*:.*$", re.MULTILINE)
    return expression.sub(line, contents) if expression.search(contents) else contents.rstrip() + "\n" + line + "\n"


def prepare_config(config, dataset_name, preset, loop_preset, temporary, artifacts):
    import copy
    import yaml
    config = copy.deepcopy(config)
    config["DATASET"]["type"] = dataset_name
    dataset = config[dataset_name]
    settings = dataset.get("settings_stereo") if dataset.get("sensor_type") == "stereo" else None
    path = Path(settings or dataset["settings"])
    path = path if path.is_absolute() else ROOT / path
    contents = override_setting(path.read_text(), "FeatureTrackerConfig.name", preset)
    contents = override_setting(contents, "LoopDetectionConfig.name", loop_preset)
    settings_path = temporary / "settings.yaml"
    settings_path.write_text(contents)
    dataset["settings"] = str(settings_path)
    dataset.pop("settings_stereo", None)
    config["GLOBAL_PARAMETERS"] = dict(config.get("GLOBAL_PARAMETERS") or {})
    config["GLOBAL_PARAMETERS"].update(
        kDoVolumetricIntegration=False, kDoSparseSemanticMappingAndSegmentation=False,
        kUseDepthEstimatorInFrontEnd=False, kUseLoopClosing=True,
        kLogsFolder=str(artifacts / "logs"))
    config.setdefault("SAVE_TRAJECTORY", {}).update(
        save_trajectory=True, output_folder=str(artifacts), basename="trajectory")
    config.setdefault("SYSTEM_STATE", {})["load_state"] = False
    path = temporary / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def collect_case_metrics(case, artifacts, requested_frames):
    """A process that stayed in INIT must not pass a SLAM/accuracy smoke gate."""
    case.pop("reason", None)
    metrics = sorted((artifacts / "plot").glob("stats_*.json"))
    case["ate"] = json.loads(metrics[-1].read_text()) if metrics else None
    loss_file = artifacts / "other_metrics_info.txt"
    if loss_file.exists():
        case["tracking_counts"] = loss_file.read_text()
    trajectory = artifacts / "trajectory_final.txt"
    case["accepted_trajectory_rows"] = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in trajectory.read_text().splitlines()) if trajectory.exists() else 0
    frames = case.get("playback", {}).get("all_frames", {}).get("frames", 0)
    case["status"] = "ok"
    if frames != requested_frames:
        case.update(status="incomplete", reason="Requested frame count was not reached")
    elif case["accepted_trajectory_rows"] == 0:
        case.update(status="incomplete", reason="No accepted trajectory; initialization/startup only")


def profile_hook(profile_dir, case_name):
    """sitecustomize snippet: cProfile every process of the case and dump on exit, including
    main_slam's os._exit() shutdown (cProfile's own dump would be skipped). One file per pid:
    spawned children (loop detection, ...) load the same sitecustomize."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(profile_dir.resolve() / case_name)
    return f'''
import atexit as _atexit, cProfile as _cProfile, os as _os
_profiler = _cProfile.Profile()
_profiler.enable()
def _dump_profile():
    try:
        _profiler.disable()
        _profiler.dump_stats({prefix!r} + "." + str(_os.getpid()) + ".prof")
    except Exception:
        pass
_original_exit = _os._exit
def _exit_with_profile(code):
    _dump_profile()
    _original_exit(code)
_os._exit = _exit_with_profile
_atexit.register(_dump_profile)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["KITTI_DATASET", "TUM_DATASET", "EUROC_DATASET"])
    parser.add_argument("--presets", nargs="+", default=["ORB2"])
    parser.add_argument("--loop-presets", nargs="+", default=["DBOW3", "SAD"])
    parser.add_argument("--cores", nargs="+", choices=("cpp", "python"), default=["cpp", "python"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path,
                        help="Run main_slam.py under cProfile (main thread: tracking) and save one .prof per case; "
                             "changes timings, compare profiled runs only with profiled runs")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-rss-gib", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.threads < 1 or any(not math.isfinite(value) or value <= 0
                                                  for value in (args.timeout, args.max_rss_gib)):
        parser.error("frames, threads, timeout and memory budget must be positive and finite")
    import psutil
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if any(name not in config for name in args.datasets):
        parser.error("Every requested dataset must exist in the input config")
    output = args.output.resolve()
    artifact_root = output.parent / (output.stem + "-artifacts")
    report = dict(schema_version=1, kind="slam_sequence_smoke", machine=machine_info(),
                  created_utc=datetime.now(timezone.utc).isoformat(), provenance=provenance(), cases=[],
                  limitations=["main_slam only; main_vo lacks the required headless/config CLI",
                               "Local-mapping and loop latency instrumentation is not yet integrated",
                               "Short bounded runs do not certify full sequence accuracy"])
    for dataset in args.datasets:
        for preset in args.presets:
            for loop in args.loop_presets:
                for core in args.cores:
                    name = f"{dataset}-{preset}-{loop}-{core}-{args.device}"
                    # Names become artifact path components, never executable shell text.
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                        parser.error("Dataset/preset names must contain only letters, numbers, underscores or hyphens")
                    artifact_root.mkdir(parents=True, exist_ok=True)
                    # Fresh directories prevent a retry from reading stale successful timings.
                    artifacts = Path(tempfile.mkdtemp(prefix=name + "-", dir=artifact_root))
                    timing = artifacts / "playback.json"
                    case = dict(dataset=dataset, preset=preset, loop_preset=loop, core=core,
                                device=args.device, requested_frames=args.frames,
                                startup_to_first_frame_ms=None, local_mapping_keyframe_ms=None,
                                loop_detection_latency_ms=None, ate=None, peak_rss_bytes=0)
                    with tempfile.TemporaryDirectory(prefix="pyslam-sequence-") as folder:
                        temporary = Path(folder)
                        child_config = prepare_config(config, dataset, preset, loop, temporary, artifacts)
                        # Disable remote downloads, retaining local multiprocessing sockets.
                        (temporary / "sitecustomize.py").write_text('''import sys
def _audit(event, args):
    if event == "socket.getaddrinfo" and args[0] not in ("localhost", "127.0.0.1", "::1", None):
        raise RuntimeError("Sequence benchmark is offline; prefetch weights first")
    if event == "socket.connect" and isinstance(args[1], tuple) and args[1][0] not in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError("Sequence benchmark is offline; prefetch weights first")
sys.addaudithook(_audit)
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = "Agg"
if "plot_usetex" in SETTINGS:
    SETTINGS.plot_usetex = False
''' + (profile_hook(args.profile_dir, artifacts.name) if args.profile_dir else ""))
                        env = dict(os.environ, PYSLAM_USE_CPP="true" if core == "cpp" else "false",
                                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                   OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                                   PYTHONPATH=str(temporary) + os.pathsep + str(ROOT),
                                   MPLBACKEND="Agg",
                                   MPLCONFIGDIR=str(temporary / "mpl"))
                        if args.device == "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = ""
                        started = time.perf_counter()
                        with (artifacts / "run.log").open("w") as log:
                            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "main_slam.py"),
                                "-c", str(child_config), "--headless", "--no_output_date", "--max-frames",
                                str(args.frames), "--timing-report", str(timing)], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            reason = None
                            try:
                                while process.poll() is None:
                                    try:
                                        parent = psutil.Process(process.pid)
                                        members = [parent] + parent.children(recursive=True)
                                        rss = sum(member.memory_info().rss for member in members if member.is_running())
                                        case["peak_rss_bytes"] = max(case["peak_rss_bytes"], rss)
                                    except psutil.NoSuchProcess:
                                        pass
                                    if case["peak_rss_bytes"] > args.max_rss_gib * 1024**3:
                                        reason = "Memory budget exceeded"
                                        break
                                    if time.perf_counter() - started > args.timeout:
                                        reason = "Timeout"
                                        break
                                    time.sleep(0.1)
                            finally:
                                if process.poll() is None:
                                    os.killpg(process.pid, signal.SIGTERM)
                                    try:
                                        process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(process.pid, signal.SIGKILL)
                                        process.wait()
                            case.update(returncode=process.returncode,
                                        elapsed_seconds=time.perf_counter() - started)
                            if reason:
                                case.update(status="incomplete", reason=reason)
                            elif process.returncode != 0 or not timing.exists():
                                case.update(status="unavailable", reason="Process failed or produced no timing report; see run.log")
                            else:
                                case["playback"] = json.loads(timing.read_text())
                                collect_case_metrics(case, artifacts, args.frames)
                    case["artifacts"] = str(artifacts)
                    report["cases"].append(case)
                    write_json(output, report)
                    print(f"{name}: {case['status']}", flush=True)
    return int(args.strict and any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
