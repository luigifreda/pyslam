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
* Shared, dependency-free metadata for visual-model inventory and benchmarks.
*
* This module intentionally does not import a wrapper, torch, or the SLAM runtime.
"""

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENUM_SOURCES = {
    "FeatureDetectorTypes": "pyslam/local_features/feature_types.py",
    "FeatureDescriptorTypes": "pyslam/local_features/feature_types.py",
    "FeatureMatcherTypes": "pyslam/local_features/feature_matcher.py",
    "FeatureTrackerTypes": "pyslam/local_features/feature_tracker.py",
    "GlobalDescriptorType": "pyslam/loop_closing/loop_detector_configs.py",
    "LocalDescriptorAggregationType": "pyslam/loop_closing/loop_detector_configs.py",
}


def command(args, cwd=ROOT, timeout=20):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enum_snapshot():
    """Read literal enum assignments without executing optional dependency imports."""
    result = {}
    for name, path in ENUM_SOURCES.items():
        tree = ast.parse((ROOT / path).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        result[name] = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in cls.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
    return result


def machine_info():
    packages = {}
    for name in ("numpy", "torch", "torchvision", "kornia", "opencv-python", "opencv-contrib-python",
                 "tensorflow", "huggingface-hub", "timm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "processor": platform.processor(),
        "python": platform.python_version(), "executable": sys.executable,
        "cpu_count": os.cpu_count(), "packages": packages,
        "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,power.limit",
                        "--format=csv,noheader"]),
        "gpu_clock_policy": "observed only; clocks and power settings are not changed",
    }


def provenance():
    paths = sorted(set(ENUM_SOURCES.values()) | {
        "pyslam/local_features/feature_manager.py", "pyslam/local_features/feature_tracker_configs.py",
        "pyslam/config_parameters.py", "benchmarking/benchmark_visual_models.py",
        "benchmarking/benchmark_visual_slam.py",
        "pyslam/slam/initializer.py", "cpp/core/slam/initializer.cpp",
        "pyslam/slam/tracking.py", "cpp/core/slam/slam.cpp",
    })
    return {
        "commit": command(["git", "rev-parse", "HEAD"]),
        "worktree_status": command(["git", "status", "--short", "--untracked-files=no"]),
        "source_sha256": {path: sha256(ROOT / path) for path in paths if (ROOT / path).is_file()},
    }


def write_json(path, data):
    """Use strict JSON; write atomically so interrupted runs leave the last complete result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
