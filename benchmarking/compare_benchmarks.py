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
* Compare visual-model pair benchmarks, rejecting incomplete/incompatible evidence.
*
* Exit 0: measured gates passed; 1: regression; 2: incomparable/incomplete.
* This command never certifies unmeasured sequence ATE or GPU configurations.
"""

import argparse
import json
import math
from pathlib import Path


def case_key(case):
    return (case["preset"], tuple(case["resolution"]), case["core"], case["device"])


def compare(before, after, tolerance=0.03, startup_tolerance=0.10, memory_tolerance=0.02,
            count_tolerance=0.03):
    """tolerance: per-frame times/fps. startup_tolerance: one-shot import/model-load
    wall-clock (a single sample per case, so noisier). memory_tolerance: peak RSS/GPU.
    count_tolerance: relative keypoint/match/inlier change allowed for cases whose outputs
    vary between repeated calls (deterministic: false); deterministic cases stay exact."""
    for value in (tolerance, startup_tolerance, memory_tolerance, count_tolerance):
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Tolerances must be finite and in [0, 1)")
    problems, regressions = [], []
    for label, data in (("before", before), ("after", after)):
        if data.get("schema_version") != 1 or data.get("kind") not in {
                "synthetic_pair_microbenchmark", "image_pair_microbenchmark"}:
            problems.append(f"{label}: unsupported or missing benchmark schema/kind")
    for field in ("schema_version", "kind", "settings"):
        if before.get(field) != after.get(field):
            problems.append(f"Different {field}")
    for field in ("system", "release", "machine", "processor", "cpu_count", "python", "packages"):
        if before.get("machine", {}).get(field) != after.get("machine", {}).get(field):
            problems.append(f"Different machine.{field}")
    # nvidia-smi's instantaneous SM clock is an observation, not GPU identity.
    # Driver, UUID and power limit must match; the observed clock may vary at idle.
    if gpu_identity(before.get("machine", {}).get("gpu")) != gpu_identity(after.get("machine", {}).get("gpu")):
        problems.append("Different machine.gpu")
    def index(data):
        entries = data.get("cases", [])
        result = {case_key(case): case for case in entries}
        if len(result) != len(entries):
            problems.append("Duplicate benchmark cases")
        return result
    old, new = index(before), index(after)
    if not old or old.keys() != new.keys():
        problems.append("Empty or different case matrices")
    measured = 0
    for key in old.keys() & new.keys():
        left, right = old[key], new[key]
        if left.get("status") != "ok" or right.get("status") != "ok":
            problems.append(f"{key}: unmeasured case ({left.get('status')}, {right.get('status')})")
            continue
        measured += 1
        for field in ("opencv", "torch_cuda"):
            if left.get(field) != right.get(field):
                problems.append(f"{key}: different runtime {field}")
        metrics = [("import_ms", startup_tolerance, False), ("model_load_ms", startup_tolerance, False),
                   ("peak_rss_bytes", memory_tolerance, False), ("peak_gpu_bytes", memory_tolerance, False),
                   ("fps", tolerance, True)]
        for metric, margin, higher_is_better in metrics:
            a, b = left.get(metric), right.get(metric)
            if not valid_number(a) or not valid_number(b):
                problems.append(f"{key}: missing/invalid {metric}")
                continue
            failed = b < a * (1 - margin) if higher_is_better else b > a * (1 + margin)
            if failed:
                regressions.append(f"{key}: {metric} {a:.4g} -> {b:.4g}")
        for stage in ("extraction", "matching", "total"):
            for stat in ("median", "p95"):
                a = ((left.get("stages_ms") or {}).get(stage) or {}).get(stat)
                b = ((right.get("stages_ms") or {}).get(stage) or {}).get(stat)
                if not valid_number(a) or not valid_number(b):
                    problems.append(f"{key}: missing/invalid {stage}.{stat}")
                elif b > a * (1 + tolerance):
                    regressions.append(f"{key}: {stage}.{stat} {a:.4g} -> {b:.4g}")
        # Older results have no "deterministic" field: treat them as deterministic, unless
        # the other side observed non-determinism for the same case.
        deterministic = left.get("deterministic", True) and right.get("deterministic", True)
        for metric in ("keypoints", "matches", "synthetic_inliers"):
            if left.get(metric) is None or right.get(metric) is None:
                problems.append(f"{key}: missing {metric}")
            elif deterministic and left[metric] != right[metric]:
                regressions.append(f"{key}: {metric} changed")
            elif not deterministic and not counts_close(left[metric], right[metric], count_tolerance):
                regressions.append(f"{key}: {metric} changed beyond {count_tolerance:.0%} (non-deterministic case)")
    return {"status": "incomplete" if problems else "regression" if regressions else "passed",
            "measured_cases": measured, "incomplete": problems, "regressions": regressions,
            "scope": "pair measurements only; sequence ATE and unmeasured configurations are not certified"}


def counts_close(a, b, tolerance):
    a, b = (a if isinstance(a, list) else [a]), (b if isinstance(b, list) else [b])
    return len(a) == len(b) and all(abs(y - x) <= tolerance * max(abs(x), 1) for x, y in zip(a, b))


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def gpu_identity(value):
    if not isinstance(value, str):
        return value
    rows = [row.split(",") for row in value.splitlines()]
    return tuple(tuple(column.strip() for i, column in enumerate(row) if i != 3)
                 if len(row) == 5 else tuple(row) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03, help="Per-frame time/fps tolerance")
    parser.add_argument("--startup-tolerance", type=float, default=0.10,
                        help="Import/model-load tolerance (one wall-clock sample per case)")
    parser.add_argument("--memory-tolerance", type=float, default=0.02, help="Peak RSS/GPU memory tolerance")
    parser.add_argument("--count-tolerance", type=float, default=0.03,
                        help="Relative count tolerance for non-deterministic cases")
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()),
                         args.tolerance, args.startup_tolerance, args.memory_tolerance,
                         args.count_tolerance)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Invalid benchmark input: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return {"passed": 0, "regression": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
