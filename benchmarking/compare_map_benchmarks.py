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
* Compare each Map backend to itself, rather than requiring C++ to beat Python.
*
* Missing/incompatible measurements fail closed. Default regression tolerance is 3%.
"""

import argparse
import json
import math
from pathlib import Path


def compare(before, after, tolerance=0.03):
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("Tolerance must be finite and in [0, 1)")
    incomplete, regressions = [], []
    for label, report in (("before", before), ("after", after)):
        if report.get("schema_version") != 1 or report.get("kind") != "map_operations":
            incomplete.append(f"{label}: unsupported benchmark schema")
    if before.get("workload") != after.get("workload") or not before.get("workload"):
        incomplete.append("Missing or different Map workload")
    for key in ("system", "release", "machine", "processor", "cpu_count", "python", "packages"):
        if any(key not in report.get("machine", {}) for report in (before, after)):
            incomplete.append(f"Missing machine.{key}")
        elif before["machine"][key] != after["machine"][key]:
            incomplete.append(f"Different machine.{key}")
    def index(report):
        entries = report.get("cases", [])
        result = {entry["name"]: entry for entry in entries}
        if len(result) != len(entries):
            incomplete.append("Duplicate Map cases")
        return result
    old, new = index(before), index(after)
    if not old or old.keys() != new.keys():
        incomplete.append("Missing or different Map cases")
    for name in old.keys() & new.keys():
        for backend in ("cpp_time", "python_time"):
            a, b = old[name].get(backend), new[name].get(backend)
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and value > 0 for value in (a, b)):
                incomplete.append(f"{name}.{backend}: invalid measurement")
            elif b > a * (1 + tolerance):
                regressions.append(f"{name}.{backend}: {a:.6g}s -> {b:.6g}s")
    return dict(status="incomplete" if incomplete else "regression" if regressions else "passed",
                incomplete=incomplete, regressions=regressions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.tolerance)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid map benchmark input: {error}\n")
    print(json.dumps(result, indent=2))
    return {"passed": 0, "regression": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
