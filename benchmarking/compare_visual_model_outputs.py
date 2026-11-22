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
* Compare numeric wrapper outputs captured by benchmark_visual_models (never unpickle).
"""

import argparse
from pathlib import Path
import math


def compare_arrays(before, after, atol=0.0, rtol=0.0):
    import numpy as np
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    problems = []
    with np.load(before, allow_pickle=False) as left, np.load(after, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            return ["Different output arrays"]
        for key in left.files:
            a, b = left[key], right[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f"{key}: shape/dtype changed: {a.shape}/{a.dtype} -> {b.shape}/{b.dtype}")
            elif not np.isfinite(a).all() or not np.isfinite(b).all():
                problems.append(f"{key}: non-finite output")
            elif np.issubdtype(a.dtype, np.integer):
                if not np.array_equal(a, b):
                    problems.append(f"{key}: integer descriptors or indices changed")
            elif not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False):
                problems.append(f"{key}: values changed beyond atol={atol}, rtol={rtol}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    try:
        problems = compare_arrays(args.before, args.after, args.atol, args.rtol)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid output fixture: {exc}\n")
    print("\n".join(problems) if problems else "Outputs equivalent")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
