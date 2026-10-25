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
* Serial recorded-data comparisons against a separately built pinhole checkout.
*
* Runs baseline/candidate pairs in alternating order, in fresh processes. A case
* manifest supplies dataset, base_path, name, sensor_type, settings and optional
* overrides (the same KEY=VALUE strings accepted by replay_sequence.py).
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def compare(reference, candidate, max_ratio=1.20, ate_relative=0.20, ate_absolute=0.02):
    """Gate throughput, tracking coverage, and ATE independently."""
    for key in ("dataset", "name", "sensor", "start", "processed_frames", "alignment",
                "local_mapping", "loop_closure"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Cannot compare different replay configurations: {key}")
    for report in (reference, candidate):
        duration = report['track_and_map_ms']['mean']
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(report.get('ate_rmse_m', -1)):
            raise ValueError('Replay has invalid timing or ATE')
    ratio = candidate['track_and_map_ms']['mean'] / reference['track_and_map_ms']['mean']
    n = reference['processed_frames']
    failures = []
    if ratio > max_ratio:
        failures.append('throughput')
    for report in (reference, candidate):
        if report.get('gt_associations', 0) < 3 or report.get('ate_rmse_m', -1) < 0:
            raise ValueError('Replay has no valid ATE evaluation')
    if candidate['states'].get('OK', 0) < reference['states'].get('OK', 0) - 0.02 * n:
        failures.append('tracking_coverage')
    if candidate['trajectory_poses'] < reference['trajectory_poses'] - 0.02 * n:
        failures.append('trajectory_coverage')
    if candidate['gt_associations'] < reference['gt_associations'] - 0.02 * n:
        failures.append('mocap_coverage')
    ate_limit = reference['ate_rmse_m'] + max(ate_absolute, ate_relative * reference['ate_rmse_m'])
    if candidate['ate_rmse_m'] > ate_limit:
        failures.append('ate')
    return dict(mean_time_ratio=ratio, baseline_ate_m=reference['ate_rmse_m'],
                candidate_ate_m=candidate['ate_rmse_m'], ate_limit_m=ate_limit, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--cases', required=True, type=Path, help='JSON list of replay cases')
    parser.add_argument('--output', required=True, type=Path, help='Directory for reports and logs')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--local-mapping', choices=('synchronous', 'threaded'),
                        default='synchronous')
    parser.add_argument('--loop-closure', action='store_true')
    parser.add_argument('--max-ratio', type=float, default=1.20)
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1 or args.max_ratio <= 1:
        parser.error('frames/repeats must be positive; max-ratio must exceed 1')
    args.output.mkdir(parents=True, exist_ok=True)
    harness = Path(__file__).with_name('replay_sequence.py')
    env = dict(os.environ, PYSLAM_USE_CPP='true', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    # A baseline may be read-only. Keep compilation caches in the report directory.
    env['NUMBA_CACHE_DIR'] = str((args.output / '.numba_cache').resolve())
    results = []
    for case in json.loads(args.cases.read_text()):
        rows = []
        for repeat in range(args.repeats):
            reports = {}
            order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
            for version in order:
                root = getattr(args, version + '_root').resolve()
                output = args.output / f"{case['id']}_{repeat}_{version}.json"
                command = [sys.executable, str(harness), '--pyslam-root', str(root),
                           '--dataset', case['dataset'], '--base-path', case['base_path'],
                           '--name', case['name'], '--sensor-type', case['sensor_type'],
                           '--frames', str(args.frames), '--output', str(output),
                           '--local-mapping', args.local_mapping]
                if args.loop_closure:
                    command.append('--loop-closure')
                if case.get('settings'):
                    command += ['--settings', case['settings']]
                for override in case.get('overrides', []):
                    command += ['--set', override]
                print(f"{case['id']} repeat {repeat}: {version}", flush=True)
                with output.with_suffix('.log').open('w') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                reports[version] = json.loads(output.read_text())
            rows.append(compare(reports['baseline'], reports['candidate'], args.max_ratio))
        failures = sorted(set(f for row in rows for f in row['failures'] if f != 'throughput'))
        ratio = statistics.median(row['mean_time_ratio'] for row in rows)
        if ratio > args.max_ratio:
            failures.append('throughput')
        results.append(dict(case=case['id'], median_paired_time_ratio=ratio, pairs=rows, failures=failures))
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    return int(any(row['failures'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
