# Benchmarking

Tools for measuring pySLAM visual models and SLAM runs and for comparing the results. They need datasets or prefetched weights (sometimes a GPU), take minutes, and write JSON. They are not part of the pytest suite. Their own unit tests are in [`tests/local_features/`](../tests/local_features/) and [`tests/camera/`](../tests/camera/). Committed results are in [`benchmarks/visual_models/`](./benchmarks/visual_models/) and [`benchmarks/tum_vi/`](./benchmarks/tum_vi/).

Run from the repository root with the pySLAM environment active.

| Script | Purpose |
| --- | --- |
| `inventory_visual_models.py` | Inventory of models, vendor sources/patches, pins, history (phase 0 evidence) |
| `benchmark_visual_models.py` | Offline pair micro-benchmarks per tracker preset, resolution, core and device; optional golden NPZ outputs and cProfile |
| `benchmark_visual_slam.py` | Bounded headless `main_slam.py` sequence runs (timings, memory, ATE) |
| `compare_benchmarks.py` | Gate: compares two pair-benchmark JSONs (exit 0 pass, 1 regression, 2 incomplete) |
| `compare_visual_model_outputs.py` | Compares golden NPZ outputs (keypoints, descriptors, matches) |
| `compare_map_benchmarks.py` | Gate for the Map binding timings written by `cpp/core/slam/tests_py/test_slam_cpp_map.py` |
| `benchmark_common.py` | Shared helpers (machine info, provenance, hashing, JSON output) |
| `camera/replay_sequence.py` | Headless replay of a recorded dataset section (ORB2, synchronous mapping): tracking states, timings, map size, ATE |
| `camera/run_pinhole_regression.py` | Paired baseline/candidate replays from a case manifest (`camera/pinhole_replay_cases.example.json`); gate on timing, coverage and ATE |
| `camera/benchmark_pinhole.py` | Paired component benchmark of the pinhole camera and optimizer factors against a git revision (exit 1 on regression) |
| `camera/evaluate_tum_vi.py` | TUM-VI translation ATE of a pySLAM NPZ or a PLVS2/ORB-SLAM3 EuRoC export |
| `camera/build_plvs2_reference.py` | Builds local PLVS2 TUM-VI examples with loop closure disabled, as a visual-odometry reference |
| `camera/trajectory_evaluation.py`, `camera/benchmark_guard.py` | Shared ATE (Sim(3)/SE(3) alignment) and the process lock/memory watchdog of the replays |

Commands and gate tolerances: [`docs/visual_features.md`](../docs/visual_features.md) for visual models; [camera models](../cpp/modules/camera_models/README.md#end-to-end-replay-and-pinhole-regression) and [TUM-VI](../docs/tum-vi.md#tests-and-execution) for the camera replays.
