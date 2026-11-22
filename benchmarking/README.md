# Benchmarking

Tools for measuring pySLAM visual models and SLAM runs and for comparing the results. They need datasets or prefetched weights (sometimes a GPU), take minutes, and write JSON. They are not part of the pytest suite. Their own unit tests are in [`tests/local_features/`](../tests/local_features/). Committed results are in [`docs/benchmarks/visual_models/`](../docs/benchmarks/visual_models/).

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

Commands and gate tolerances: [`docs/visual_features.md`](../docs/visual_features.md).
