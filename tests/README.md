# Tests

Automated checks for pySLAM. Usage scripts live in [`examples/`](../examples/).

Run everything from the repository root with the pySLAM environment active (`. pyenv-activate.sh`), the C++ modules built (`cpp/build.sh`), and the C++ core built (`cpp/core/slam/build.sh`). Rebuild after changing any C++ source, otherwise the tests run against the old binaries.

```bash
./tests/run_all.sh
```

## What runs

`run_all.sh` runs these steps in order and reports which ones failed:

1. `pytest tests` on the C++ sparse-SLAM core (the default).
2. `PYSLAM_USE_CPP=false pytest tests/python_core tests/slam` on the Python core.
3. `cpp/core/slam/tests_cpp/run_all_tests.sh`: C++ unit tests.
4. `cpp/core/slam/tests_py/run_all_tests.sh`: Python tests of the C++ core bindings (KD-tree, matchers, optimizers, map, and a full tracking/mapping lifecycle).
5. The script-style checks in `tests/modules/` (volumetric integration, Sim3 solver, PnP solver).
6. `cpp/modules/test_opencv_casters/run_tests.sh`, when that binding is built.

There are two pytest passes because `pyslam.slam` binds its classes to one core on first import, so a process can test only one core. `tests/conftest.py` keeps `tests/python_core` out of the C++ pass.

| Folder | Scope |
|--------|-------|
| `slam/` | Core-agnostic front end and back end: tracking, motion model, two-view initialization, adaptive thresholds, local BA interrupted by its abort flag. Runs in both passes. |
| `python_core/` | Python-core-only back end: local-mapping culling, essential-graph optimization (g2o, GTSAM), and local BA aborted just before g2o's first iteration (only the Python core can raise the flag at that exact point; the timed aborts in `slam/` cover both cores). |
| `camera/` | Camera models, local-map limits, triangulation, trajectory evaluation (C++ core). |
| `cv/` | Pose estimators and depth from point clouds. |
| `loopclosing/` | Loop detector database, VPR extractors, loop group consistency. |
| `semantics/` | Semantic fusion methods. |
| `dataset/`, `dust3r/` | Dataset readers, image prefetching, DUSt3R processing. |
| `modules/` | Script-style module checks, run directly by `run_all.sh`. |

The C++ sources stay next to their CMake projects (`cpp/core/slam/tests_cpp`, `cpp/core/slam/tests_py`, `cpp/modules/test_opencv_casters`). This folder is the Python suite and the single entry point.

## A single test

```bash
python -m pytest tests/slam/test_tracking_pipeline.py -v
PYSLAM_USE_CPP=false python -m pytest tests/slam/test_tracking_pipeline.py -v
python cpp/core/slam/tests_py/test_slam_cpp_search_keyframe_by_projection.py
```

## Adding a test

Put a `unittest.TestCase` or a pytest function in `tests/<topic>/test_*.py`. Keep helpers out of that filename pattern so pytest does not collect them (for example `tests/slam/slam_test_support.py`, the replay and benchmark harnesses in `tests/camera/`).

- A test that should hold on both cores imports from `pyslam.slam` and goes in `tests/slam/`.
- A test of Python-only core code goes in `tests/python_core/`, and so does a check that only the Python core can reach. Keep the C++ pass free of skips.
- A test of C++ internals goes in `cpp/core/slam/tests_cpp` or `tests_py`, and in that directory's `run_all_tests.sh`.

See the [C++ core README](../cpp/core/slam/README.md) for details on the core itself.
