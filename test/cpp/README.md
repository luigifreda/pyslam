# Testing C++ SLAM core 

All tests assume the pySLAM environment is active (`. pyenv-activate.sh`) and the C++ core is built (`pyslam/slam/cpp/build.sh`). Rebuild the core after changing any C++ source, otherwise the tests run against the old binary.

## Quick check

```bash
./pyslam/slam/cpp/tests_cpp/run_all_tests.sh   # C++ unit tests
./pyslam/slam/cpp/tests_py/run_all_tests.sh    # Python tests of the C++ core (prints a pass/fail summary)
```

## Running a single test

Each test file is a standalone `unittest` script:

```bash
python pyslam/slam/cpp/tests_py/test_slam_cpp_search_keyframe_by_projection.py
```

## Regression tests

These cover past crashes and matching bugs. Run them after touching the matchers, relocalization, or loop closing.

| Test | Covers |
|------|--------|
| `pyslam/slam/cpp/tests_py/test_slam_cpp_search_and_fuse.py` | `search_and_fuse` on monocular keyframes (empty `kps_ur`; used to segfault in local mapping) |
| `pyslam/slam/cpp/tests_py/test_slam_cpp_search_by_sim3.py` | `search_by_sim3` expands beyond the seed matches (Python and C++) |
| `pyslam/slam/cpp/tests_py/test_slam_cpp_search_keyframe_by_projection.py` | skip indices are compact `get_matched_points()` positions; out-of-range ones are ignored (used to corrupt the heap during relocalization) |
| `test/loopclosing/test_loop_detector_dbow3_entry_id.py` | DBoW3 database entry ids map to the right frame ids |
| `test/dataset/test_folder_dataset_timestamps.py` | `FolderDataset` timestamps parsed from filenames |

The first three run in `tests_py/run_all_tests.sh`; the last two run on their own with `python <file>`.

See this [README](../../pyslam/slam/cpp/README.md) file for further details on the C++ core.
