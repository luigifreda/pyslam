# OpenCV Caster Tests

This folder contains:

- `test_casters.py`: functional pybind11 round-trip checks for the active caster
- `bench_opencv_casters.cpp`: a native microbenchmark for `numpy.ndarray -> cv::Mat` conversion

## Build

```bash
./build.sh
```

By default the build uses the currently active Python environment (`python` from `PATH`,
falling back to `python3`). You can override that explicitly with:

```bash
PYTHON_BIN=/path/to/python ./build.sh
```

## Run tests

```bash
./run_tests.sh
```

`run_tests.sh` uses the same interpreter selection logic as `build.sh`, so the build,
pytest run, and embedded benchmark all use one consistent Python environment.
If you only want the build and pytest checks, you can skip the benchmark phase with:

```bash
RUN_BENCHMARKS=0 ./run_tests.sh
```

The benchmark phase is intentionally lightweight by default. You can make it heavier or lighter with:

```bash
BENCH_ITER_SCALE_PCT=100 BENCH_SAMPLES=9 ./run_tests.sh
```

Defaults:

- `BENCH_ITER_SCALE_PCT=20`
- `BENCH_SAMPLES=5`

## Run benchmarks

Compare the active header with the archived `v1` implementation in `cpp/casters/old_versions/`:

```bash
./build/bench_opencv_casters_current
./build/bench_opencv_casters_v1
```

The benchmark reports median timings across repeated samples, per-op cost,
an adjusted per-op estimate with loop overhead removed, and an alias rate that
shows how often the `cv::Mat` reuses the original NumPy buffer.

Covered cases include:

- contiguous grayscale arrays
- contiguous color arrays
- a positive-stride sliced grayscale view
- a transposed grayscale view
- a negative-stride color view
- an `int64` grayscale array that exercises dtype normalization
