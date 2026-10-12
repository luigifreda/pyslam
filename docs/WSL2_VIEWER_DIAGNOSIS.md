# Viewer3D startup on WSL2 / Intel D3D12

Diagnosed on Ubuntu 26.04 under WSL2 with Python 3.11.9, on 2026-10-12.

The apparent hang at `Waiting for Viewer3D to be ready` was a dead child
process. Startup tracing stopped at `pangolin.CreateWindowAndBind`; the
child exited with SIGSEGV (multiprocessing exit code `-11`). Moving viewer
creation before `Slam`, disabling its drawing thread, and using `spawn`
alone did not fix it.

A native backtrace from a minimal reproduction (import the original
`main_slam`, then create a Pangolin window) showed:

```text
llvm::DenseMapBase<...>::LookupBucketFor  [libtensorflow_framework.so.2]
llvm::...::lookupOrInsertIntoBucket      [libtensorflow_framework.so.2]
CommandLineParser::addOption            [libtensorflow_framework.so.2]
llvm::cl::Option::addArgument           [libtensorflow_framework.so.2]
                                       [Intel WSL driver/libLLVM-9.so]
dlopen                                 [libc.so.6]
                                       [libigd12um64xel.so]
OpenAdapter12                          [libigd12umd64.so]
                                       [libd3d12core.so]
```

Intel's LLVM library resolved symbols against TensorFlow's incompatible
LLVM implementation. The isolated Pangolin/Viewer3D tests passed because
they did not load the application's optional TensorFlow dependencies.

The correction uses a dedicated `spawn` context for Viewer3D and all its
queues/shared flags. Application imports live inside `main_slam.py`'s
`__main__` guard, so the spawned viewer does not reimport optional native
libraries. The viewer's process and drawing thread are excluded from its
pickle state. The global multiprocessing method for other SLAM workers
is unchanged. Readiness now detects child exit and defaults to a 120-second
timeout; failed startup terminates and joins the child.

Set `PYSLAM_VIEWER_TRACE=1` to print startup lines and the OpenGL renderer,
and enable fault/timeout stack dumps in the viewer child. Tracing stops
before the rendering loop.

Validation performed:

- Full GUI run: frames 0–1100, 1101 processed, 0 tracking-lost frames.
  Closed Map Viewer with `WM_DELETE_WINDOW` after dataset end.
  Metrics: `results/metrics_20261012_155046/other_metrics_info.txt`.
- Full headless run: 1101 processed, exit code 0.
  Metrics: `results/metrics_20261012_154918/other_metrics_info.txt`.
- Viewer with TensorFlow imported in its parent and drawing thread enabled:
  ready, rendering process alive, clean viewer shutdown. Renderer:
  `D3D12 (Intel(R) UHD Graphics)`.
- Four regression tests passed, without requiring a display:
  import isolation, spawn/data transfer with a drawing thread, child failure
  reporting, and timeout termination.

Run the regression checks from the repository root in the pySLAM environment:

```bash
PYTHONPATH=. python -m unittest discover -s test/viz -p test_viewer_startup.py
```
