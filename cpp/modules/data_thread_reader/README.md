# Dataset image prefetching

File-based datasets automatically decode images on a C++ background thread when the `data_thread_reader` extension is available. This applies to KITTI, EuRoC, TUM-VI, TUM (including ICL-NUIM and Bonn), ScanNet, Replica, TartanAir, 7-Scenes, Neural RGB-D, CLIO, Rover, and image folders. Video, live cameras, ROS bags, MCAP, and semantic label loading keep their existing readers.

Build the extension in the active pySLAM Python environment with `./cpp/build.sh`. For an already configured build, only this target needs rebuilding:

```bash
cmake --build cpp/build --target data_thread_reader -j 2
```

If importing the extension fails, the dataset prints a warning and uses its Python loaders. To explicitly select those loaders, set the environment variable before constructing the dataset:

```bash
PYSLAM_DISABLE_IMAGE_PREFETCH=1 python main_slam.py
```

`getImageColor`, `getImageColorRight`, `getDepth`, and TartanAir's `getDepthRight` use the queues. Direct `getImage` and `getImageRight` calls use the Python loaders. Folder reads on both paths honor the requested frame index and `start_frame_id`; timestamps are stable across repeated reads and seeks.

One worker handles all streams in a dataset, including CLAHE, rectification, resizing, grayscale-to-RGB conversion, and depth conversion as needed. Each stream holds up to six unpublished frames by default; a full, unread stream does not stall another stream. Requests block with the Python GIL released until their stream's frame is ready. Forward jumps and repeated or backward reads reposition that stream's worker queue. Concurrent `get` requests are serialized so their seeks cannot invalidate another waiting request.

The NumPy result shares the decoded OpenCV buffer without copying pixels. Returned frames remain valid after later reads and after closing the dataset, and mutations do not affect later reads of the same file. The queue limit bounds unpublished frames per stream, with one additional decode possibly in flight; arrays retained by the caller consume additional memory. `unpublished_count()` reports the largest individual stream queue, rather than the sum of all queues.

TartanAir depth uses a small `.npy` reader for C-contiguous arrays with supported little-endian numeric types (including its usual float32 depth). Unsupported or corrupt files and failed image reads/conversions return `None`. Dataset getters set `is_ok` to false on a failed prefetched read.

Call `dataset.close()` when finished to stop and join the worker promptly. Destruction also closes it. Prefetching overlaps loading with processing; the first frame, a seek, or a consumer faster than the decoder can still wait for I/O.

Run the synthetic reader and dataset regression checks from the repository root:

```bash
PYTHONPATH=. python -m unittest discover -s tests/dataset -p 'test_image_prefetcher.py' -v
PYTHONPATH=. python -m unittest discover -s tests/dataset -p 'test_folder_dataset_timestamps.py' -v
```
