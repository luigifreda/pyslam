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
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from pyslam.io.dataset import (
    FolderDataset, TumDataset, KittiDataset, ScannetDataset, ReplicaDataset,
    TartanairDataset, SevenScenesDataset, NeuralRGBDDataset, ClioDataset, RoverDataset, EurocDataset,
)
from pyslam.io.dataset_types import DatasetType, SensorType
from pyslam.io.image_prefetcher import ImagePrefetcher, make_stream


def _write_color(path, value):
    image = np.full((6, 8, 3), value, np.uint8)
    image[0, 0] = (value, (value + 20) % 255, (value + 40) % 255)
    if not cv2.imwrite(path, image):
        raise RuntimeError(path)


class TestImagePrefetcher(TestCase):
    def test_concurrent_seeks_and_repeated_reads_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index in range(20):
                path = os.path.join(folder, f"{index}.png")
                _write_color(path, index)
                paths.append(path)
            reader = ImagePrefetcher([make_stream("image", paths)], capacity=1)
            self.addCleanup(reader.close)
            barrier = threading.Barrier(4)
            errors = []

            def read(offset):
                try:
                    barrier.wait()
                    for index in [offset, 19, 0, offset, 10, 1] * 3:
                        np.testing.assert_array_equal(reader.get("image", index), cv2.imread(paths[index]))
                except Exception as error:
                    errors.append(error)

            threads = [threading.Thread(target=read, args=(offset,)) for offset in (0, 3, 8, 15)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(2.0)
            stalled = any(thread.is_alive() for thread in threads)
            reader.close()
            for thread in threads:
                thread.join(1.0)
            self.assertFalse(stalled)
            self.assertEqual(errors, [])

    def test_stopped_reader_and_held_array_lifetime(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "image.png")
            _write_color(path, 17)
            reader = ImagePrefetcher([make_stream("image", [path])])
            held = reader.get("image", 0)
            impl = reader._impl
            reader.close()
            reader.close()
            self.assertIsNone(impl.get("image", 0))
            del impl, reader
            np.testing.assert_array_equal(held, cv2.imread(path))

    def test_scaled_depth_matches_numpy_rounding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "depth.png")
            depth = np.arange(65536, dtype=np.uint16).reshape(256, 256)
            self.assertTrue(cv2.imwrite(path, depth))
            reader = ImagePrefetcher([make_stream(
                "depth", [path], imread_flag=cv2.IMREAD_UNCHANGED,
                ops=[{"type": "scale_float", "factor": 1000.0}],
            )])
            self.addCleanup(reader.close)
            np.testing.assert_array_equal(reader.get("depth", 0), (depth / 1000.0).astype(np.float32))

    def test_corrupt_npy_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, shape in enumerate(((4, 5), (10**50, 2))):
                path = os.path.join(folder, f"bad{index}.npy")
                with open(path, "wb") as handle:
                    np.lib.format.write_array_header_1_0(handle, {
                        "descr": "<f4", "fortran_order": False, "shape": shape,
                    })
                    handle.write(b"short")
                paths.append(path)
            reader = ImagePrefetcher([make_stream("depth", paths, loader="npy")])
            self.addCleanup(reader.close)
            for index in range(len(paths)):
                self.assertIsNone(reader.get("depth", index))

    def test_sequential_reads_match_imread_and_keep_earlier_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index in range(10):
                path = os.path.join(folder, f"{index:02d}.png")
                _write_color(path, 10 * index + 3)
                paths.append(path)
            reader = ImagePrefetcher([make_stream("image", paths)], capacity=6)
            self.addCleanup(reader.close)

            deadline = time.time() + 5.0
            while reader.unpublished_count() < 6 and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(reader.unpublished_count(), 6)
            self.assertLessEqual(reader.unpublished_count(), reader._impl.capacity())

            first = reader.get("image", 0)
            self.assertIsNotNone(first)
            self.assertIsNotNone(first.base)
            self.assertTrue(first.flags["C_CONTIGUOUS"])
            expected_first = cv2.imread(paths[0])
            np.testing.assert_array_equal(first, expected_first)

            again = reader.get("image", 0)
            np.testing.assert_array_equal(again, expected_first)

            jumped = reader.get("image", 9)
            np.testing.assert_array_equal(jumped, cv2.imread(paths[9]))
            np.testing.assert_array_equal(first, expected_first)

            for index in range(1, 9):
                got = reader.get("image", index)
                np.testing.assert_array_equal(got, cv2.imread(paths[index]))
                self.assertLessEqual(reader.unpublished_count(), 6)
            np.testing.assert_array_equal(first, expected_first)

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "only.png")
            _write_color(path, 7)
            missing = os.path.join(folder, "missing.png")
            reader = ImagePrefetcher([make_stream("image", [path, missing])], capacity=2)
            self.addCleanup(reader.close)
            np.testing.assert_array_equal(reader.get("image", 0), cv2.imread(path))
            self.assertIsNone(reader.get("image", 1))
            self.assertIsNone(reader.get("image", 5))

    def test_gray_to_rgb_uint16_depth_npy_and_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            gray_path = os.path.join(folder, "gray.png")
            gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
            self.assertTrue(cv2.imwrite(gray_path, gray))
            depth_path = os.path.join(folder, "depth.png")
            depth = (np.arange(12, dtype=np.uint16).reshape(3, 4) + 1) * 10
            self.assertTrue(cv2.imwrite(depth_path, depth))
            scaled_path = os.path.join(folder, "scaled.png")
            self.assertTrue(cv2.imwrite(scaled_path, depth))
            npy_path = os.path.join(folder, "depth.npy")
            npy_depth = np.arange(12, dtype=np.float32).reshape(3, 4) + 0.5
            np.save(npy_path, npy_depth)

            color = cv2.imread(gray_path, cv2.IMREAD_COLOR)
            map_x, map_y = np.meshgrid(
                np.arange(color.shape[1], dtype=np.float32),
                np.arange(color.shape[0], dtype=np.float32),
            )
            remap_path = os.path.join(folder, "remap.png")
            self.assertTrue(cv2.imwrite(remap_path, color))

            reader = ImagePrefetcher(
                [
                    make_stream(
                        "image",
                        [gray_path],
                        imread_flag=cv2.IMREAD_GRAYSCALE,
                        ops=[{"type": "gray_to_rgb"}],
                    ),
                    make_stream("depth", [depth_path], imread_flag=cv2.IMREAD_UNCHANGED),
                    make_stream(
                        "scaled",
                        [scaled_path],
                        imread_flag=cv2.IMREAD_UNCHANGED,
                        ops=[{"type": "scale_float", "factor": 2.0}],
                    ),
                    make_stream("npy", [npy_path], loader="npy"),
                    make_stream(
                        "remap",
                        [remap_path],
                        ops=[{"type": "remap", "map1": map_x, "map2": map_y}],
                    ),
                ],
                capacity=2,
            )
            self.addCleanup(reader.close)

            gray_loaded = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(
                reader.get("image", 0), cv2.cvtColor(gray_loaded, cv2.COLOR_GRAY2RGB)
            )
            np.testing.assert_array_equal(
                reader.get("depth", 0), cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            )
            scaled = cv2.imread(scaled_path, cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(
                reader.get("scaled", 0), (scaled / 2.0).astype(np.float32)
            )
            np.testing.assert_array_equal(reader.get("npy", 0), np.load(npy_path))
            remapped = cv2.imread(remap_path)
            np.testing.assert_array_equal(
                reader.get("remap", 0), cv2.remap(remapped, map_x, map_y, cv2.INTER_LINEAR)
            )

    def test_unread_stream_does_not_stall_the_one_being_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            images = []
            depths = []
            for index in range(10):
                image_path = os.path.join(folder, f"img{index}.png")
                depth_path = os.path.join(folder, f"depth{index}.png")
                _write_color(image_path, index + 1)
                self.assertTrue(
                    cv2.imwrite(
                        depth_path, np.full((4, 4), index, np.uint16)
                    )
                )
                images.append(image_path)
                depths.append(depth_path)
            reader = ImagePrefetcher(
                [
                    make_stream("image", images),
                    make_stream("depth", depths, imread_flag=cv2.IMREAD_UNCHANGED),
                ],
                capacity=3,
            )
            self.addCleanup(reader.close)
            errors = []

            def drain():
                try:
                    for index, path in enumerate(images):
                        got = reader.get("image", index)
                        if got is None or not np.array_equal(got, cv2.imread(path)):
                            errors.append(index)
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=drain)
            thread.start()
            thread.join(5.0)
            stalled = thread.is_alive()
            reader.close()
            thread.join(1.0)
            self.assertFalse(stalled)
            self.assertEqual(errors, [])


class TestDatasetPrefetch(TestCase):
    def test_missing_module_falls_back_and_disable_closes_existing_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "image.png")
            _write_color(path, 9)
            with patch("pyslam.io.image_prefetcher._load_module", side_effect=ImportError("not built")):
                dataset = FolderDataset(folder, "*.png", fps=10)
            self.addCleanup(dataset.close)
            self.assertIsNone(dataset._prefetcher)
            np.testing.assert_array_equal(dataset.getImageColor(0), cv2.imread(path))
            streams = [make_stream("image", [path])]
            dataset.start_prefetch(streams)
            impl = dataset._prefetcher._impl
            with patch.dict(os.environ, {"PYSLAM_DISABLE_IMAGE_PREFETCH": "1"}):
                dataset.start_prefetch(streams)
            self.assertIsNone(dataset._prefetcher)
            self.assertIsNone(impl.get("image", 0))

    def test_folder_offsets_seeks_and_timestamps_match_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            for index in range(3):
                _write_color(os.path.join(folder, f"frame{index}.png"), index + 10)
            for disabled in ("0", "1"):
                with self.subTest(disabled=disabled), patch.dict(os.environ, {"PYSLAM_DISABLE_IMAGE_PREFETCH": disabled}):
                    dataset = FolderDataset(folder, "*.png", fps=10, start_frame_id=1)
                    self.addCleanup(dataset.close)
                    for frame in (0, 1, 0):
                        np.testing.assert_array_equal(
                            dataset.getImageColor(frame),
                            cv2.imread(os.path.join(folder, f"frame{frame + 1}.png")),
                        )
                        self.assertAlmostEqual(dataset.getTimestamp(), (frame + 2) * 0.1)
                    dataset.close()

    def test_kitti_constructor_and_color_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            sequence = Path(folder) / "sequences/00"
            sequence.mkdir(parents=True)
            (sequence / "times.txt").write_text("0.0\n0.1\n")
            for side in range(4):
                images = sequence / f"image_{side}"
                images.mkdir()
                for index in range(2):
                    _write_color(str(images / f"{index:06d}.png"), side * 10 + index)
            dataset = KittiDataset(folder, "00", start_frame_id=1)
            self.addCleanup(dataset.close)
            self.assertIsNotNone(dataset._prefetcher)
            for is_color in (False, True, False):
                dataset.set_is_color(is_color)
                for getter, side in ((dataset.getImageColor, 2 if is_color else 0),
                                     (dataset.getImageColorRight, 3 if is_color else 1)):
                    np.testing.assert_array_equal(getter(0), cv2.imread(str(sequence / f"image_{side}/000001.png")))
                self.assertAlmostEqual(dataset.getTimestamp(), 0.1)

    def test_euroc_rectified_streams_match_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            timestamps = [1000000000, 1050000000]
            (root / "euroc_timestamps").mkdir()
            (root / "euroc_timestamps/MH01.txt").write_text("\n".join(map(str, timestamps)))
            for side in (0, 1):
                target = root / f"MH01/mav0/cam{side}/data"
                target.mkdir(parents=True)
                for index, timestamp in enumerate(timestamps):
                    image = np.arange(48, dtype=np.uint8).reshape(6, 8) + side + index
                    self.assertTrue(cv2.imwrite(str(target / f"{timestamp}.png"), image))
            K = np.array([[5., 0, 4.], [0, 5., 3.], [0, 0, 1.]])
            calibration = {"K": K, "D": np.array([0.1, 0., 0., 0.]),
                           "R": np.eye(3), "P": np.column_stack((K, np.zeros(3)))}
            config = SimpleNamespace(cam_settings={"Camera.width": 8, "Camera.height": 6},
                                     cam_stereo_settings={"left": calibration, "right": calibration})
            with patch("pyslam.io.dataset.kSettingsFolder", folder):
                fast = EurocDataset(folder, "MH01", start_frame_id=1, config=config)
                with patch.dict(os.environ, {"PYSLAM_DISABLE_IMAGE_PREFETCH": "1"}):
                    legacy = EurocDataset(folder, "MH01", start_frame_id=1, config=config)
            self.addCleanup(fast.close)
            self.addCleanup(legacy.close)
            self.assertIsNotNone(fast._prefetcher)
            for getter in ("getImageColor", "getImageColorRight"):
                np.testing.assert_array_equal(getattr(fast, getter)(0), getattr(legacy, getter)(0))
                self.assertEqual(fast.getTimestamp(), legacy.getTimestamp())
                self.assertEqual(fast.getNextTimestamp(), legacy.getNextTimestamp())

    def test_rgbd_dataset_streams_match_legacy(self):
        # Check each distinct file-list convention, transformation, and timestamp
        # policy with start_frame_id applied exactly once.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            layouts = {
                "scans/scan/color": "{i}.jpg", "scans/scan/depth": "{i}.png",
                "replica/results": "frame{i:06d}.jpg",
                "seven/seq-01": "frame-{i:06d}.color.png",
                "neural/images": "img{i}.png", "neural/depth_filtered": "depth{i}.png",
                "clio/images": "rgb_{i}.jpg", "clio/depth": "depth_{i}.png",
                "rover/cam/rgb": "{i}.png", "rover/cam/depth": "{i}.png",
                "tartan/image_left": "{i:06d}_left.png",
                "tartan/image_right": "{i:06d}_right.png",
            }
            for directory, pattern in layouts.items():
                target = root / directory
                target.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    _write_color(str(target / pattern.format(i=i)), i * 10 + 3)
            depth = np.arange(48, dtype=np.uint16).reshape(6, 8) * 121
            for i in range(2):
                for path in (f"replica/results/depth{i:06d}.png", f"seven/seq-01/frame-{i:06d}.depth.png",
                             f"scans/scan/depth/{i}.png", f"neural/depth_filtered/depth{i}.png",
                             f"clio/depth/depth_{i}.png", f"rover/cam/depth/{i}.png"):
                    self.assertTrue(cv2.imwrite(str(root / path), depth + i))
                for side in ("left", "right"):
                    target = root / f"tartan/depth_{side}"
                    target.mkdir(exist_ok=True)
                    np.save(target / f"{i:06d}_{side}_depth.npy", depth.astype(np.float32) + i)
            (root / "tartan/pose_left.txt").write_text("pose0\npose1\n")
            (root / "rover/cam/associations.txt").write_text(
                "# timestamp rgb timestamp depth\n\n1.5 rgb/0.png 1.5 depth/0.png\n2.5 rgb/1.png 2.5 depth/1.png\n")
            config = SimpleNamespace(dataset_settings={"label_version": "nyu40"},
                                     cam_settings={"Camera.width": 4, "Camera.height": 3})
            cases = [
                (ScannetDataset, "scan", {"config": config}),
                (ReplicaDataset, "replica", {}), (SevenScenesDataset, "seven", {}),
                (NeuralRGBDDataset, "neural", {}), (ClioDataset, "clio", {}),
                (RoverDataset, "rover", {"camera_name": "cam", "associations": "associations.txt"}),
                (TartanairDataset, "tartan", {"sensor_type": SensorType.STEREO}),
            ]
            for cls, name, kwargs in cases:
                with self.subTest(dataset=cls.__name__):
                    options = dict(start_frame_id=1, **kwargs)
                    with patch.dict(os.environ, {"PYSLAM_DISABLE_IMAGE_PREFETCH": "0"}):
                        fast = cls(folder, name, **options)
                    with patch.dict(os.environ, {"PYSLAM_DISABLE_IMAGE_PREFETCH": "1"}):
                        legacy = cls(folder, name, **options)
                    self.addCleanup(fast.close)
                    self.addCleanup(legacy.close)
                    self.assertIsNotNone(fast._prefetcher)
                    getters = ["getImageColor", "getDepth"]
                    if cls is TartanairDataset:
                        getters += ["getImageColorRight", "getDepthRight"]
                    for getter in getters:
                        np.testing.assert_array_equal(getattr(fast, getter)(0), getattr(legacy, getter)(0))
                        self.assertEqual(fast.getTimestamp(), legacy.getTimestamp())
                        self.assertEqual(fast.getNextTimestamp(), legacy.getNextTimestamp())
                    fast.close()
                    legacy.close()

    def test_disable_switch_uses_the_legacy_loader(self):
        previous = os.environ.get("PYSLAM_DISABLE_IMAGE_PREFETCH")
        os.environ["PYSLAM_DISABLE_IMAGE_PREFETCH"] = "1"
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "000.png")
                _write_color(path, 9)
                dataset = FolderDataset(
                    folder,
                    "*.png",
                    sensor_type=SensorType.MONOCULAR,
                    fps=10,
                    type=DatasetType.FOLDER,
                )
                self.addCleanup(dataset.close)
                self.assertIsNone(dataset._prefetcher)
                np.testing.assert_array_equal(dataset.getImageColor(0), cv2.imread(path))
        finally:
            if previous is None:
                os.environ.pop("PYSLAM_DISABLE_IMAGE_PREFETCH", None)
            else:
                os.environ["PYSLAM_DISABLE_IMAGE_PREFETCH"] = previous

    def test_folder_and_tum_getters_match_the_files(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ["000.png", "001.png", "002.png"]
            for name in names:
                _write_color(os.path.join(folder, name), 15)
            dataset = FolderDataset(
                folder,
                "*.png",
                sensor_type=SensorType.MONOCULAR,
                fps=10,
                type=DatasetType.FOLDER,
            )
            self.addCleanup(dataset.close)
            self.assertIsNotNone(dataset._prefetcher)
            got = []
            for frame_id, name in enumerate(names):
                image = dataset.getImageColor(frame_id)
                np.testing.assert_array_equal(image, cv2.imread(os.path.join(folder, name)))
                got.append((dataset.getTimestamp(), dataset.getNextTimestamp()))
            self.assertEqual(got, [(0.0, 1.0), (1.0, 2.0), (2.0, 2.1)])

        with tempfile.TemporaryDirectory() as root:
            sequence = os.path.join(root, "seq")
            os.makedirs(sequence)
            rows = []
            for index in range(2):
                rgb = f"rgb/{index}.png"
                depth = f"depth/{index}.png"
                os.makedirs(os.path.join(sequence, "rgb"), exist_ok=True)
                os.makedirs(os.path.join(sequence, "depth"), exist_ok=True)
                _write_color(os.path.join(sequence, rgb), 30 + index)
                depth_image = np.full((4, 5), 1000 + index, np.uint16)
                self.assertTrue(cv2.imwrite(os.path.join(sequence, depth), depth_image))
                rows.append(f"{1.5 + index} {rgb} {1.5 + index} {depth}")
            with open(os.path.join(sequence, "associations.txt"), "w") as handle:
                handle.write("\n".join(rows) + "\n")
            dataset = TumDataset(
                root,
                "seq",
                sensor_type=SensorType.RGBD,
                associations="associations.txt",
                start_frame_id=1,
                type=DatasetType.TUM,
            )
            self.addCleanup(dataset.close)
            np.testing.assert_array_equal(
                dataset.getImageColor(0), cv2.imread(os.path.join(sequence, "rgb/1.png"))
            )
            np.testing.assert_array_equal(
                dataset.getDepth(0),
                cv2.imread(os.path.join(sequence, "depth/1.png"), cv2.IMREAD_UNCHANGED),
            )
            self.assertAlmostEqual(dataset.getTimestamp(), 2.5)
            self.assertIsNone(dataset.getImageColor(1))

            mono = TumDataset(
                root,
                "seq",
                sensor_type=SensorType.MONOCULAR,
                associations="associations.txt",
                type=DatasetType.TUM,
            )
            self.addCleanup(mono.close)
            self.assertIsNone(mono.getDepth(0))
            self.assertIsNotNone(mono.getImageColor(0))


if __name__ == "__main__":
    unittest.main()
