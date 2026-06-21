"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* Helpers for CLIO / Hydra RGB-D scenes:
* - read COLMAP sparse models (images.bin / images.txt)
* - convert COLMAP world-to-camera poses to PYSLAM camera-to-world (T_wc)
* - estimate capture rate from the scene ROS1 bag (if present)
* - align reference poses to ClioDataset playback indices (timestamp / ATE sync)
* - resolve playback fps consistently for dataset and ground truth
*
* COLMAP pose convention: qvec = (qw, qx, qy, qz), tvec such that x_cam = R * x_world + t.
* PYSLAM stores Twc (camera pose in world frame), so we invert the w2c transform.
"""

from __future__ import annotations

import glob
import os
import re
import struct
from collections import namedtuple

import numpy as np

from pyslam.utilities.geometry import qvec2rotmat
from pyslam.utilities.logging import Printer


# Fallback playback rate when no *.bag is available in the scene folder.
# Measured on the CLIO office scene: 1460 frames / 194.6 s ~= 7.5 Hz (dt ~= 133 ms).
CLIO_DEFAULT_FPS = 7.5
# Default color topic in CLIO/Hydra ROS1 bags (e.g. office.bag).
CLIO_COLOR_TOPIC = "/dominic/forward/color/image_raw"
# Metric odometry published by the Hydra/CLIO pipeline (depth-scaled COLMAP odometry).
CLIO_ODOM_TOPIC = "/dominic/forward/colmap_odom"


ColmapImage = namedtuple(
    "ColmapImage", ["id", "qvec", "tvec", "camera_id", "name"]
)


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Unpack the next chunk from a COLMAP binary file (little-endian by default)."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file: str) -> dict[int, ColmapImage]:
    """Read registered images from COLMAP images.bin (binary reconstruction format)."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]

            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D)

            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
            )
    return images


def read_images_text(path_to_model_file: str) -> dict[int, ColmapImage]:
    """Read registered images from COLMAP images.txt (text reconstruction format)."""
    images = {}
    with open(path_to_model_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])), dtype=np.float64)
            tvec = np.array(list(map(float, elems[5:8])), dtype=np.float64)
            camera_id = int(elems[8])
            image_name = elems[9]
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
            )
            # Skip the 2D observations line.
            fid.readline()
    return images


def colmap_qvec_tvec_to_Twc(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP (qw,qx,qy,qz) + tvec (w2c) to a 4x4 camera-to-world pose (T_wc)."""
    # COLMAP qvec order is (qw, qx, qy, qz); pyslam qvec2rotmat expects (qx, qy, qz, qw).
    qw, qx, qy, qz = qvec
    R_w2c = qvec2rotmat(np.array([qx, qy, qz, qw], dtype=np.float64))
    T_w2c = np.eye(4, dtype=np.float64)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = tvec
    return np.linalg.inv(T_w2c)


def clio_image_name_to_frame_id(name: str) -> int | None:
    """Map a CLIO RGB filename (rgb_123.jpg) to its integer frame id, or None if unmatched."""
    match = re.search(r"rgb_(\d+)\.jpg$", name)
    if match:
        return int(match.group(1))
    return None


def list_clio_dataset_frame_ids(base_path: str) -> list[int]:
    """
    Return sorted rgb frame ids from images/rgb_*.jpg.

    Must match ClioDataset.image_paths / frame_ids exactly so reference poses and
    playback timestamps use the same frame ordering.
    """
    images_path = os.path.join(base_path, "images")
    image_pattern = os.path.join(images_path, "rgb_*.jpg")
    image_paths = sorted(
        glob.glob(image_pattern),
        key=lambda path: clio_image_name_to_frame_id(os.path.basename(path)) or -1,
    )
    frame_ids = []
    for path in image_paths:
        frame_id = clio_image_name_to_frame_id(os.path.basename(path))
        if frame_id is not None:
            frame_ids.append(frame_id)
    return frame_ids


def align_clio_poses_to_dataset(
    poses: list[np.ndarray],
    reference_frame_ids: list[int],
    dataset_frame_ids: list[int],
    *,
    reference_is_index_ordered: bool = False,
) -> tuple[list[np.ndarray | None], int]:
    """
    Map reference poses onto dataset playback indices.

    ClioDataset assigns timestamps as (dataset_index * Ts), not (rgb_id * Ts).
    Reference trajectories must therefore be indexed by dataset playback order:

      - COLMAP sparse: poses are keyed by rgb filename id (rgb_10 -> id 10). A sparse
        model may omit frames that exist on disk; lookup is by id, not by pose list index.
      - Bag colmap_odom: one odometry message per frame, in the same order as the sorted
        image list; poses[i] corresponds to dataset_frame_ids[i].

    Returns (pose per dataset index or None if missing, number of missing poses).
    """
    aligned: list[np.ndarray | None] = [None] * len(dataset_frame_ids)
    missing = 0

    if reference_is_index_ordered:
        # Bag odometry: sequential messages align with sorted images by index.
        if len(poses) != len(dataset_frame_ids):
            Printer.yellow(
                "[colmap_io] CLIO bag pose count "
                f"({len(poses)}) differs from dataset frame count ({len(dataset_frame_ids)}); "
                "aligning by index up to the shorter sequence"
            )
        for i in range(min(len(poses), len(dataset_frame_ids))):
            aligned[i] = poses[i]
        missing = sum(p is None for p in aligned)
        return aligned, missing

    # COLMAP sparse: map each rgb id to its pose; missing registrations stay None.
    pose_by_id = dict(zip(reference_frame_ids, poses))
    for i, rgb_id in enumerate(dataset_frame_ids):
        pose = pose_by_id.get(rgb_id)
        if pose is None:
            missing += 1
        else:
            aligned[i] = pose
    return aligned, missing


def resolve_colmap_sparse_path(base_path: str, settings: dict | None = None) -> str:
    """Return the COLMAP sparse model folder for a CLIO scene (default: sparse/0)."""
    # Optional override from config: colmap_sparse_path
    if settings is not None and "colmap_sparse_path" in settings:
        sparse_path = settings["colmap_sparse_path"]
        if os.path.isabs(sparse_path):
            return sparse_path
        return os.path.join(base_path, sparse_path)

    default = os.path.join(base_path, "sparse", "0")
    if os.path.isdir(default):
        return default

    sparse_root = os.path.join(base_path, "sparse")
    if os.path.isdir(sparse_root):
        subdirs = sorted(
            d
            for d in os.listdir(sparse_root)
            if os.path.isdir(os.path.join(sparse_root, d))
        )
        if len(subdirs) == 1:
            return os.path.join(sparse_root, subdirs[0])
    return default


def load_colmap_poses_for_clio(
    base_path: str, settings: dict | None = None
) -> tuple[list[np.ndarray], list[int]]:
    """
    Load camera-to-world poses from a CLIO scene COLMAP model.

    Returns poses sorted by rgb frame id and the corresponding frame id list.
    Used as reference trajectory for ClioGroundTruth (not independent motion-capture GT).
    """
    sparse_path = resolve_colmap_sparse_path(base_path, settings=settings)
    images_bin = os.path.join(sparse_path, "images.bin")
    images_txt = os.path.join(sparse_path, "images.txt")

    if os.path.isfile(images_bin):
        images = read_images_binary(images_bin)
    elif os.path.isfile(images_txt):
        images = read_images_text(images_txt)
    else:
        raise FileNotFoundError(
            f"COLMAP images file not found in {sparse_path} (expected images.bin or images.txt)"
        )

    poses_by_frame_id: dict[int, np.ndarray] = {}
    for image in images.values():
        frame_id = clio_image_name_to_frame_id(image.name)
        if frame_id is None:
            continue
        poses_by_frame_id[frame_id] = colmap_qvec_tvec_to_Twc(image.qvec, image.tvec)

    if not poses_by_frame_id:
        raise ValueError(f"No CLIO rgb_*.jpg poses found in COLMAP model: {sparse_path}")

    frame_ids = sorted(poses_by_frame_id.keys())
    poses = [poses_by_frame_id[frame_id] for frame_id in frame_ids]
    return poses, frame_ids


def load_clio_poses_from_bag(
    base_path: str, odom_topic: str = CLIO_ODOM_TOPIC
) -> tuple[list[np.ndarray], list[int]] | None:
    """
    Load metric reference poses from a CLIO scene ROS1 bag (colmap_odom topic).

    Returns (poses, frame_ids) or None if the bag/topic is unavailable.
    Failures are logged (not silent) so callers can distinguish missing bag vs read error.
    """
    try:
        from pathlib import Path

        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        # Fall back to COLMAP sparse in load_clio_reference_poses when auto mode is used.
        Printer.yellow(
            "[colmap_io] rosbags not installed; cannot load CLIO bag colmap_odom poses"
        )
        return None

    from pyslam.utilities.geometry import xyzq2Tmat

    bag_files = sorted(glob.glob(os.path.join(base_path, "*.bag")))
    if not bag_files:
        return None

    typestore = get_typestore(Stores.ROS1_NOETIC)
    poses: list[np.ndarray] = []
    frame_ids: list[int] = []

    try:
        with AnyReader([Path(bag_files[0])]) as reader:
            connections = [c for c in reader.connections if c.topic == odom_topic]
            if not connections:
                Printer.yellow(
                    f"[colmap_io] Topic {odom_topic} not found in {bag_files[0]}"
                )
                return None

            for frame_id, (_, _, raw) in enumerate(
                reader.messages(connections=[connections[0]])
            ):
                msg = typestore.deserialize_ros1(raw, connections[0].msgtype)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                poses.append(
                    xyzq2Tmat(p.x, p.y, p.z, q.x, q.y, q.z, q.w).astype(np.float64)
                )
                frame_ids.append(frame_id)
    except Exception as e:
        Printer.yellow(
            f"[colmap_io] Failed to read CLIO bag odometry from {bag_files[0]}: {e}"
        )
        return None

    if not poses:
        return None
    return poses, frame_ids


def load_clio_reference_poses(
    base_path: str, settings: dict | None = None
) -> tuple[list[np.ndarray], list[int], str, bool]:
    """
    Load a CLIO reference trajectory for evaluation/visualization.

    Priority (clio_reference_poses setting):
      - auto (default): bag colmap_odom, else COLMAP sparse/0
      - bag: ROS bag colmap_odom only
      - sparse: COLMAP sparse/0 only (often non-metric)

    Returns: poses, frame_ids, source_label, is_metric
    """
    reference_mode = "auto"
    if settings is not None and "clio_reference_poses" in settings:
        reference_mode = str(settings["clio_reference_poses"]).lower()

    if reference_mode in ("auto", "bag"):
        bag_poses = load_clio_poses_from_bag(base_path)
        if bag_poses is not None:
            return bag_poses[0], bag_poses[1], "bag_colmap_odom", True
        if reference_mode == "bag":
            raise FileNotFoundError(
                f"CLIO bag colmap_odom not found in {base_path} (clio_reference_poses: bag)"
            )

    poses, frame_ids = load_colmap_poses_for_clio(base_path, settings=settings)
    return poses, frame_ids, "sparse_colmap", False


def clio_fps_from_bag(base_path: str, color_topic: str = CLIO_COLOR_TOPIC) -> float | None:
    """
    Estimate CLIO capture rate from a scene ROS1 bag (*.bag in base_path).

    Uses the median inter-frame interval on the color image topic.
    Returns None if rosbags is unavailable, no bag is found, or reading fails.
    Warnings are printed on failure (used by resolve_clio_fps fallback chain).
    """
    try:
        from pathlib import Path

        from rosbags.highlevel import AnyReader
    except ImportError:
        return None

    bag_files = sorted(glob.glob(os.path.join(base_path, "*.bag")))
    if not bag_files:
        return None

    timestamps = []
    try:
        with AnyReader([Path(bag_files[0])]) as reader:
            connections = [c for c in reader.connections if c.topic == color_topic]
            if not connections:
                connections = [
                    c
                    for c in reader.connections
                    if "image" in c.topic.lower()
                    and "depth" not in c.topic.lower()
                    and "color" in c.topic.lower()
                ]
            if not connections:
                Printer.yellow(
                    f"[colmap_io] No color image topic found in {bag_files[0]} for fps detection"
                )
                return None

            for _, timestamp, _ in reader.messages(connections=[connections[0]]):
                timestamps.append(timestamp)
                if len(timestamps) >= 300:
                    break
    except Exception as e:
        Printer.yellow(
            f"[colmap_io] Failed to estimate CLIO fps from {bag_files[0]}: {e}"
        )
        return None

    if len(timestamps) < 2:
        return None

    dts = np.diff(np.asarray(timestamps, dtype=np.float64)) * 1e-9
    median_dt = float(np.median(dts))
    if median_dt <= 0:
        return None
    return 1.0 / median_dt


def resolve_clio_fps(
    base_path: str,
    settings: dict | None = None,
    cam_settings: dict | None = None,
) -> float:
    """
    Resolve CLIO playback fps (shared by ClioDataset and ClioGroundTruth).

    Dataset and GT must use the same fps so synthetic timestamps match during
    find_poses_associations / ATE evaluation. Priority:
      settings['fps'] > bag auto-detection > Camera.fps (settings yaml) > CLIO_DEFAULT_FPS.

    cam_settings is passed from config.cam_settings (see groundtruth_factory).
    """
    if settings is not None and "fps" in settings:
        return float(settings["fps"])

    bag_fps = clio_fps_from_bag(base_path)
    if bag_fps is not None:
        return bag_fps

    # Without a bag, honor Camera.fps from settings/CLIO.yaml (same fallback as ClioDataset).
    if cam_settings is not None and "Camera.fps" in cam_settings:
        return float(cam_settings["Camera.fps"])

    return CLIO_DEFAULT_FPS
