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

"""Compare Python and C++ Tracking / LocalMapping on one synthetic RGBD sequence.

Loop closing stays off. Local mapping runs on the caller thread so both
implementations insert keyframes in the same order. The two runs are separate
processes because USE_CPP_CORE is selected once, at import.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

kRootFolder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../.."))
if kRootFolder not in sys.path:
    sys.path.insert(0, kRootFolder)

NUM_FRAMES = 20
WIDTH = 640
HEIGHT = 480
FX = 500.0
FY = 500.0
CX = 320.0
CY = 240.0
PLANE_Z = 2.0
MAX_ROTATION_DEG = 1.0
MAX_TRANSLATION_M = 0.05


def _texture():
    rng = np.random.RandomState(1)
    texture = np.zeros((1500, 2000), dtype=np.uint8)
    for _ in range(500):
        x = int(rng.randint(0, 1900))
        y = int(rng.randint(0, 1400))
        w = int(rng.randint(10, 48))
        h = int(rng.randint(10, 48))
        texture[y : y + h, x : x + w] = int(rng.randint(0, 256))
    noise = rng.randint(0, 30, texture.shape, dtype=np.uint8)
    return np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _frame(texture, frame_id):
    import cv2

    tx = 0.02 * frame_id
    xs = (np.arange(WIDTH, dtype=np.float32) - CX) / FX * PLANE_Z + tx
    ys = (np.arange(HEIGHT, dtype=np.float32) - CY) / FY * PLANE_Z
    map_x = ((xs + 2.0) / 4.0 * (texture.shape[1] - 1)).astype(np.float32)
    map_y = ((ys + 1.5) / 3.0 * (texture.shape[0] - 1)).astype(np.float32)
    grid_x = np.broadcast_to(map_x, (HEIGHT, WIDTH))
    grid_y = np.broadcast_to(map_y.reshape(HEIGHT, 1), (HEIGHT, WIDTH))
    gray = cv2.remap(texture, grid_x, grid_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    depth = np.full((HEIGHT, WIDTH), PLANE_Z, dtype=np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(depth)


def _camera_settings():
    return {
        "Camera.width": WIDTH,
        "Camera.height": HEIGHT,
        "Camera.fx": FX,
        "Camera.fy": FY,
        "Camera.cx": CX,
        "Camera.cy": CY,
        "Camera.k1": 0.0,
        "Camera.k2": 0.0,
        "Camera.p1": 0.0,
        "Camera.p2": 0.0,
        "Camera.k3": 0.0,
        "Camera.fps": 30,
        "Camera.bf": 40.0,
        "DepthMapFactor": 1.0,
        "ThDepth": 50.0,
    }


def _make_camera(use_cpp):
    from pyslam.io.dataset_types import SensorType
    from pyslam.slam import PinholeCamera

    settings = _camera_settings()
    if use_cpp:
        return PinholeCamera(
            {"cam_settings": settings, "dataset_settings": {"sensor_type": "rgbd"}}
        )

    class _Config:
        cam_settings = settings
        DistCoef = [0.0, 0.0, 0.0, 0.0, 0.0]
        sensor_type = SensorType.RGBD

    return PinholeCamera(_Config())


def _pose_record(tracking, slam):
    state = tracking.state
    state_name = state.name if hasattr(state, "name") else str(state)
    rotation = None if tracking.cur_R is None else np.asarray(tracking.cur_R, dtype=np.float64).reshape(3, 3)
    translation = None if tracking.cur_t is None else np.asarray(tracking.cur_t, dtype=np.float64).reshape(3)
    return {
        "state": state_name,
        "R": None if rotation is None else rotation.tolist(),
        "t": None if translation is None else translation.tolist(),
        "num_keyframes": int(slam.map.num_keyframes()),
        "num_points": int(slam.map.num_points()),
    }


def run_sequence(output_path, python_frontend=False):
    import pyslam.config as config  # noqa: F401  — core lib paths before the SLAM import

    from pyslam.config_parameters import Parameters
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.slam import Slam, USE_CPP
    if python_frontend:
        from pyslam.slam.slam import Slam
    from pyslam.io.dataset_types import SensorType

    Parameters.kUseLoopClosing = False
    Parameters.kLocalMappingOnSeparateThread = False
    Parameters.kDoVolumetricIntegration = False
    Parameters.kDoSparseSemanticMappingAndSegmentation = False
    Parameters.kUseLargeWindowBA = False
    Parameters.kUseParallelProcessLBA = False
    Parameters.kLocalMappingParallelKpsMatching = False
    Parameters.kUseMotionBlurDection = False

    tracker_config = dict(FeatureTrackerConfigs.ORB2)
    tracker_config["deterministic"] = True
    tracker_config["num_features"] = 800

    camera = _make_camera(USE_CPP)
    slam = Slam(
        camera,
        tracker_config,
        None,
        None,
        SensorType.RGBD,
        headless=True,
    )
    texture = _texture()
    records = []
    track_seconds = []
    try:
        for img_id in range(NUM_FRAMES):
            image, depth = _frame(texture, img_id)
            started = time.perf_counter()
            slam.track(image, None, depth, img_id, float(img_id))
            track_seconds.append(time.perf_counter() - started)
            record = _pose_record(slam.tracking, slam)
            record["id"] = img_id
            records.append(record)
            print(
                f"frame {img_id:02d} state={record['state']} "
                f"kfs={record['num_keyframes']} pts={record['num_points']}",
                flush=True,
            )
        poses, timestamps, ids = slam.get_final_trajectory()
        final_trajectory = {"poses": [np.asarray(p).tolist() for p in poses],
                            "timestamps": list(timestamps), "ids": list(ids)}
    finally:
        slam.quit()

    payload = {"use_cpp": bool(USE_CPP), "python_frontend": python_frontend,
               "frames": records, "track_seconds": track_seconds,
               "final_trajectory": final_trajectory}
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(payload, out)
    print(f"wrote {output_path}", flush=True)


def _rotation_deg(rotation_a, rotation_b):
    cosine = (np.trace(rotation_a.T @ rotation_b) - 1.0) * 0.5
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compare(python_path, cpp_path):
    with open(python_path, encoding="utf-8") as handle:
        python_run = json.load(handle)
    with open(cpp_path, encoding="utf-8") as handle:
        cpp_run = json.load(handle)
    if cpp_run["use_cpp"] is not True or cpp_run.get("python_frontend", False):
        raise SystemExit("C++ run did not use the native frontend")
    if python_run["use_cpp"] and not python_run.get("python_frontend", False):
        raise SystemExit("Python run did not use the Python frontend")
    python_frames = python_run["frames"]
    cpp_frames = cpp_run["frames"]
    if len(python_frames) != len(cpp_frames):
        raise SystemExit(f"frame count {len(python_frames)} vs {len(cpp_frames)}")

    worst_rotation = 0.0
    worst_translation = 0.0
    ok_frames = 0
    for python_frame, cpp_frame in zip(python_frames, cpp_frames):
        if python_frame["state"] != cpp_frame["state"]:
            raise SystemExit(
                f"frame {python_frame['id']}: state {python_frame['state']} vs {cpp_frame['state']}"
            )
        if python_frame["state"] != "OK":
            continue
        if python_frame["num_keyframes"] != cpp_frame["num_keyframes"]:
            raise SystemExit(
                f"frame {python_frame['id']}: keyframes {python_frame['num_keyframes']} vs {cpp_frame['num_keyframes']}"
            )
        if python_frame["R"] is None or cpp_frame["R"] is None:
            if python_frame["R"] is None and cpp_frame["R"] is None:
                continue
            raise SystemExit(f"frame {python_frame['id']}: OK without a pose")
        ok_frames += 1
        rotation = _rotation_deg(
            np.asarray(python_frame["R"], dtype=np.float64),
            np.asarray(cpp_frame["R"], dtype=np.float64),
        )
        translation = float(
            np.linalg.norm(
                np.asarray(python_frame["t"], dtype=np.float64)
                - np.asarray(cpp_frame["t"], dtype=np.float64)
            )
        )
        worst_rotation = max(worst_rotation, rotation)
        worst_translation = max(worst_translation, translation)
        print(
            f"frame {python_frame['id']:02d} dR={rotation:.4f} deg dt={translation:.5f} m "
            f"py_kfs={python_frame['num_keyframes']} cpp_kfs={cpp_frame['num_keyframes']} "
            f"py_pts={python_frame['num_points']} cpp_pts={cpp_frame['num_points']}"
        )
        if rotation > MAX_ROTATION_DEG or translation > MAX_TRANSLATION_M:
            raise SystemExit(
                f"frame {python_frame['id']}: pose delta {rotation:.4f} deg, {translation:.5f} m"
            )
    if ok_frames < 5:
        raise SystemExit(f"only {ok_frames} OK frames; tracking did not stay initialized")
    print(
        f"matched {ok_frames} OK frames, worst rotation {worst_rotation:.4f} deg, "
        f"worst translation {worst_translation:.5f} m"
    )


def _spawn(use_cpp, output_path, python_frontend=False):
    env = os.environ.copy()
    env["PYSLAM_USE_CPP"] = "1" if use_cpp else "0"
    env["PYTHONPATH"] = kRootFolder + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, os.path.abspath(__file__), "--run", output_path]
    if python_frontend:
        command.append("--python-frontend")
    completed = subprocess.run(command, env=env, cwd=kRootFolder, timeout=180)
    if completed.returncode != 0:
        raise SystemExit(f"{'C++' if use_cpp else 'Python'} run failed with code {completed.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Write one trajectory and exit. Used by the child process.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("/tmp", "pyslam_cpp_python_trajectory"),
    )
    parser.add_argument("--python-frontend", action="store_true",
                        help="Use the Python orchestrators with the selected core (child run).")
    parser.add_argument("--same-core", action="store_true",
                        help="Compare both frontends using the native map and kernels.")
    args = parser.parse_args()
    if args.run:
        run_sequence(args.run, args.python_frontend)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    python_path = os.path.join(args.output_dir, "python.json")
    cpp_path = os.path.join(args.output_dir, "cpp.json")
    _spawn(args.same_core, python_path, python_frontend=args.same_core)
    _spawn(True, cpp_path)
    compare(python_path, cpp_path)
    with open(python_path, encoding="utf-8") as handle:
        python_run = json.load(handle)
    with open(cpp_path, encoding="utf-8") as handle:
        cpp_run = json.load(handle)
    for label, run in (("Python", python_run), ("C++", cpp_run)):
        print(f"{label} median tracking time after initialization: "
              f"{1000 * np.median(run['track_seconds'][2:]):.2f} ms")
    py_final, cpp_final = python_run["final_trajectory"], cpp_run["final_trajectory"]
    if py_final["ids"] != cpp_final["ids"] or py_final["timestamps"] != cpp_final["timestamps"]:
        raise SystemExit("final trajectory frame IDs or timestamps differ")
    for py_pose, cpp_pose in zip(py_final["poses"], cpp_final["poses"]):
        py_pose, cpp_pose = np.asarray(py_pose), np.asarray(cpp_pose)
        if (_rotation_deg(py_pose[:3, :3], cpp_pose[:3, :3]) > MAX_ROTATION_DEG or
                np.linalg.norm(py_pose[:3, 3] - cpp_pose[:3, 3]) > MAX_TRANSLATION_M):
            raise SystemExit("final trajectory pose delta exceeds tolerance")


if __name__ == "__main__":
    main()
