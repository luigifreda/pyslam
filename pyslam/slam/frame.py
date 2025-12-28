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

import sys
import cv2
import numpy as np

# import json
import ujson as json

from threading import RLock, Lock, Thread, current_thread
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.spatial import cKDTree

from pyslam.utilities.timer import Timer

from pyslam.io.dataset_types import SensorType
from pyslam.config_parameters import Parameters

from .camera import Camera, PinholeCamera
from .camera_pose import CameraPose
from .feature_tracker_shared import FeatureTrackerShared

from pyslam.utilities.geometry import add_ones, poseRt
from pyslam.utilities.geom_triangulation import triangulate_normalized_points
from pyslam.utilities.logging import Printer
from pyslam.utilities.colors import Colors, ColorTableGenerator

from pyslam.utilities.drawing import draw_feature_matches
from pyslam.utilities.features import (
    compute_NSAD_between_matched_keypoints,
    descriptor_sigma_mad,
    descriptor_sigma_mad_v2,
    stereo_match_subpixel_correlation,
)
from pyslam.utilities.serialization import (
    deserialize_array_flexible,
    NumpyB64Json,
    vector3_serialize,
    vector3_deserialize,
    cv_mat_to_json_raw,
)

import atexit
import traceback


from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.local_features.feature_tracker import FeatureTracker
    from pyslam.local_features.feature_matcher import FeatureMatcher
    from pyslam.local_features.feature_manager import FeatureManager
    from pyslam.slam.map_point import MapPoint


kMinDepth = Parameters.kMinDepth
kDrawFeatureRadius = [
    r * 5 for r in range(1, 100)
]  # NOTE: This is just a convenience representation of the feature radius for drawing purposes.
kDrawOctaveColor = np.linspace(0, 255, 12)


# for parallel stereo processing
def detect_and_compute(img, left=True):
    if left:
        return FeatureTrackerShared.feature_tracker.detectAndCompute(img)
    else:
        assert FeatureTrackerShared.feature_tracker_right is not None
        return FeatureTrackerShared.feature_tracker_right.detectAndCompute(img)


# Base object class for frame info management;
# it collects methods for managing:
# - camera intrinsics
# - camera pose
# - points projections
# - checking points visibility
class FrameBase(object):
    _id = 0  # shared frame counter
    _id_lock = Lock()

    def __init__(self, camera: Camera, pose=None, id=None, timestamp=None, img_id=None):
        self._lock_pose = Lock()
        # frame camera info
        self.camera: Camera = camera
        # self._pose is a CameraPose() representing Tcw (pc = Tcw * pw)
        if pose is None:
            self._pose = CameraPose()
        else:
            self._pose = CameraPose(
                pose.copy()
            )  # We need to copy the pose to avoid sharing the same reference
        # frame id management
        if id is not None:
            self.id = id
        else:
            with FrameBase._id_lock:
                self.id = FrameBase._id
                FrameBase._id += 1
        # frame timestamp
        self.timestamp = timestamp
        self.img_id = img_id

        self.median_depth = -1  # median depth of the frame
        self.fov_center_c = None  # fov center 3D position w.r.t. camera
        self.fov_center_w = None  # fov center 3D position w.r.t world

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the Lock from the state (don't pickle it)
        if "_lock_pose" in state:
            del state["_lock_pose"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'Lock' initially)
        self.__dict__.update(state)
        # Recreate the Lock after unpickling
        self._lock_pose = Lock()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return isinstance(rhs, FrameBase) and self.id == rhs.id

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    @staticmethod
    def next_id():
        with FrameBase._id_lock:
            return FrameBase._id

    @staticmethod
    def set_id(id):
        with FrameBase._id_lock:
            FrameBase._id = id

    @property
    def width(self):
        return self.camera.width

    @property
    def height(self):
        return self.camera.height

    def isometry3d(self):  # pose as g2o.Isometry3d
        with self._lock_pose:
            return self._pose.isometry3d

    def Tcw(self):
        with self._lock_pose:
            return self._pose.Tcw.copy()

    def Twc(self):
        with self._lock_pose:
            return self._pose.get_inverse_matrix().copy()

    def Rcw(self):
        with self._lock_pose:
            return self._pose.Rcw.copy()

    def Rwc(self):
        with self._lock_pose:
            return self._pose.Rwc.copy()

    def tcw(self):
        with self._lock_pose:
            return self._pose.tcw.copy()

    def Ow(self):
        with self._lock_pose:
            return self._pose.Ow.copy()

    def pose(self):
        with self._lock_pose:
            return self._pose.Tcw.copy()

    def quaternion(self):  # g2o.Quaternion(),  quaternion_cw
        with self._lock_pose:
            return self._pose.quaternion

    def orientation(self):  # g2o.Quaternion(),  quaternion_cw
        with self._lock_pose:
            return self._pose.orientation

    def position(self):  # 3D vector tcw (world origin w.r.t. camera frame)
        with self._lock_pose:
            return self._pose.position

    # update pose_cw from transformation matrix or g2o.Isometry3d, input pose
    def update_pose(self, pose):
        with self._lock_pose:
            self._pose.set(pose)
            if self.fov_center_c is not None:
                self.fov_center_w = self._pose.Rwc @ self.fov_center_c + self._pose.Ow.reshape(3, 1)

    # update pose from transformation matrix
    def update_translation(self, tcw):
        with self._lock_pose:
            self._pose.set_translation(tcw)
            if self.fov_center_c is not None:
                self.fov_center_w = self._pose.Rwc @ self.fov_center_c + self._pose.Ow.reshape(3, 1)

    # update pose from transformation matrix
    def update_rotation_and_translation(self, Rcw, tcw):
        with self._lock_pose:
            self._pose.set_from_rotation_and_translation(Rcw, tcw)
            if self.fov_center_c is not None:
                self.fov_center_w = self._pose.Rwc @ self.fov_center_c + self._pose.Ow.reshape(3, 1)

    # transform a world point into a camera point
    def transform_point(self, pw):
        with self._lock_pose:
            return (self._pose.Rcw @ pw) + self._pose.tcw  # p w.r.t. camera

    # transform a world points into camera points [Nx3]
    # out: points  w.r.t. camera frame  [Nx3]
    def transform_points(self, points):
        with self._lock_pose:
            Rcw = self._pose.Rcw.copy()
            tcw = self._pose.tcw.reshape((3, 1)).copy()
        points = np.ascontiguousarray(points)
        return (Rcw @ points.T + tcw).T  # get points  w.r.t. camera frame  [Nx3]

    # project an [Nx3] array of map point vectors on this frame
    # out: [Nx2] image projections (u,v) or [Nx3] array of stereo projections (u,v,ur) in case do_stereo_projet=True,
    #      [Nx1] array of map point depths
    def project_points(self, points, do_stereo_project=False):
        points = np.ascontiguousarray(points)
        pcs = self.transform_points(points)
        if do_stereo_project:
            return self.camera.project_stereo(pcs)
        else:
            return self.camera.project(pcs)

    # project a list of N MapPoint objects on this frame
    # out: [Nx2] image projections (u,v) or [Nx3] array of stereo projections (u,v,ur) in case do_stereo_projet=True,
    #      [Nx1] array of map point depths
    def project_map_points(self, map_points, do_stereo_project=False):
        points = np.ascontiguousarray([p.pt() for p in map_points])
        return self.project_points(points, do_stereo_project)

    # project a 3d point vector pw on this frame
    # out: image point, depth
    def project_point(self, pw: np.ndarray, do_stereo_project: bool = False):
        pc = self.transform_point(pw)  # p w.r.t. camera
        if do_stereo_project:
            return self.camera.project_stereo(pc)
        else:
            return self.camera.project(pc)

    # project a MapPoint object on this frame
    # out: image point, depth
    def project_map_point(self, map_point: "MapPoint", do_stereo_project: bool = False):
        return self.project_point(map_point.pt(), do_stereo_project)

    def is_in_image(self, uv, z):
        return self.camera.is_in_image(uv, z)

    # input: [Nx2] array of uvs, [Nx1] of zs
    # output: [Nx1] array of visibility flags
    def are_in_image(self, uvs, zs):
        return self.camera.are_in_image(uvs, zs)

    # input: map_point
    # output: visibility flag, projection uv, depth z
    def is_visible(self, map_point):
        # with self._lock_pose:    (no need, project_map_point already locks the pose)
        uv, z = self.project_map_point(map_point)
        PO = map_point.pt() - self.Ow()

        if not self.is_in_image(uv, z):
            return False, uv, z

        dist3D = np.linalg.norm(PO)
        # point depth must be inside the scale pyramid of the image
        if dist3D < map_point.min_distance() or dist3D > map_point.max_distance():
            return False, uv, z
        # viewing angle must be less than 60 deg
        if np.dot(PO, map_point.get_normal()) < Parameters.kViewingCosLimitForPoint * dist3D:
            return False, uv, z
        return True, uv, z

    # input: a list of map points
    # output: [Nx1] array of visibility flags,
    #         [Nx2] array of projections (u,v) or [Nx3] array of stereo image points (u,v,ur) in case do_stereo_projet=True,
    #         [Nx1] array of depths,
    #         [Nx1] array of distances PO
    # check a) points are in image b) good view angle c) good distance range
    def are_visible(self, map_points: list["MapPoint"], do_stereo_project=False):
        # n = len(map_points)
        # points = np.zeros((n, 3), dtype=np.float32)
        # point_normals = np.zeros((n, 3), dtype=np.float32)
        # min_dists = np.zeros(n, dtype=np.float32)
        # max_dists = np.zeros(n, dtype=np.float32)
        # for i, p in enumerate(map_points):
        #     pt, normal, min_dist, max_dist = p.get_all_pos_info() # just one lock here
        #     points[i] = pt
        #     point_normals[i] = normal # corresponding to p.get_normal()
        #     min_dists[i] = min_dist   # corresponding to p.min_distance()
        #     max_dists[i] = max_dist   # corresponding to p.min_distance()
        points, normals, min_dists, max_dists = zip(*(p.get_all_pos_info() for p in map_points))
        points = np.ascontiguousarray(np.vstack(points))  # shape (N, 3)
        normals = np.ascontiguousarray(np.vstack(normals))  # shape (N, 3)
        min_dists = np.ascontiguousarray(min_dists)  # shape (N,)
        max_dists = np.ascontiguousarray(max_dists)  # shape (N,)

        uvs, zs = self.project_points(points, do_stereo_project)
        POs = points - self.Ow()
        dists = np.linalg.norm(POs, axis=-1, keepdims=True)
        POs /= dists
        cos_view = np.sum(normals * POs, axis=1)

        are_in_image = self.are_in_image(uvs, zs)
        are_in_good_view_angle = cos_view > Parameters.kViewingCosLimitForPoint
        dists = dists.reshape(
            -1,
        )
        are_in_good_distance = (dists > min_dists) & (dists < max_dists)

        out_flags = are_in_image & are_in_good_view_angle & are_in_good_distance
        return out_flags, uvs, zs, dists


# A Frame mainly collects keypoints, descriptors and their corresponding 3D points
class Frame(FrameBase):
    is_store_imgs = False  # to store images when needed for debugging or processing purposes
    is_compute_median_depth = False  # to compute median depth when needed
    feature_detect_and_compute_callback = None  # symmetric to C++
    feature_detect_and_compute_right_callback = None  # symmetric to C++

    def __init__(
        self,
        camera: Camera,
        img,
        img_right=None,
        depth=None,
        pose=None,
        id=None,
        timestamp=None,
        img_id=None,
        semantic_img=None,
        frame_data_dict=None,
    ):
        super().__init__(camera, pose=pose, id=id, timestamp=timestamp, img_id=img_id)

        self._lock_features = Lock()
        self._lock_semantics = Lock()
        self.is_keyframe = False

        self._kd = None  # kdtree for fast-search of keypoints

        # image keypoints information arrays (unpacked from array of cv::KeyPoint())
        # NOTE: in the stereo case, we assume the images are rectified (and unistortion becomes redundant)
        self.kps = None  # left keypoint coordinates                                      [Nx2]
        self.kps_r = None  # righ keypoint coordinates (extracted on right image)           [Nx2]
        self.kpsu = None  # [u]ndistorted keypoint coordinates                             [Nx2]
        self.kpsn = None  # [n]ormalized keypoint coordinates                              [Nx2] (Kinv * [kp,1])
        self.kps_sem = None  # [sem]antic keypoint information                                [NxD] where D is the semantic information length
        self.octaves = None  # keypoint octaves                                               [Nx1]
        self.octaves_r = (
            None  # keypoint octaves                                               [Nx1]
        )
        self.sizes = None  # keypoint sizes                                                 [Nx1]
        self.angles = None  # keypoint sizes                                                 [Nx1]
        self.des = None  # keypoint descriptors                                           [NxD] where D is the descriptor length
        self.des_r = None  # right keypoint descriptors                                     [NxD] where D is the descriptor length
        self.depths = None  # keypoint depths                                                [Nx1]
        self.kps_ur = None  # corresponding right u-coordinates for left keypoints           [Nx1] (computed with stereo matching and assuming rectified stereo images)

        # map points information arrays
        self.points: list[MapPoint] | None = (
            None  # map points => self.points[idx] (if is not None) is the map point matched with self.kps[idx]
        )
        self.outliers = None  # outliers flags for map points (reset and set by pose_optimization())

        self.kf_ref = None  # reference keyframe

        self.img = None  # image (copy of img if available)
        self.img_right = None  # right image (copy of img_right if available)
        self.depth_img = None  # depth (copy of depth if available)
        self.semantic_img = None  # semantics (copy of semantic_img if available)
        self.semantic_instances_img = None  # semantic instances

        self.is_blurry = False
        self.laplacian_var = None

        if img is not None:
            # self.H, self.W = img.shape[0:2]
            if Frame.is_store_imgs:
                self.img = img.copy()

        if img_right is not None:
            if Frame.is_store_imgs:
                self.img_right = img_right.copy()

        if depth is not None:
            if self.camera is not None and self.camera.depth_factor != 1.0:
                depth = depth * self.camera.depth_factor
            if Frame.is_store_imgs:
                self.depth_img = depth.copy()

        if semantic_img is not None:
            if Frame.is_store_imgs:
                self.semantic_img = semantic_img.copy()

        if frame_data_dict is not None:
            self.is_keyframe = frame_data_dict["is_keyframe"]
            self.median_depth = frame_data_dict["median_depth"]
            self.fov_center_c = frame_data_dict["fov_center_c"]
            self.fov_center_w = frame_data_dict["fov_center_w"]

            self.is_blurry = frame_data_dict["is_blurry"]
            self.laplacian_var = frame_data_dict["laplacian_var"]

            self.kps = frame_data_dict["kps"]
            self.kps_r = frame_data_dict["kps_r"]
            self.kpsu = frame_data_dict["kpsu"]
            self.kpsn = frame_data_dict["kpsn"]
            self.kps_sem = frame_data_dict["kps_sem"]
            self.octaves = frame_data_dict["octaves"]
            self.octaves_r = frame_data_dict["octaves_r"]
            self.sizes = frame_data_dict["sizes"]
            self.angles = frame_data_dict["angles"]
            self.des = frame_data_dict["des"]
            self.des_r = frame_data_dict["des_r"]
            self.depths = frame_data_dict["depths"]
            self.kps_ur = frame_data_dict["kps_ur"]
            self.points = frame_data_dict["points"]
            self.outliers = frame_data_dict["outliers"]
            self.kf_ref = frame_data_dict["kf_ref"]

            if self.img is None:
                self.img = frame_data_dict["img"]

            if self.img_right is None:
                self.img_right = frame_data_dict["img_right"]

            if self.depth_img is None:
                self.depth_img = frame_data_dict["depth_img"]

            if self.semantic_img is None:
                self.semantic_img = frame_data_dict["semantic_img"]

            if self.semantic_instances_img is None:
                self.semantic_instances_img = frame_data_dict["semantic_instances_img"]

            self.ensure_contiguous_arrays()
            return

        if img is not None:
            if img_right is not None:
                do_parallel = True
                if do_parallel:
                    with ThreadPoolExecutor() as executor:
                        future_l = executor.submit(detect_and_compute, img)
                        future_r = executor.submit(detect_and_compute, img_right, left=False)
                        self.kps, self.des = future_l.result()
                        self.kps_r, self.des_r = future_r.result()
                else:
                    self.kps, self.des = FeatureTrackerShared.feature_tracker.detectAndCompute(img)
                    self.kps_r, self.des_r = FeatureTrackerShared.feature_tracker.detectAndCompute(
                        img_right
                    )
                # print(f'kps: {len(self.kps)}, des: {self.des.shape}, kps_r: {len(self.kps_r)}, des_r: {self.des_r.shape}')
            else:
                self.kps, self.des = FeatureTrackerShared.feature_tracker.detectAndCompute(img)

            # convert from a list of keypoints to arrays of points, octaves, sizes
            if self.kps is not None and len(self.kps) > 0:
                kps_data = np.ascontiguousarray(
                    [[x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps],
                    dtype=np.float32,
                )
                self.kps = kps_data[:, :2] if kps_data is not None else None
                self.octaves = np.uint8(kps_data[:, 2])  # print('octaves: ', self.octaves)
                self.sizes = kps_data[:, 3]
                self.angles = kps_data[:, 4]
                if self.camera is None:
                    raise Exception("Frame.init: camera is None")
                self.kpsu = self.camera.undistort_points(
                    self.kps
                )  # convert to undistorted keypoint coordinates
                self.kpsn = self.camera.unproject_points(self.kpsu)
                if len(self.kps) != len(self.kpsu):
                    raise Exception("Frame.init: len(self.kps) != len(self.kpsu)")
                num_kps = len(self.kps)
                self.points = np.full(num_kps, None, dtype=object)  # init map points
                self.outliers = np.full(num_kps, False, dtype=bool)

            if self.kps_r is not None:
                kps_data_r = np.ascontiguousarray(
                    [[x.pt[0], x.pt[1], x.octave, x.size, x.angle] for x in self.kps_r],
                    dtype=np.float32,
                )
                self.kps_r = kps_data_r[:, :2] if kps_data_r is not None else None
                self.octaves_r = np.uint8(kps_data_r[:, 2])  # print('octaves: ', self.octaves)

            if self.kps is not None and len(self.kps) > 0:
                # compute stereo matches: if there is depth available, use it, otherwise use stereo matching
                if depth is not None:
                    self.compute_stereo_from_rgbd(kps_data, depth)
                elif img_right is not None:
                    self.depths = np.full(len(self.kps), -1, dtype=float)
                    self.kps_ur = np.full(len(self.kps), -1, dtype=float)
                    self.compute_stereo_matches(img, img_right)

            self.ensure_contiguous_arrays()

    # ensure contiguous arrays
    def ensure_contiguous_arrays(self):
        for attr in [
            "kps",
            "kpsu",
            "kpsn",
            "kps_r",
            "des",
            "des_r",
            "depths",
            "kps_ur",
            "angles",
            "sizes",
            "octaves",
            "octaves_r",
            "outliers",
            "points",
            "img",
            "img_right",
            "depth_img",
            "semantic_img",
            "semantic_instances_img",
        ]:
            val = getattr(self, attr, None)
            if isinstance(val, np.ndarray) and not val.flags["C_CONTIGUOUS"]:
                setattr(self, attr, np.ascontiguousarray(val))

    def set_img_right(self, img_right):
        self.img_right = np.ascontiguousarray(img_right.copy())

    def set_depth_img(self, depth_img):
        if self.camera is not None:  # and self.camera.depth_factor != 1.0:
            depth_img = depth_img * self.camera.depth_factor
            self.depth_img = np.ascontiguousarray(depth_img.copy())
        else:
            message = "Frame.set_depth_img: camera is None or depth_factor is not set"
            Printer.error(message)
            raise Exception(message)

    def set_semantics(self, semantic_img):
        with self._lock_semantics:
            self.semantic_img = np.ascontiguousarray(semantic_img.copy())
        if self.kps is not None:
            with self._lock_features:
                if len(semantic_img.shape) == 3:
                    self.kps_sem = semantic_img[
                        self.kps[:, 1].astype(int), self.kps[:, 0].astype(int), :
                    ]
                elif len(semantic_img.shape) == 2:
                    self.kps_sem = semantic_img[
                        self.kps[:, 1].astype(int), self.kps[:, 0].astype(int)
                    ]
                self.kps_sem = (
                    np.ascontiguousarray(self.kps_sem) if self.kps_sem is not None else None
                )

    def set_semantic_instances(self, semantic_instances_img):
        with self._lock_semantics:
            self.semantic_instances_img = np.ascontiguousarray(
                semantic_instances_img.copy(), dtype=np.int32
            )

    def is_semantics_available(self) -> bool:
        with self._lock_semantics:
            return self.semantic_img is not None

    def update_points_semantics(self, semantic_fusion_method):
        cur_points = self.get_points()
        for p in cur_points:
            if p is not None and not p.is_bad():
                p.update_semantics(semantic_fusion_method)

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the Lock from the state (don't pickle it)
        if "_lock_pose" in state:  # from FrameBase
            del state["_lock_pose"]
        if "_lock_features" in state:
            del state["_lock_features"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the Lock after unpickling
        self._lock_pose = Lock()  # from FrameBase
        self._lock_features = Lock()

    def to_json(self):
        ret = {
            "id": int(self.id),
            "timestamp": float(self.timestamp),
            "img_id": int(self.img_id),
            "pose": (self._pose.Tcw.astype(float).tolist() if self._pose.Tcw is not None else None),
            "camera": self.camera.to_json(),
            "is_keyframe": bool(self.is_keyframe),
            "median_depth": float(self.median_depth),
            "fov_center_c": vector3_serialize(self.fov_center_c),
            "fov_center_w": vector3_serialize(self.fov_center_w),
            "is_blurry": bool(self.is_blurry),
            "laplacian_var": float(self.laplacian_var) if self.laplacian_var is not None else None,
            "kps": (self.kps.astype(float).tolist() if self.kps is not None else None),
            "kps_r": (self.kps_r.astype(float).tolist() if self.kps_r is not None else None),
            "kpsu": (self.kpsu.astype(float).tolist() if self.kpsu is not None else None),
            "kpsn": (self.kpsn.astype(float).tolist() if self.kpsn is not None else None),
            "kps_sem": (
                cv_mat_to_json_raw(self.kps_sem)
                if self.kps_sem is not None and len(self.kps_sem) > 0
                else None
            ),
            "octaves": (self.octaves.tolist() if self.octaves is not None else None),
            "octaves_r": (self.octaves_r.tolist() if self.octaves_r is not None else None),
            "sizes": (self.sizes.tolist() if self.sizes is not None else None),
            "angles": (self.angles.astype(float).tolist() if self.angles is not None else None),
            "des": (NumpyB64Json.numpy_to_json(self.des) if self.des is not None else None),
            "des_r": (NumpyB64Json.numpy_to_json(self.des_r) if self.des_r is not None else None),
            "depths": (
                self.depths.astype(float).tolist()
                if self.depths is not None and len(self.depths) > 0
                else None
            ),
            "kps_ur": (
                self.kps_ur.astype(float).tolist()
                if self.kps_ur is not None and len(self.kps_ur) > 0
                else None
            ),
            "points": (
                [p.id if p is not None else -1 for p in self.points]
                if self.points is not None
                else None
            ),
            # "points": serialize_points_array(self),
            "outliers": (
                self.outliers.astype(bool).tolist() if self.outliers is not None else None
            ),
            "kf_ref": self.kf_ref.id if self.kf_ref is not None else -1,
            "img": (NumpyB64Json.numpy_to_json(self.img) if self.img is not None else None),
            "depth_img": (
                NumpyB64Json.numpy_to_json(self.depth_img) if self.depth_img is not None else None
            ),
            "img_right": (
                NumpyB64Json.numpy_to_json(self.img_right) if self.img_right is not None else None
            ),
            "semantic_img": (
                NumpyB64Json.numpy_to_json(self.semantic_img)
                if self.semantic_img is not None
                else None
            ),
            "semantic_instances_img": (
                NumpyB64Json.numpy_to_json(self.semantic_instances_img)
                if self.semantic_instances_img is not None
                else None
            ),
        }
        return ret

    @staticmethod
    def from_json(json_str):
        camera = PinholeCamera.from_json(json_str["camera"])
        pose = deserialize_array_flexible(json_str["pose"], dtype=np.float64)

        frame_data_dict = {}
        frame_data_dict["is_keyframe"] = json_str["is_keyframe"]
        frame_data_dict["median_depth"] = json_str["median_depth"]
        frame_data_dict["fov_center_c"] = vector3_deserialize(json_str.get("fov_center_c", None))
        frame_data_dict["fov_center_w"] = vector3_deserialize(json_str.get("fov_center_w", None))

        try:
            frame_data_dict["is_blurry"] = json_str["is_blurry"]
        except:
            frame_data_dict["is_blurry"] = False
        try:
            frame_data_dict["laplacian_var"] = json_str["laplacian_var"]
        except:
            frame_data_dict["laplacian_var"] = None

        frame_data_dict["kps"] = deserialize_array_flexible(json_str["kps"], dtype=np.float32)
        frame_data_dict["kps_r"] = deserialize_array_flexible(json_str["kps_r"], dtype=np.float32)
        frame_data_dict["kpsu"] = deserialize_array_flexible(json_str["kpsu"], dtype=np.float32)
        frame_data_dict["kpsn"] = deserialize_array_flexible(json_str["kpsn"], dtype=np.float32)
        frame_data_dict["kps_sem"] = deserialize_array_flexible(
            json_str["kps_sem"], dtype=np.float32
        )
        frame_data_dict["octaves"] = deserialize_array_flexible(json_str["octaves"], dtype=np.uint8)
        frame_data_dict["octaves_r"] = deserialize_array_flexible(
            json_str["octaves_r"], dtype=np.uint8
        )
        frame_data_dict["sizes"] = deserialize_array_flexible(json_str["sizes"], dtype=np.float32)
        frame_data_dict["angles"] = deserialize_array_flexible(json_str["angles"], dtype=np.float32)

        # Descriptors - direct dict format (NumpyB64Json)
        # Normalize to 2D shape (N, D) for frames to prevent shape mismatches
        frame_data_dict["des"] = (
            NumpyB64Json.json_to_numpy_descriptor(json_str["des"], expected_ndim=2)
            if json_str["des"] is not None
            else None
        )
        frame_data_dict["des_r"] = (
            NumpyB64Json.json_to_numpy_descriptor(json_str["des_r"], expected_ndim=2)
            if json_str["des_r"] is not None
            else None
        )

        frame_data_dict["depths"] = deserialize_array_flexible(json_str["depths"], dtype=np.float32)
        frame_data_dict["kps_ur"] = deserialize_array_flexible(json_str["kps_ur"], dtype=np.float32)

        pts_val = json_str["points"]
        if pts_val is None:
            frame_data_dict["points"] = None
        else:
            # Convert -1 to None for compatibility with C++ serialization
            # C++ uses -1 for null map points, Python uses None
            pts_converted = [None if (p is None or p == -1) else p for p in pts_val]
            # Convert to numpy array with object dtype to preserve None values
            frame_data_dict["points"] = np.array(pts_converted, dtype=object)

        frame_data_dict["outliers"] = (
            np.array(json_str["outliers"], dtype=bool) if json_str["outliers"] is not None else None
        )
        # Handle kf_ref: C++ uses -1 for null, Python uses None
        kf_ref_val = json_str.get("kf_ref", None)
        frame_data_dict["kf_ref"] = None if (kf_ref_val is None or kf_ref_val == -1) else kf_ref_val

        # Images - direct dict format (NumpyB64Json)
        frame_data_dict["img"] = (
            NumpyB64Json.json_to_numpy(json_str["img"]) if json_str["img"] is not None else None
        )
        frame_data_dict["depth_img"] = (
            NumpyB64Json.json_to_numpy(json_str["depth_img"])
            if json_str["depth_img"] is not None
            else None
        )
        frame_data_dict["img_right"] = (
            NumpyB64Json.json_to_numpy(json_str["img_right"])
            if json_str["img_right"] is not None
            else None
        )
        frame_data_dict["semantic_img"] = (
            NumpyB64Json.json_to_numpy(json_str["semantic_img"])
            if json_str["semantic_img"] is not None
            else None
        )
        frame_data_dict["semantic_instances_img"] = (
            NumpyB64Json.json_to_numpy(json_str["semantic_instances_img"])
            if json_str["semantic_instances_img"] is not None
            else None
        )

        if "kps" in frame_data_dict and "points" in frame_data_dict:
            if len(frame_data_dict["kps"]) != len(frame_data_dict["points"]):
                Printer.red(
                    f"Frame.from_json: kps: {frame_data_dict['kps']}, points: {frame_data_dict['points']}"
                )
                assert False

        f = Frame(
            camera=camera,
            img=None,
            img_right=None,
            depth=None,
            semantic_img=None,
            pose=pose,
            id=json_str["id"],
            timestamp=json_str["timestamp"],
            img_id=json_str["img_id"],
            frame_data_dict=frame_data_dict,
        )

        return f

    # post processing after deserialization to replace saved ids with reloaded objects
    def replace_ids_with_objects(self, points, frames, keyframes):
        # Pre-build dictionaries for faster lookups
        points_dict = {obj.id: obj for obj in points if obj is not None}
        keyframes_dict = {obj.id: obj for obj in keyframes if obj is not None}

        def get_object_with_id(id, lookup_dict):
            if id is None or id == -1:  # Handle -1 from C++ serialization
                return None
            return lookup_dict.get(id, None)

        # Get actual points - replace IDs with actual map point objects
        if self.points is not None and len(self.points) > 0:
            # Replace IDs with actual map point objects, preserving None values
            self.points = np.array(
                [get_object_with_id(id, points_dict) for id in self.points], dtype=object
            )

        # Get actual kf_ref (handle -1 from C++ serialization)
        if self.kf_ref is not None and self.kf_ref != -1:
            self.kf_ref = get_object_with_id(self.kf_ref, keyframes_dict)
        else:
            self.kf_ref = None

    # KD tree of undistorted keypoints
    @property
    def kd(self):
        if self._kd is None:
            self._kd = cKDTree(self.kpsu)
        return self._kd

    def delete(self):
        with self._lock_pose:
            with self._lock_features:
                del self

    def get_point_match(self, idx):
        with self._lock_features:
            return self.points[idx]

    def set_point_match(self, p: "MapPoint", idx: int):
        with self._lock_features:
            self.points[idx] = p

    def remove_point_match(self, idx: int):
        with self._lock_features:
            self.points[idx] = None

    def replace_point_match(self, p: "MapPoint", idx: int):
        self.points[idx] = p  # replacing is not critical (it does not create a 'None jump')

    def remove_point(self, p: "MapPoint"):
        with self._lock_features:
            if self.points is not None:
                mask = np.array([point == p for point in self.points])
                if np.any(mask):
                    self.points[mask] = None
                # idx_o = p.get_observation_idx(self)
                # if idx_o >= 0:
                #     self.points[idx_o] = None
                # idx_f = p.get_frame_view_idx(self)
                # if idx_f >= 0:
                #     self.points[idx_f] = None

    def remove_frame_views(self, idxs):
        if len(idxs) == 0:
            return

        # Collect frame views to be removed while holding the lock
        frame_views_to_remove = []
        with self._lock_features:
            for idx, p in zip(idxs, self.points[idxs]):
                if p is not None:
                    frame_views_to_remove.append((p, idx))

        # Make external calls outside of lock context
        for p, idx in frame_views_to_remove:
            p.remove_frame_view(self, idx)

    def reset_points(self):
        with self._lock_features:
            num_keypoints = len(self.kps)
            self.points = np.full(num_keypoints, None, dtype=object)
            self.outliers = np.full(num_keypoints, False, dtype=bool)

    def get_points(self) -> list["MapPoint"] | None:
        with self._lock_features:
            return np.ascontiguousarray(self.points.copy()) if self.points is not None else None

    def get_matched_points(self):
        with self._lock_features:
            matched_idxs = np.flatnonzero(self.points != None)
            matched_points = self.points[matched_idxs]
            return matched_points  # , matched_idxs

    def get_matched_points_idxs(self):
        with self._lock_features:
            return np.flatnonzero(self.points != None)

    def get_unmatched_points_idxs(self):
        with self._lock_features:
            unmatched_idxs = np.flatnonzero(self.points == None)
            return unmatched_idxs

    def get_matched_inlier_points(self):
        with self._lock_features:
            matched_idxs = np.flatnonzero((self.points != None) & (self.outliers == False))
            matched_points = self.points[matched_idxs]
            return matched_points, matched_idxs

    def get_matched_good_points(self):
        with self._lock_features:
            good_points = (
                [p for p in self.points if p is not None and not p.is_bad()]
                if self.points is not None
                else []
            )
            return np.asarray(good_points)

    def get_matched_good_points_and_idxs(self):
        with self._lock_features:
            return np.asarray(
                [(p, i) for i, p in enumerate(self.points) if p is not None and not p.is_bad()]
            )

    # without locking since used in the tracking thread
    def get_matched_good_points_idxs(self):
        return np.asarray(
            [i for i, p in enumerate(self.points) if p is not None and not p.is_bad()], dtype=int
        )

    def num_tracked_points(self, minObs=1):
        with self._lock_features:
            # num_points = 0
            # for i,p in enumerate(self.points):
            #     if p is not None and not p.is_bad():
            #         if p.num_observations() >= minObs:
            #             num_points += 1
            # return num_points
            return sum(1 for p in self.points if p is not None and p.is_good_with_min_obs(minObs))

    def num_matched_inlier_map_points(self):
        with self._lock_features:
            # num_matched_points = 0
            # for i,p in enumerate(self.points):
            #     if p is not None and not self.outliers[i]:
            #         if p.num_observations() > 0:
            #             num_matched_points += 1
            # return num_matched_points
            return sum(
                1
                for i, p in enumerate(self.points)
                if p is not None and not self.outliers[i] and p.num_observations() > 0
            )

    # without locking since used in the tracking thread
    def get_tracked_mask(self):
        # Create a mask for tracked points (not None and not an outlier)
        points_mask = np.array([p is not None for p in self.points])
        outliers_mask = ~np.array(self.outliers)
        tracked_mask = points_mask & outliers_mask
        return tracked_mask

    # update found count for map points
    def update_map_points_statistics(self, sensor_type=SensorType.MONOCULAR):
        num_matched_inlier_points = 0
        with self._lock_features:
            for i, p in enumerate(self.points):
                if p is not None:
                    if not self.outliers[i]:
                        p.increase_found()  # update point statistics
                        if p.num_observations() > 0:
                            num_matched_inlier_points += 1
                    elif sensor_type == SensorType.STEREO:
                        self.points[i] = None
        return num_matched_inlier_points

    # reset outliers detected in last pose optimization
    def clean_outlier_map_points(self):
        num_matched_points = 0
        assert len(self.outliers) == len(self.points), "len(self.outliers) == len(self.points)"

        # Collect pairs (point,idx) that define the frame views to be removed outside the lock
        frame_views_to_remove = []

        with self._lock_features:
            for i, p in enumerate(self.points):
                if p is not None:
                    if self.outliers[i]:
                        frame_views_to_remove.append((p, i))  # Collect pairs (point,idx)
                        p.last_frame_id_seen = self.id
                        self.points[i] = None
                        self.outliers[i] = False
                    else:
                        if p.num_observations() > 0:
                            num_matched_points += 1

        # Do the external calls outside the lock
        for p, i in frame_views_to_remove:
            p.remove_frame_view(self, i)

        return num_matched_points

    # reset bad map points and update visibility count
    def clean_bad_map_points(self):
        # Collect pairs (point,idx) that define the frame views to be removed outside the lock
        frame_views_to_remove = []

        with self._lock_features:
            for i, p in enumerate(self.points):
                if p is None:
                    continue
                if p.is_bad():
                    frame_views_to_remove.append((p, i))  # Collect pairs (point,idx)
                    self.points[i] = None
                    self.outliers[i] = False
                else:
                    p.last_frame_id_seen = self.id
                    p.increase_visible()

        # Do the external calls outside the lock
        for p, i in frame_views_to_remove:
            p.remove_frame_view(self, i)

    # clean VO matches
    def clean_vo_matches(self):
        with self._lock_features:
            # num_cleaned_points = 0
            for i, p in enumerate(self.points):
                if p and p.num_observations() < 1:
                    self.points[i] = None
                    self.outliers[i] = False
                    # num_cleaned_points += 1
            # print("#cleaned vo points: ", num_cleaned_points)

    # check for point replacements
    def check_replaced_map_points(self):
        with self._lock_features:
            num_replaced_points = 0
            for i, p in enumerate(self.points):
                if p is not None:
                    replacement = p.get_replacement()
                    if replacement is not None:
                        self.points[i] = replacement
                        num_replaced_points += 1
            print("#replaced points: ", num_replaced_points)

    def compute_stereo_from_rgbd(self, kps_data, depth):
        kps_int = np.ascontiguousarray(kps_data[:, :2], dtype=np.uint32)
        depth_values = depth[kps_int[:, 1], kps_int[:, 0]]  # Depth at keypoint locations (v, u)
        valid_depth_mask = (depth_values > kMinDepth) & np.isfinite(depth_values)
        self.depths = np.where(valid_depth_mask, depth_values, -1.0)
        safe_depth_values = np.where(valid_depth_mask, depth_values, 1.0)  # 1.0 avoids div-by-zero
        self.kps_ur = np.where(
            valid_depth_mask, self.kpsu[:, 0] - self.camera.bf / safe_depth_values, -1.0
        )
        if len(self.kps_ur) != len(self.kpsu) or len(self.kps_ur) != len(self.depths):
            raise Exception(
                "Frame.compute_stereo_from_rgbd: len(self.kps_ur) != len(self.kpsu) or len(self.kps_ur) != len(self.depths)"
            )
        # print(f'depth: {self.depths}, kps_ur: {self.kps_ur}')
        # compute median depth
        if Frame.is_compute_median_depth:
            self.median_depth = (
                np.median(depth_values[valid_depth_mask]) if np.any(valid_depth_mask) else 0
            )
            self.fov_center_c = self.camera.unproject_3d(
                self.camera.cx, self.camera.cy, self.median_depth
            )
            self.fov_center_w = self._pose.Rwc @ self.fov_center_c + self._pose.Ow

    def compute_stereo_matches(self, img, img_right):
        min_z = self.camera.b
        min_disparity = 0
        max_disparity = self.camera.bf / min_z
        # we enforce matching on the same row here by using the flag row_matching (epipolar constraint)
        row_matching = True
        ratio_test = 0.9
        stereo_matching_result = FeatureTrackerShared.feature_matcher.match(
            img,
            img_right,
            des1=self.des,
            des2=self.des_r,
            kps1=self.kps,
            kps2=self.kps_r,
            ratio_test=ratio_test,
            row_matching=row_matching,
            max_disparity=max_disparity,
        )
        if len(stereo_matching_result.idxs1) == 0 or len(stereo_matching_result.idxs2) == 0:
            Printer.yellow(f"[compute_stereo_matches] no stereo matches found")
            return
        matched_kps_l = np.array(self.kps[stereo_matching_result.idxs1], dtype=float)
        matched_kps_r = np.array(self.kps_r[stereo_matching_result.idxs2], dtype=float)

        # check disparity range
        disparities = np.array(
            matched_kps_l[:, 0] - matched_kps_r[:, 0], dtype=float
        )  # assuming keypoints are extracted from rectified images
        good_disparities_mask = np.logical_and(
            disparities > min_disparity, disparities < max_disparity
        )
        good_disparities = disparities[good_disparities_mask]
        good_matched_idxs1 = stereo_matching_result.idxs1[good_disparities_mask]
        good_matched_idxs2 = stereo_matching_result.idxs2[good_disparities_mask]

        # compute fundamental matrix and check inliers, just for the hell of it (debugging) ... this is totally redundant here due to the row_matching
        do_check_with_fundamental_mat = False
        if do_check_with_fundamental_mat:
            ransac_method = None
            try:
                ransac_method = cv2.USAC_MAGSAC
            except:
                ransac_method = cv2.RANSAC
            fmat_err_thld = 3.0  # threshold for fundamental matrix estimation: maximum allowed distance from a point to an epipolar line in pixels (to treat a point pair as an inlier)
            F, mask_inliers = cv2.findFundamentalMat(
                self.kps[good_matched_idxs1],
                self.kps_r[good_matched_idxs2],
                ransac_method,
                fmat_err_thld,
                confidence=0.999,
            )
            mask_inliers = mask_inliers.flatten() == 1
            print(
                f"[compute_stereo_matches] perc good fundamental-matrix matches: {100 * np.sum(mask_inliers) / len(good_matched_idxs1)}"
            )
            good_disparities = good_disparities[mask_inliers]
            good_matched_idxs1 = good_matched_idxs1[mask_inliers]
            good_matched_idxs2 = good_matched_idxs2[mask_inliers]

        img_bw_ = None
        img_right_ = None

        # subpixel stereo matching
        do_subpixel_stereo_matching = True
        if do_subpixel_stereo_matching:
            if img.ndim > 2:
                img_bw_ = (
                    np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
                    if img_bw_ is None
                    else img_bw_
                )
            if img_right.ndim > 2:
                img_right_ = (
                    np.ascontiguousarray(cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY))
                    if img_right_ is None
                    else img_right_
                )
            refined_disparities, refined_us_right, valid_idxs = stereo_match_subpixel_correlation(
                self.kps[good_matched_idxs1],
                self.kps_r[good_matched_idxs2],
                min_disparity=min_disparity,
                max_disparity=max_disparity,
                image_left=img_bw_,
                image_right=img_right_,
            )
            good_disparities = refined_disparities[valid_idxs]
            good_matched_idxs1 = good_matched_idxs1[valid_idxs]
            good_matched_idxs2 = good_matched_idxs2[valid_idxs]
            self.kps_ur[good_matched_idxs1] = refined_us_right[valid_idxs]

        # check normalized sum of absolute differences at matched points (at level 0)
        do_check_sads = False
        if (
            do_check_sads and not do_subpixel_stereo_matching
        ):  # this is redundant if we did subpixel stereo matching
            window_size = 5
            # TODO: optimize this conversions (probably we can store them in the class if this has not been done before)
            if img.ndim > 2:
                img_bw_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img_bw_ is None else img_bw_
            if img_right.ndim > 2:
                img_right_ = (
                    cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)
                    if img_right_ is None
                    else img_right_
                )
            sads = compute_NSAD_between_matched_keypoints(
                img_bw_,
                img_right_,
                self.kps[good_matched_idxs1],
                self.kps_r[good_matched_idxs2],
                window_size,
            )
            # print(f'sads: {sads}')
            sads_median = np.median(sads)  # MAD, approximating dists_median=0
            sigma_mad = 1.4826 * sads_median
            good_sads_mask = sads < 1.5 * sigma_mad
            print(
                f"[compute_stereo_matches] perc good sads: {100 * np.sum(good_sads_mask) / len(sads)}"
            )
            good_disparities = good_disparities[good_sads_mask]
            good_matched_idxs1 = good_matched_idxs1[good_sads_mask]
            good_matched_idxs2 = good_matched_idxs2[good_sads_mask]

        # check chi2 of reprojections errors, just for the hell of it (debugging)
        do_chi2_check = False
        if do_chi2_check:
            # triangulate the points
            pose_l = np.eye(4)
            t_rl = np.array([-self.camera.b, 0, 0], dtype=float)
            pose_rl = poseRt(np.eye(3), t_rl)  # assuming stereo rectified images
            # print(f'pose_l: {pose_l}, pose_lr: {pose_lr}')
            kpsn_r = self.camera.unproject_points(
                self.kps_r
            )  # assuming rectified stereo images and so kps_r is undistorted

            pts3d, mask_pts3d = triangulate_normalized_points(
                pose_l, pose_rl, self.kpsn[good_matched_idxs1], kpsn_r[good_matched_idxs2]
            )

            pts3d = pts3d[mask_pts3d]
            good_disparities = good_disparities[mask_pts3d]
            good_matched_idxs1 = good_matched_idxs1[mask_pts3d]
            good_matched_idxs2 = good_matched_idxs2[mask_pts3d]

            # check reprojection errors

            uvs_l, depths_l = self.camera.project(pts3d)  # left projections
            good_depths_l = depths_l > 0

            pts3d_r = pts3d + t_rl
            uvs_r, depths_r = self.camera.project(pts3d_r)  # right projections
            good_depths_r = depths_r > 0

            good_depths_mask = np.logical_and(good_depths_l, good_depths_r)
            uvs_l = uvs_l[good_depths_mask][:]
            uvs_r = uvs_r[good_depths_mask][:]
            good_disparities = good_disparities[good_depths_mask]
            good_matched_idxs1 = good_matched_idxs1[good_depths_mask]
            good_matched_idxs2 = good_matched_idxs2[good_depths_mask]

            # compute mono reproj errors on left image
            errs_l_vec = uvs_l - self.kps[good_matched_idxs1]

            errs_l_sqr = np.sum(errs_l_vec * errs_l_vec, axis=1)  # squared reprojection errors
            kpsl_levels = self.octaves[good_matched_idxs1]
            invSigmas2_1 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kpsl_levels]
            chis2_l = errs_l_sqr * invSigmas2_1  # chi-squared
            # print(f'chis2_l: {chis2_l}')
            good_chi2_l_mask = chis2_l < Parameters.kChi2Mono
            num_good_chi2_l = good_chi2_l_mask.sum()
            # print(f'perc good chis2_l: {100*num_good_chi2_l/len(good_chi2_l_mask)}')

            errs_r_vec = uvs_r - self.kps_r[good_matched_idxs2]
            errs_r_sqr = np.sum(errs_r_vec * errs_r_vec, axis=1)  # squared reprojection errors
            kpsr_levels = self.octaves_r[good_matched_idxs2]
            invSigmas2_2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kpsr_levels]
            chis2_r = errs_r_sqr * invSigmas2_2  # chi-squared
            # print(f'chis2_r: {chis2_r}')
            good_chi2_r_mask = chis2_r < Parameters.kChi2Mono
            num_good_chi2_r = good_chi2_r_mask.sum()
            # print(f'perc good chis2_r: {100*num_good_chi2_r/len(good_chi2_r_mask)}')

            good_chis2_mask = np.logical_and(
                chis2_l < Parameters.kChi2Mono, chis2_r < Parameters.kChi2Mono
            )
            num_good_chis2 = good_chis2_mask.sum()
            print(
                f"[compute_stereo_matches] perc good chis2: {100*num_good_chis2/len(good_matched_idxs1)}"
            )
            good_disparities = good_disparities[good_chis2_mask]
            good_matched_idxs1 = good_matched_idxs1[good_chis2_mask]
            good_matched_idxs2 = good_matched_idxs2[good_chis2_mask]

        # filter descriptor distances with sigma-mad
        do_check_with_des_distances_sigma_mad = False
        if do_check_with_des_distances_sigma_mad:
            des1 = stereo_matching_result.des1[good_matched_idxs1]
            des2 = stereo_matching_result.des2[good_matched_idxs2]
            if Parameters.kUseDescriptorSigmaMadv2:
                sigma_mad, dists_median, des_dists = descriptor_sigma_mad_v2(
                    des1, des2, descriptor_distances=FeatureTrackerShared.descriptor_distances
                )
                good_des_dists_mask = des_dists < 1.5 * sigma_mad + dists_median
            else:
                sigma_mad, dists_median, des_dists = descriptor_sigma_mad(
                    des1, des2, descriptor_distances=FeatureTrackerShared.descriptor_distances
                )
                good_des_dists_mask = des_dists < 1.5 * sigma_mad
            num_good_des_dists = good_des_dists_mask.sum()
            print(
                f"[compute_stereo_matches] perc good des distances: {100*num_good_des_dists/len(good_des_dists_mask)}"
            )
            good_disparities = good_disparities[good_des_dists_mask]
            good_matched_idxs1 = good_matched_idxs1[good_des_dists_mask]
            good_matched_idxs2 = good_matched_idxs2[good_des_dists_mask]

        # update depths and kps_ur
        self.depths[good_matched_idxs1] = self.camera.bf * np.reciprocal(
            good_disparities.astype(float)
        )
        if not do_subpixel_stereo_matching:
            self.kps_ur[good_matched_idxs1] = self.kps_r[good_matched_idxs2][:, 0]
            print(f"[compute_stereo_matches] found final {len(good_matched_idxs1)} stereo matches")

        if Frame.is_compute_median_depth:
            valid_dephts_mask = self.depths > kMinDepth
            self.median_depth = np.median(self.depths[valid_dephts_mask])
            self.fov_center_c = self.camera.unproject_3d(
                self.camera.cx, self.camera.cy, self.median_depth
            )
            self.fov_center_w = self._pose.Rwc @ self.fov_center_c + self._pose.Ow
            print(f"[compute_stereo_matches] median depth: {self.median_depth}")

        if Parameters.kStereoMatchingShowMatchedPoints:  # debug stereo matching
            # print(f'[compute_stereo_matches] found intial {len(good_matched_idxs1)} stereo matches')
            stereo_img_matches = draw_feature_matches(
                img,
                img_right,
                self.kps[good_matched_idxs1],
                self.kps_r[good_matched_idxs2],
                horizontal=False,
            )
            # cv2.namedWindow('stereo_img_matches', cv2.WINDOW_NORMAL)
            cv2.imshow("stereo_img_matches", stereo_img_matches)
            cv2.waitKey(1)

        if False:
            from pyslam.viz.rerun_interface import Rerun
            import rerun as rr  # pip install rerun-sdk

            if not Rerun.is_initialized:
                Rerun.init()
            points, _ = self.unproject_points_3d(good_matched_idxs1, transform_in_world=False)
            # print(f'[compute_stereo_matches] #points 3D: {len(points)}')
            rr.log("points", rr.Points3D(points))

    # unproject keypoints where the depth is available
    def unproject_points_3d(self, idxs, transform_in_world=False):
        if self.depths is not None:
            depth_values = self.depths[idxs][:, np.newaxis]
            valid_depth_mask = depth_values > kMinDepth
            kpsn = add_ones(self.kpsn[idxs])
            pts3d_mask = valid_depth_mask.flatten()
            pts3d = kpsn * depth_values
            pts3d[~pts3d_mask] = 0  # set invalid points to 0
            if transform_in_world:
                # print(f'unproject_points_3d: Rwc: {self._pose.Rwc}, Ow: {self._pose.Ow}')
                pts3d = (self._pose.Rwc @ pts3d.T + self._pose.Ow[:, np.newaxis]).T
            return pts3d, pts3d_mask
        else:
            return None, None

    def compute_points_median_depth(self, points3d=None, percentile=0.5):
        with self._lock_pose:
            Rcw2 = self._pose.Rcw[2, :3]  # just 2-nd row
            tcw2 = self._pose.tcw[2]  # just 2-nd row
        if points3d is None:
            with self._lock_features:
                points3d = np.array([p.pt() for p in self.points if p is not None])
        if len(points3d) > 0:
            z = np.dot(Rcw2, points3d[:, :3].T) + tcw2
            z = np.sort(z)
            idx = min(int(len(z) * percentile), len(z) - 1)
            return z[idx]
        else:
            Printer.red("frame.compute_points_median_depth() with no points")
            return -1

    # draw tracked features on the image for selected keypoint indexes
    def draw_feature_trails_without_radius_(
        self,
        img: np.ndarray,
        kps_idxs: list[int],
        with_level_radius: bool = False,
        trail_max_length: int = 16,
    ):
        img = img.copy()

        color_table_generator = ColorTableGenerator.instance()
        with self._lock_features:
            uvs = np.rint(self.kps[kps_idxs]).astype(
                np.intp
            )  # image keypoints coordinates  # use distorted coordinates when drawing on distorted original image
            # for each keypoint idx
            for i, kp_idx in enumerate(kps_idxs):
                # u1, v1 = int(round(self.kps[kp_idx][0])), int(round(self.kps[kp_idx][1]))  # use distorted coordinates when drawing on distorted original image
                uv = tuple(uvs[i])

                point = self.points[kp_idx]  # MapPoint
                if point is not None and not point.is_bad():
                    # radius = self.sizes[kp_idx] # actual size
                    # radius = kDrawFeatureRadius[self.octaves[kp_idx]]  # fake size for visualization
                    p_frame_views = point.frame_views()  # list of (Frame, idx)
                    if p_frame_views:
                        # there's a corresponding 3D map point
                        color = color_table_generator.color_from_int_without_hash(point.id)
                        cv2.circle(
                            img, uv, color=color, radius=4, thickness=-1
                        )  # draw keypoint size as a circle
                        # draw the trail (for each keypoint, its trail_max_length corresponding points in previous frames)
                        pts = []
                        lfid = None  # last frame id
                        for f, idx in reversed(p_frame_views[-trail_max_length:]):
                            if f is None:
                                continue
                            if lfid is not None and lfid - 1 != f.id:
                                # stop when there is a jump in the ids of frame observations
                                break
                            pts.append(tuple(f.kps[idx].astype(int)))
                            lfid = f.id
                        if len(pts) > 1:
                            cv2.polylines(
                                img,
                                np.array([pts], dtype=np.int32),
                                False,
                                color,
                                thickness=1,
                                lineType=16,
                            )
                else:
                    # no corresponding 3D point
                    cv2.circle(img, uv, color=(0, 0, 0), radius=2)  # radius=radius)
            return img

    # draw tracked features on the image for selected keypoint indexes
    def draw_feature_trails_with_radius_(
        self,
        img: np.ndarray,
        kps_idxs: list[int],
        with_level_radius: bool = False,
        trail_max_length: int = 16,
    ):
        img = img.copy()
        with self._lock_features:
            uvs = np.rint(self.kps[kps_idxs]).astype(
                np.intp
            )  # image keypoints coordinates  # use distorted coordinates when drawing on distorted original image
            # for each keypoint idx
            for i, kp_idx in enumerate(kps_idxs):
                # u1, v1 = int(round(self.kps[kp_idx][0])), int(round(self.kps[kp_idx][1]))  # use distorted coordinates when drawing on distorted original image
                uv = tuple(uvs[i])

                # color = Colors.myjet_color_x_255(self.octaves[i1])
                point = self.points[kp_idx]  # MapPoint
                if point is not None and not point.is_bad():
                    # radius = self.sizes[kp_idx] # actual size
                    radius = kDrawFeatureRadius[self.octaves[kp_idx]]  # fake size for visualization
                    p_frame_views = point.frame_views()  # list of (Frame, idx)
                    if p_frame_views:
                        # there's a corresponding 3D map point
                        color = (0, 255, 0) if len(p_frame_views) > 2 else (255, 0, 0)
                        cv2.circle(
                            img, uv, color=color, radius=radius, thickness=1
                        )  # draw keypoint size as a circle
                        # draw the trail (for each keypoint, its trail_max_length corresponding points in previous frames)
                        pts = []
                        lfid = None  # last frame id
                        for f, idx in reversed(p_frame_views[-trail_max_length:]):
                            if f is None:
                                continue
                            if lfid is not None and lfid - 1 != f.id:
                                # stop when there is a jump in the ids of frame observations
                                break
                            pts.append(tuple(f.kps[idx].astype(int)))
                            lfid = f.id
                        num_pts = len(pts)
                        if num_pts > 1:
                            color = Colors.myjet_color_x_255(num_pts)
                            cv2.polylines(
                                img,
                                np.array([pts], dtype=np.int32),
                                False,
                                color,
                                thickness=1,
                                lineType=16,
                            )
                else:
                    # no corresponding 3D point
                    cv2.circle(img, uv, color=(0, 0, 0), radius=2)  # radius=radius)
            return img

    # draw tracked features on the image
    def draw_all_feature_trails(
        self, img: np.ndarray, with_level_radius: bool = False, trail_max_length: int = 16
    ):
        kps_idxs = range(len(self.kps))
        if with_level_radius:
            return self.draw_feature_trails_with_radius_(
                img, kps_idxs, trail_max_length=trail_max_length
            )
        else:
            return self.draw_feature_trails_without_radius_(
                img, kps_idxs, trail_max_length=trail_max_length
            )


####################################################################################
#  Frame helper functions
####################################################################################


# input: a list of map points,
#        a target frame,
#        a suggested se3 transformation Rcw, tcw for target frame
# output: [Nx1] array of visibility flags, (target map points from 1 on 2)
#         [Nx1] array of depths,
#         [Nx1] array of distances PO
# check a) map points are in image b) good view angle c) good distance range
def are_map_points_visible_in_frame(
    map_points: list["MapPoint"], frame: Frame, Rcw: np.ndarray, tcw: np.ndarray
):
    # similar to frame.are_visible()
    n = len(map_points)
    if n == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )  # Return empty arrays if no map points

    points_w = np.zeros((n, 3), dtype=np.float64)
    normals = np.zeros((n, 3), dtype=np.float32)
    min_dists = np.zeros(n, dtype=np.float32)
    max_dists = np.zeros(n, dtype=np.float32)
    for i, p in enumerate(map_points):
        pt, normal, min_dist, max_dist = p.get_all_pos_info()
        points_w[i] = pt
        normals[i] = normal  # in world frame
        min_dists[i] = min_dist
        max_dists[i] = max_dist

    points_c = (Rcw @ points_w.T + tcw.reshape((3, 1))).T  # in camera 1
    uvs, zs = frame.camera.project(points_c)  # project on camera 2

    Ow = (-Rcw.T @ tcw.reshape((3, 1))).T
    POs = points_w - Ow
    dists = np.linalg.norm(POs, axis=-1, keepdims=True)
    POs /= dists
    cos_view = np.sum(normals * POs, axis=1)

    are_in_image = frame.are_in_image(uvs, zs)
    are_in_good_view_angle = cos_view > Parameters.kViewingCosLimitForPoint
    dists = dists.reshape(
        -1,
    )
    are_in_good_distance = (dists > min_dists) & (dists < max_dists)

    out_flags = are_in_image & are_in_good_view_angle & are_in_good_distance
    return out_flags, uvs, zs, dists


# input: two frames frame1 and frame2,
#        a list of target map points of frame1,
#        a suggested se3 or sim3 transformation sR21, t21 between the frames
# output: [Nx1] array of visibility flags, (target map points from 1 on 2)
#         [Nx1] array of depths,
#         [Nx1] array of distances PO
# check a) map points are in image b) good view angle c) good distance range
def are_map_points_visible(
    frame1: Frame, frame2: Frame, map_points1, sR21: np.ndarray, t21: np.ndarray
):
    # similar to frame.are_visible()
    n = len(map_points1)
    if n == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )  # Return empty arrays if no map points

    points_w = np.zeros((n, 3), dtype=np.float64)
    point_normals = np.zeros((n, 3), dtype=np.float32)
    min_dists = np.zeros(n, dtype=np.float32)
    max_dists = np.zeros(n, dtype=np.float32)
    for i, p in enumerate(map_points1):
        pt, normal, min_dist, max_dist = p.get_all_pos_info()
        points_w[i] = pt
        point_normals[i] = normal  # in world frame
        min_dists[i] = min_dist
        max_dists[i] = max_dist

    points_c1 = frame1.transform_points(points_w)  # in camera 1
    points_c2 = (sR21 @ points_c1.T + t21.reshape((3, 1))).T  # in camera 2 by using input sim3

    uvs_2, zs_2 = frame2.camera.project(points_c2)  # project on camera 2

    # PO2s = points_w - frame2.Ow()
    # dists_2 = np.linalg.norm(PO2s, axis=-1, keepdims=True)
    dists_2 = np.linalg.norm(points_c2, axis=-1, keepdims=True)
    # PO2s /= dists_2
    # cos_view = np.sum(point_normals * PO2s, axis=1)

    are_in_image_2 = frame2.are_in_image(uvs_2, zs_2)
    # are_in_good_view_angle_2 = cos_view > Parameters.kViewingCosLimitForPoint
    dists_2 = dists_2.reshape(
        -1,
    )
    are_in_good_distance_2 = (dists_2 > min_dists) & (dists_2 < max_dists)

    # out_flags = are_in_image_2 & are_in_good_view_angle_2 & are_in_good_distance_2
    out_flags = are_in_image_2 & are_in_good_distance_2
    return out_flags, uvs_2, zs_2, dists_2


# match frames f1 and f2
# out: a vector of match index pairs [idx1[i],idx2[i]] such that the keypoint f1.kps[idx1[i]] is matched with f2.kps[idx2[i]]
def match_frames(f1: Frame, f2: Frame, ratio_test=None):
    matching_result = FeatureTrackerShared.feature_matcher.match(
        f1.img, f2.img, f1.des, f2.des, kps1=f1.kps, kps2=f2.kps, ratio_test=ratio_test
    )
    return matching_result


def compute_frame_matches_threading(
    target_frame: Frame,
    other_frames: list,
    match_idxs,
    max_workers=2,
    ratio_test=None,
    print_fun=None,
):
    # do parallell computation using multithreading
    def thread_match_function(kf_pair):
        kf1, kf2 = kf_pair
        matching_result = FeatureTrackerShared.feature_matcher.match(
            kf1.img, kf2.img, kf1.des, kf2.des, kps1=kf1.kps, kps2=kf2.kps, ratio_test=ratio_test
        )
        idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2
        match_idxs[(kf1, kf2)] = (np.array(idxs1), np.array(idxs2))

    kf_pairs = [
        (target_frame, kf) for kf in other_frames if kf is not target_frame and not kf.is_bad()
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            thread_match_function, kf_pairs
        )  # automatic join() at the end of the `width` block
    return match_idxs


# compute matches between target frame and other frames
# out:
#   match_idxs: dictionary of keypoint matches  (kf_i, kf_j) -> (idxs_i,idxs_j)
def compute_frame_matches(
    target_frame: Frame,
    other_frames: list,
    match_idxs,
    do_parallel=True,
    max_workers=2,
    ratio_test=None,
    print_fun=None,
):
    timer = Timer()
    # do_parallel = False # force disable
    if not do_parallel:
        # Do serial computation
        for kf in other_frames:
            if kf is target_frame or kf.is_bad():
                continue
            matching_result = FeatureTrackerShared.feature_matcher.match(
                target_frame.img,
                kf.img,
                target_frame.des,
                kf.des,
                kps1=target_frame.kps,
                kps2=kf.kps,
                ratio_test=ratio_test,
            )
            idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2
            match_idxs[(target_frame, kf)] = (idxs1, idxs2)
    else:
        match_idxs = compute_frame_matches_threading(
            target_frame, other_frames, match_idxs, max_workers, ratio_test, print_fun
        )
    if print_fun is not None:
        print_fun(
            f"compute_frame_matches: #compared pairs: {len(match_idxs)}, do_parallel: {do_parallel}, timing: {timer.elapsed()}"
        )
    return match_idxs


# prepare input data for sim3 solver
# in:
#   f1, f2: two frames
#   idxs1, idxs2: indices of matches between f1 and f2
# out:
#   points_3d_1, points_3d_2: 3D points in f1 and f2
#   sigmas2_1, sigmas2_2: feature sigmas in f1 and f2
#   idxs1_out, idxs2_out: indices of good point matches in f1 and f2
def prepare_input_data_for_sim3solver(f1: Frame, f2: Frame, idxs1, idxs2):
    level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2
    points_3d_1 = []
    points_3d_2 = []
    sigmas2_1 = []
    sigmas2_2 = []
    idxs1_out = []
    idxs2_out = []
    # get matches for current keyframe and candidate keyframes
    for i1, i2 in zip(idxs1, idxs2):
        p1 = f1.get_point_match(i1)
        if p1 is None or p1.is_bad():
            continue
        p2 = f2.get_point_match(i2)
        if p2 is None or p2.is_bad():
            continue
        points_3d_1.append(p1.pt())
        points_3d_2.append(p2.pt())
        sigmas2_1.append(level_sigmas2[f1.octaves[i1]])
        sigmas2_2.append(level_sigmas2[f2.octaves[i2]])
        idxs1_out.append(i1)
        idxs2_out.append(i2)
    return (
        np.ascontiguousarray(points_3d_1),
        np.ascontiguousarray(points_3d_2),
        np.ascontiguousarray(sigmas2_1),
        np.ascontiguousarray(sigmas2_2),
        np.ascontiguousarray(idxs1_out),
        np.ascontiguousarray(idxs2_out),
    )


# prepare input data for pnp solver
# in:
#   f1, f2: two frames
#   idxs1, idxs2: indices of matches between f1 and f2
# out:
#   points_3d: 3D points in f2
#   points_2d: 2D points in f1
#   sigmas2, feature sigmas in f1
#   idxs1_out, idxs2_out: indices of good point matches in f1 and f2
def prepare_input_data_for_pnpsolver(f1: Frame, f2: Frame, idxs1, idxs2, print=print):
    level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2
    points_3d = []
    points_2d = []
    sigmas2 = []
    idxs1_out = []
    idxs2_out = []

    # get matches for current keyframe and candidate keyframes
    for i1, i2 in zip(idxs1, idxs2):
        p2 = f2.get_point_match(i2)
        if p2 is None or p2.is_bad():
            continue
        points_3d.append(p2.pt())
        points_2d.append(f1.kps[i1])
        sigmas2.append(level_sigmas2[f1.octaves[i1]])
        idxs1_out.append(i1)
        idxs2_out.append(i2)
    return (
        np.ascontiguousarray(points_3d),
        np.ascontiguousarray(points_2d),
        np.ascontiguousarray(sigmas2),
        np.ascontiguousarray(idxs1_out),
        np.ascontiguousarray(idxs2_out),
    )
