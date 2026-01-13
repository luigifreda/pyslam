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

import cv2
import numpy as np

# import json
import ujson as json

from scipy.spatial import cKDTree

from threading import Lock

from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Printer
from pyslam.utilities.serialization import extract_tcw_matrix_from_pose_data

from collections import defaultdict, OrderedDict, Counter

from .frame import Frame
from .camera_pose import CameraPose

import typing

if typing.TYPE_CHECKING:
    from .map_point import MapPoint
    from .keyframe import KeyFrame


class KeyFrameGraph(object):
    def __init__(self):
        self._lock_connections = Lock()
        # spanning tree
        self.init_parent = False  # is parent initialized?
        self.parent = None
        self.children = set()
        # loop edges
        self.loop_edges = set()
        self.not_to_erase = False  # if there is a loop edge then you cannot erase this keyframe
        # covisibility graph
        self.connected_keyframes_weights = Counter()  # defaultdict(int)
        self.ordered_keyframes_weights = (
            OrderedDict()
        )  # ordered list of connected keyframes (on the basis of the number of map points with this keyframe)
        #
        self.is_first_connection = True

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the Lock from the state (don't pickle it)
        if "_lock_connections" in state:
            del state["_lock_connections"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the Lock after unpickling
        self._lock_connections = Lock()

    def to_json(self):
        with self._lock_connections:
            return {
                "parent": self.parent.id if self.parent is not None else None,
                "children": [k.id for k in self.children],
                "loop_edges": [k.id for k in self.loop_edges],
                "init_parent": bool(
                    self.init_parent
                ),  # Missing field - critical for proper keyframe hierarchy
                "not_to_erase": bool(self.not_to_erase),
                "connected_keyframes_weights": [
                    (k.id, w) for k, w in self.connected_keyframes_weights.items() if k is not None
                ],
                "ordered_keyframes_weights": [
                    (k.id, w) for k, w in self.ordered_keyframes_weights.items() if k is not None
                ],
                "is_first_connection": bool(self.is_first_connection),
            }

    def init_from_json(self, json_str):
        with self._lock_connections:
            self.parent = json_str["parent"]
            self.children = json_str["children"]  # converted to set in replace_ids_with_objects()
            self.loop_edges = json_str[
                "loop_edges"
            ]  # converted to set in replace_ids_with_objects()
            self.init_parent = bool(json_str.get("init_parent", False))  # Restore init_parent flag
            self.not_to_erase = json_str["not_to_erase"]
            self.connected_keyframes_weights = json_str[
                "connected_keyframes_weights"
            ]  # converted to Counter in replace_ids_with_objects()
            self.ordered_keyframes_weights = json_str[
                "ordered_keyframes_weights"
            ]  # converted to OrderedDict in replace_ids_with_objects()
            self.is_first_connection = json_str["is_first_connection"]

    # post processing after deserialization to replace saved ids with reloaded objects
    def replace_ids_with_objects(self, points, frames, keyframes):
        # Pre-build a dictionary for efficient lookups
        keyframes_dict = {obj.id: obj for obj in keyframes if obj is not None}

        def get_object_with_id(id, lookup_dict):
            return lookup_dict.get(id, None)

        # Replace parent
        if self.parent is not None:
            self.parent = get_object_with_id(self.parent, keyframes_dict)
        # Replace children
        if self.children is not None:
            self.children = {get_object_with_id(id, keyframes_dict) for id in self.children}
        # Replace loop edges
        if self.loop_edges is not None:
            self.loop_edges = {get_object_with_id(id, keyframes_dict) for id in self.loop_edges}
        # Replace connected_keyframes_weights
        if self.connected_keyframes_weights is not None:
            self.connected_keyframes_weights = Counter(
                {
                    get_object_with_id(id, keyframes_dict): weight
                    for id, weight in self.connected_keyframes_weights
                }
            )
        # Replace ordered_keyframes_weights
        if self.ordered_keyframes_weights is not None:
            self.ordered_keyframes_weights = OrderedDict(
                {
                    get_object_with_id(id, keyframes_dict): weight
                    for id, weight in self.ordered_keyframes_weights
                }
            )

    # ===============================
    # spanning tree

    def add_child_no_lock_(self, keyframe):
        self.children.add(keyframe)

    def add_child(self, keyframe):
        with self._lock_connections:
            self.add_child_no_lock_(keyframe)

    def erase_child_no_lock_(self, keyframe):
        try:
            self.children.remove(keyframe)
        except:
            pass

    def erase_child(self, keyframe):
        with self._lock_connections:
            self.erase_child_no_lock_(keyframe)

    def set_parent_no_lock_(self, keyframe):
        self.parent = keyframe
        keyframe.add_child(self)

    def set_parent(self, keyframe):
        with self._lock_connections:
            if self == keyframe:
                if __debug__:
                    Printer.orange("KeyFrameGraph.set_parent - trying to set self as parent")
                return
            self.set_parent_no_lock_(keyframe)

    def get_children(self):
        with self._lock_connections:
            return self.children.copy()

    def get_parent(self):
        with self._lock_connections:
            return self.parent

    def has_child(self, keyframe):
        with self._lock_connections:
            return keyframe in self.children

    # ===============================
    # loop edges
    def add_loop_edge(self, keyframe):
        with self._lock_connections:
            self.not_to_erase = True
            if keyframe not in self.loop_edges:
                self.loop_edges.add(keyframe)

    def get_loop_edges(self):
        with self._lock_connections:
            return self.loop_edges.copy()

    # ===============================
    # covisibility

    def reset_covisibility(self):
        self.connected_keyframes_weights = Counter()
        self.ordered_keyframes_weights = OrderedDict()

    def update_best_covisibles_no_lock_(self):
        self.ordered_keyframes_weights = OrderedDict(
            sorted(self.connected_keyframes_weights.items(), key=lambda x: x[1], reverse=True)
        )  # order by value (decreasing order)

    def add_connection_no_lock_(self, keyframe, weight):
        self.connected_keyframes_weights[keyframe] = weight
        self.update_best_covisibles_no_lock_()

    def add_connection(self, keyframe, weight):
        with self._lock_connections:
            self.add_connection_no_lock_(keyframe, weight)

    def erase_connection_no_lock_(self, keyframe):
        try:
            del self.connected_keyframes_weights[keyframe]
            self.update_best_covisibles_no_lock_()
        except:
            pass

    def get_connected_keyframes_no_lock_(self):
        return list(self.connected_keyframes_weights.keys())  # returns a copy

    # get a list of all the keyframe that shares points
    def get_connected_keyframes(self):
        with self._lock_connections:
            return self.get_connected_keyframes_no_lock_()

    def get_covisible_keyframes_no_lock_(self):
        return list(self.ordered_keyframes_weights.keys())  # returns a copy

    # get an ordered list of covisible keyframes
    def get_covisible_keyframes(self):
        with self._lock_connections:
            return self.get_covisible_keyframes_no_lock_()

    # get an ordered list of covisible keyframes
    def get_best_covisible_keyframes(self, N):
        with self._lock_connections:
            return list(self.ordered_keyframes_weights.keys())[:N]  # returns a copy

    def get_covisible_by_weight(self, weight):
        with self._lock_connections:
            return [kf for kf, w in self.ordered_keyframes_weights.items() if w > weight]

    def get_weight_no_lock_(self, keyframe):
        return self.connected_keyframes_weights[keyframe]

    def get_weight(self, keyframe):
        with self._lock_connections:
            return self.get_weight_no_lock_(keyframe)

    def get_connected_keyframes_weights(self):
        """Get all connected keyframes with their weights in a thread-safe way.

        Returns:
            dict: Dictionary mapping keyframe_id to weight for all connected keyframes.
        """
        with self._lock_connections:
            return {
                kf.id: w for kf, w in self.connected_keyframes_weights.items() if kf is not None
            }


class KeyFrame(Frame, KeyFrameGraph):
    def __init__(self, frame: Frame, img=None, img_right=None, depth=None, kid=None):
        KeyFrameGraph.__init__(self)
        Frame.__init__(
            self,
            img=None,
            camera=frame.camera,
            pose=frame.pose(),
            id=frame.id,
            timestamp=frame.timestamp,
            img_id=frame.img_id,
        )  # here we MUST have img=None in order to avoid recomputing keypoint info

        if frame.img is not None:
            self.img = frame.img  # this is already a copy of an image
        else:
            if img is not None:
                self.img = img  # .copy()

        if frame.img_right is not None:
            self.img_right = frame.img_right
        else:
            if img_right is not None:
                self.img_right = img_right  # .copy()

        if frame.depth_img is not None:
            self.depth_img = frame.depth_img
        else:
            if depth is not None:
                self.set_depth_img(depth)

        self.map = None

        self.is_keyframe = True
        self.kid = kid  # keyframe id (keyframe counter-id, different from frame.id)

        self._is_bad = False
        self.to_be_erased = False

        self.lba_count = 0  # how many time this keyframe has adjusted by LBA

        self.is_blurry = frame.is_blurry
        self.laplacian_var = frame.laplacian_var

        # pose relative to parent: self.Tcw() @ self.parent.Twc() (this is computed when bad flag is activated)
        self._pose_Tcp = CameraPose()

        # share keypoints info with frame (these are computed once for all on frame initialization and they are not changed anymore)
        self.kps = frame.kps  # keypoint coordinates                  [Nx2]
        self.kpsu = frame.kpsu  # [u]ndistorted keypoint coordinates    [Nx2]
        self.kpsn = frame.kpsn  # [n]ormalized keypoint coordinates     [Nx2] (Kinv * [kp,1])
        self.kps_sem = (
            frame.kps_sem
        )  # [sem]antic keypoint information       [NxD] where D is the semantic information length
        self.octaves = frame.octaves  # keypoint octaves                      [Nx1]
        self.sizes = frame.sizes  # keypoint sizes                        [Nx1]
        self.angles = frame.angles  # keypoint angles                       [Nx1]
        self.des = (
            frame.des
        )  # keypoint descriptors                  [NxD] where D is the descriptor length
        self.depths = frame.depths  # keypoint depths                       [Nx1]
        self.kps_ur = frame.kps_ur  # right keypoint coordinates            [Nx1]

        self.median_depth = frame.median_depth
        self.fov_center_c = frame.fov_center_c
        self.fov_center_w = frame.fov_center_w

        # for loop closing
        self.g_des = None  # global (image-wise) descriptor for loop closing
        self.loop_query_id = None
        self.num_loop_words = 0
        self.loop_score = 0

        # for relocalization
        self.reloc_query_id = None
        self.num_reloc_words = 0
        self.reloc_score = 0

        # for GBA
        self.GBA_kf_id = 0
        self.is_Tcw_GBA_valid = False  # For easing C++ equivalent code
        self.Tcw_GBA = None
        self.Tcw_before_GBA = None

        if hasattr(frame, "_kd"):
            self._kd = frame._kd
        else:
            Printer.orange("KeyFrame %d computing kdtree for input frame %d" % (self.id, frame.id))
            self._kd = cKDTree(self.kpsu)

        # map points information arrays (copy points coming from frame)
        self.points = (
            frame.get_points()
        )  # map points => self.points[idx] is the map point matched with self.kps[idx] (if is not None)
        self.outliers = (
            np.full(self.kpsu.shape[0], False, dtype=bool) if self.kpsu is not None else None
        )  # used just in TrackingCore.propagate_map_point_matches()

    def to_json(self):
        frame_json = Frame.to_json(self)

        frame_json["is_keyframe"] = self.is_keyframe
        frame_json["kid"] = self.kid
        frame_json["_is_bad"] = self._is_bad
        frame_json["lba_count"] = self.lba_count
        frame_json["to_be_erased"] = self.to_be_erased
        frame_json["_pose_Tcp"] = (
            self._pose_Tcp.Tcw.astype(float).tolist() if self._pose_Tcp.Tcw is not None else None
        )
        frame_json["is_Tcw_GBA_valid"] = self.is_Tcw_GBA_valid

        # Loop closing and relocalization fields
        frame_json["loop_query_id"] = self.loop_query_id
        frame_json["num_loop_words"] = self.num_loop_words
        frame_json["loop_score"] = self.loop_score
        frame_json["reloc_query_id"] = self.reloc_query_id
        frame_json["num_reloc_words"] = self.num_reloc_words
        frame_json["reloc_score"] = self.reloc_score

        # GBA fields
        frame_json["GBA_kf_id"] = self.GBA_kf_id
        frame_json["Tcw_GBA"] = (
            self.Tcw_GBA.astype(float).tolist() if self.Tcw_GBA is not None else None
        )
        frame_json["Tcw_before_GBA"] = (
            self.Tcw_before_GBA.astype(float).tolist() if self.Tcw_before_GBA is not None else None
        )

        keyframe_graph_json = KeyFrameGraph.to_json(self)
        return {**frame_json, **keyframe_graph_json}

    @staticmethod
    def from_json(json_str):
        f = Frame.from_json(json_str)
        kf = KeyFrame(f)

        kf.is_keyframe = bool(json_str["is_keyframe"])
        kf.kid = json_str["kid"]
        kf._is_bad = bool(json_str["_is_bad"])
        kf.lba_count = int(json_str["lba_count"])
        kf.to_be_erased = bool(json_str["to_be_erased"])
        # Handle _pose_Tcp - extract Tcw matrix, then instantiate CameraPose
        tcw_matrix = extract_tcw_matrix_from_pose_data(json_str.get("_pose_Tcp"))
        if tcw_matrix is not None:
            kf._pose_Tcp = CameraPose(tcw_matrix)
        else:
            kf._pose_Tcp = CameraPose()
        kf.is_Tcw_GBA_valid = bool(json_str.get("is_Tcw_GBA_valid", False))

        # Loop closing and relocalization fields
        kf.loop_query_id = json_str.get("loop_query_id", None)
        kf.num_loop_words = int(json_str.get("num_loop_words", 0))
        kf.loop_score = float(json_str.get("loop_score", 0.0))
        kf.reloc_query_id = json_str.get("reloc_query_id", None)
        kf.num_reloc_words = int(json_str.get("num_reloc_words", 0))
        kf.reloc_score = float(json_str.get("reloc_score", 0.0))

        # GBA fields
        kf.GBA_kf_id = int(json_str.get("GBA_kf_id", 0))
        tcw_gba_matrix = extract_tcw_matrix_from_pose_data(json_str.get("Tcw_GBA"))
        if tcw_gba_matrix is not None:
            kf.Tcw_GBA = tcw_gba_matrix
        else:
            kf.Tcw_GBA = None
        tcw_before_gba_matrix = extract_tcw_matrix_from_pose_data(json_str.get("Tcw_before_GBA"))
        if tcw_before_gba_matrix is not None:
            kf.Tcw_before_GBA = tcw_before_gba_matrix
        else:
            kf.Tcw_before_GBA = None

        kf.init_from_json(json_str)
        return kf

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the Lock from the state (don't pickle it)
        if "_lock_pose" in state:  # from FrameBase
            del state["_lock_pose"]
        if "_lock_features" in state:  # from Frame
            del state["_lock_features"]
        if "_lock_connections" in state:  # from KeyFrameGraph
            del state["_lock_connections"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'Lock' initially)
        self.__dict__.update(state)
        # Recreate the Lock after unpickling
        self._lock_pose = Lock()  # from FrameBase
        self._lock_features = Lock()
        self._lock_connections = Lock()

    # post processing after deserialization to replace saved ids with reloaded objects
    def replace_ids_with_objects(self, points, frames, keyframes):
        Frame.replace_ids_with_objects(self, points, frames, keyframes)
        KeyFrameGraph.replace_ids_with_objects(self, points, frames, keyframes)

    # associate matched map points to observations
    def init_observations(self):
        with self._lock_features:
            for idx, p in enumerate(self.points):
                if p is not None and not p.is_bad():
                    if p.add_observation(self, idx):
                        p.update_info()

    def update_connections(self):
        # for all map points of this keyframe check in which other keyframes they are seen
        # build a counter for these other keyframes
        points: list[MapPoint] = self.get_matched_good_points()
        num_points = len(points)
        if num_points == 0:
            Printer.orange("KeyFrame: update_connections - frame without points")
            return

        viewing_keyframes = Counter()
        for p in points:
            for kf in p.keyframes():
                if kf.kid != self.kid:
                    viewing_keyframes[kf] += 1

        if (
            not viewing_keyframes
        ):  # if empty   (https://www.pythoncentral.io/how-to-check-if-a-list-tuple-or-dictionary-is-empty-in-python/)
            return

        # order the keyframes: sort by weight in descending order
        covisible_keyframes = viewing_keyframes.most_common()
        # print('covisible_keyframes: ', covisible_keyframes)

        # get keyframe that shares most points
        kf_max, w_max = covisible_keyframes[0]

        # if the counter is greater than threshold add connection
        # otherwise add the one with maximum counter
        with self._lock_connections:
            self.connected_keyframes_weights = viewing_keyframes
            if w_max >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                self.ordered_keyframes_weights = OrderedDict()
                for kf, w in covisible_keyframes:
                    if w >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                        kf.add_connection_no_lock_(self, w)
                        self.ordered_keyframes_weights[kf] = w
                    else:
                        break
            else:
                # self.connected_keyframes_weights = Counter({kf_max: w_max})
                self.ordered_keyframes_weights = OrderedDict([(kf_max, w_max)])
                kf_max.add_connection_no_lock_(self, w_max)

            # update spanning tree
            # we need to avoid setting the parent to None or self or a bad keyframe
            if (
                self.is_first_connection
                and self.kid != 0
                and kf_max is not None
                and kf_max != self
                and not kf_max.is_bad()
            ):
                self.set_parent_no_lock_(kf_max)
                self.is_first_connection = False
        # print('ordered_keyframes_weights: ', self.ordered_keyframes_weights)

    def Tcp(self):
        with self._lock_connections:
            return (
                self._pose_Tcp.get_matrix()
            )  # pose relative to parent: self.Tcw() @ self.parent.Twc() (this is computed when bad flag is activated)

    def is_bad(self):
        with self._lock_connections:
            return self._is_bad

    def set_not_erase(self):
        with self._lock_connections:
            self.not_to_erase = True

    def set_erase(self):
        with self._lock_connections:
            if len(self.loop_edges) == 0:
                self.not_to_erase = False
        if self.to_be_erased:
            self.set_bad()

    def set_bad(self):
        with self._lock_connections:
            if not self.kid:  # check if kid is not None and not 0
                return

            if self.not_to_erase:
                self.to_be_erased = True
                return

            # --- 1. Remove covisibility connections ---
            for kf_connected in self.get_connected_keyframes_no_lock_():
                kf_connected.erase_connection_no_lock_(self)

            # --- 2. Remove feature observations ---
            for idx, p in enumerate(self.points):
                if p is not None:
                    p.remove_observation(self, idx)

            self.reset_covisibility()

            # --- 3. Update the spanning tree ---
            # Each children must be connected to a new parent

            assert self.parent is not None
            parent_candidates = {self.parent}

            # Prevent infinite loop due to malformed graph
            max_iters = len(self.children) * 100
            iters = 0

            # Reassign children based on covisibility weights
            remaining_children = list(self.children)
            self.children.clear()

            while remaining_children and iters < max_iters:
                iters += 1
                best_child = None
                best_parent = None
                max_weight = -1

                for kf_child in remaining_children:
                    if kf_child.is_bad():
                        continue

                    covisibles = kf_child.get_covisible_keyframes_no_lock_()
                    # Intersect with parent candidates
                    for candidate in parent_candidates:
                        if candidate in covisibles:
                            w = kf_child.get_weight_no_lock_(candidate)
                            if w > max_weight:
                                best_child = kf_child
                                best_parent = candidate
                                max_weight = w

                if best_child and best_parent:
                    best_child.set_parent_no_lock_(best_parent)
                    parent_candidates.add(best_child)
                    remaining_children.remove(best_child)
                else:
                    break  # No valid parent found; exit

                if iters >= max_iters:
                    Printer.orange("KeyFrame: set_bad - max iterations reached")

            # --- 4. Reassign unconnected children to original parent ---
            for kf_child in remaining_children:
                kf_child.set_parent_no_lock_(self.parent)

            # --- 5. Cleanup ---
            self.parent.erase_child_no_lock_(self)
            self._pose_Tcp.update(self.Tcw() @ self.parent.Twc())
            self._is_bad = True

        if self.map is not None:
            self.map.remove_keyframe(self)
