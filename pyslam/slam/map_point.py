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

import math
import time
import numpy as np
from threading import RLock, Lock, Thread

from pyslam.utilities.geometry import poseRt, add_ones, normalize_vector, normalize_vector2
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.utilities.system import Printer
from pyslam.config_parameters import Parameters

from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared
from pyslam.semantics.semantic_serialization import serialize_semantic_des, deserialize_semantic_des

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame
    from pyslam.slam.map import Map


class MapPointBase(object):
    _id = 0  # shared point counter
    _id_lock = Lock()  # shared lock for id

    def __init__(self, id=None):
        if id is not None:
            self.id = id
        else:
            with MapPointBase._id_lock:
                self.id = MapPointBase._id
                MapPointBase._id += 1

        self._lock_pos = Lock()
        self._lock_features = Lock()

        self.map: Map | None = (
            None  # this is used by the object for automatically removing itself from the map when it becomes bad (see below)
        )

        self._observations = dict()  # keyframe observations (used by mapping methods)
        # for kf, kidx in self._observations.items(): kf.points[kidx] = this point
        self._frame_views = (
            dict()
        )  # frame observations (used for drawing the tracking keypoint trails, frame by frame)
        # for f, idx in self._frame_views.items(): f.points[idx] = this point

        self._is_bad = False  # a map point becomes bad when its num_observations < 2 (cannot be considered for bundle ajustment or other related operations)
        self._num_observations = 0  # number of keyframe observations
        self.num_times_visible = 1  # number of times the point is visible in the camera
        self.num_times_found = 1  # number of times the point was actually matched and not rejected as outlier by the pose optimization in Tracking.track_local_map()
        self.last_frame_id_seen = -1  # last frame id in which this point was seen

        # self.is_replaced = False    # is True when the point was replaced by another point
        self.replacement = None  # replacing point

        # for loop correction
        self.corrected_by_kf = 0  # use kf.kid here!
        self.corrected_reference = 0  # use kf.kid here!
        self.kf_ref = None  # reference keyframe

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return isinstance(rhs, MapPointBase) and self.id == rhs.id

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    def observations_string(self):
        obs = sorted(
            [(kf.id, kidx, kf.get_point_match(kidx) != None) for kf, kidx in self.observations()],
            key=lambda x: x[0],
        )
        return "observations: " + str(obs)

    def frame_views_string(self):
        obs = sorted(
            [(f.id, idx, f.get_point_match(idx) != None) for f, idx in self.frame_views()],
            key=lambda x: x[0],
        )
        return "views: " + str(obs)

    def __str__(self):
        # return str(self.__class__) + ": " + str(self.__dict__)
        return (
            "MapPoint "
            + str(self.id)
            + " { "
            + self.observations_string()
            + ", "
            + self.frame_views_string()
            + " }"
        )

    # return a copy of the dictionary’s list of (key, value) pairs
    def observations(self):
        with self._lock_features:
            return list(self._observations.items())  # https://www.python.org/dev/peps/pep-0469/

    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe
    def observations_iter(self):
        return iter(self._observations.items())  # https://www.python.org/dev/peps/pep-0469/

    # return a copy of the dictionary’s list of keys
    def keyframes(self):
        with self._lock_features:
            return list(self._observations.keys())

    # return an iterator of the dictionary’s list of keys
    # NOT thread-safe
    def keyframes_iter(self):
        return iter(self._observations.keys())

    def is_in_keyframe(self, keyframe: "KeyFrame"):
        assert keyframe.is_keyframe
        with self._lock_features:
            return keyframe in self._observations

    def get_observation_idx(self, keyframe: "KeyFrame"):
        assert keyframe.is_keyframe
        with self._lock_features:
            return self._observations.get(keyframe, -1)

    def get_frame_view_idx(self, frame: "Frame"):
        with self._lock_features:
            return self._frame_views.get(frame, -1)

    def add_observation_no_lock_(self, keyframe: "KeyFrame", idx):
        success = False
        if keyframe not in self._observations:
            self._observations[keyframe] = idx
            if keyframe.kps_ur is not None and keyframe.kps_ur[idx] >= 0:
                self._num_observations += 2
            else:
                self._num_observations += 1
            success = True
        # elif self._observations[keyframe] != idx:     # if the keyframe is already there but it is incoherent then fix it!
        #    self._observations[keyframe] = idx
        if success:
            keyframe.set_point_match(self, idx)
        return success

    def add_observation(self, keyframe: "KeyFrame", idx):
        assert keyframe.is_keyframe
        with self._lock_features:
            success = self.add_observation_no_lock_(keyframe, idx)
        return success

    def remove_observation(self, keyframe: "KeyFrame", idx=None, map_no_lock=False):
        assert keyframe.is_keyframe
        kf_remove_point_match = False
        kf_remove_point = False
        set_bad = False

        with self._lock_features:
            # remove point association
            if idx is not None:
                kf_remove_point_match = True
                if __debug__:
                    assert self == keyframe.get_point_match(idx)
                    assert not self in keyframe.points  # checking there are no multiple instances
            else:
                kf_remove_point = True
            try:
                del self._observations[keyframe]
                if keyframe.kps_ur is not None and keyframe.kps_ur[idx] >= 0:
                    self._num_observations = max(0, self._num_observations - 2)
                else:
                    self._num_observations = max(0, self._num_observations - 1)
                set_bad = self._num_observations <= 2
                if self.kf_ref is keyframe and self._observations:
                    self.kf_ref = list(self._observations.keys())[0]
            except KeyError:
                pass

        # Make external calls outside of lock context
        if kf_remove_point_match:
            keyframe.remove_point_match(idx)
        if kf_remove_point:
            keyframe.remove_point(self)
        if set_bad:
            self.set_bad(map_no_lock=map_no_lock)

    # return a copy of the dictionary’s list of (key, value) pairs
    def frame_views(self):
        with self._lock_features:
            return list(self._frame_views.items())

    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe
    def frame_views_iter(self):
        return iter(self._frame_views.items())

    # return a copy of the dictionary’s list of keys
    def frames(self):
        with self._lock_features:
            return list(self._frame_views.keys())

    # return an iterator of the dictionary’s list of keys
    # NOT thread-safe
    def frames_iter(self):
        return iter(self._frame_views.keys())

    def is_in_frame(self, frame):
        with self._lock_features:
            return frame in self._frame_views

    # add a frame observation
    def add_frame_view(self, frame, idx):
        assert not frame.is_keyframe
        with self._lock_features:
            if (
                frame not in self._frame_views
            ):  # do not allow a point to be matched to diffent keypoints of the same frame
                self._frame_views[frame] = idx
                success = True
            # elif self._frame_views[keyframe] != idx:     # if the frame is already there but it is incoherent then fix it!
            #   self._frame_views[keyframe] = idx
            #   return True
            else:
                success = False

        if success:
            # Call external method outside of lock context
            frame.set_point_match(self, idx)

        return success

    def remove_frame_view(self, frame: "Frame", idx=None):
        assert not frame.is_keyframe
        frame_remove_point_match = False
        frame_remove_point = False

        with self._lock_features:
            # remove point from frame
            if idx is not None:
                if __debug__:
                    assert self == frame.get_point_match(idx)
                frame_remove_point_match = True
                if __debug__:
                    assert (
                        not self in frame.get_points()
                    )  # checking there are no multiple instances
            else:
                frame_remove_point = True
            try:
                del self._frame_views[frame]
            except KeyError:
                pass

        # Make external calls outside of lock context
        if frame_remove_point_match:
            frame.remove_point_match(idx)
        if frame_remove_point:
            frame.remove_point(self)

    def is_bad(self):
        with self._lock_features:
            # with self._lock_pos:
            return self._is_bad

    def is_bad_or_is_in_keyframe(self, keyframe: "KeyFrame"):
        with self._lock_features:
            assert keyframe.is_keyframe
            return self._is_bad or (keyframe in self._observations)

    def num_observations(self):
        with self._lock_features:
            return self._num_observations

    def is_good_with_min_obs(self, minObs):
        # with self._lock_features:
        return (not self._is_bad) and (self._num_observations >= minObs)

    def is_bad_and_is_good_with_min_obs(self, minObs):
        with self._lock_features:
            return (self._is_bad, (not self._is_bad) and (self._num_observations >= minObs))

    def increase_visible(self, num_times=1):
        with self._lock_features:
            self.num_times_visible += num_times

    def increase_found(self, num_times=1):
        with self._lock_features:
            self.num_times_found += num_times

    def get_found_ratio(self):
        with self._lock_features:
            return self.num_times_found / self.num_times_visible


# A Point is a 3-D point in the world
# Each Point is observed in multiple Frames
# NOTE: LOCK ORDERING RULE (to prevent deadlocks), always acquire locks in this order!
# 1. global_lock (if needed)
# 2. _lock_features
# 3. _lock_pos
class MapPoint(MapPointBase):
    global_lock = Lock()  # shared global lock for blocking point position update

    def __init__(self, position, color, keyframe=None, idxf=None, id=None):
        super().__init__(id)
        self._pt = np.ascontiguousarray(position)  # position in the world frame

        self.color = color
        self.semantic_des = None

        self.des = None  # best descriptor (continuously updated)
        self._min_distance, self._max_distance = 0, float("inf")  # depth infos
        self.normal = np.array([0, 0, 1])  # just a default 3D vector

        self.kf_ref = keyframe
        self.first_kid = -1  # first observation keyframe id

        if keyframe is not None:
            if keyframe.is_keyframe:
                self.first_kid = keyframe.kid
            # update normal and depth infos
            po = self._pt - self.kf_ref.Ow()
            self.normal, dist = normalize_vector(po)
            if idxf is not None:
                self.des = keyframe.des[idxf]
                level = keyframe.octaves[idxf]
            else:
                self.des = None
                level = 0
            level_scale_factor = FeatureTrackerShared.feature_manager.scale_factors[level]
            self._max_distance = dist * level_scale_factor
            self._min_distance = (
                self._max_distance
                / FeatureTrackerShared.feature_manager.scale_factors[
                    FeatureTrackerShared.feature_manager.num_levels - 1
                ]
            )

        self.num_observations_on_last_update_des = 1  # must be 1!
        self.num_observations_on_last_update_normals = 1  # must be 1!
        self.num_observations_on_last_update_semantics = 1  # must be 1!

        # for GBA
        self.pt_GBA = None
        self.is_pt_GBA_valid = False  # For easing C++ equivalent code
        self.GBA_kf_id = 0

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the unpickable RLock from the state (can't pickle it)
        if "_lock_pos" in state:
            del state["_lock_pos"]
        if "_lock_features" in state:
            del state["_lock_features"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'RLock' initially)
        self.__dict__.update(state)
        self._lock_pos = Lock()
        self._lock_features = Lock()

    def to_json(self):
        return {
            "id": int(self.id) if self.id is not None else None,
            "_observations": [
                (int(kf.id), int(idx)) for kf, idx in self._observations.items() if kf is not None
            ],
            "_frame_views": [
                (int(f.id), int(idx)) for f, idx in self._frame_views.items() if f is not None
            ],
            "_is_bad": bool(self._is_bad),
            "_num_observations": self._num_observations,
            "num_times_visible": self.num_times_visible,
            "num_times_found": self.num_times_found,
            "last_frame_id_seen": self.last_frame_id_seen,
            "pt": self.pt.tolist(),
            "color": (
                self.color.tolist()
                if self.color is not None and isinstance(self.color, np.ndarray)
                else self.color
            ),
            "semantic_des": serialize_semantic_des(
                self.semantic_des, SemanticMappingShared.semantic_feature_type
            ),
            "des": self.des.tolist() if self.des is not None else None,
            "_min_distance": self._min_distance,
            "_max_distance": self._max_distance,
            "normal": self.normal.tolist(),
            "first_kid": int(self.first_kid),
            "kf_ref": int(self.kf_ref.id) if self.kf_ref is not None else None,
        }

    @staticmethod
    def from_json(json_str):
        p = MapPoint(json_str["pt"], json_str["color"], keyframe=None, idxf=None, id=json_str["id"])

        p._observations = json_str["_observations"]
        p._frame_views = json_str["_frame_views"]
        p._is_bad = json_str["_is_bad"]
        p._num_observations = json_str["_num_observations"]
        p.num_times_visible = json_str["num_times_visible"]
        p.num_times_found = json_str["num_times_found"]
        p.last_frame_id_seen = json_str["last_frame_id_seen"]

        p.des = np.array(json_str["des"])
        p._min_distance = json_str["_min_distance"]
        p._max_distance = json_str["_max_distance"]
        p.normal = np.array(json_str["normal"])
        p.first_kid = json_str["first_kid"]
        p.kf_ref = json_str["kf_ref"]

        p.semantic_des, semantic_type = deserialize_semantic_des(json_str["semantic_des"])
        return p

    def replace_ids_with_objects(self, points, frames, keyframes):
        # Pre-build dictionaries for efficient lookups
        keyframes_dict = {obj.id: obj for obj in keyframes if obj is not None}
        frames_dict = {obj.id: obj for obj in frames if obj is not None}

        def get_object_with_id(id, lookup_dict):
            return lookup_dict.get(id, None)

        # Replace _observations
        if self._observations is not None:
            self._observations = {
                get_object_with_id(fid, keyframes_dict): idx for fid, idx in self._observations
            }
        # Replace _frame_views
        if self._frame_views is not None:
            self._frame_views = {
                get_object_with_id(fid, frames_dict): idx for fid, idx in self._frame_views
            }
        # Replace kf_ref
        if self.kf_ref is not None:
            self.kf_ref = get_object_with_id(self.kf_ref, keyframes_dict)

    def pt(self):
        with self._lock_pos:
            return self._pt.copy()

    def homogeneous(self):
        with self._lock_pos:
            # return add_ones(self._pt)
            return np.concatenate([self._pt, np.array([1.0])], axis=0)

    def update_position(self, position):
        with self.global_lock:
            with self._lock_pos:
                self._pt = position

    def min_distance(self):
        with self._lock_pos:
            # return FeatureTrackerShared.feature_manager.inv_scale_factor * self._min_distance  # give it one level of margin (can be too much with scale factor = 2)
            return Parameters.kMinDistanceToleranceFactor * self._min_distance

    def max_distance(self):
        with self._lock_pos:
            # return FeatureTrackerShared.feature_manager.scale_factor * self._max_distance  # give it one level of margin (can be too much with scale factor = 2)
            return Parameters.kMaxDistanceToleranceFactor * self._max_distance

    def get_all_pos_info(self):
        with self._lock_pos:
            return (
                self._pt,
                self.normal,
                Parameters.kMinDistanceToleranceFactor * self._min_distance,
                Parameters.kMaxDistanceToleranceFactor * self._max_distance,
            )

    def get_reference_keyframe(self):
        with self._lock_features:
            return self.kf_ref

    # return array of corresponding descriptors
    def descriptors(self):
        with self._lock_features:
            return [kf.des[idx] for kf, idx in self._observations.items()]

    # minimum distance between input descriptor and map point corresponding descriptors
    def min_des_distance(self, descriptor):
        with self._lock_features:
            # return min([FeatureTrackerShared.descriptor_distance(d, descriptor) for d in self.descriptors()])
            return FeatureTrackerShared.descriptor_distance(self.des, descriptor)

    def delete(self):
        if not self._is_bad:
            self.set_bad()
        del self  # delete if self is the last reference

    def set_bad(self, map_no_lock=False):
        if self._is_bad:
            return
        with self._lock_features:
            with self._lock_pos:
                self._is_bad = True
                self._num_observations = 0
                observations = list(self._observations.items())
                self._observations.clear()
        for kf, idx in observations:
            kf.remove_point_match(idx)
        if self.map is not None:
            if map_no_lock:
                self.map.remove_point_no_lock(self)
            else:
                self.map.remove_point(self)

    def get_replacement(self):
        with self._lock_features:
            with self._lock_pos:
                return self.replacement

    def get_normal(self):
        with self._lock_pos:
            return self.normal

    # replace this point with map point p
    def replace_with(self, p: "MapPoint"):
        if p.id == self.id:
            return
        # if __debug__:
        #    Printer.orange('replacing ', self, ' with ', p)
        observations, num_times_visible, num_times_found = None, 0, 0
        with self._lock_features:
            with self._lock_pos:
                observations = list(self._observations.items())
                self._observations.clear()
                num_times_visible = self.num_times_visible
                num_times_found = self.num_times_found
                self._is_bad = True
                self._num_observations = 0
                # self.is_replaced = True    # tell the delete() method not to remove observations and frame views
                self.replacement = p

        # replace point observations in keyframes
        for kf, kidx in observations:  # we have kf.get_point_match(kidx) = self
            # if p.is_in_keyframe(kf):
            #     # point p is already in kf => just remove this point match from kf
            #     # (do NOT remove the observation otherwise self._num_observations is decreased in the replacement)
            #     kf.remove_point_match(kidx)
            # else:
            #     # point p is not in kf => add new observation in p
            #     kf.replace_point_match(p,kidx)
            #     p.add_observation(kf,kidx)
            if p.add_observation(kf, kidx):
                # point p was NOT in kf => added new observation in p
                kf.replace_point_match(p, kidx)
            else:
                # point p is already in kf => just remove this point match from kf
                # (do NOT remove the observation otherwise self._num_observations is decreased in the replacement)
                kf.remove_point_match(kidx)
                # if p.get_observation_idx(kf) != kidx:
                #    kf.remove_point_match(kidx)
                # else:
                #    kf.replace_point_match(p,kidx)

        p.increase_visible(num_times_visible)
        p.increase_found(num_times_found)
        # p.update_info()
        p.update_best_descriptor(force=True)

        # replace point observations in frames (done by frame.check_replaced_map_points())
        # for f, idx in frame_views:   # we have f.get_point_match(idx) = self
        #     if p.is_in_frame(f):
        #         if not f.is_keyframe: # if not already managed above in keyframes
        #             # point p is already in f => just remove this point match from f
        #             f.remove_point_match(idx)
        #     else:
        #         # point p is not in f => add new frame view in p
        #         f.replace_point_match(p,idx)
        #         p.add_frame_view(f,idx)

        if self.map is not None:
            self.map.remove_point(self)
        else:
            Printer.warn("MapPoint: replace_with() - map is None")
        # if __debug__:
        #    Printer.green('after replacement ', p)

    # update normal and depth representations
    def update_normal_and_depth(self, force=False):
        skip = False
        with self._lock_features:
            with self._lock_pos:
                if self._is_bad:
                    return
                if self._num_observations > self.num_observations_on_last_update_normals or force:
                    # implicit if self._num_observations > 1
                    self.num_observations_on_last_update_normals = self._num_observations
                    observations = list(self._observations.items())
                    kf_ref = self.kf_ref
                    idx_ref = self._observations[kf_ref]
                    position = self._pt.copy()
                else:
                    skip = True
        if skip or not observations:
            return

        normals = np.array(
            [normalize_vector2(position - kf.Ow()) for kf, idx in observations]
        ).reshape(-1, 3)
        normal = normalize_vector2(np.mean(normals, axis=0))
        # print('normals: ', normals)
        # print('mean normal: ', self.normal)

        level = kf_ref.octaves[idx_ref]
        level_scale_factor = FeatureTrackerShared.feature_manager.scale_factors[level]
        dist = np.linalg.norm(position - kf_ref.Ow())

        with self._lock_pos:
            self._max_distance = dist * level_scale_factor
            self._min_distance = (
                self._max_distance
                / FeatureTrackerShared.feature_manager.scale_factors[
                    FeatureTrackerShared.feature_manager.num_levels - 1
                ]
            )
            self.normal = normal

    def update_best_descriptor(self, force=False):
        skip = False
        with self._lock_features:
            if self._is_bad:
                return
            if (
                self._num_observations > self.num_observations_on_last_update_des or force
            ):  # implicit if self._num_observations > 1
                self.num_observations_on_last_update_des = self._num_observations
                observations = list(self._observations.items())
            else:
                skip = True
        if skip or len(observations) == 0:
            return
        descriptors = [kf.des[idx] for kf, idx in observations if not kf.is_bad()]

        N = len(descriptors)
        if N >= 2:
            # median_distances = [ np.median([FeatureTrackerShared.descriptor_distance(d, descriptors[i]) for d in descriptors]) for i in range(N) ]
            # median_distances = [ np.median(FeatureTrackerShared.descriptor_distances(descriptors[i], descriptors)) for i in range(N)]
            D = np.array(
                [FeatureTrackerShared.descriptor_distances(d, descriptors) for d in descriptors]
            )
            median_distances = np.median(D, axis=1)
            with self._lock_features:
                self.des = descriptors[np.argmin(median_distances)].copy()
            # print('descriptors: ', descriptors)
            # print('median_distances: ', median_distances)
            # print('des: ', self.des)

    def update_semantics(self, semantic_fusion_method, force=False):
        skip = False
        with self._lock_features:
            if self._is_bad:
                return
            if (
                self._num_observations > self.num_observations_on_last_update_semantics or force
            ):  # implicit if self._num_observations > 1
                self.num_observations_on_last_update_semantics = self._num_observations
                observations = list(self._observations.items())
            else:
                skip = True
        if skip or len(observations) == 0:
            return
        semantics = [
            kf.kps_sem[idx]
            for kf, idx in observations
            if not kf.is_bad() and kf.kps_sem is not None
        ]
        if len(semantics) >= 2:
            fused_semantics = semantic_fusion_method(semantics)
            with self._lock_features:
                self.semantic_des = fused_semantics

    def update_info(self):
        # if self._is_bad:
        #    return
        self.update_normal_and_depth()
        self.update_best_descriptor()

    # predict detection level from pyslam.slam.map point distance
    def predict_detection_level(self, dist):
        with self._lock_pos:
            ratio = self._max_distance / dist
        level = math.ceil(math.log(ratio) / FeatureTrackerShared.feature_manager.log_scale_factor)
        if level < 0:
            level = 0
        elif level >= FeatureTrackerShared.feature_manager.num_levels:
            level = FeatureTrackerShared.feature_manager.num_levels - 1
        return level

    # predict detection levels from pyslam.slam.map point distances
    @staticmethod
    def predict_detection_levels(points: list["MapPoint"], dists: np.ndarray):
        assert len(points) == len(dists)
        max_distances = np.array([p._max_distance for p in points])
        ratios = max_distances / dists
        ratios = np.maximum(ratios, 1e-8)  # prevent log(0) or log(negative)
        levels = np.ceil(
            np.log(ratios) / FeatureTrackerShared.feature_manager.log_scale_factor
        ).astype(np.intp)
        levels = np.clip(levels, 0, FeatureTrackerShared.feature_manager.num_levels - 1)
        return levels
