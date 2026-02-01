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
import time
import threading
import signal
import torch.multiprocessing as mp

import cv2
import numpy as np

from pyslam.slam import Camera, Map, KeyFrame, Frame, USE_CPP

from pyslam.io.dataset_types import DatasetEnvironmentType, SensorType

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import set_rlimit
from pyslam.utilities.logging import LoggerQueue
from pyslam.utilities.file_management import create_folder
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue, push_to_front, static_fields_to_dict
from pyslam.utilities.depth import filter_shadow_points
from pyslam.utilities.multi_threading import SimpleTaskTimer
from pyslam.utilities.timer import TimerFps
from pyslam.utilities.pickling import ensure_picklable, filter_unpicklable_recursive
from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared

from pyslam.config_parameters import Parameters, get_np_dtype

from pyslam.dense.volumetric_integrator_types import VolumetricIntegratorType

from volumetric import ObjectData, ObjectDataGroup

import traceback

from collections import deque


from enum import Enum

import torch
from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)

import open3d as o3d

import torch.multiprocessing as mp


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.config import Config
    from pyslam.slam.keyframe import KeyFrame


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


kVolumetricIntegratorProcessName = "VolumetricIntegratorProcess"


class VolumetricIntegrationTaskType(Enum):
    NONE = 0
    INTEGRATE = 1
    SAVE = 2
    LOAD = 3
    RESET = 4
    UPDATE_OUTPUT = 5


# keyframe (picklable) data that are needed for volumetric integration
class VolumetricIntegrationKeyframeData:
    def __init__(
        self,
        keyframe: KeyFrame,
        img=None,
        img_right=None,
        depth=None,
        semantic_img=None,
        semantic_instances_img=None,
    ):
        # keyframe data
        self.id = keyframe.id if keyframe is not None else -1
        self.kid = keyframe.kid if keyframe is not None else -1
        self.img_id = keyframe.img_id if keyframe is not None else -1
        self.timestamp = keyframe.timestamp if keyframe is not None else -1
        self.pose = keyframe.pose() if keyframe is not None else None  # Tcw
        self.camera = keyframe.camera if keyframe is not None else None

        self.img = img if img is not None else (keyframe.img if keyframe is not None else None)
        self.img_right = (
            img_right
            if img_right is not None
            else (keyframe.img_right if keyframe is not None else None)
        )
        self.depth = (
            depth if depth is not None else (keyframe.depth_img if keyframe is not None else None)
        )
        self.semantic_img = (
            semantic_img
            if semantic_img is not None
            else (keyframe.semantic_img if keyframe is not None else None)
        )
        self.semantic_instances_img = (
            semantic_instances_img
            if semantic_instances_img is not None
            else (keyframe.semantic_instances_img if keyframe is not None else None)
        )


class VolumetricIntegrationTask:
    def __init__(
        self,
        keyframe: KeyFrame = None,
        img=None,
        img_right=None,
        depth=None,
        semantic_img=None,
        task_type=VolumetricIntegrationTaskType.NONE,
        load_save_path=None,
    ):
        self.task_type = task_type
        self.keyframe_data = VolumetricIntegrationKeyframeData(
            keyframe, img, img_right, depth, semantic_img
        )
        self.load_save_path = load_save_path


# picklable point cloud obtained from o3d.geometry.PointCloud
class VolumetricIntegrationPointCloud:
    def __init__(
        self,
        point_cloud: o3d.geometry.PointCloud = None,
        points=None,
        colors=None,
        semantics=None,
        object_ids=None,
        semantic_colors=None,
        object_colors=None,
    ):
        if point_cloud is not None:
            self.points = np.asarray(point_cloud.points)
            self.colors = np.asarray(point_cloud.colors)
            if hasattr(point_cloud, "semantics"):
                self.semantics = (
                    np.asarray(point_cloud.semantics) if point_cloud.semantics is not None else None
                )
            else:
                self.semantics = None
            if hasattr(point_cloud, "object_ids"):
                self.object_ids = (
                    np.asarray(point_cloud.object_ids)
                    if point_cloud.object_ids is not None
                    else None
                )
            else:
                self.object_ids = None
            if hasattr(point_cloud, "semantic_colors"):
                self.semantic_colors = (
                    np.asarray(point_cloud.semantic_colors)
                    if point_cloud.semantic_colors is not None
                    else None
                )
            else:
                self.semantic_colors = None
        else:
            self.points = np.asarray(points) if points is not None else None
            self.colors = np.asarray(colors) if colors is not None else None
            self.semantics = np.asarray(semantics) if semantics is not None else None
            self.object_ids = np.asarray(object_ids) if object_ids is not None else None
            self.semantic_colors = (
                np.asarray(semantic_colors) if semantic_colors is not None else None
            )
            self.object_colors = np.asarray(object_colors) if object_colors is not None else None

    def to_o3d(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        pc.colors = o3d.utility.Vector3dVector(self.colors)
        return pc


# picklable mesh obtained from o3d.geometry.TriangleMesh
class VolumetricIntegrationMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.vertex_normals = np.asarray(mesh.vertex_normals)

    def to_o3d(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        return mesh


# This seems to be very slow to manage
# class VolumetricIntegrationObjects:
#     def __init__(
#         self,
#         objects_data: ObjectDataGroup,
#         semantic_colors: np.ndarray,
#         object_colors: np.ndarray,
#         num_objects: int,
#     ):
#         self.objects_data = objects_data  # ObjectDataGroup contains N objects (ObjectData objects)
#         self.semantic_colors = semantic_colors  # [N, 3] in [0, 1] range
#         self.object_colors = object_colors  # [N, 3] in [0, 1] range
#         self.num_objects = (
#             num_objects  # number of objects (ObjectData objects) in the ObjectDataGroup
#         )


class VolumetricIntegratinOrientedBoundingBox3D:
    def __init__(
        self,
        box_matrix: np.ndarray,
        box_size: np.ndarray,
    ):
        # precision consistency with the C++ code in cpp/glutils/globject.h and cpp/volumetric/bounding_boxes_3d.h
        matrix = np.asarray(box_matrix, dtype=np.float64)
        if matrix.shape == (4, 4):
            # OpenGL consumes column-major matrices; pybind/eigen expose column-major data.
            # Update bindings expect C-contiguous buffers, so transpose once here.
            matrix = matrix.T
        self.box_matrix: np.ndarray = np.ascontiguousarray(matrix, dtype=np.float64)
        self.box_size: np.ndarray = np.ascontiguousarray(box_size, dtype=np.float64)


class VolumetricIntegrationObject:
    def __init__(
        self,
        object_data: ObjectData,
    ):
        self.points = np.ascontiguousarray(
            object_data.points, dtype=Parameters.kDenseMappingDtypeVertices
        )
        self.colors = np.ascontiguousarray(
            object_data.colors, dtype=Parameters.kDenseMappingDtypeColors
        )
        self.class_id: int = object_data.class_id
        self.object_id: int = object_data.object_id
        self.confidence_min: float = object_data.confidence_min
        self.confidence_max: float = object_data.confidence_max
        self.oriented_bounding_box: VolumetricIntegratinOrientedBoundingBox3D = (
            VolumetricIntegratinOrientedBoundingBox3D(
                object_data.oriented_bounding_box.get_matrix(),
                object_data.oriented_bounding_box.size,
            )
        )


class VolumetricIntegrationObjectList:
    def __init__(
        self,
        object_data_group: ObjectDataGroup,
        semantic_colors: np.ndarray,
        object_colors: np.ndarray,
        num_objects: int,
    ):
        self.object_list: list[VolumetricIntegrationObject] = [
            VolumetricIntegrationObject(object_data)
            for object_data in object_data_group.object_vector
        ]  # N objects
        self.semantic_colors: np.ndarray = np.ascontiguousarray(
            semantic_colors, dtype=Parameters.kDenseMappingDtypeColors
        )  # [N, 3] in [0, 1] range
        self.object_colors: np.ndarray = np.ascontiguousarray(
            object_colors, dtype=Parameters.kDenseMappingDtypeColors
        )  # [N, 3] in [0, 1] range
        self.num_objects: int = (
            num_objects  # number of objects (ObjectData objects) in the ObjectDataGroup
        )


class VolumetricIntegrationOutput:
    def __init__(
        self,
        task_type,
        id=-1,
        point_cloud: VolumetricIntegrationPointCloud = None,
        mesh: VolumetricIntegrationMesh = None,
        objects: VolumetricIntegrationObjectList = None,
    ):
        self.task_type = task_type
        self.id = id
        self.point_cloud: VolumetricIntegrationPointCloud = point_cloud
        self.mesh: VolumetricIntegrationMesh = mesh
        self.objects: VolumetricIntegrationObjectList = objects
        self.timestamp = time.perf_counter()


# ==================================================================================================


class VolumetricIntegratorBase:
    """
    Base class for volumetric integrators.
    """

    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
    logging_manager, logger = None, None

    def __init__(
        self,
        camera,
        environment_type,
        sensor_type,
        volumetric_integrator_type,
        viewer_queue=None,
        **kwargs,
    ):
        self.volumetric_integrator_type = volumetric_integrator_type
        self.constructor_kwargs = kwargs

        if (
            volumetric_integrator_type == VolumetricIntegratorType.TSDF
            or volumetric_integrator_type == VolumetricIntegratorType.GAUSSIAN_SPLATTING
            or Parameters.kVolumetricIntegrationUseDepthEstimator == True
        ):

            # NOTE: The following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).
            if mp.get_start_method() != "spawn":
                mp.set_start_method(
                    "spawn", force=True
                )  # NOTE: This may generate some pickling problems with multiprocessing
                #    in combination with torch and we need to check it in other places.
                #    This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

            set_rlimit()

        self.camera = camera
        self.environment_type = environment_type
        self.sensor_type = sensor_type
        self.viewer_queue = viewer_queue

        self.keyframe_queue_timer = SimpleTaskTimer(
            interval=0.5,  # NOTE: This is the interval in seconds for flushing the keyframe queue.
            callback=self.flush_keyframe_queue,
            single_shot=False,
            name="KeyframeQueueTimer",
        )
        self.keyframe_queue_lock = threading.Lock()  # Thread-safe access to keyframe_queue
        self.keyframe_queue = (
            deque()
        )  # We use a deque to accumulate keyframes for volumetric integration.
        # We integrate only the keyframes that have been processed by LBA at least once.

        self.volume = None

        self.time_volumetric_integration = mp.Value("d", 0.0)

        self.last_input_task = None
        self.last_output = None
        self.last_integrated_id = -1

        self.depth_estimator = None
        self.img_id_to_depth = None

        self.reset_mutex = mp.Lock()
        self.reset_requested = mp.Value("i", -1)

        self.load_request_completed = mp.Value("i", -1)
        self.load_request_condition = mp.Condition()
        self.save_request_completed = mp.Value("i", -1)
        self.save_request_condition = mp.Condition()

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.q_in = (
            self.mp_manager.Queue()
        )  # This queue is used for regular tasks of the volumetric integration process (e.g., integrate, update output, save, etc.)
        self.q_out = (
            self.mp_manager.Queue()
        )  # This queue is used for output tasks of the volumetric integration process (e.g., visualize, save, etc.)
        self.q_management = (
            self.mp_manager.Queue()
        )  # This queue is used for special management tasks of the volumetric integration process (e.g., reset, rebuild, etc.)

        # NOTE: The Parameters static fields are not sinced in the parallel process launched below (which has its own memory space).
        #       Here we explicitly copy the current state at _initialization time_ and pass it to the parallel process.
        #       In this way, the parallel process will be able to access the updated prameters (until the parallel process is launched).
        self.parameters_dict = self.mp_manager.Dict()
        self.parameters_dict = static_fields_to_dict(Parameters)

        self.q_in_condition = (
            mp.Condition()
        )  # Shared condition for self.q_in and self.q_management queues
        self.q_out_condition = mp.Condition()

        self.is_running = mp.Value("i", 0)
        self.is_looping = mp.Value("i", 0)

        # Export SemanticMappingShared state for passing to spawned process
        # This is needed when using spawn multiprocessing method, as static fields
        # are not inherited by spawned processes
        self.semantic_mapping_shared_state = None
        if (
            SemanticMappingShared.is_semantic_mapping_enabled()
            # and Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess
        ):
            try:
                Printer.green(
                    "VolumetricIntegratorBase: Exporting SemanticMappingShared state for passing to spawned process"
                )
                VolumetricIntegratorBase.print(
                    "VolumetricIntegratorBase: Exporting SemanticMappingShared state for passing to spawned process"
                )
                self.semantic_mapping_shared_state = SemanticMappingShared.export_state()
            except Exception as e:
                Printer.orange(
                    f"WARNING: VolumetricIntegratorBase: Failed to export SemanticMappingShared state: {e}"
                )
                traceback.print_exc()

        self.init_print()
        self.start()

    def is_ready(self):
        return self.is_running.value == 1 and self.is_looping.value == 1

    def init_print(self):
        if kVerbose:
            if Parameters.kDoVolumetricIntegration:
                if Parameters.kVolumetricIntegrationDebugAndPrintToFile:
                    # redirect the prints of volumetric integration to the file (by default) logs/volumetric_integration.log
                    # you can watch the output in separate shell by running:
                    # $ tail -f logs/volumetric_integration.log

                    logging_file = Parameters.kLogsFolder + "/volumetric_integrator.log"
                    create_folder(logging_file)
                    if VolumetricIntegratorBase.logging_manager is None:
                        # Note: Each process has its own memory space, so singleton pattern works per-process
                        VolumetricIntegratorBase.logging_manager = LoggerQueue.get_instance(
                            logging_file
                        )
                        VolumetricIntegratorBase.logger = (
                            VolumetricIntegratorBase.logging_manager.get_logger(
                                "volumetric_integrator_main"
                            )
                        )

                    def print_file(*args, **kwargs):
                        try:
                            if VolumetricIntegratorBase.logger is not None:
                                message = " ".join(
                                    str(arg) for arg in args
                                )  # Convert all arguments to strings and join with spaces
                                return VolumetricIntegratorBase.logger.info(message, **kwargs)
                        except:
                            print("Error printing: ", args, kwargs)

                else:

                    def print_file(*args, **kwargs):
                        message = " ".join(
                            str(arg) for arg in args
                        )  # Convert all arguments to strings and join with spaces
                        return print(message, **kwargs)

                VolumetricIntegratorBase.print = staticmethod(print_file)

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()

        # Remove from the state the things you don't want to pickle
        if "keyframe_queue_timer" in state:
            del state["keyframe_queue_timer"]
        if "keyframe_queue_lock" in state:
            del state["keyframe_queue_lock"]

        # Filter out any classmethod or staticmethod objects that cannot be pickled
        keys_to_remove = []
        for key, value in state.items():
            # Check if value is a classmethod or staticmethod object (these cannot be pickled)
            if isinstance(value, (classmethod, staticmethod)):
                keys_to_remove.append(key)
            # For dictionaries and other structures, recursively filter them
            elif isinstance(value, (dict, list, tuple)):
                try:
                    filtered = filter_unpicklable_recursive(value)
                    if filtered is not None:
                        state[key] = filtered
                    else:
                        keys_to_remove.append(key)
                except Exception:
                    # If filtering fails, remove the key to be safe
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del state[key]

        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the lock after unpickling
        if not hasattr(self, "keyframe_queue_lock"):
            self.keyframe_queue_lock = threading.Lock()
        # self.keyframe_queue_timer = SimpleTaskTimer(interval=1, callback=self.flush_keyframe_queue, single_shot=False)

    def start(self):
        # self.is_running.value = 1
        self.process = mp.Process(
            target=self.run,
            args=(
                self.camera,
                self.environment_type,
                self.sensor_type,
                self.viewer_queue,
                self.q_in,
                self.q_in_condition,
                self.q_out,
                self.q_out_condition,
                self.q_management,
                self.is_running,
                self.is_looping,
                self.reset_mutex,
                self.reset_requested,
                self.load_request_completed,
                self.load_request_condition,
                self.save_request_completed,
                self.save_request_condition,
                self.time_volumetric_integration,
                self.parameters_dict,
                self.constructor_kwargs,
                self.semantic_mapping_shared_state,  # Pass SemanticMappingShared state
            ),
            name=kVolumetricIntegratorProcessName,
        )

        # self.process.daemon = True
        self.process.start()
        self.keyframe_queue_timer.start()

    # This method must be implemented in the derived classes and is called in the quit() method
    def _stop_volume_integrator_implementation(self):
        pass

    def save(self, path):
        VolumetricIntegratorBase.print("VolumetricIntegratorBase: saving...")
        try:
            if self.save_request_completed.value == 0:
                VolumetricIntegratorBase.print("VolumericIntegratorBase: saving: already saving...")
                return
            filepath = path + "/dense_map.ply"
            task_type = VolumetricIntegrationTaskType.SAVE
            task = VolumetricIntegrationTask(task_type=task_type, load_save_path=filepath)
            self.save_request_completed.value = 0
            self.add_task(task, front=True)
            with self.save_request_condition:
                while self.save_request_completed.value == 0:
                    self.save_request_condition.wait()
            VolumetricIntegratorBase.print("VolumetricIntegratorBase: saving done")
        except Exception as e:
            VolumetricIntegratorBase.print(f"VolumetricIntegratorBase: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

    def load(self, path):
        if False:
            task_type = VolumetricIntegrationTaskType.LOAD
            task = VolumetricIntegrationTask(None, None, task_type, load_save_path=path)
            self.load_request_completed.value = 0
            self.add_task(task)
            with self.load_request_condition:
                while self.load_request_completed.value == 0:
                    self.load_request_condition.wait()
        # TODO: Clarify how to use this

    def request_reset(self):
        VolumetricIntegratorBase.print("VolumetricIntegratorBase: Requesting reset...")
        with self.reset_mutex:
            if self.reset_requested.value == 1:
                VolumetricIntegratorBase.print(
                    "VolumetricIntegratorBase: reset already requested..."
                )
                return
            self.reset_requested.value = 1
        while True:
            with self.reset_mutex:
                with self.q_in_condition:
                    self.q_in_condition.notify_all()  # to unblock q_in_condition.wait() in run() method
                if self.reset_requested.value == 0:
                    break
            time.sleep(0.1)
        with self.keyframe_queue_lock:
            self.keyframe_queue.clear()
        VolumetricIntegratorBase.print("VolumetricIntegratorBase: ...Reset done.")

    def reset_if_requested(
        self, reset_mutex, reset_requested, q_in, q_in_condition, q_out, q_out_condition
    ):
        # acting within the launched process with the passed mp.Value() (received in input)
        with reset_mutex:
            if reset_requested.value == 1:
                VolumetricIntegratorBase.print("VolumetricIntegratorBase: reset_if_requested()...")
                with q_in_condition:
                    empty_queue(q_in)
                    q_in_condition.notify_all()
                with q_out_condition:
                    empty_queue(q_out)
                    q_out_condition.notify_all()
                # Now reset the volume integrator in the launched parallel process
                try:
                    VolumetricIntegratorBase.print("VolumetricIntegratorBase: resetting volume...")
                    self.volume.reset()
                except Exception as e:
                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: reset_if_requested: Exception: {e}"
                    )
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")
                reset_requested.value = 0

    def quit(self):
        try:
            if self.is_running.value == 1:
                VolumetricIntegratorBase.print("VolumetricIntegratorBase: quitting...")
                self.is_running.value = 0
                self.keyframe_queue_timer.stop()

                with self.q_in_condition:
                    self.q_in.put(None)  # put a None in the queue to signal we have to exit
                    self.q_in_condition.notify_all()
                with self.q_out_condition:
                    self.q_out_condition.notify_all()

                # check if "spawn" method is used, so we increse the default timeout
                timeout = (
                    Parameters.kMultiprocessingProcessJoinDefaultTimeout
                    if mp.get_start_method() != "spawn"
                    else 2 * Parameters.kMultiprocessingProcessJoinDefaultTimeout
                )
                self.process.join(timeout=timeout)

                if self.process.is_alive():
                    Printer.orange(
                        "Warning: Volumetric integration process did not terminate in time, forced kill."
                    )
                    self.process.terminate()

                # Clean up LoggerQueue before terminating process
                if VolumetricIntegratorBase.logging_manager is not None:
                    try:
                        VolumetricIntegratorBase.logging_manager.stop_listener()
                        VolumetricIntegratorBase.logging_manager = None
                        VolumetricIntegratorBase.logger = None
                    except Exception as e:
                        VolumetricIntegratorBase.print(f"Error cleaning up logging manager: {e}")

                # Shutdown the manager AFTER the process has exited
                if hasattr(self, "mp_manager") and self.mp_manager is not None:
                    try:
                        self.mp_manager.shutdown()
                    except Exception as e:
                        print(f"Warning: Error shutting down manager: {e}")

                # Use regular print instead of VolumetricIntegratorBase.print after cleanup
                print("VolumetricIntegratorBase: done")
        except Exception as e:
            VolumetricIntegratorBase.print(f"VolumetricIntegratorBase: quit: Exception: {e}")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

    def init(
        self,
        camera: Camera,
        environment_type: DatasetEnvironmentType,
        sensor_type: SensorType,
        parameters_dict,
        constructor_kwargs,
    ):

        self.last_output = None
        self.depth_factor = 1.0  # /camera.depth_factor # Now, depth factor is already included in the set kf depth image (see Frame constructor)
        self.environment_type = environment_type
        self.sensor_type = sensor_type

        self.img_id_to_depth = {}

        self.parameters_dict = parameters_dict

        depth_estimator_type_str = parameters_dict["kVolumetricIntegrationDepthEstimatorType"]
        depth_estimator_type = DepthEstimatorType.from_string(depth_estimator_type_str)
        use_depth_estimator = parameters_dict["kVolumetricIntegrationUseDepthEstimator"]
        if use_depth_estimator:
            min_depth = 0
            max_depth = 50 if environment_type == DatasetEnvironmentType.OUTDOOR else 10
            precision = torch.float16  # for depth_pro
            if sensor_type == SensorType.MONOCULAR:
                Printer.red(
                    "*************************************************************************************"
                )
                Printer.red(
                    "VolumetricIntegratorBase: ERROR: You cannot use a MONOCULAR depth estimator here when"
                )
                Printer.red(
                    "you are using a MONOCULAR SLAM system: The scale of the metric depth estimator will"
                )
                Printer.red("conflict with the independent scale of the SLAM system!")
                Printer.red(
                    "*************************************************************************************"
                )
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorBase: init: depth_estimator_type={depth_estimator_type}, min_depth={min_depth}, max_depth={max_depth}, precision={precision}"
            )
            self.depth_estimator = depth_estimator_factory(
                depth_estimator_type=depth_estimator_type,
                min_depth=min_depth,
                max_depth=max_depth,
                dataset_env_type=environment_type,
                precision=precision,
                camera=camera,
            )
        else:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorBase: init: depth_estimator=None, depth_estimator_type={depth_estimator_type}, depth_factor={self.depth_factor}"
            )

        # Prepare maps to undistort color and depth images
        h, w = camera.height, camera.width
        D = camera.D
        if D is None:
            D = np.zeros((5,), dtype=float)
        else:
            D = np.asarray(D).astype(float)
        K = np.asarray(camera.K).astype(float)
        # Printer.green(f'VolumetricIntegratorBase: init: h={h}, w={w}, D={D}, K={K}')
        if np.linalg.norm(D) <= 1e-10:
            self.new_K = K
            self.calib_map1 = None
            self.calib_map2 = None
        else:
            if Parameters.kDepthImageUndistortionUseOptimalNewCameraMatrixWithAlphaScale:
                alpha = Parameters.kDepthImageUndistortionOptimalNewCameraMatrixWithAlphaScaleValue
                self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
            else:
                self.new_K = K
            self.calib_map1, self.calib_map2 = cv2.initUndistortRectifyMap(
                K, D, None, self.new_K, (w, h), cv2.CV_32FC1
            )

        # Store rectified camera intrinsics for use when depth is rectified
        # Extract fx, fy, cx, cy from new_K
        self.rectified_fx = float(self.new_K[0, 0])
        self.rectified_fy = float(self.new_K[1, 1])
        self.rectified_cx = float(self.new_K[0, 2])
        self.rectified_cy = float(self.new_K[1, 2])

    # main loop of the volume integration process
    def run(
        self,
        camera,
        environment_type,
        sensor_type,
        viewer_queue,
        q_in,
        q_in_condition,
        q_out,
        q_out_condition,
        q_management,
        is_running,
        is_looping,
        reset_mutex,
        reset_requested,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_volumetric_integration,
        parameters_dict,
        constructor_kwargs,
        semantic_mapping_shared_state=None,
    ):
        is_running.value = 1
        VolumetricIntegratorBase.print("VolumetricIntegratorBase: starting...")

        # Restore SemanticMappingShared state in spawned process if using spawn method
        if semantic_mapping_shared_state is not None:
            try:
                SemanticMappingShared.import_state(semantic_mapping_shared_state, force=True)
                Printer.green(
                    "VolumetricIntegratorBase: SemanticMappingShared state restored in spawned process"
                )
                VolumetricIntegratorBase.print(
                    "VolumetricIntegratorBase: SemanticMappingShared state restored in spawned process"
                )
            except Exception as e:
                Printer.orange(
                    f"WARNING: VolumetricIntegratorBase: Failed to restore SemanticMappingShared state: {e}"
                )
                traceback.print_exc()

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorBase: received signal {signum}, setting is_running to 0"
            )
            is_running.value = 0
            # Notify condition to wake up from wait
            with q_in_condition:
                q_in_condition.notify_all()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        self.init(camera, environment_type, sensor_type, parameters_dict, constructor_kwargs)
        is_looping.value = 1

        # FPS-based burst control parameters
        is_fps_throttle_enabled = Parameters.kVolumetricIntegrationFpsThrottleEnabled
        fps_max_threshold = Parameters.kVolumetricIntegrationFpsMaxThreshold
        fps_throttle_base_delay = Parameters.kVolumetricIntegrationFpsThrottleBaseDelay
        fps_throttle_scale = Parameters.kVolumetricIntegrationFpsThrottleScale

        # Initialize FPS timer for measuring integration rate
        timer_fps = TimerFps("VolumetricIntegratorBase")
        timer_fps.start()

        # Define dtype for the volumetric integrator
        self.dtype_vertices = get_np_dtype(Parameters.kDenseMappingDtypeVertices)
        self.dtype_colors = get_np_dtype(Parameters.kDenseMappingDtypeColors)
        self.dtype_depths = get_np_dtype(Parameters.kDenseMappingDtypeDepth)
        self.dtype_semantics = get_np_dtype(Parameters.kDenseMappingDtypeSemantics)
        self.dtype_object_ids = get_np_dtype(Parameters.kDenseMappingDtypeObjectIds)

        # main loop
        while is_running.value == 1:
            try:
                # Check if we have management tasks or regular tasks to process
                has_management_task = not q_management.empty()
                has_regular_task = not q_in.empty()

                with q_in_condition:
                    # Wait only if both q_in and q_management queues are empty and reset is not requested
                    while (
                        not has_management_task
                        and not has_regular_task
                        and is_running.value == 1
                        and reset_requested.value != 1
                    ):
                        VolumetricIntegratorBase.print(
                            "VolumetricIntegratorBase: waiting for new task..."
                        )
                        # Use timeout to periodically check is_running flag
                        q_in_condition.wait(timeout=1.0)
                        # Re-check queues after wait
                        has_management_task = not q_management.empty()
                        has_regular_task = not q_in.empty()

                # Check if we should exit before processing
                if is_running.value == 0:
                    VolumetricIntegratorBase.print(
                        "VolumetricIntegratorBase: is_running is 0, exiting..."
                    )
                    break

                # Process tasks if available
                if has_regular_task or has_management_task:
                    # check q_in size and dump a warn message if it is too big
                    q_in_size = q_in.qsize()
                    if q_in_size >= 10:
                        warn_msg = f"!VolumetricIntegratorBase: WARNING: q_in size: {q_in_size}"
                        VolumetricIntegratorBase.print(warn_msg)
                        Printer.orange(warn_msg)

                    self.volume_integration(
                        q_in,
                        q_out,
                        q_out_condition,
                        q_management,
                        viewer_queue,
                        is_running,
                        load_request_completed,
                        load_request_condition,
                        save_request_completed,
                        save_request_condition,
                        time_volumetric_integration,
                    )

                    # check fps after integration
                    timer_fps.refresh()
                    current_fps = timer_fps.get_fps()

                    do_fps_throttling = (
                        is_fps_throttle_enabled
                        and q_in_size > Parameters.kVolumetricIntegrationFpsThrottleMinQueueSize
                    )

                    # FPS-based throttling: if FPS exceeds threshold, add adaptive delay
                    if do_fps_throttling and current_fps > 0.0 and current_fps > fps_max_threshold:
                        # Calculate how much we're over the threshold
                        fps_excess = current_fps - fps_max_threshold
                        # Adaptive delay: base delay + scaled delay based on excess FPS
                        throttle_delay = fps_throttle_base_delay + fps_excess * fps_throttle_scale
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorBase: FPS throttling: current_fps={current_fps:.2f} "
                            f"(threshold={fps_max_threshold:.2f}), adding delay: {throttle_delay:.3f}s"
                        )
                        time.sleep(throttle_delay)

                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: current fps={current_fps:.2f}"
                    )

                else:
                    VolumetricIntegratorBase.print("VolumetricIntegratorBase: q_in is empty...")
                    time.sleep(0.1)  # sleep for a bit before checking the queue again

                self.reset_if_requested(
                    reset_mutex, reset_requested, q_in, q_in_condition, q_out, q_out_condition
                )

            except Exception as e:
                VolumetricIntegratorBase.print("VolumetricIntegratorBase: Exception: ", e)
                traceback.print_exc()

        is_looping.value = 0

        # Stop the volume integrator: This is expected to be an implementation-specific operation
        self._stop_volume_integrator_implementation()

        empty_queue(q_in)  # empty the queue before exiting
        empty_queue(q_out)  # empty the queue before exiting

        # Clean up LoggerQueue instances in this spawned process before exiting
        LoggerQueue.stop_all_instances()

        VolumetricIntegratorBase.print("VolumetricIntegratorBase: loop exit...")

    def estimate_depth_if_needed_and_rectify(
        self, keyframe_data: VolumetricIntegrationKeyframeData
    ):
        color = keyframe_data.img
        color_right = keyframe_data.img_right
        depth = keyframe_data.depth
        semantic = keyframe_data.semantic_img
        semantic_instances = keyframe_data.semantic_instances_img

        pts3d = None
        semantic_undistorted = None
        semantic_instances_undistorted = None

        if depth is None or depth.size == 0:
            if self.depth_estimator is None:
                Printer.yellow(
                    "VolumetricIntegratorBase: depth_estimator is None, depth is None, skipping the keyframe..."
                )
                return None, None, None, None  # skip this keyframe
            else:
                inference_start_time = time.time()
                if keyframe_data.id in self.img_id_to_depth:
                    depth = self.img_id_to_depth[keyframe_data.id]
                else:
                    if self.sensor_type == SensorType.MONOCULAR:
                        Printer.error(
                            "VolumetricIntegratorBase: You cannot use a MONOCULAR depth estimator in the back-end with a MONOCULAR SLAM system!\n\t Their scale factors will be inconsistent!"
                        )
                    depth, pts3d = self.depth_estimator.infer(color, color_right)
                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: depth inference time: {time.time() - inference_start_time}"
                    )
                    if self.parameters_dict[
                        "kVolumetricIntegrationDepthEstimationFilterShadowPoints"
                    ]:
                        depth = filter_shadow_points(depth, delta_depth=None)

        # if not depth.dtype in [np.uint8, np.uint16, np.float32]:
        if not depth.dtype == np.float32:
            if USE_CPP:
                # In C++ mode, the depth conversion is performed inside the C++ Frame code
                # and not propagated to the Python code, so we need to make it here again
                # TODO: fix this and uniform depth management between Python and C++
                depth = depth.astype(np.float32) * self.camera.depth_factor
            else:
                depth = depth.astype(np.float32)
            keyframe_data.depth = depth

        if self.calib_map1 is not None and self.calib_map2 is not None:
            def _remap_label_image(label_img):
                """Remap label/instance images using cv2 with safe dtype."""
                if label_img is None:
                    return None
                # cv2.remap does not support int32; use float32 and restore int32.
                if label_img.dtype not in (np.uint8, np.uint16, np.float32):
                    label_img = label_img.astype(np.float32, copy=False)
                remapped = cv2.remap(
                    label_img,
                    self.calib_map1,
                    self.calib_map2,
                    interpolation=cv2.INTER_NEAREST,
                )
                # Restore int32 labels for downstream integration
                return np.ascontiguousarray(remapped, dtype=np.int32)

            color_undistorted = cv2.remap(
                color, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_LINEAR
            )
            depth_undistorted = cv2.remap(
                depth, self.calib_map1, self.calib_map2, interpolation=cv2.INTER_NEAREST
            )
            if semantic is not None:
                semantic_undistorted = _remap_label_image(semantic)
            if semantic_instances is not None:
                semantic_instances_undistorted = _remap_label_image(semantic_instances)
        else:
            color_undistorted = color
            depth_undistorted = depth
            semantic_undistorted = semantic
            semantic_instances_undistorted = semantic_instances

        if self.depth_estimator is not None:
            if not keyframe_data.id in self.img_id_to_depth:
                self.img_id_to_depth[keyframe_data.id] = depth_undistorted

        color_undistorted = cv2.cvtColor(color_undistorted, cv2.COLOR_BGR2RGB)

        return (
            color_undistorted,
            depth_undistorted,
            pts3d,
            semantic_undistorted,
            semantic_instances_undistorted,
        )

    def get_camera_intrinsics_for_depth(self):
        """
        Returns the appropriate camera intrinsics (fx, fy, cx, cy) to use with the depth image.
        If depth rectification was performed (calib_map1/calib_map2 are not None), returns rectified intrinsics.
        Otherwise, returns original camera intrinsics.
        """
        if self.calib_map1 is not None and self.calib_map2 is not None:
            # Depth was rectified, use rectified intrinsics
            return self.rectified_fx, self.rectified_fy, self.rectified_cx, self.rectified_cy
        else:
            # No rectification performed, use original intrinsics
            return self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy

    def get_camera_for_rectified_depth(self):
        """
        Returns a camera object with rectified intrinsics if depth rectification was performed.
        Otherwise, returns the original camera object.
        This is useful for systems that require a Camera object (e.g., Gaussian Splatting).
        """
        if self.calib_map1 is not None and self.calib_map2 is not None:
            # Depth was rectified, create a copy of camera with rectified intrinsics
            # Create a shallow copy and modify intrinsics
            import copy

            rectified_camera = copy.copy(self.camera)
            rectified_camera.fx = self.rectified_fx
            rectified_camera.fy = self.rectified_fy
            rectified_camera.cx = self.rectified_cx
            rectified_camera.cy = self.rectified_cy
            # Update intrinsic matrices
            rectified_camera.set_intrinsic_matrices()
            return rectified_camera
        else:
            # No rectification performed, return original camera
            return self.camera

    def volume_integration(
        self,
        q_in,
        q_out,
        q_out_condition,
        q_management,
        viewer_queue,
        is_running,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_volumetric_integration,
    ):
        # to be overridden in the derived classes
        raise NotImplementedError(
            "Volume integration is not implemented for this volumetric integrator"
        )

    # Called by add_keyframe() and periodically by the keyframe_queue_timer
    def flush_keyframe_queue(self):
        # iterate over the keyframe queue and flush the keyframes into the task queue
        # Thread-safe pattern using lock to protect concurrent access
        verbose = False
        with self.keyframe_queue_lock:
            if len(self.keyframe_queue) == 0:
                if verbose:
                    VolumetricIntegratorBase.print(
                        "VolumetricIntegratorBase: flush_keyframe_queue: keyframe queue is empty"
                    )
                return

            # Process all items currently in the queue, but limit iterations to prevent infinite loops
            # if all items keep rotating without meeting criteria
            max_checks = len(self.keyframe_queue)
            processed_count = 0

            while len(self.keyframe_queue) > 0:
                try:
                    kf_to_process = self.keyframe_queue[0]
                except IndexError:
                    VolumetricIntegratorBase.print(
                        "VolumetricIntegratorBase: flush_keyframe_queue: IndexError: keyframe queue is empty"
                    )
                    break

                # If semantic mapping is enabled and the keyframe has no semantic image, wait for it so we can process it during volumetric integration
                wait_for_semantic_mapping = False
                if SemanticMappingShared.is_semantic_mapping_enabled():
                    if not kf_to_process.is_semantics_available():
                        wait_for_semantic_mapping = True
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorBase: flush_keyframe_queue: waiting for semantic mapping for keyframe {kf_to_process.id} (kid: {kf_to_process.kid})"
                        )
                    else:
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorBase: flush_keyframe_queue: keyframe {kf_to_process.id} (kid: {kf_to_process.kid}) has semantic image {kf_to_process.semantic_img.shape}"
                        )

                # We integrate only the keyframes that have been processed by LBA at least once.
                if (
                    kf_to_process.lba_count >= Parameters.kVolumetricIntegrationMinNumLBATimes
                    and not wait_for_semantic_mapping
                ):
                    # Remove from queue and schedule task
                    try:
                        self.keyframe_queue.popleft()
                        processed_count += 1
                    except IndexError:
                        break
                    # Release lock before adding task to avoid holding lock during potentially slow operation
                    kf_to_add = kf_to_process
                    task_type = VolumetricIntegrationTaskType.INTEGRATE
                    task = VolumetricIntegrationTask(kf_to_add, task_type=task_type)
                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: Adding integration task with keyframe id: {kf_to_add.id} (kid: {kf_to_add.kid})"
                    )
                    # Add task outside the lock to avoid blocking other threads
                    self.add_task(task)
                else:
                    # Move the current head to the tail to examine others next
                    if len(self.keyframe_queue) <= 1:
                        # Only one item left, no point in rotating
                        break
                    self.keyframe_queue.rotate(-1)
                    processed_count += 1

                if processed_count >= max_checks:
                    break

    # This lives in the main process and is called by the local mapping process.
    def add_keyframe(self, keyframe: KeyFrame, img, img_right, depth, print=print):
        VolumetricIntegratorBase.print(
            f"VolumetricIntegratorBase: add_keyframe: adding frame {keyframe.id}"
        )

        use_depth_estimator = Parameters.kVolumetricIntegrationUseDepthEstimator
        if (depth is None or depth.size == 0) and not use_depth_estimator:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorBase: add_keyframe: depth is None -> skipping frame {keyframe.id}"
            )
            return
        try:
            # We accumulate the keyframe in a queue.
            # We integrate only the keyframes that have been processed by LBA at least once.
            with self.keyframe_queue_lock:
                self.keyframe_queue.append(keyframe)
            # Flush outside the lock to avoid deadlock (flush_keyframe_queue will acquire the lock)
            self.flush_keyframe_queue()

        except Exception as e:
            VolumetricIntegratorBase.print(f"VolumetricIntegratorBase: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

    def add_task(self, task: VolumetricIntegrationTask, front=True):
        if self.is_running.value == 1:
            with self.q_in_condition:
                try:
                    if front:
                        push_to_front(self.q_in, task)
                    else:
                        # Use timeout to avoid indefinite blocking
                        self.q_in.put(task, timeout=1.0)
                    self.q_in_condition.notify_all()
                except Exception as e:
                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: add_task: EXCEPTION: {e} !!!"
                    )
                    if kPrintTrackebackDetails:
                        traceback_details = traceback.format_exc()
                        VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

    def add_update_output_task(self):
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(
                    VolumetricIntegrationTask(task_type=VolumetricIntegrationTaskType.UPDATE_OUTPUT)
                )
                self.q_in_condition.notify_all()

    def rebuild(self, map: Map):
        VolumetricIntegratorBase.print("")
        VolumetricIntegratorBase.print(
            f"VolumetricIntegratorBase: rebuild() rebuilding volumetric mapping..."
        )
        if self.is_running.value == 1:
            # Clear the task queue first to ensure RESET is processed before any INTEGRATE tasks
            # that might be added by flush_keyframe_queue(). This prevents RESET from being postponed.
            with self.q_in_condition:
                empty_queue(self.q_in)
                VolumetricIntegratorBase.print(
                    "VolumetricIntegratorBase: rebuild() cleared task queue before adding RESET task"
                )

            task = VolumetricIntegrationTask(task_type=VolumetricIntegrationTaskType.RESET)
            # Add to management queue and notify using q_in_condition (shared for both q_in and q_management queues)
            try:
                with self.q_in_condition:
                    self.q_management.put(task, timeout=1.0)
                    self.q_in_condition.notify_all()  # Wake up worker to process management task
            except Exception as e:
                VolumetricIntegratorBase.print(
                    f"VolumetricIntegratorBase: rebuild: EXCEPTION: q_management.put(task): {e} !!!"
                )
                if kPrintTrackebackDetails:
                    traceback_details = traceback.format_exc()
                    VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

            # Wait for the RESET task to be processed with timeout
            max_wait_time = 5.0  # seconds
            wait_start = time.time()
            while self.is_running.value == 1 and not self.q_management.empty():
                if time.time() - wait_start > max_wait_time:
                    VolumetricIntegratorBase.print(
                        f"VolumetricIntegratorBase: rebuild() WARNING: Timeout waiting for RESET task. "
                        f"Queue may still contain management tasks."
                    )
                    break
                time.sleep(0.1)

            # Add all the map keyframes to the queue for re-integration
            with self.keyframe_queue_lock:
                self.keyframe_queue.clear()
                for kf in map.keyframes:
                    if (
                        not kf.is_bad()
                        and kf.lba_count >= Parameters.kVolumetricIntegrationMinNumLBATimes
                    ):
                        if kf.depth_img is None:
                            VolumetricIntegratorBase.print(
                                f"VolumetricIntegratorBase: rebuild: depth is None -> skipping frame {kf.id}"
                            )
                            continue
                        self.keyframe_queue.append(kf)
                    else:
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorBase: rebuild: keyframe {kf.id} (kid: {kf.kid}) is bad or has not been processed by LBA at least once -> skipping"
                        )
            # Flush outside the lock to avoid deadlock
            self.flush_keyframe_queue()

    def pop_output(
        self, timeout=Parameters.kLoopDetectingTimeoutPopKeyframe
    ) -> VolumetricIntegrationOutput:
        q_out = self.q_out
        q_out_condition = self.q_out_condition

        if self.is_running.value == 0:
            return None
        with q_out_condition:
            while q_out.empty() and self.is_running.value == 1:
                ok = q_out_condition.wait(timeout=timeout)
                if not ok:
                    VolumetricIntegratorBase.print("VolumetricIntegratorBase: pop_output: timeout")
                    break  # Timeout occurred
        if q_out.empty():
            return None
        try:
            return q_out.get(timeout=timeout)
        except Exception as e:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorBase: pop_output: encountered exception: {e}"
            )
            return None

    def draw_output(self, output: VolumetricIntegrationOutput):
        from pyslam.viz.viewer3D import Viewer3DDenseInput

        if output is None:
            return
        if self.viewer_queue is None:
            Printer.orange("WARNING: viewer_queue is None")
            return
        point_cloud = output.point_cloud
        mesh = output.mesh
        dense_state = Viewer3DDenseInput()
        if mesh is not None:
            dense_state.mesh = (
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                np.asarray(mesh.vertex_colors),
            )  # ,np.asarray(mesh.vertex_normals))
        else:
            if point_cloud is not None:
                points = np.asarray(point_cloud.points)
                colors = np.asarray(point_cloud.colors)
                if colors.shape[1] == 4:
                    colors = colors[:, 0:3]
                print(
                    f"Viewer3D: draw_dense_geometry - points.shape: {points.shape}, colors.shape: {colors.shape}"
                )
                semantic_colors = None
                if hasattr(point_cloud, "semantic_colors"):
                    semantic_colors = (
                        np.asarray(point_cloud.semantic_colors)
                        if point_cloud.semantic_colors is not None
                        else None
                    )
                object_colors = None
                if hasattr(point_cloud, "object_colors"):
                    object_colors = (
                        np.asarray(point_cloud.object_colors)
                        if point_cloud.object_colors is not None
                        else None
                    )
                print(
                    f"Viewer3D: draw_dense_geometry - points.shape: {points.shape}, colors.shape: {colors.shape}"
                )
                if semantic_colors is not None and semantic_colors.shape[0] > 0:
                    print(
                        f"Viewer3D: draw_dense_geometry - semantic_colors.shape: {semantic_colors.shape}"
                    )
                dense_state.point_cloud = (points, colors, semantic_colors, object_colors)
            else:
                Printer.orange("WARNING: both point_cloud and mesh are None")
        self.viewer_queue.put(dense_state)
