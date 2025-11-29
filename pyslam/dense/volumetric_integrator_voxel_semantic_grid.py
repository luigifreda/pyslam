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
import math
import torch.multiprocessing as mp

import cv2
import numpy as np

from pyslam.slam import Camera, Map, KeyFrame, Frame, USE_CPP

from pyslam.io.dataset_types import DatasetEnvironmentType, SensorType

from pyslam.utilities.system import Printer
from pyslam.utilities.timer import TimerFps
from pyslam.config_parameters import Parameters
from pyslam.utilities.depth import depth2pointcloud, filter_shadow_points
from pyslam.utilities.geometry import inv_T
from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared

from volumetric_grid import (
    VoxelSemanticGrid,
    VoxelBlockSemanticGrid,
    VoxelBlockSemanticProbabilisticGrid,
    TBBUtils,
)

import traceback

from collections import deque


from enum import Enum

import logging

import open3d as o3d

from pyslam.dense.volumetric_integrator_base import (
    VolumetricIntegrationTaskType,
    VolumetricIntegrationTask,
    VolumetricIntegrationOutput,
    VolumetricIntegrationPointCloud,
    VolumetricIntegratorBase,
    VolumetricIntegrationKeyframeData,
)


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


class VolumetricIntegratorVoxelSemanticGrid(VolumetricIntegratorBase):
    def __init__(self, camera, environment_type, sensor_type, volumetric_integrator_type, **kwargs):
        super().__init__(
            camera, environment_type, sensor_type, volumetric_integrator_type, **kwargs
        )

    def init(
        self,
        camera: Camera,
        environment_type: DatasetEnvironmentType,
        sensor_type: SensorType,
        parameters_dict,
        constructor_kwargs,
    ):
        VolumetricIntegratorBase.init(
            self, camera, environment_type, sensor_type, parameters_dict, constructor_kwargs
        )
        self.init_print()

        self.volumetric_integration_depth_trunc = (
            Parameters.kVolumetricIntegrationDepthTruncIndoor
            if environment_type == DatasetEnvironmentType.INDOOR
            else Parameters.kVolumetricIntegrationDepthTruncOutdoor
        )

        TBBUtils.set_max_threads(Parameters.kVolumetricIntegrationTBBThreads)
        use_voxel_blocks = constructor_kwargs.get("use_voxel_blocks", True)
        use_semantic_probabilistic = constructor_kwargs.get("use_semantic_probabilistic", False)
        if use_voxel_blocks:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorVoxelSemanticGrid: using voxel blocks"
            )
            if use_semantic_probabilistic:
                VolumetricIntegratorBase.print(
                    f"VolumetricIntegratorVoxelSemanticGrid: using probabilistic semantic integration"
                )
                voxel_grid_type = VoxelBlockSemanticProbabilisticGrid
            else:
                voxel_grid_type = VoxelBlockSemanticGrid
            self.volume = voxel_grid_type(
                voxel_size=Parameters.kVolumetricIntegrationVoxelLength,
                block_size=Parameters.kVolumetricIntegrationBlockSize,
            )
        else:
            self.volume = VoxelSemanticGrid(voxel_size=Parameters.kVolumetricIntegrationVoxelLength)

        try:
            # Settings for semantic integration:
            #       - The decay rate is just used with semantic probabilistic integration
            #       - With non-probabilistic integration, the confidence counter is only updated when the depth is below the depth threshold
            #       - With probabilistic integration, we use the depth to compute the confidence weights for semantics:
            #         if depth < depth_threshold, the confidence weight is 1.0, otherwise it exponentially decays with the depth decay rate
            if environment_type == DatasetEnvironmentType.INDOOR:
                self.volume.set_depth_threshold(
                    Parameters.kVolumetricSemanticProbabilisticIntegrationDepthThresholdIndoor
                )
                self.volume.set_depth_decay_rate(
                    Parameters.kVolumetricSemanticProbabilisticIntegrationDepthDecayRateIndoor
                )
            else:
                self.volume.set_depth_threshold(
                    Parameters.kVolumetricSemanticProbabilisticIntegrationDepthThresholdOutdoor
                )
                self.volume.set_depth_decay_rate(
                    Parameters.kVolumetricSemanticProbabilisticIntegrationDepthDecayRateOutdoor
                )
        except Exception as e:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorVoxelSemanticGrid: EXCEPTION: {e} !!!"
            )
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

        if not SemanticMappingShared.is_semantic_mapping_enabled():
            Printer.red(
                f"\nVolumetricIntegratorVoxelSemanticGrid: semantic mapping is not enabled -> skipping semantic integration"
            )
            Printer.red(
                f"\tYou can enable semantic mapping by setting the kSemanticSegmentationType parameter in config_parameters.py or select a non-semantic volumetric integrator."
            )

    def volume_integration(
        self,
        q_in,
        q_out,
        q_out_condition,
        q_management,
        is_running,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_volumetric_integration,
    ):
        # print('VolumetricIntegratorVoxelSemanticGrid: volume_integration')
        last_output = None
        do_output = False
        timer = TimerFps("VolumetricIntegratorVoxelSemanticGrid", is_verbose=kTimerVerbose)
        timer.start()
        try:
            if is_running.value == 1:

                self.last_management_task = None
                try:
                    self.last_management_task = (
                        q_management.get_nowait()
                    )  # non-blocking call to get a new management task for volume integration
                except:
                    pass
                if (
                    self.last_management_task is not None
                    and self.last_management_task.task_type == VolumetricIntegrationTaskType.RESET
                ):
                    VolumetricIntegratorBase.print(
                        "VolumetricIntegratorVoxelSemanticGrid: resetting..."
                    )
                    self.volume.reset()

                self.last_input_task: VolumetricIntegrationTask = (
                    q_in.get()
                )  # blocking call to get a new input task for volume integration

                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                else:

                    if self.last_input_task.task_type == VolumetricIntegrationTaskType.INTEGRATE:
                        keyframe_data: VolumetricIntegrationKeyframeData = (
                            self.last_input_task.keyframe_data
                        )

                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorVoxelSemanticGrid: processing keyframe_data: {keyframe_data}"
                        )

                        color_undistorted, depth_undistorted, pts3d, semantic_undistorted = (
                            self.estimate_depth_if_needed_and_rectify(keyframe_data)
                        )

                        pose = keyframe_data.pose  # Tcw
                        inv_pose = inv_T(pose)  # Twc

                        # print(f"VolumetricIntegratorVoxelSemanticGrid: color_undistorted: shape: {color_undistorted.shape}, type: {color_undistorted.dtype}")

                        if kVerbose:
                            if depth_undistorted is not None:
                                VolumetricIntegratorBase.print(
                                    f"\t\tdepth_undistorted: shape: {depth_undistorted.shape}, type: {depth_undistorted.dtype}"
                                )
                                if depth_undistorted.size > 0:
                                    min_depth = np.min(depth_undistorted)
                                    max_depth = np.max(depth_undistorted)
                                    VolumetricIntegratorBase.print(
                                        f"\t\tmin_depth: {min_depth}, max_depth: {max_depth}"
                                    )
                                else:
                                    VolumetricIntegratorBase.print(
                                        f"\t\tdepth_undistorted is empty, skipping min/max computation"
                                    )

                            if pts3d is not None:
                                VolumetricIntegratorBase.print(
                                    f"\t\tpts3d: shape: {pts3d.shape}, type: {pts3d.dtype}"
                                )
                            if (
                                semantic_undistorted is not None
                                and semantic_undistorted.shape[0] > 0
                            ):
                                VolumetricIntegratorBase.print(
                                    f"\t\tsemantic_undistorted: shape: {semantic_undistorted.shape}, type: {semantic_undistorted.dtype}"
                                )

                        # Filter depth
                        filter_depth = (
                            Parameters.kVolumetricIntegratorGridShadowPointsFilter
                        )  # do you want to filter the depth?
                        if filter_depth:
                            depth_filtered = filter_shadow_points(
                                depth_undistorted, delta_depth=None
                            )
                        else:
                            depth_filtered = depth_undistorted

                        point_cloud = depth2pointcloud(
                            depth_filtered,
                            color_undistorted,
                            self.camera.fx,
                            self.camera.fy,
                            self.camera.cx,
                            self.camera.cy,
                            self.volumetric_integration_depth_trunc,
                            semantic_image=semantic_undistorted,
                            instance_image=None,  # TODO: at present time, we do not use instance segmentation for volumetric integration
                        )

                        # camera depths are used to generate confidence weights for semantics
                        # if depth is less than kDepthThreshold, the confidence is 1.0
                        # if depth is greater than kDepthThreshold, the confidence decays exponentially with depth
                        depths = point_cloud.points[:, 2]
                        depths = np.ascontiguousarray(depths, dtype=np.float32)

                        # transform point cloud from camera coordinate system to world coordinate system
                        points_world = (
                            inv_pose[:3, :3] @ point_cloud.points.T + inv_pose[:3, 3].reshape(3, 1)
                        ).T
                        point_cloud.points = points_world

                        # Ensure colors are in the correct format for C++ integration
                        # Colors from depth2pointcloud are already in [0, 1] range as float
                        # Convert to contiguous float32 array for C++ binding
                        if point_cloud.colors is not None and point_cloud.colors.size > 0:
                            colors = np.ascontiguousarray(point_cloud.colors, dtype=np.float32)
                            # Ensure colors are in valid [0, 1] range
                            # colors = np.clip(colors, 0.0, 1.0)
                        else:
                            # If no colors, create default black colors
                            colors = np.zeros((point_cloud.points.shape[0], 3), dtype=np.float32)
                        points = np.ascontiguousarray(point_cloud.points, dtype=np.float64)

                        if point_cloud.semantics is not None and point_cloud.semantics.size > 0:
                            # extract semantics from semantic_undistorted
                            semantics = np.ascontiguousarray(point_cloud.semantics, dtype=np.int32)
                        else:
                            semantics = np.ascontiguousarray(
                                np.ones(point_cloud.points.shape[0], dtype=np.int32), dtype=np.int32
                            )

                        if (
                            point_cloud.instance_ids is not None
                            and point_cloud.instance_ids.size > 0
                        ):
                            instance_ids = np.ascontiguousarray(
                                point_cloud.instance_ids, dtype=np.int32
                            )
                        else:
                            # instance_ids = np.ascontiguousarray(np.ones(point_cloud.points.shape[0], dtype=np.int32), dtype=np.int32)
                            instance_ids = None

                        self.volume.integrate(
                            points,
                            colors,
                            instance_ids=instance_ids,
                            class_ids=semantics,
                            depths=depths,  # used to compute confidence weights for semantics
                        )

                        self.last_integrated_id = keyframe_data.id

                        do_output = True
                        if self.last_output is not None:
                            elapsed_time = time.perf_counter() - self.last_output.timestamp
                            if elapsed_time < Parameters.kVolumetricIntegrationOutputTimeInterval:
                                do_output = False

                    elif self.last_input_task.task_type == VolumetricIntegrationTaskType.SAVE:
                        save_path = self.last_input_task.load_save_path
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorVoxelSemanticGrid: saving point cloud to: {save_path}"
                        )
                        points, colors, semantics, instance_ids = self.volume.get_voxel_data(
                            min_count=Parameters.kVolumetricIntegratorGridMinCount
                        )
                        points = np.asarray(points, dtype=np.float64)
                        colors = np.asarray(colors, dtype=np.float32)
                        # Ensure colors are in [0, 1] range (they should already be, but clamp to be safe)
                        # colors = np.clip(colors, 0.0, 1.0)
                        point_cloud_o3d = o3d.geometry.PointCloud()
                        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
                        point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
                        o3d.io.write_point_cloud(save_path, point_cloud_o3d)

                        last_output = VolumetricIntegrationOutput(self.last_input_task.task_type)

                    elif (
                        self.last_input_task.task_type
                        == VolumetricIntegrationTaskType.UPDATE_OUTPUT
                    ):
                        do_output = True

                    if do_output:
                        mesh_out, pc_out = None, None
                        points, colors, semantics, instance_ids = self.volume.get_voxel_data(
                            min_count=Parameters.kVolumetricIntegratorGridMinCount
                        )
                        # Convert C++ vectors to numpy arrays with proper shape and dtype
                        points = np.ascontiguousarray(points, dtype=np.float64)
                        colors = np.ascontiguousarray(colors, dtype=np.float32)
                        semantics = (
                            np.ascontiguousarray(semantics) if semantics is not None else None
                        )
                        instance_ids = (
                            np.ascontiguousarray(instance_ids) if instance_ids is not None else None
                        )
                        semantic_colors = (
                            np.ascontiguousarray(
                                SemanticMappingShared.sem_img_to_rgb(semantics, bgr=True),
                                dtype=np.float32,
                            )
                            / 255.0
                            if semantics is not None
                            and SemanticMappingShared.is_semantic_mapping_enabled()
                            else None
                        )

                        # Ensure colors are in [0, 1] range (they should already be, but clamp to be safe)
                        # colors = np.clip(colors, 0.0, 1.0)
                        pc_out = VolumetricIntegrationPointCloud(
                            points=points,
                            colors=colors,
                            semantics=semantics,
                            instance_ids=instance_ids,
                            semantic_colors=semantic_colors,
                        )
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorVoxelSemanticGrid: id: {self.last_integrated_id} -> PointCloud: points: {pc_out.points.shape}"
                        )

                        last_output = VolumetricIntegrationOutput(
                            self.last_input_task.task_type,
                            self.last_integrated_id,
                            pc_out,
                            mesh_out,
                        )
                        self.last_output = last_output

                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorVoxelSemanticGrid: last output id: {last_output.id if last_output is not None else None}"
                        )

                    if is_running.value == 1 and last_output is not None:
                        if (
                            last_output.task_type == VolumetricIntegrationTaskType.INTEGRATE
                            or last_output.task_type == VolumetricIntegrationTaskType.UPDATE_OUTPUT
                        ):
                            with q_out_condition:
                                # push the computed output in the output queue (for viz or other tasks)
                                last_output.timestamp = time.perf_counter()
                                q_out.put(last_output)
                                q_out_condition.notify_all()
                                VolumetricIntegratorBase.print(
                                    f"VolumetricIntegratorVoxelSemanticGrid: pushed new output to q_out size: {q_out.qsize()}"
                                )
                        elif last_output.task_type == VolumetricIntegrationTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()

        except Exception as e:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorVoxelSemanticGrid: EXCEPTION: {e} !!!"
            )
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

        timer.refresh()
        time_volumetric_integration.value = timer.last_elapsed
        id_info = f"last output id: {self.last_output.id}" if self.last_output is not None else ""
        VolumetricIntegratorBase.print(
            f"VolumetricIntegratorVoxelSemanticGrid: {id_info}, last integrated id: {self.last_integrated_id}, q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, volume-integration elapsed time: {time_volumetric_integration.value}"
        )
