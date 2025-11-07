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

import traceback

from collections import deque


from enum import Enum

import logging

import open3d as o3d

from pyslam.dense.volumetric_integrator_base import (
    VolumetricIntegrationTaskType,
    VolumetricIntegrationOutput,
    VolumetricIntegrationMesh,
    VolumetricIntegrationPointCloud,
    VolumetricIntegratorBase,
)


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


class VolumetricIntegratorTsdf(VolumetricIntegratorBase):
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

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=Parameters.kVolumetricIntegrationVoxelLength,
            sdf_trunc=Parameters.kVolumetricIntegrationSdfTrunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        self.o3d_camera = o3d.camera.PinholeCameraIntrinsic(
            width=camera.width,
            height=camera.height,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
        )

    def volume_integration(
        self,
        q_in,
        q_out,
        q_out_condition,
        is_running,
        load_request_completed,
        load_request_condition,
        save_request_completed,
        save_request_condition,
        time_volumetric_integration,
    ):
        # print('VolumetricIntegratorTsdf: volume_integration')
        last_output = None
        do_output = False
        timer = TimerFps("VolumetricIntegratorTsdf", is_verbose=kTimerVerbose)
        timer.start()
        try:
            if is_running.value == 1:

                self.last_input_task = (
                    q_in.get()
                )  # blocking call to get a new input task for volume integration

                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                else:

                    if self.last_input_task.task_type == VolumetricIntegrationTaskType.INTEGRATE:
                        keyframe_data = self.last_input_task.keyframe_data

                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorTsdf: processing keyframe_data: {keyframe_data}"
                        )

                        color_undistorted, depth_undistorted, pts3d, semantic_undistorted = (
                            self.estimate_depth_if_needed_and_rectify(keyframe_data)
                        )

                        pose = keyframe_data.pose  # Tcw
                        # inv_pose = inv_T(pose)   # Twc

                        # print(f"VolumetricIntegratorTsdf: color_undistorted: shape: {color_undistorted.shape}, type: {color_undistorted.dtype}")

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

                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(color_undistorted),
                            o3d.geometry.Image(depth_undistorted),
                            depth_scale=self.depth_factor,
                            depth_trunc=self.volumetric_integration_depth_trunc,
                            convert_rgb_to_intensity=False,
                        )

                        self.volume.integrate(rgbd, self.o3d_camera, pose)

                        self.last_integrated_id = keyframe_data.id

                        do_output = True
                        if self.last_output is not None:
                            elapsed_time = time.perf_counter() - self.last_output.timestamp
                            if elapsed_time < Parameters.kVolumetricIntegrationOutputTimeInterval:
                                do_output = False

                    elif self.last_input_task.task_type == VolumetricIntegrationTaskType.SAVE:
                        save_path = self.last_input_task.load_save_path
                        if Parameters.kVolumetricIntegrationExtractMesh:
                            VolumetricIntegratorBase.print(
                                f"VolumetricIntegratorTsdf: saving mesh to: {save_path}"
                            )
                            mesh = self.volume.extract_triangle_mesh()
                            # mesh.compute_vertex_normals()
                            o3d.io.write_triangle_mesh(save_path, mesh)
                        else:
                            VolumetricIntegratorBase.print(
                                f"VolumetricIntegratorTsdf: saving point cloud to: {save_path}"
                            )
                            point_cloud = self.volume.extract_point_cloud()
                            o3d.io.write_point_cloud(save_path, point_cloud)

                        last_output = VolumetricIntegrationOutput(self.last_input_task.task_type)

                    elif self.last_input_task.task_type == VolumetricIntegrationTaskType.RESET:
                        self.volume.reset()
                    elif (
                        self.last_input_task.task_type
                        == VolumetricIntegrationTaskType.UPDATE_OUTPUT
                    ):
                        do_output = True

                    if do_output:
                        mesh_out, pc_out = None, None
                        if Parameters.kVolumetricIntegrationExtractMesh:
                            mesh = self.volume.extract_triangle_mesh()
                            # mesh.compute_vertex_normals()
                            mesh_out = VolumetricIntegrationMesh(mesh)
                            VolumetricIntegratorBase.print(
                                f"VolumetricIntegratorTsdf: id: {self.last_integrated_id} -> Mesh: points: {mesh_out.vertices.shape}"
                            )
                        else:
                            point_cloud = self.volume.extract_point_cloud()
                            pc_out = VolumetricIntegrationPointCloud(point_cloud)
                            VolumetricIntegratorBase.print(
                                f"VolumetricIntegratorTsdf: id: {self.last_integrated_id} -> PointCloud: points: {pc_out.points.shape}"
                            )

                        last_output = VolumetricIntegrationOutput(
                            self.last_input_task.task_type,
                            self.last_integrated_id,
                            pc_out,
                            mesh_out,
                        )
                        self.last_output = last_output

                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorTsdf: last output id: {last_output.id if last_output is not None else None}"
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
                                    f"VolumetricIntegratorTsdf: pushed new output to q_out size: {q_out.qsize()}"
                                )
                        elif last_output.task_type == VolumetricIntegrationTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()

        except Exception as e:
            VolumetricIntegratorBase.print(f"VolumetricIntegratorTsdf: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

        timer.refresh()
        time_volumetric_integration.value = timer.last_elapsed
        id_info = f"last output id: {self.last_output.id}" if self.last_output is not None else ""
        VolumetricIntegratorBase.print(
            f"VolumetricIntegratorTsdf: {id_info}, last integrated id: {self.last_integrated_id}, q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, volume-integration elapsed time: {time_volumetric_integration.value}"
        )
