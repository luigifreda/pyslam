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

import pyslam.config as config

config.cfg.set_lib("gaussian_splatting")

import cv2
import numpy as np

from pyslam.slam import Camera, Map, KeyFrame, Frame

from pyslam.io.dataset_types import DatasetEnvironmentType, SensorType
from pyslam.utilities.geometry import inv_T
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

from monogs.gaussian_splatting_manager import GaussianSplattingManager
from monogs.utils.config_utils import load_config


kVerbose = True
kTimerVerbose = False

kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"
kGaussianSplattingOutputFolder = kRootFolder + "/results/gaussian_splatting"

kGaussianSplattingConfigDefaultPath = kRootFolder + "/settings/gaussian_splatting_base_config.yaml"


class VolumetricIntegratorGaussianSplatting(VolumetricIntegratorBase):
    def __init__(self, camera, environment_type, sensor_type, volumetric_integrator_type, **kwargs):
        super().__init__(
            camera, environment_type, sensor_type, volumetric_integrator_type, **kwargs
        )
        Parameters.kVolumetricIntegrationMinNumLBATimes = 0  # In MonoGS backend with optimize for keyframes poses too. For this reason, we don't need to wait for enough LBA passes over the keyframes.
        Printer.yellow(
            f"VolumetricIntegratorGaussianSplatting: init - setting Parameters.kVolumetricIntegrationMinNumLBATimes to zero"
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

        if not os.path.exists(kGaussianSplattingOutputFolder):
            os.makedirs(kGaussianSplattingOutputFolder, exist_ok=True)

        self.gs_config = load_config(kGaussianSplattingConfigDefaultPath)
        self.gsm = GaussianSplattingManager(
            self.gs_config,
            save_results=True,
            save_dir=kGaussianSplattingOutputFolder,
            monocular=(sensor_type == SensorType.MONOCULAR),
            live_mode=False,
            use_gui=True,
            eval_rendering=False,
            use_dataset=False,
            print_fun=print,
        )
        self.gsm.start()
        self.volume = self.gsm  # self.volume is called in the base class

    def _stop_volume_integrator_implementation(self):
        self.gsm.stop()

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
        # print('VolumetricIntegratorGaussianSplatting: volume_integration')
        last_output = None
        do_output = False
        timer = TimerFps("VolumetricIntegratorGaussianSplatting", is_verbose=kTimerVerbose)
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
                        "VolumetricIntegratorGaussianSplatting: resetting..."
                    )
                    self.volume.reset()

                self.last_input_task = (
                    q_in.get()
                )  # blocking call to get a new input task for volume integration

                if self.last_input_task is None:
                    is_running.value = 0  # got a None to exit
                else:

                    if self.last_input_task.task_type == VolumetricIntegrationTaskType.INTEGRATE:
                        keyframe_data = self.last_input_task.keyframe_data

                        color_undistorted, depth_undistorted, pts3d, semantic_undistorted = (
                            self.estimate_depth_if_needed_and_rectify(keyframe_data)
                        )

                        pose = keyframe_data.pose  # Tcw
                        # inv_pose = inv_T(pose)   # Twc

                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorGaussianSplatting: keyframe id: {keyframe_data.id}, depth_undistorted: shape: {depth_undistorted.shape}, type: {depth_undistorted.dtype}"
                        )

                        self.gsm.add_keyframe(
                            keyframe_data.id,
                            keyframe_data.camera,
                            color_undistorted,
                            depth_undistorted,
                            pose=pose,
                            gt_pose=None,
                        )

                        self.last_integrated_id = keyframe_data.id

                        do_output = True
                        if self.last_output is not None:
                            elapsed_time = time.perf_counter() - self.last_output.timestamp
                            if elapsed_time < Parameters.kVolumetricIntegrationOutputTimeInterval:
                                do_output = False

                    elif self.last_input_task.task_type == VolumetricIntegrationTaskType.SAVE:
                        VolumetricIntegratorBase.print(
                            "VolumetricIntegratorGaussianSplatting: saving rough point cloud to {save_path}..."
                        )
                        save_path = self.last_input_task.load_save_path

                        # This is just a rough point cloud for visualization purposes.
                        # It is not the encoded gaussian splatting representation that is saved
                        # by the GaussianSplattingManager.
                        points, colors = self.volume.extract_point_cloud()
                        pc_out = VolumetricIntegrationPointCloud(points=points, colors=colors)
                        VolumetricIntegratorBase.print(
                            f"VolumetricIntegratorGaussianSplatting: saving rough point cloud to: {save_path}"
                        )
                        o3d.io.write_point_cloud(save_path, pc_out.to_o3d())

                        folder_save_path = os.path.dirname(save_path)
                        self.gsm.save(
                            folder_save_path + "/gaussian_splatting"
                        )  # save the Gaussian splatting model

                        last_output = VolumetricIntegrationOutput(self.last_input_task.task_type)
                    elif (
                        self.last_input_task.task_type
                        == VolumetricIntegrationTaskType.UPDATE_OUTPUT
                    ):
                        do_output = True

                    if do_output:
                        points, colors = self.volume.extract_point_cloud()
                        if points is not None and colors is not None:
                            if points.shape[0] > 0:

                                pc_out = VolumetricIntegrationPointCloud(
                                    points=points, colors=colors
                                )
                                VolumetricIntegratorBase.print(
                                    f"VolumetricIntegratorGaussianSplatting: id: {self.last_integrated_id} -> PointCloud: points: {pc_out.points.shape}"
                                )

                                last_output = VolumetricIntegrationOutput(
                                    self.last_input_task.task_type, self.last_integrated_id, pc_out
                                )
                                self.last_output = last_output

                    if is_running.value == 1 and last_output is not None:
                        if (
                            last_output.task_type == VolumetricIntegrationTaskType.INTEGRATE
                            or last_output.task_type == VolumetricIntegrationTaskType.UPDATE_OUTPUT
                        ):
                            with q_out_condition:
                                # push the computed output in the output queue (for viz or other tasks)
                                q_out.put(last_output)
                                q_out_condition.notify_all()
                                VolumetricIntegratorBase.print(
                                    f"VolumetricIntegratorGaussianSplatting: pushed new output to q_out size: {q_out.qsize()}"
                                )
                        elif last_output.task_type == VolumetricIntegrationTaskType.SAVE:
                            with save_request_condition:
                                save_request_completed.value = 1
                                save_request_condition.notify_all()

        except Exception as e:
            VolumetricIntegratorBase.print(
                f"VolumetricIntegratorGaussianSplatting: EXCEPTION: {e} !!!"
            )
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                VolumetricIntegratorBase.print(f"\t traceback details: {traceback_details}")

        timer.refresh()
        time_volumetric_integration.value = timer.last_elapsed
        id_info = f"last output id: {self.last_output.id}" if self.last_output is not None else ""
        VolumetricIntegratorBase.print(
            f"VolumetricIntegratorGaussianSplatting: {id_info}, last integrated id: {self.last_integrated_id}, q_in size: {q_in.qsize()}, q_out size: {q_out.qsize()}, volume-integration elapsed time: {time_volumetric_integration.value}"
        )
