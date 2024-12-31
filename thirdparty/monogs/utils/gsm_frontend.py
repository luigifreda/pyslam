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
*
* This file is part of MonoGS
* See https://github.com/muskie82/MonoGS for further information.
*
"""


import time

import numpy as np
import torch
import torch.multiprocessing as mp

import os
import re
import cv2

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils

from .camera_utils import CameraViewpoint, CameraMsg
from .eval_utils import eval_ate, save_gaussians
from .logging_utils import Log
from .multiprocessing_utils import clone_obj
from .pose_utils import update_pose
from .slam_utils import get_loss_tracking, get_median_depth

import traceback
import builtins


class GsmLoadRequest:
    def __init__(self, base_path=None, ply_path=None):
        self.base_path = base_path
        self.ply_path = ply_path


class GsmFrontEnd(mp.Process):
    def __init__(self, config, use_frontend_tracking, print_fun=None):
        super().__init__()
        self.config = config
        
        self.print = print_fun if print_fun is not None else builtins.print
        
        self.dataset = None
        self.input_frames_queue = None
        
        self.use_frontend_tracking = use_frontend_tracking # True => use frontend tracking 
                                                           # False => use input keyframe poses or gt poses if self.dataset is set
        
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.is_running = False
        self.initialized = False
        self.sent_first_gaussian_packet = False
        self.num_keyframes = 0
        
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.load_request = None   # type: GsmLoadRequest

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        
        self.request_save = False

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        
    def stop(self):
        self.is_running = False
        if self.is_alive():
            self.join()
        
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

        self.use_gui = self.config["Results"]["use_gui"]
        self.constant_velocity_warmup = 200 # TODO: fix hardcoding
		
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    # TODO: do_reset=False should be used in the case we request to init without resetting the loaded map
    def initialize(self, cur_frame_idx, viewpoint, do_reset=True):
        self.initialized = not self.monocular
        self.sent_first_gaussian_packet = False
        self.num_keyframes = 0
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose if we have a dataset
        # Otherwise, use the current frame pose set outside 
        if self.dataset is not None:
            self.print(f"GsmFrontend: Initializing first pose from dataset ground truth (do_reset: {do_reset})")
            viewpoint.T = viewpoint.T_gt

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        # TODO: do_reset=False should be used in the case we request to init without resetting the loaded map
        self.request_init(cur_frame_idx, viewpoint, depth_map, do_reset)
        self.reset = False
        self.load_request = None 
        
    def tracking(self, cur_frame_idx, viewpoint):
        
        if self.initialized and cur_frame_idx > self.constant_velocity_warmup and self.monocular:
            prev_prev = self.cameras[cur_frame_idx - self.use_every_n_frames -1 ]
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            
            pose_prev_prev = prev_prev.T
            pose_prev = prev.T
            velocity = pose_prev @ torch.linalg.inv(pose_prev_prev)
            pose_new = velocity @ pose_prev
            viewpoint.T = pose_new
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T
            
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 50 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=CameraMsg(viewpoint),
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break
        
        self.median_depth = get_median_depth(depth, opacity)         
        return render_pkg

    def render(self, viewpoint):
        render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                current_frame=CameraMsg(viewpoint),
                                gtcolor=viewpoint.original_image,
                                gtdepth=viewpoint.depth
                                if not self.monocular
                                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                            )
                        )
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
    
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = curr_frame.T
        last_kf_CW = last_kf.T
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        if not last_keyframe_idx in occ_aware_visibility:
            self.print(f'GsmFrontend: Warning: is_keyframe: last_keyframe_idx {last_keyframe_idx} not in occ_aware_visibility yet')
            return False 
        
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            if not kf_idx in occ_aware_visibility:
                self.print(f'GsmFrontend: Warning: kf_idx {kf_idx} not in occ_aware_visibility yet')
                continue
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(curr_frame.T)

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = kf_i.T
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(kf_j.T)

                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())

                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map, do_reset=True):
        msg = ["init", cur_frame_idx, viewpoint, depth_map, do_reset]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_load(self, base_path, ply_path):
        msg = ["load", base_path, ply_path]
        self.backend_queue.put(msg)

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        viewpoint_path = data[4]  # used for rendering after load request
        
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T.clone()
            
        if viewpoint_path is not None:
            viewpoint = CameraViewpoint.load(viewpoint_path)
            self.print(f'GsmFrontEnd: Loaded camera viewpoint from {viewpoint_path}')
            self.render(viewpoint)

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        self.print('GsmFrontEnd: starting...')
        
        if self.load_request is not None: 
            self.request_load(self.load_request.base_path, self.load_request.ply_path)
            self.reset = False
            self.pause = True
            # viewpoint_path = os.path.join(self.load_request.base_path, 'curr_camera.json')
            # viewpoint = CameraViewpoint.load(viewpoint_path)
            #self.render(viewpoint)
        
        cur_frame_idx = 0
        if self.dataset is not None:
            projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.dataset.fx,
                fy=self.dataset.fy,
                cx=self.dataset.cx,
                cy=self.dataset.cy,
                W=self.dataset.width,
                H=self.dataset.height,
            ).transpose(0, 1)
            projection_matrix = projection_matrix.to(device=self.device)
        else: 
            projection_matrix = None

        while self.is_running:
            try: 
                if self.q_vis2main.empty():
                    if self.pause:
                        pass
                        #continue
                else:
                    data_vis2main = self.q_vis2main.get()
                    rec_pause = data_vis2main.flag_pause
                    self.print(f'GsmFrontEnd: rec_pause: {rec_pause}')
                    if rec_pause is not None:                           
                        self.pause = rec_pause
                        self.print(f'GsmFrontEnd: pause: {self.pause}')                        
                        if self.pause:
                            self.backend_queue.put(["pause"])
                            continue
                        else:
                            self.backend_queue.put(["unpause"])                      

                    self.refine_map = data_vis2main.flag_refine_map
                    if self.refine_map:
                        self.backend_queue.put(["refine_map", data_vis2main.num_iterations])
                        self.refine_map = False
                        continue
                    
                    self.refine_color = data_vis2main.flag_refine_color
                    if self.refine_color:
                        self.backend_queue.put(["refine_color", data_vis2main.num_iterations])
                        self.refine_color = False
                        continue
                    
                    self.gui_reset = data_vis2main.flag_reset
                    if self.gui_reset:
                        self.reset = True
                        continue
                    
                    self.on_close = data_vis2main.flag_on_close
                    if self.on_close:
                        self.is_running = False
                        break
                    
                    self.save_results = data_vis2main.flag_save
                    if self.save_results and self.save_dir is not None:
                        self.print(f"GsmFrontEnd: Saving Gaussians as point cloud to {self.save_dir} ...")
                        save_gaussians(self.gaussians, self.save_dir, cur_frame_idx, final=False)
                        curr_camera = self.cameras[self.current_window[-1]]
                        curr_camera.save(os.path.join(self.save_dir,'curr_camera.json'))
                        continue
                    else: 
                        self.print(f'GsmFrontEnd: save_results: {self.save_results}, save_dir: {self.save_dir}')

                if self.frontend_queue.empty() and not self.pause:
                    if self.dataset is not None:
                        if cur_frame_idx >= len(self.dataset):
                            print("GsmFrontEnd: No more frames to process, saving results ...")
                            self.is_running = False
                            if self.save_results:
                                eval_ate(
                                    self.cameras,
                                    self.kf_indices,
                                    self.save_dir,
                                    0,
                                    final=True,
                                    monocular=self.monocular,
                                )
                                self.print(f"GsmFrontEnd: Saving Gaussians as point cloud to {self.save_dir} ...")
                                save_gaussians(self.gaussians, self.save_dir, "final", final=True, print_fun=self.print)
                            break

                    if self.requested_init:
                        time.sleep(0.01)
                        continue

                    if self.single_thread and self.requested_keyframe > 0:
                        time.sleep(0.01)
                        continue

                    if not self.initialized and self.requested_keyframe > 0:
                        time.sleep(0.01)
                        continue

                    if self.dataset is not None:
                        
                        viewpoint = CameraViewpoint.init_from_dataset(self.dataset, cur_frame_idx, projection_matrix)
                    
                    else: 
                        
                        # Let's wait the backend to complete the last keyframe insertion before inserting a new one
                        if self.requested_keyframe > 0:
                            continue 
                        
                        if self.input_frames_queue.empty():
                            continue 
                        
                        new_frame_data = self.input_frames_queue.get()
                        if new_frame_data is None:
                            break
                        cur_frame_idx, camera, image, depth, pose, gt_pose = new_frame_data
                        image = (
                            torch.from_numpy(image / 255.0)
                            .clamp(0.0, 1.0)
                            .permute(2, 0, 1)
                            .to(device=self.device, dtype=torch.float32)
                        )
                        if projection_matrix is None:
                            projection_matrix = torch.from_numpy(camera.get_render_projection_matrix()).to(self.device, dtype=torch.float32).transpose(0, 1)
                        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device, dtype=torch.float32)
                        gt_pose = torch.from_numpy(gt_pose.astype(np.float32)).to(self.device, dtype=torch.float32) if gt_pose is not None else None
                        viewpoint = CameraViewpoint(uid=cur_frame_idx, 
                                                    color=image, depth=depth.copy(), 
                                                    gt_T=gt_pose, projection_matrix=projection_matrix,
                                                    fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy, 
                                                    fovx=camera.fovx, fovy=camera.fovy, image_height=camera.height, image_width=camera.width, 
                                                    device=self.device)
                        viewpoint.T = pose # NOTE: Here we are setting the initial pose
                        self.num_keyframes += 1
                        
                    self.print("GsmFrontEnd: processing frame ", cur_frame_idx)
                    
                    
                    viewpoint.compute_grad_mask(self.config)

                    self.cameras[cur_frame_idx] = viewpoint

                    if self.reset:
                        # NOTE: When we have a load request self.reset is set to False
                        self.initialize(cur_frame_idx, viewpoint, do_reset=self.reset)
                        self.current_window.append(cur_frame_idx)
                        cur_frame_idx += 1
                        continue
                    else: 
                        if len(self.current_window) == 0:
                            self.current_window.append(cur_frame_idx)

                    self.initialized = self.initialized or (
                        len(self.current_window) == self.window_size
                    )

                    if self.use_frontend_tracking:
                        # Tracking
                        render_pkg = self.tracking(cur_frame_idx, viewpoint)
                    else:
                        if self.dataset is not None:
                            self.print(f"GsmFrontEnd: Updating frame {cur_frame_idx} with ground truth pose")
                            viewpoint.T = viewpoint.T_gt           
                        render_pkg = self.render(viewpoint)
                        time.sleep(0.01)
                        
                    #self.print(f'cur_frame_idx: {cur_frame_idx}, T: {viewpoint.T_gt}, R: {viewpoint.R_gt}')

                    current_window_dict = {}
                    current_window_dict[self.current_window[0]] = self.current_window[1:]
                    
                    gui_gaussians_update_rate = 5 if self.dataset is not None else 2 
                    gui_check_idx = cur_frame_idx if self.dataset is not None else self.num_keyframes
                    if (gui_check_idx % gui_gaussians_update_rate == 0) or (not self.sent_first_gaussian_packet):
                        # sent keyframes and gaussians to the gui
                        keyframes = [CameraMsg(self.cameras[kf_idx])
                                    for kf_idx in self.current_window]
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians),
                                keyframes=keyframes,
                                kf_window=current_window_dict,
                            )
                        )
                        if self.initialized:
                            self.sent_first_gaussian_packet = True
                    else:
                        # sent only keyframes to the gui
                        keyframes = [CameraMsg(self.cameras[kf_idx])
                                    for kf_idx in self.current_window]
                        self.q_main2vis.put(
                            gui_utils.GaussianPacket(
                                keyframes=keyframes,
                                kf_window=current_window_dict,
                            )
                        )

                    if self.dataset is not None:
                        if self.requested_keyframe > 0:
                            self.cleanup(cur_frame_idx)
                            cur_frame_idx += 1
                            continue

                    last_keyframe_idx = self.current_window[0]
                    check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                    curr_visibility = (render_pkg["n_touched"] > 0).long()
                    
                    if self.dataset is not None:
                        create_kf = self.is_keyframe(
                            cur_frame_idx,
                            last_keyframe_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                        )
                        if len(self.current_window) < self.window_size and last_keyframe_idx in self.occ_aware_visibility:
                            union = torch.logical_or(
                                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                            ).count_nonzero()
                            intersection = torch.logical_and(
                                curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                            ).count_nonzero()
                            point_ratio = intersection / union
                            create_kf = (
                                check_time
                                and point_ratio < self.config["Training"]["kf_overlap"]
                            )
                        if self.single_thread:
                            create_kf = check_time and create_kf
                    else:
                        # always create keyframe if not using dataset
                        create_kf = True
                        
                    if create_kf:
                        self.print(f'GsmFrontend: Creating keyframe at frame {cur_frame_idx}, current window: {self.current_window}')
                        self.current_window, removed = self.add_to_window(
                            cur_frame_idx,
                            curr_visibility,
                            self.occ_aware_visibility,
                            self.current_window,
                        )
                        if removed is not None:
                            self.print(f'GsmFrontend: Removed frames {removed} from window')
                        if self.monocular and not self.initialized and removed is not None:
                            self.reset = True
                            Log(
                                "Keyframes lacks sufficient overlap to initialize the map, resetting."
                            )
                            continue
                        depth_map = self.add_new_keyframe(
                            cur_frame_idx,
                            depth=render_pkg["depth"],
                            opacity=render_pkg["opacity"],
                            init=False,
                        )
                        self.request_keyframe(
                            cur_frame_idx, viewpoint, self.current_window, depth_map
                        )
                    else:
                        self.cleanup(cur_frame_idx)

                    if self.dataset is not None:
                        cur_frame_idx += 1                        
                        if (
                            self.save_results
                            and self.save_trj
                            and create_kf
                            and len(self.kf_indices) % self.save_trj_kf_intv == 0
                        ):
                            Log("GsmFrontend: Evaluating ATE at frame: ", cur_frame_idx)
                            eval_ate(
                                self.cameras,
                                self.kf_indices,
                                self.save_dir,
                                cur_frame_idx,
                                monocular=self.monocular,
                            )
                                
                else:
                    
                    if not self.frontend_queue.empty():
                        data = self.frontend_queue.get()
                        
                        if data[0] == "sync_backend":
                            self.sync_backend(data)
                            
                        elif data[0] == "gui_refresh":
                            # update gui with the latest gaussians
                            keyframes = [CameraMsg(self.cameras[kf_idx]) for kf_idx in self.cameras.keys()]
                            self.q_main2vis.put(
                                gui_utils.GaussianPacket(
                                gaussians=clone_obj(self.gaussians), keyframes=keyframes)
                            )

                        elif data[0] == "keyframe":
                            self.sync_backend(data)
                            self.requested_keyframe -= 1

                        elif data[0] == "init":
                            self.sync_backend(data)
                            self.requested_init = False
                            
                        elif data[0] == "unpause":
                            self.pause = False

                        elif data[0] == "stop":
                            Log("Frontend Stopped.")
                            break
                    
                if self.pause:
                    time.sleep(0.01)
                    
            except Exception as e:
                Log("Frontend Error: ", e)
                traceback.print_exc()

        while not self.input_frames_queue.empty():
            self.input_frames_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        while not self.q_vis2main.empty():
            self.q_vis2main.get()
        # while not self.q_main2vis.empty():
        #     self.q_main2vis.get()
        # while not self.backend_queue.empty():
        #     self.backend_queue.get()            
        
        torch.cuda.empty_cache()
        self.print(f'GsmFrontEnd finished...')