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

import random
import time
import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import numpy as np

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim

from .logging_utils import Log
from .multiprocessing_utils import clone_obj
from .pose_utils import update_pose
from .slam_utils import get_loss_mapping
from .eval_utils import load_gaussians
from .camera_utils import CameraViewpoint

import traceback
import builtins


class GsmBackEnd(mp.Process):
    def __init__(self, config, print_fun=None):
        super().__init__()
        self.config = config
        
        self.print = print_fun if print_fun is not None else builtins.print
        
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        
        self.is_running = False
        
    def stop(self):
        self.is_running = False
        if self.is_alive():
            self.join()

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        self.print("GsmBackEnd: Initializing map...")
        iterator = tqdm(range(1, self.init_itr_num + 1), desc="GsmBackEnd: Initializing map") if self.init_itr_num >= 100 else range(self.init_itr_num)
        render_success = False
        n_touched = None
        render_pkg = None
        for mapping_iteration in iterator:
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )

            # Safeguard: render may fail and return None depending on GPU/state
            if render_pkg is None:
                self.print(
                    "GsmBackEnd: initialize_map - render returned None, aborting initialization."
                )
                # Abort initialization cleanly; nothing to record for this frame
                return None
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            render_success = True
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        if not render_success or n_touched is None:
            # No valid render iterations completed; nothing to record
            self.print(
                "GsmBackEnd: initialize_map - no successful render iterations, skipping occ_aware_visibility update."
            )
            return render_pkg

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        self.print(f"GsmBackEnd: Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1, random_window_size=2, message="curr"):
        if len(current_window) == 0:
            return
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        if iters >= 100:
            iterator = tqdm(range(1,iters+1), desc="GsmBackEnd: Mapping")
        else:
            iterator = range(iters)
            
        self.print(f"GsmBackEnd: Mapping: {message} window: {current_window} random_window_size: {random_window_size},  (iters: {iters})...")
        
        for ii in iterator:
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            #random_window = torch.randperm(len(random_viewpoint_stack))[:random_window_size]
            random_window = np.random.permutation(len(random_viewpoint_stack))[:random_window_size]
            for cam_idx in random_window:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
        
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            self.print(f"GsmBackEnd: Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.print(f"GsmBackEnd: Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                
            # if ii % 10 == 0 and ii > 0:
            #     self.push_to_frontend()
                 
        return gaussian_split


    def map_all(self, iters=1):
        try:
            window = []
            for cam_idx, viewpoint in self.viewpoints.items():
                window.append(cam_idx)
            self.map(window, iters=iters, message="all")
            self.push_to_frontend()       
            self.print(f"GsmBackEnd: Map refinement done")       
        except Exception as e:
            self.print(f"GsmBackEnd: Map refinement failed: {e}")
            print(traceback.format_exc())
              

    def map_randow_complementary_window(self, curr_window, iters=1):
        random_win_size = len(curr_window)
        complementary_window = []
        for cam_idx, viewpoint in self.viewpoints.items():
            if not cam_idx in curr_window:
                complementary_window.append(cam_idx)
        if len(complementary_window) < random_win_size:
            return               
        # Select a starting index for the contiguous subsequence
        max_start_idx = len(complementary_window) - random_win_size
        start_idx = random.randint(0, max_start_idx)
        random_window = complementary_window[start_idx:(start_idx + random_win_size)]    
        self.map(random_window, iters=iters, message="random")
        
    def color_refinement(self, iteration_total=26000):
        try:
            self.print("GsmBackEnd: Color refinement...")
            self.print(f"GsmBackEnd: Starting color refinement: num iterations {iteration_total}")

            iteration_total = iteration_total
            for iteration in tqdm(range(1, iteration_total + 1), desc="GsmBackEnd: Color refinement"):
                viewpoint_idx_stack = list(self.viewpoints.keys())
                viewpoint_cam_idx = viewpoint_idx_stack.pop(
                    random.randint(0, len(viewpoint_idx_stack) - 1)
                )
                viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                render_pkg = render(
                    viewpoint_cam, self.gaussians, self.pipeline_params, self.background
                )
                image, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - self.opt_params.lambda_dssim) * (
                    Ll1
                ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()
                with torch.no_grad():
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(iteration)
            self.print(f"GsmBackEnd: Color refinement done")
        except Exception as e:
            self.print(f"GsmBackEnd: Color refinement failed: {e}")
            print(traceback.format_exc()) 
        

    def push_to_frontend(self, tag=None, viewpoint_path=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.T.clone()))

        if tag is None:
            tag = "sync_backend"

        # NOTE: viewpoint_path is used for rendering after load request
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes, viewpoint_path]
        self.frontend_queue.put(msg)
    
    def push_to_frontend_simple(self, tag):
        msg = [tag]
        self.frontend_queue.put(msg)
    
    def run(self):
        while self.is_running:
            try: 
                
                if self.backend_queue.empty():
                    
                    if self.pause:
                        time.sleep(0.01)
                        continue
                    
                    if len(self.current_window) == 0:
                        time.sleep(0.01)
                        continue

                    if self.single_thread:
                        time.sleep(0.01)
                        continue
                                        
                    if self.last_sent >= 10:
                        self.map(self.current_window, prune=True, iters=10)
                        self.push_to_frontend()
                        time.sleep(0.005)
                    else: 
                        #self.map_all(iters=1)
                        self.map(self.current_window, iters=1, random_window_size=len(self.current_window))
                        time.sleep(0.005)                                                                              
                else:
                    data = self.backend_queue.get()
                    
                    if data[0] == "stop":
                        self.print(f'GsmBackEnd: Received stop signal...')
                        break
                    
                    elif data[0] == "pause":
                        self.pause = True
                        
                    elif data[0] == "unpause":
                        self.pause = False
                        
                    elif data[0] == "load":
                        print(f'GsmBackEnd: Loading point cloud from {data[1]}...')
                        base_path = data[1]
                        ply_path = data[2]
                        load_gaussians(self.gaussians, ply_path)
                        print(f'GsmBackEnd: Loaded point cloud from {ply_path}...')
                        viewpoint_path = os.path.join(base_path, 'last_camera.json')  # we need this for rendering purposes when loading the model                                      
                        self.push_to_frontend('sync_backend', viewpoint_path=viewpoint_path)
                        self.push_to_frontend('gui_refresh')
                        self.push_to_frontend('unpause')
                        
                    elif data[0] == "init":
                        cur_frame_idx = data[1]
                        viewpoint = data[2]
                        depth_map = data[3]
                        do_reset = data[4]
                        if do_reset:
                            self.print(f"GsmBackEnd: Resetting the system")
                            self.reset()

                        self.viewpoints[cur_frame_idx] = viewpoint
                        self.add_next_kf(
                            cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                        )
                        self.initialize_map(cur_frame_idx, viewpoint)
                        self.push_to_frontend("init")

                    elif data[0] == "keyframe":
                        cur_frame_idx = data[1]
                        viewpoint = data[2]
                        current_window = data[3]
                        depth_map = data[4]
                        
                        self.print(f"GsmBackEnd: Inserting keyframe {cur_frame_idx} ...")                        

                        self.viewpoints[cur_frame_idx] = viewpoint
                        self.current_window = current_window
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                        opt_params = []
                        frames_to_optimize = self.config["Training"]["pose_window"]
                        iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                        if not self.initialized:
                            if (
                                len(self.current_window)
                                == self.config["Training"]["window_size"]
                            ):
                                frames_to_optimize = (
                                    self.config["Training"]["window_size"] - 1
                                )
                                iter_per_kf = 50 if self.live_mode else 100 #300
                                self.print(f"GsmBackEnd: Performing initial BA for initialization")
                            else:
                                iter_per_kf = self.mapping_itr_num
                        for cam_idx in range(len(self.current_window)):
                            if self.current_window[cam_idx] == 0:
                                continue
                            viewpoint = self.viewpoints[current_window[cam_idx]]
                            if cam_idx < frames_to_optimize:
                                opt_params.append(
                                    {
                                        "params": [viewpoint.cam_rot_delta],
                                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                        * 0.5,
                                        "name": "rot_{}".format(viewpoint.uid),
                                    }
                                )
                                opt_params.append(
                                    {
                                        "params": [viewpoint.cam_trans_delta],
                                        "lr": self.config["Training"]["lr"][
                                            "cam_trans_delta"
                                        ]
                                        * 0.5,
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
                        self.keyframe_optimizers = torch.optim.Adam(opt_params)

                        self.map(self.current_window, iters=iter_per_kf)
                        self.map(self.current_window, prune=True)
                        self.push_to_frontend("keyframe")
												
                        self.print(f'GsmBackEnd: Finished with keyframe {cur_frame_idx}')
						                        
                    elif data[0] == "refine_map":
                        iters = data[1]
                        if len(data) > 1:
                            iters = data[1]
                        self.map_all(iters=iters)
                        self.push_to_frontend("gui_refresh")
                        self.push_to_frontend_simple("gui_unpause")
                        
                    elif data[0] == "refine_color":
                        iters = data[1]
                        if len(data) > 1:
                            iters = data[1]
                        self.color_refinement(iters)
                        self.push_to_frontend("gui_refresh")
                        self.push_to_frontend_simple("gui_unpause")
                    
                    else:
                        raise Exception("Unprocessed data", data)
                    
            except Exception as e:
                self.print(f"GsmBackEnd: Error in backend: {e}")
                print(traceback.format_exc())
                
        while not self.backend_queue.empty():
            self.backend_queue.get()
        # while not self.frontend_queue.empty():
        #     self.frontend_queue.get()
        
        torch.cuda.empty_cache()
        self.print(f'GsmBackEnd finished...')
        return
