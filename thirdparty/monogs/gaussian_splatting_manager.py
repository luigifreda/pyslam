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

import os
import sys
import time
import re
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import traceback

import threading
from queue import Queue

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui

import builtins

try: 
    from utils.config_utils import load_config
    from utils.dataset import load_dataset
    from utils.eval_utils import eval_ate, eval_rendering, save_gaussians, load_gaussians
    from utils.logging_utils import Log
    from utils.multiprocessing_utils import FakeQueue
    from utils.gsm_backend import GsmBackEnd
    from utils.gsm_frontend import GsmFrontEnd, GsmLoadRequest
except:
    from .utils.config_utils import load_config
    from .utils.dataset import load_dataset
    from .utils.eval_utils import eval_ate, eval_rendering, save_gaussians, load_gaussians
    from .utils.logging_utils import Log
    from .utils.multiprocessing_utils import FakeQueue
    from .utils.gsm_backend import GsmBackEnd
    from .utils.gsm_frontend import GsmFrontEnd, GsmLoadRequest


# This is a generalized version of the SLAM class. 
# It offers the possibility to push "posed" keyframes and use the the MonoGS backend. 
class GaussianSplattingManager:
    def __init__(self, config, save_results=False, save_dir=None, \
                 live_mode = True, monocular = False, \
                 use_gui = True, eval_rendering = False, \
                 use_dataset=True, \
                 print_fun=None,
                 use_frontend_tracking=False): # By default, we don't use the frontend tracking since we expect to push posed keyframes
                                               # use_frontend_tracking: True => use frontend tracking, 
                                               #                        False => use input keyframe poses or gt poses if self.dataset is set

        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
            
        self.print = print_fun if print_fun is not None else builtins.print
            
        self.save_dir = save_dir            
        if save_results:
            self.init_saving_results(config, eval_mode=eval_rendering)
        else: 
            config["Results"]["save_results"] = False
                    
        self.print(f'GaussianSplattingManager: Window size: {config["Training"]["window_size"]}')
            
        self.config = config
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = live_mode #self.config["Dataset"]["type"] == "realsense"
        self.monocular = monocular # self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = use_gui # self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = eval_rendering # self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        
        self.dataset = None
        if use_dataset:
            self.dataset = load_dataset(
                model_params, model_params.source_path, config=config
            )
        if self.dataset is None:
            self.eval_rendering = eval_rendering = False

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        input_frames_queue = Queue()
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = self.save_dir 
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = GsmFrontEnd(self.config, use_frontend_tracking, print_fun=print_fun)
        self.backend = GsmBackEnd(self.config, print_fun=print_fun)

        self.frontend.dataset = self.dataset
        self.frontend_thread = None
        
        self.frontend.input_frames_queue = input_frames_queue        
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.frontend_queue = frontend_queue
        self.backend_queue = backend_queue
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            name = "pySLAM & MonoGS"
        )
        
        self.is_stopped = False
        self.gui_process = None

    def __del__(self):
        self.stop()
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        self.config["Results"]["save_dir"] = save_dir

    def init_saving_results(self, config, eval_mode=False):
        if eval_mode:
            Log("Running MonoGS in Evaluation Mode")
            Log("Following config will be overriden")
            Log("\tsave_results=True")
            config["Results"]["save_results"] = True
            Log("\tuse_gui=False")
            config["Results"]["use_gui"] = False
            Log("\teval_rendering=True")
            config["Results"]["eval_rendering"] = True
            Log("\tuse_wandb=True")
            config["Results"]["use_wandb"] = True

        if config["Results"]["save_results"]:
            if self.save_dir is None:
                save_dir = config["Results"]["save_dir"]
            else: 
                save_dir = self.save_dir
            # At this point, save_dir represents the base directory for saving results
            
            mkdir_p(save_dir)
            
            current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = config["Dataset"]["dataset_path"].split("/")
            try:
                dataset_name = os.path.join(
                    save_dir, path[-3] + "_" + path[-2], current_datetime
                )
            except:
                dataset_name = os.path.join(
                    save_dir, path[-1], current_datetime
                )
            save_dir = dataset_name
            #tmp = config
            #tmp = tmp.split(".")[0]
            config["Results"]["save_dir"] = save_dir
            
            mkdir_p(save_dir)
            
            with open(os.path.join(save_dir, "config.yml"), "w") as file:
                documents = yaml.dump(config, file)
            
            run = wandb.init(
                project="MonoGS",
                name=f"{dataset_name}_{current_datetime}",
                config=config,
                mode=None if config["Results"]["use_wandb"] else "disabled",
            )
            wandb.define_metric("frame_idx")
            wandb.define_metric("ate*", step_metric="frame_idx")
            
            self.save_dir = save_dir
            print(f'GaussianSplattingManager: Saving results to {self.save_dir}')


    def load(self, base_path, ply_path):
        if self.use_gui and self.gui_process is None:
            self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process.start()
            time.sleep(5)
        
        print("GaussianSplattingManager: Loading map from: " + base_path)
        self.frontend.reset = False
        self.frontend.load_request = GsmLoadRequest(base_path, ply_path)         

    def start(self):
        
        self.torch_start = torch.cuda.Event(enable_timing=True)
        self.torch_end = torch.cuda.Event(enable_timing=True)
        self.torch_start.record()
                
        self.backend_process = mp.Process(target=self.backend.run)
        if self.use_gui and self.gui_process is None:
            self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process.start()
            time.sleep(5)

        self.backend.is_running = True
        self.backend_process.start()
        
        self.frontend.is_running = True
        self.frontend_thread = threading.Thread(target=self.frontend.run, name="GsmFrontend")
        self.frontend_thread.start()
        
        
    def reset(self):
        packet = gui_utils.Packet_vis2main()
        packet.flag_reset = True
        self.q_vis2main.put(packet)         
        

    def extract_point_cloud(self):
        if self.frontend.gaussians is None:
            return None, None
        else:
            points, colors = self.frontend.gaussians.extract_point_cloud_with_rgb()
            return points, colors


    def stop(self):
        if self.is_stopped:
            return
        self.is_stopped = True
        
        self.print('GaussianSplattingManager: stopping...')
                
        try: 
            self.frontend.is_running = False
            if self.frontend_thread is not None:
                if self.frontend_thread.is_alive():
                    self.frontend_thread.join()
            
            self.torch_end.record()
            self.backend_queue.put(["pause"])

            torch.cuda.synchronize()
            
            # empty the frontend queue
            N_frames = len(self.frontend.cameras)
            FPS = N_frames / (self.torch_start.elapsed_time(self.torch_end) * 0.001)
            Log("Total time", self.torch_start.elapsed_time(self.torch_end) * 0.001, tag="Eval")
            Log("Total FPS", N_frames / (self.torch_start.elapsed_time(self.torch_end) * 0.001), tag="Eval")

            if self.eval_rendering and self.save_dir is not None:
                self.print('GaussianSplattingManager: evaluating rendering...')
                self.gaussians = self.frontend.gaussians
                kf_indices = self.frontend.kf_indices
                ATE = eval_ate(
                    self.frontend.cameras,
                    self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="before_opt",
                )
                columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
                metrics_table = wandb.Table(columns=columns)
                metrics_table.add_data(
                    "Before",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )

                # re-used the frontend queue to retrive the gaussians from the backend.
                while not self.frontend_queue.empty():
                    self.frontend_queue.get()
                self.backend_queue.put(["refine_color"])
                while True:
                    if self.frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = self.frontend_queue.get()
                    if data[0] == "sync_backend" and self.frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="after_opt",
                )
                metrics_table.add_data(
                    "After",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )
                wandb.log({"Metrics": metrics_table})
                self.print(f"GsmFrontEnd: Saving Gaussians as point cloud to {self.save_dir} ...")
                save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

            self.backend_queue.put(["stop"])
            if self.backend_process.is_alive():
                self.print('GaussianSplattingManager: Stopping backend...')
                self.backend_process.join(timeout=5)
            
            self.print("GaussianSplattingManager: Backend stopped and joined the main thread")
            if self.use_gui:
                self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                if self.gui_process.is_alive():
                    self.print("GaussianSplattingManager: Stopping GUI...")
                    self.gui_process.join()
                self.print("GaussianSplattingManager: GUI Stopped and joined the main thread")
                
            self.backend.is_running = False
            self.backend.stop()
        
        except Exception as e:
            self.print('GaussianSplattingManager: Stop error', e)
            self.print(traceback.format_exc())
        
        self.print('GaussianSplattingManager: Stop completed')
        
    
    # Add a keyframe to the frontend
    def add_keyframe(self, frame_id, camera, color, depth, pose, gt_pose):
        self.frontend.input_frames_queue.put((frame_id, camera, color, depth, pose, gt_pose))
        

# get from the checkpoint folder the point cloud path corresponding to the final or highest availabel iteration
def get_target_folder(base_path):
    # List all subfolders in the base path
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # Check if the "final" subfolder exists
    if "final" in subfolders:
        return os.path.join(base_path, "final")

    # Find the highest iteration subfolder
    iteration_pattern = re.compile(r"iteration_(\d+)")
    highest_iteration = -1
    highest_folder = None

    for folder in subfolders:
        match = iteration_pattern.match(folder)
        if match:
            iteration_number = int(match.group(1))
            if iteration_number > highest_iteration:
                highest_iteration = iteration_number
                highest_folder = folder

    # Return the folder with the highest iteration or None if not found
    return os.path.join(base_path, highest_folder) if highest_folder else None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", type=str, default="", help="Path to checkpoint to load") 

    #args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()

    mp.set_start_method("spawn")
    
    if args.load != "":
        args.config = args.load + "/config.yml"
        if not os.path.exists(args.config):
            sys.exit("Checkpoint not found")
        #print(f'Loading config from {args.config}')
        args.eval = False

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None
    
    save_results=True        
    if args.load != "":
        save_results=False
        args.eval = False    

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    gsm = GaussianSplattingManager(config, save_results=save_results)
    if args.load != "":
        target_folder_path = get_target_folder(args.load + "/point_cloud")
        ply_path = os.path.join(target_folder_path, "point_cloud.ply")
        gsm.load(args.load, ply_path)
    gsm.start()
    
    #while gsm.frontend.is_running:
    while gsm.gui_process.is_alive():
        time.sleep(1) # wait for the frontend to finish
    
    gsm.stop()
    wandb.finish()

    # All done
    Log("Done.")
