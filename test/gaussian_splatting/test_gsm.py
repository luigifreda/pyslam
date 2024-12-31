import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('gaussian_splatting') 

import argparse
import time 
import re 
import os 

import wandb 
import yaml

import torch.multiprocessing as mp

from monogs.gaussian_splatting_manager import GaussianSplattingManager
from monogs.utils.config_utils import load_config


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kResultsFolder = kRootFolder + '/results/gaussian_splatting'


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


# To test the GaussianSplatterManager:
#  python test_gsm.py --config ../../thirdparty/monogs/configs/rgbd/tum/fr3_office.yaml
# To reload a saved "checkpoint" and view the model
#  python test_gsm.py --load <checkpoint_path>
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default='../../thirdparty/monogs/configs/rgbd/tum/fr3_office.yaml')
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", type=str, default="", help="Path to checkpoint to load") 

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
        print("Running MonoGS in Evaluation Mode")
        print("Following config will be overriden")
        print("\tsave_results=True")
        config["Results"]["save_results"] = True
        print("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        print("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        print("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True
                
    gsm = GaussianSplattingManager(config, save_results=save_results, save_dir=kResultsFolder)
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
    print("Done.")
