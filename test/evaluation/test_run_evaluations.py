import sys
sys.path.append("../../")
import config

import argparse
import csv
import re
import numpy as np
import math 

import os
import subprocess
from datetime import datetime
import shutil

import yaml
import json 
import concurrent.futures

from utils_data import merge_dicts
from utils_sys import Printer
from utils_run import run_command_async, run_command_sync

from slam_evaluation_manager import SlamEvaluationManager

try:
  import hjson
except ImportError:
  print("hjson not installed. Please install it with: pip install hjson") # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../../'
kSettingsFolder = kRootFolder + '/settings'
kResultsFolder = kRootFolder + '/results'
kEvaluationFolder = kRootFolder + '/evaluation'


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime('%Y_%m_%d-%H_%M_%S')
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

config_dir_path = os.path.abspath(os.path.join(kEvaluationFolder, 'configs'))
default_config_file_path= os.path.join(config_dir_path, 'evaluations.json')
default_template_config_file_path = os.path.abspath(os.path.join(kRootFolder, 'config.template.yaml'))
     
  
# Run an evaluation by calling main_slam.py on a set of datasets with a set of presets.
# Then, create a summary table.
if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Run a set of evaluations by calling main_slam.py on a set of datasets. Finally, create a summary table.')
  argparser.add_argument("-c", "--config-file", type=str, default=default_config_file_path, help="Path of the input configuration file for the evaulations.")
  argparser.add_argument("-t", "--template-config-file", type=str, default=default_template_config_file_path, help="Path of the template configuration file for the evaulations.")
  args = argparser.parse_args()

  evaluation_manager = SlamEvaluationManager(args.config_file, args.template_config_file)
  evaluation_manager.run_evaluations()
  evaluation_manager.create_final_table()
    
  print("Done.")