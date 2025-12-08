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

import sys

import pyslam.config as config

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

from pyslam.utilities.data_management import merge_dicts
from pyslam.utilities.system import Printer
from pyslam.utilities.run_command import run_command_async, run_command_sync

from slam_evaluation_manager import SlamEvaluationManager

try:
    import hjson
except ImportError:
    print(
        "hjson not installed. Please install it with: pip install hjson"
    )  # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../../"
kSettingsFolder = kRootFolder + "/settings"
kResultsFolder = kRootFolder + "/results"
kEvaluationFolder = kRootFolder + "/evaluation"


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime("%Y_%m_%d-%H_%M_%S")
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

config_dir_path = os.path.abspath(os.path.join(kEvaluationFolder, "configs"))
default_config_file_path = os.path.join(config_dir_path, "evaluation.json")
default_template_config_file_path = os.path.abspath(
    os.path.join(config_dir_path, "config.template.yaml")
)


# Run an evaluation by calling main_slam.py on a set of datasets with a set of presets.
# Then, create a summary table.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run a set of evaluations by calling main_slam.py on a set of datasets. Finally, create a summary table."
    )
    argparser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=default_config_file_path,
        help="Path of the input configuration file for the evaulations.",
    )
    argparser.add_argument(
        "-t",
        "--template-config-file",
        type=str,
        default=default_template_config_file_path,
        help="Path of the template configuration file for the evaulations.",
    )
    args = argparser.parse_args()

    evaluation_manager = SlamEvaluationManager(args.config_file, args.template_config_file)
    evaluation_manager.run_evaluation()
    evaluation_manager.create_final_table()

    print("Done.")
