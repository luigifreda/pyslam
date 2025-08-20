#!/usr/bin/env -S python3 -O
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
from pathlib import Path

from pyslam.utilities.utils_data import merge_dicts
from pyslam.utilities.utils_sys import Printer
from pyslam.utilities.utils_run import run_command_async, run_command_sync

from pyslam.evaluation.slam_evaluation_manager import SlamEvaluationManager

try:
    import hjson
except ImportError:
    print(
        "hjson not installed. Please install it with: pip install hjson"
    )  # why hjson? because it allows comments in the json file and it does not complain about the trailing commas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kSettingsFolder = kRootFolder + "/settings"
kResultsFolder = kRootFolder + "/results"
kEvaluationFolder = kRootFolder + "/pyslam/evaluation"


date_time_now = datetime.now()
date_time_now_string = date_time_now.strftime("%Y_%m_%d-%H_%M_%S")
eval_path_prefix = kResultsFolder + "/eval_" + date_time_now_string

config_dir_path = os.path.abspath(os.path.join(kEvaluationFolder, "configs"))
default_config_file_path = os.path.join(config_dir_path, "evaluation_tum.json")
default_template_config_file_path = os.path.abspath(
    os.path.join(config_dir_path, "config.template.yaml")
)


# You can use this script for two objectives:
# 1. Run an evaluation by calling main_slam.py on a set of datasets with a set of presets.
#   $ ./main_slam_evaluation.py -c configs/evaluation_<my awesome dataset>.json
# 2. If needed, you can also create a report from an existing data folder without running an evaluation (a bit hacky, for instance, after having merged data results from previous evaluation runs)
#   $ ./main_slam_evaluation.py --just-create-report -o <path to results folder containing evaluation*.json>
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Run a set of evaluations by calling main_slam.py on a set of datasets. Finally, create a summary table."
    )
    argparser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=default_config_file_path,
        help="Path of the input configuration file for the evaluations.",
    )
    argparser.add_argument(
        "-t",
        "--template-config-file",
        type=str,
        default=default_template_config_file_path,
        help="Path of the template configuration file for the evaulations.",
    )
    #
    argparser.add_argument(
        "--just-create-report", action="store_true", help="Create a report from the results folder"
    )
    argparser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default="/home/luigi/Work/slam_wss/pyslam-master-new/results/eval_2025_04_13-21_16_14/tum",
        help="Path of the output folder",
    )
    args = argparser.parse_args()

    if args.just_create_report:
        # just create a report from an existing data folder
        assert args.output_path, "Please specify the output path"
        folder = Path(args.output_path)
        config_json_file = json_file = next(folder.glob("evaluation*.json"), None)
        evaluation_manager = SlamEvaluationManager(
            str(config_json_file),
            args.template_config_file,
            just_create_report=args.just_create_report,
        )
        print(f"Creating a report from the results folder: {args.output_path}")
        evaluation_manager.output_path = args.output_path
    else:
        # run the evaluation
        evaluation_manager = SlamEvaluationManager(
            args.config_file, args.template_config_file, just_create_report=args.just_create_report
        )
        evaluation_manager.run_evaluation()

    evaluation_manager.create_final_table()

    print("Done.")
