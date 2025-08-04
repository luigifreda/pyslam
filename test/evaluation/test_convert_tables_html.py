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

from pyslam.utilities.utils_eval_html import csv_list_to_html

# Example usage
if __name__ == "__main__":
    csv_list_to_html(
        csv_paths=[
            "data/table_max_10cols.csv",
            "data/table_rmse_10cols.csv",
            "data/table_max.csv",
            "data/table_rmse.csv",   
        ],
        output_html_path="pyslam_evaluation_report.html",
        title="pySLAM Evaluation Report"
    )
