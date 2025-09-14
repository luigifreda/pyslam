import sys
import pyslam.config as config

import os

from pyslam.utilities.utils_sys import DataDownloader

data_path = "../../data/models/"

download_json = {}

if __name__ == "__main__":

    data_downloader = DataDownloader(download_json)
    data_downloader.start()
