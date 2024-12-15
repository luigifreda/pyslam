import sys 
sys.path.append("../../")
import config
import os

from utils_sys import DataDownloader
data_path = '../../data/models/'

download_json = {
}

if __name__ == "__main__":

    data_downloader = DataDownloader(download_json)
    data_downloader.start()
