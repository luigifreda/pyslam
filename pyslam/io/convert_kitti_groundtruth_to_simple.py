import sys
import numpy as np
import os

from pyslam.io.ground_truth import groundtruth_factory

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kResultsFolder = kRootFolder + "/results"

groundtruth_settings = {}
groundtruth_settings["type"] = "kitti"
groundtruth_settings["base_path"] = "/home/luigi/Work/datasets/rgbd_datasets/kitti/dataset"
groundtruth_settings["name"] = "06"


# Convert the configured kitti ground truth in a simple format which can be used with video datasets
def main(settings=groundtruth_settings, out_filename=kResultsFolder + "simple_groundtruth.txt"):
    print(f"converting {groundtruth_settings}")
    groundtruth = groundtruth_factory(groundtruth_settings)
    groundtruth.convertToSimpleDataset(filename=out_filename)


if __name__ == "__main__":
    main()
