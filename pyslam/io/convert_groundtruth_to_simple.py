import sys
import numpy as np
import os

from pyslam.config import Config

from .ground_truth import groundtruth_factory


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kResultsFolder = kRootFolder + "/results"


# Read the groundtruth from the config file and convert it into simple format
def main(out_filename=kResultsFolder + "/simple_groundtruth.txt"):

    # reading the groundtruth from config file, from a known format and converting it into simple format
    config = Config()

    print(f"converting {config.dataset_settings}")
    groundtruth = groundtruth_factory(config.dataset_settings)
    groundtruth.convertToSimpleDataset(filename=out_filename)


if __name__ == "__main__":
    main()
