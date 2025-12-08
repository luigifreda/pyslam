import json
import os
import cv2
import numpy as np
import argparse
import sys


from pyslam.config import Config

config = Config()

from pyslam.slam import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


def load_json_config(json_path):
    """Load the JSON configuration file."""
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


def get_image_paths(paths, extensions, frame_step, filter_out, recursive):
    image_paths = []

    paths = set(paths)  # Ensure paths are unique to begin with

    for path in paths:
        if recursive:
            for root, dirs, files in os.walk(path):
                dirs.sort()  # Sort directories to ensure a consistent traversal order
                files = sorted(
                    f
                    for f in files
                    if f.endswith(tuple(extensions)) and (filter_out not in f or filter_out == "")
                )

                # Use frame_step independently per directory
                for i, file in enumerate(files):
                    if i % frame_step == 0:  # Include only every frame_step-th frame per directory
                        file_path = os.path.join(root, file)
                        image_paths.append(file_path)
        else:
            files = sorted(
                f
                for f in os.listdir(path)
                if f.endswith(tuple(extensions)) and (filter_out not in f or filter_out == "")
            )

            # Use frame_step for the non-recursive case
            for i, file in enumerate(files):
                if i % frame_step == 0:  # Include only every frame_step-th frame
                    file_path = os.path.join(path, file)
                    image_paths.append(file_path)

    # Ensure paths are unique and sorted
    return sorted(set(image_paths))


def extract_descriptors(image_paths, feature_tracker):
    """Extract descriptors from images using OpenCV."""
    descriptors_list = []
    descriptor_dim = None

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not open {image_path}")
            continue
        print(f"Extracting descriptors from {image_path}")

        keypoints, descriptors = feature_tracker.detectAndCompute(image)
        if descriptors is not None:
            if descriptor_dim is None:
                descriptor_dim = descriptors.shape[1]
            descriptors_list.append(descriptors)

    # Stack all descriptors into a single array
    if descriptors_list:
        print(f"Extracted {len(descriptors_list)} descriptors of dimension {descriptor_dim}")
        all_descriptors = np.vstack(descriptors_list)
    else:
        all_descriptors = np.empty((0, descriptor_dim if descriptor_dim is not None else 0))
    return all_descriptors


def save_descriptors(descriptors, output_path):
    """Save the descriptors array to a file."""
    print(f"Saving descriptors {descriptors.shape} to ", output_path)
    np.save(output_path, descriptors)
    print(f"...done")


def main(json_path, output_path, feature_tracker):
    config = load_json_config(json_path)
    paths = config["paths"]
    extensions = config["extensions"]
    recursive = config.get("recursive", False)
    filter_out = config.get("filter_out", "")
    frame_step = config.get(
        "frame_step", 1
    )  # if set to N, extract descriptors from every Nth frame in each folder

    # Get image paths and extract descriptors
    image_paths = get_image_paths(paths, extensions, frame_step, filter_out, recursive)
    descriptors = extract_descriptors(image_paths, feature_tracker)

    # Save descriptors
    save_descriptors(descriptors, output_path)


# Generate an array of descriptors from images
# An example of input JSON config:
# {
#     "paths": ["/home/luigi/Work/slam_wss/pyslam-master-new/data/images/GardensPoint"],
#     "extensions": [".jpg", ".png"],
#     "filter_out": "",
#     "frame_step": 5,
#     "recursive": true
# }
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--json_config_path",
        required=False,
        type=str,
        default=kDataFolder + "/vocabulary_config.json",
        help="Path to your JSON config",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        type=str,
        default=kDataFolder + "/descriptors.npy",
        help="Path to save the descriptors array",
    )
    args = parser.parse_args()

    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 2000
    print("tracker_config: ", tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared.
    FeatureTrackerShared.set_feature_tracker(feature_tracker)

    main(args.json_config_path, args.output_file, feature_tracker)
