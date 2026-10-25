import os
import glob
import re
import argparse

import numpy as np
import time

try:
    import cv2
except ImportError:
    cv2 = None

from pyslam.io.mcap.mcap_reader import McapReader
from pyslam.io.mcap.mcap_image_previewer import McapImagePreviewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the MCAP file or directory",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="ROS2 topic(s) to filter on (comma-separated list, e.g., 'topic1,topic2,topic3')",
    )
    args = parser.parse_args()

    previewer = McapImagePreviewer(args.path, selected_topics=args.topics)
    previewer.visualize_images()

    time.sleep(1)
    print("Done.")
