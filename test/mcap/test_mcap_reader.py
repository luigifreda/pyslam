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
        "--ros2",
        action="store_true",
        help="Decode messages as ROS2 using mcap-ros2-support",
    )
    parser.add_argument(
        "--decoded",
        action="store_true",
        help="Use NumPy-friendly decoded helpers for common message types",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="ROS2 topic(s) to filter on (comma-separated list, e.g., 'topic1,topic2,topic3')",
    )
    args = parser.parse_args()

    reader = McapReader(args.path, selected_topics=args.topics)
    summary = reader.get_summary()
    print(summary)

    time.sleep(1)

    if args.ros2:
        if args.decoded:
            for path, m, decoded in reader.iter_decoded():
                print(
                    "FILE:",
                    path,
                    "TOPIC:",
                    m.channel.topic,
                    "TYPE:",
                    m.schema.name,
                    "DECODED_KEYS:",
                    list(decoded.keys()),
                )
        else:
            for path, m in reader.iter_ros2():
                print(
                    "FILE:",
                    path,
                    "TOPIC:",
                    m.channel.topic,
                    "TYPE:",
                    m.schema.name,
                    "MSG_PYTYPE:",
                    type(m.ros_msg),
                )
    else:
        # NOTE: no support yet for ROS1 decoding
        for path, schema, channel, msg in reader:
            print("FILE:", path, "TOPIC:", channel.topic, "TS:", msg.log_time)
