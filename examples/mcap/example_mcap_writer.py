import os
import sys

import cv2
from pyslam.config import Config

config = Config()

from pyslam.io.mcap.mcap_writer import McapWriter

from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType

from pyslam.io.mcap.ros2_schemas import Ros2Schemas


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kOutputFolder = kRootFolder + "/results/mcap"


if __name__ == "__main__":

    if not os.path.exists(kOutputFolder):
        os.makedirs(kOutputFolder)

    dataset = dataset_factory(config)

    writer = McapWriter(os.path.join(kOutputFolder, "test_mcap_writer_output.mcap"))
    image_schema = Ros2Schemas.get_schema("image")
    writer.register_schema(image_schema["name"], image_schema["text"])
    # pointcloud_schema = Ros2Schemas.get_schema("pointcloud2")
    # writer.register_schema(pointcloud_schema["name"], pointcloud_schema["text"])
    # tf_schema = Ros2Schemas.get_schema("tf")
    # writer.register_schema(tf_schema["name"], tf_schema["text"])

    img_id = 0
    key = None
    while True:

        timestamp, img = None, None

        if dataset.is_ok:
            print("..................................")
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            img_right = (
                dataset.getImageColorRight(img_id)
                if dataset.sensor_type == SensorType.STEREO
                else None
            )

            if img is not None:
                timestamp = dataset.getTimestamp()  # get current timestamp (in seconds)
                next_timestamp = dataset.getNextTimestamp()  # get next timestamp (in seconds)
                frame_duration = (
                    next_timestamp - timestamp
                    if (timestamp is not None and next_timestamp is not None)
                    else -1
                )

                # Convert timestamp from seconds to nanoseconds (integer)
                timestamp_ns = int(timestamp * 1e9) if timestamp is not None else None

                writer.write_image_from_numpy(
                    topic=f"/camera/rgb/image_color",
                    image=img,
                    frame_id="camera_color_optical_frame",
                    encoding="rgb8",
                    stamp_ns=timestamp_ns,
                )
                print(
                    f"writing image {img_id}, shape {img.shape}, type {img.dtype} at timestamp {timestamp_ns}"
                )

                if img_right is not None:
                    writer.write_image_from_numpy(
                        topic=f"/camera/rgb/left/image_color",
                        image=img_right,
                        frame_id="camera_color_optical_frame",
                        encoding="rgb8",
                        stamp_ns=timestamp_ns,
                    )
                    print(
                        f"writing right image {img_id}, shape {img_right.shape}, type {img_right.dtype} at timestamp {timestamp_ns}"
                    )

                if depth is not None:
                    writer.write_image_from_numpy(
                        topic=f"/camera/depth/image",
                        image=depth,
                        frame_id="camera_depth_optical_frame",
                        encoding="32FC1",
                        stamp_ns=timestamp_ns,
                    )
                    print(
                        f"writing depth image {img_id}, shape {depth.shape}, type {depth.dtype} at timestamp {timestamp_ns}"
                    )

            if True and img is not None:
                cv2.imshow("img", img)
                if img_right is not None:
                    cv2.imshow("img_right", img_right)
                if depth is not None:
                    cv2.imshow("depth", depth)
                key = cv2.waitKey(1)
                if key == ord("q") or key == 27:
                    break

            img_id += 1
        else:
            break

    print("Finished writing to MCAP file")
    writer.finish()
    print("Writer finished")
