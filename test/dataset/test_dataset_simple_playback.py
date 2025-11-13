import os
import sys

# import multiprocessing as mp
import torch.multiprocessing as mp

import cv2
import time

from pyslam.config import Config

config = Config()


from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.utilities.timer import TimerFps


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kOutputFolder = kRootFolder + "/results/mcap"


kShowImages = True
kSyncPlayback = True  # NOTE: sync playback to respect the frame durations

if __name__ == "__main__":

    # if mp.get_start_method() != "spawn":
    #     mp.set_start_method(
    #         "spawn", force=True
    #     )  # NOTE: This may generate some pickling problems with multiprocessing
    #     #       in combination with torch and we need to check it in other places.
    #     #       This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

    if not os.path.exists(kOutputFolder):
        os.makedirs(kOutputFolder)

    dataset = dataset_factory(config)

    timer_fps = TimerFps(name="dataset_playback", average_width=10, is_verbose=True)

    img_id = 0
    key = None
    while True:

        start_loop_time = time.time()

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
                timer_fps.refresh()

                timestamp = dataset.getTimestamp()  # get current timestamp (in seconds)
                next_timestamp = dataset.getNextTimestamp()  # get next timestamp (in seconds)
                frame_duration = (
                    next_timestamp - timestamp
                    if (timestamp is not None and next_timestamp is not None)
                    else -1
                )

                print(
                    f"image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}, shape: {img.shape}, type: {img.dtype}"
                )
                if depth is not None:
                    print(f"depth: {depth.shape}, type: {depth.dtype}")
                if img_right is not None:
                    print(f"img_right: {img_right.shape}, type: {img_right.dtype}")

            if kShowImages and img is not None:
                cv2.imshow("img", img)
                if img_right is not None:
                    cv2.imshow("img_right", img_right)
                if depth is not None:
                    cv2.imshow("depth", depth)

                key = cv2.waitKey(1)
                if key == ord("q") or key == 27:
                    break

                if kSyncPlayback:
                    delta_time = time.time() - start_loop_time
                    delta_sleep = frame_duration - delta_time
                    if delta_sleep > 0:
                        time.sleep(delta_sleep)

            img_id += 1
        else:
            break

    print("done")
