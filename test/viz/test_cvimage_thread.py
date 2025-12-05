import sys
import numpy as np
import cv2
import time

from pyslam.config import Config
from pyslam.viz.cvimage_thread import CvImageViewer
from pyslam.utilities.timer import TimerFps

if __name__ == "__main__":
    viewer = CvImageViewer.get_instance()
    do_loop = True
    timer = TimerFps("Test CvImageViewer")

    desired_fps = 150.0
    frame_duration = 1 / desired_fps
    i = 0

    while do_loop:
        time_start = time.time()

        timer.refresh()
        fps = timer.get_fps()

        image1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        viewer.draw(image1, "random image1")

        image2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        viewer.draw(image2, "random image2")

        key = viewer.get_key()

        if key and key != chr(0):
            if key == "q":
                do_loop = False
                break

        i += 1
        if i % 50 == 0:
            print(f"fps: {fps:.1f}")

        time_end = time.time()
        processing_duration = time_end - time_start
        delta_time_sleep = frame_duration - processing_duration
        if delta_time_sleep > 1e-3:
            time.sleep(delta_time_sleep)

    viewer.quit()
    print("Done")
