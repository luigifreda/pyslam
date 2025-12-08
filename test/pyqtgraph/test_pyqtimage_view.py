import sys

sys.path.append("../../")
from pyslam.config import Config

import numpy as np
import cv2
import time

from pyslam.viz.qimage_thread import QimageViewer

# Example Usage
if __name__ == "__main__":
    viewer = QimageViewer.get_instance()
    do_loop = True

    border_thickness = 10

    i = 0
    j = 1

    use_parallel_cv = True  # to test if there is any issue in using a parallel cv2.imshow()

    while do_loop:
        # Create a random RGB image for testing
        random_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        viewer.draw(random_image, "random image 1")

        if use_parallel_cv:
            cv2.imshow("random image 1 - Opencv", random_image)
            cv2.waitKey(1)

        random_image2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        random_image2 = cv2.cvtColor(random_image2, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # Draw a black rectangle along the border
        height, width = random_image2.shape[:2]
        cv2.rectangle(
            random_image2, (0, 0), (width - 1, height - 1), (0, 0, 0), thickness=border_thickness
        )

        random_image2 = np.tile(random_image2, (j, 1, 1))
        viewer.draw(random_image2, "random image 2")

        time.sleep(0.04)  # Simulate dynamic updates
        i += 1

        if i % 30 == 0:
            j += 1
            if j % 10 == 0:
                j = 1

        key = viewer.get_key()

        if key and key != chr(0):
            print("key: ", key)
            if key == "q":
                viewer.quit()
                do_loop = False

    print("Done")
