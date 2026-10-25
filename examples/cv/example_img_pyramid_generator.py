import sys

from pyslam.config import Config

import time
import cv2
import numpy as np

from pyslam.utilities.img_management import pyramid


img = cv2.imread("../data/kitti06-12.png", cv2.IMREAD_COLOR)

gauss_filter = True
pyramid_generator = pyramid(img, 1.2, gauss_filter=gauss_filter)
for pyr_img, level in pyramid_generator:
    name = "level " + str(level)
    cv2.imshow(name, pyr_img)
    print(name, " size: ", pyr_img.shape)

time_start = time.time()
pyramid_generator = pyramid(img, 1.2, gauss_filter=gauss_filter)
for pyr_img, level in pyramid_generator:
    print("shape: ", pyr_img.shape)
duration = time.time() - time_start
print("duration: ", duration)

k = cv2.waitKey(0)

cv2.destroyAllWindows()
