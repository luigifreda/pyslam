import sys

from pyslam.config import Config

import time 
import cv2
import numpy as np

from pyslam.utilities.pyramid import Pyramid, PyramidType
from pyslam.utilities.utils_img import combine_images_horizontally


img = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)

pyramid_params = dict(
    num_levels=8, 
    scale_factor=1.2, 
    sigma0=1.,    
    first_level=-1,  # 0: start from image; -1: start from image*scale_factor  
    pyramid_type=PyramidType.RESIZE
)    
    
pyramid = Pyramid(**pyramid_params)
print('pyramid_type: ', pyramid.pyramid_type.name)
print('first_level: ', pyramid.first_level)

time_start = time.time()  
pyramid.compute(img)
duration = time.time() - time_start
print('duration: ', duration)

for i in range(0,pyramid.num_levels):
    name = 'level ' + str(i) + ':  img  - img_filtered'
    img_pyr = combine_images_horizontally(pyramid.imgs[i],pyramid.imgs_filtered[i])
    cv2.imshow(name,img_pyr)
    print(name, ' size: ', pyramid.imgs[i].shape)
    
k= cv2.waitKey(0)

cv2.destroyAllWindows()