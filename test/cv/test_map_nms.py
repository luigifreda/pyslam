import sys
sys.path.append("../../")
import config 

import cv2
import numpy as np
from scipy import ndimage, misc

from utils_img import img_blocks 
from utils_features import nms_from_map, get_best_score_idxs, get_best_points_coordinates 


shape = (500,500)
im = np.abs(np.random.normal(loc=0, scale=255./3, size=shape)).astype(np.uint8)
cv2.imshow('image',im)

size_dilate=5
dilate_kernel = np.ones((size_dilate,size_dilate),np.uint8)
im_dilate = cv2.dilate(im, dilate_kernel)
#im_dilate = ndimage.maximum_filter(im, size=size_dilate)
cv2.imshow('image dilate',im_dilate)

im_nms = nms_from_map(im,size=50)
cv2.imshow('image nms',im_nms)

pts = get_best_points_coordinates(im_nms, num_points=10000)
img_pts = np.zeros(shape)
for p in pts: 
    cv2.circle(img_pts, (p[0],p[1]), color=(1, 0, 0), radius=2)
cv2.imshow('image pts',img_pts)

cv2.waitKey(0)
