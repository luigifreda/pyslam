"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2
import math 

from utils_geom import add_ones, homography_matrix
from utils_draw import draw_random_img
from utils import Printer


# combine two images horizontally
def combine_images_horizontally(img1, img2): 
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)    
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)                     
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[:h2, w1:w1+w2,:3] = img2
    return img3 


# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks 
def img_blocks(img, row_divs, col_divs):
    rows, cols = img.shape[:2]
    #print('img.shape: ', img.shape)
    xs = np.uint32(np.rint(np.linspace(0, cols, num=col_divs+1)))   # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs+1)))
    #print('img_blocks xs: ', xs)
    #print('img_blocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], y1, x1    # return block, row, col
            
            
def mask_block(mask,x1,x2,y1,y2):
    if mask is None:
        return None 
    else: 
        return mask[y1:y2, x1:x2]           
    
# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks 
def img_mask_blocks(img, mask, row_divs, col_divs):
    rows, cols = img.shape[:2]
    #print('img.shape: ', img.shape)
    xs = np.uint32(np.rint(np.linspace(0, cols, num=col_divs+1)))   # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs+1)))
    #print('img_blocks xs: ', xs)
    #print('img_blocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], mask_block(mask,x1,x2,y1,y2), y1, x1    # return block, row, col            

  
# create a generator over an image to produce a pyramid of images in the scale space by using the input scale factor 
# N.B: check the newer Pyramid class in pyramid.py! 
def pyramid(image, scale=1.2, minSize=(30, 30), gauss_filter=True, sigma0=1.0):
    level = 0 
    inv_scale = 1./scale
        
    # from https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L212
    # \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sigma_nominal = 0.5   # no filtering on the original image from https://www.vlfeat.org/api/sift.html#sift-tech-ss
    #sigma0 = 1.0       # N.B.: SIFT use 1.6 for this value
    sigma_prev = sigma_nominal      
    
    sigma_total = math.pow(scale,level) * sigma0
    print('level %d, sigma_total: %f' %(level,sigma_total))
    sigma_cur = math.sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev)    
    sigma_prev = sigma_cur     
    
    if gauss_filter: 
        image = cv2.GaussianBlur(image,ksize=(0,0),sigmaX=sigma_cur)          
    
    # yield the original image
    yield image, level

    while True:
        level += 1
        
        sigma_total = math.pow(scale,level) * sigma0
        print('level %d, sigma_total: %f' %(level,sigma_total))
        sigma_cur = math.sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev)
        sigma_prev = sigma_cur 
                            
        if gauss_filter: 
            blur = cv2.GaussianBlur(image,ksize=(0,0),sigmaX=sigma_cur)
            image = cv2.resize(blur,(0,0),fx=inv_scale,fy=inv_scale)#,interpolation = cv2.INTER_NEAREST)      
        else:          
            image = cv2.resize(image,(0,0),fx=inv_scale,fy=inv_scale)#,interpolation = cv2.INTER_NEAREST)        

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        # yield the next image in the pyramid
        yield image, level
        
        
# N.B.: if you want the mask indexs, you can return  mask_idxs = (mask.ravel() == 1)          
def mask_from_polygon(size,pts):
    pts = pts.astype(np.int32) #reshape(-1,1,2)
    mask = np.zeros(size[:2],np.uint8)
    mask = cv2.fillConvexPoly(mask,pts,255)     
    return mask    


# rotate an image by adjusting the output image size in order to contain the rotated image
# angle in degrees        
def rotate_img(img, center=None, angle=0, scale=1):
    (h, w) = img.shape[:2]
    if center is None: 
        center = (w / 2, h / 2)   
    img_box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])     
    #print('img_box:',img_box)          
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # grab sin and cos from matrix 
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h * cos))    
    # adjust the rotation matrix to take into account translation (in the new image)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]    
    rotated_img_box = (M @ add_ones(img_box).T).T   
    #print('rotated_img_box:',rotated_img_box)          
    img_out = cv2.warpAffine(img, M, (new_w, new_h))
    return img_out, rotated_img_box, M 


# transform an image by rotating and translating the camera (camera-x along image-x, camera-y along image-y, camera-z along the optical axis)
# the image is assumed to lie on the plane Z=1 (in front of the camera at distance d=1 along the optical axis);
# we compute the homography induced by the plane Z=1 when the camera is moved from [I|0] to [R|t] (see homography_matrix());
# adjust_frame => adjust the frame or not in order to contain the transformed image, in this case tx,ty are useless 
# tx=0.5 correspond to half image width (see homography_matrix());
# angles input are in degrees  
def transform_img(img,rotx,roty,rotz,tx=0,ty=0,scale=1,adjust_frame=True):
    roll  = rotx*math.pi/180.0
    pitch = roty*math.pi/180.0
    yaw   = rotz*math.pi/180.0  
    # N.B.: in the computed homography_matrix we set d=1 (see homography_matrix()) 
    # u=fx*X/Z => on Z=d=1 one has u=fx*X/1  
    # if we shift the camera of tz along Z, then one has u'=fx*X/(1-tz) 
    # hence we have a zoom_factor = 1/(1-tz) => tz = (zoom_factor - 1)/zoom_factor
    tz = (scale - 1)/scale
    (h, w) = img.shape[:2]
    center =  np.float32([w / 2, h / 2, 1])       
    img_box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])       
    #print('img_box:',img_box)       
    H = homography_matrix(img,roll,pitch,yaw,tx,ty,tz)
    #print('H:',H)
    transformed_img_box = (H @ add_ones(img_box).T) 
    transformed_img_box = (transformed_img_box[:2]/transformed_img_box[2]).T   
    transformed_center = (H @ center.T).T
    #print('transformed_img_box:',transformed_img_box)     
    if adjust_frame:   
        # adjust the frame in order to contain the transformed image 
        min_u = math.floor(transformed_img_box[:,0].min())
        max_u = math.ceil(transformed_img_box[:,0].max())
        min_v = math.floor(transformed_img_box[:,1].min())
        max_v = math.ceil(transformed_img_box[:,1].max())  
        new_w = max_u-min_u
        new_h = max_v-min_v
        if H[2,2] != 0:
            H = H/H[2, 2]
        T = np.array([[ 1,  0, -min_u],
                      [ 0,  1, -min_v],
                      [ 0,  0,     1]])    
        H = T @ H   
        transformed_img_box = (H @ add_ones(img_box).T) 
        transformed_img_box = (transformed_img_box[:2]/transformed_img_box[2]).T   
        transformed_center = (H @ center.T).T        
    else:
        # simulate the camera pose change 
        new_w = w
        new_h = h                      
    img_out = cv2.warpPerspective(img, H, (new_w,new_h))    
    return img_out, transformed_img_box, H 


# add 'disturbing' background on `img` outside the given bounding `img_box` 
def add_background(img, img_box, img_background=None):
    if img_background is None: 
        # create random image   
        img_background = draw_random_img(img.shape)
    else:
        # check if we have to resize img_background
        if img_background.shape != img.shape: 
            #print('resizing img background')
            (h, w) = img.shape[:2]
            img_background = cv2.resize(img_background,(w,h))
            # check if we have to convert to gray image            
            if img.ndim == 2:   
                img_background = cv2.cvtColor(img_background,cv2.COLOR_RGB2GRAY) 
        #print('img.shape:',img.shape,', img_background.shape:',img_background.shape)            
    mask = mask_from_polygon(img.shape,img_box) 
    inverse_mask = cv2.bitwise_not(mask)
    img_background = cv2.bitwise_or(img_background, img_background, mask=inverse_mask)
    # combine foreground+background
    final = cv2.bitwise_or(img, img_background)
    return final


def proc_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


# create a scaled image of uint8 from a image of floats 
def img_from_floats(img_flt, eps=1e-9):
    assert(img_flt.dtype in [np.float32, np.float64, np.double, np.single])
    img_max = np.amax(img_flt)
    img_min = np.amin(img_flt)
    img_range = img_max - img_min
    if img_range < eps:
        img_range = 1
    img = ((img_flt-img_min)*255/img_range).astype(np.uint8)    
    return img 


# remove borders from img 
def remove_borders(image, borders):
    shape = image.shape
    new_im = np.zeros_like(image)
    if len(shape) == 4:
        shape = [shape[1], shape[2], shape[3]]
        new_im[:, borders:shape[0]-borders, borders:shape[1]-borders, :] = image[:, borders:shape[0]-borders, borders:shape[1]-borders, :]
    elif len(shape) == 3:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders, :] = image[borders:shape[0] - borders, borders:shape[1] - borders, :]
    else:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders] = image[borders:shape[0] - borders, borders:shape[1] - borders]
    return new_im
