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

import random
import string


# draw a list of points with different random colors on a input image 
def draw_points(img, pts, radius=5): 
    if img.ndim < 3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for pt in pts:
        color = tuple(np.random.randint(0,255,3).tolist())
        img = cv2.circle(img,tuple(pt),radius,color,-1)
    return img    


# draw corresponding points with the same random color on two separate images
def draw_points2(img1, img2, pts1, pts2, radius=5): 
    if img1.ndim < 3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if img2.ndim < 3:        
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),radius,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),radius,color,-1)
    return img1,img2    


# draw lines on a image; line_edges is assumed to be a list of 2D img points
def draw_lines(img, line_edges, pts=None, radius=5):
    pt = None 
    for i,l in enumerate(line_edges):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = l[0]
        x1,y1 = l[1]
        img = cv2.line(img, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
        if pts is not None: 
            pt = pts[i]        
            img = cv2.circle(img,tuple(pt),radius,color,-1)
    return img


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


# combine two images vertically
def combine_images_vertically(img1, img2): 
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)    
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)                     
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((h1+h2, max(w1, w2),3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[h1:h1+h2,:w2,:3] = img2
    return img3 


# draw features matches (images are combined horizontally)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_horizontally(img1,img2)    
    h1,w1 = img1.shape[:2]    
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size        
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i] 
        size2 = kps2_sizes[i]    
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)    # optic flow style         
        cv2.line(img3, (a,b),(c+w1,d), color, 1)  # join corrisponding points 
        cv2.circle(img3,(a,b),2, color,-1)   
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=size1, thickness=1)  # draw keypoint size as a circle 
        cv2.circle(img3,(c+w1,d),2, color,-1) 
        cv2.circle(img3,(c+w1,d), color=(0, 255, 0), radius=size2, thickness=1)  # draw keypoint size as a circle  
    return img3    


# draw features matches (images are combined vertically)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_vertically(img1,img2) 
    h1,w1 = img1.shape[:2]           
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size        
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i] 
        size2 = kps2_sizes[i]    
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)      # optic flow style   
        cv2.line(img3, (a,b),(c,d+h1), color, 1)   # join corrisponding points 
        cv2.circle(img3,(a,b),2, color,-1)   
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=size1, thickness=1)  # draw keypoint size as a circle 
        cv2.circle(img3,(c,d+h1),2, color,-1) 
        cv2.circle(img3,(c,d+h1), color=(0, 255, 0), radius=size2, thickness=1)  # draw keypoint size as a circle  
    return img3   


# draw features matches (images are combined horizontally)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None, horizontal=True):
    if horizontal: 
        return draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)    
    else:
        return draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)


def draw_random_lines(img,N=200):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        pt1x, pt2x = np.random.randint( -0.5*w, w*1.5, 2)
        pt1y, pt2y = np.random.randint( -0.5*h, h*1.5, 2)
        color = tuple(np.random.randint(0,255,3).tolist())
        thickness = np.random.randint(1, 10)
        cv2.line(img, (pt1x,pt1y), (pt2x,pt2y), color, thickness, lineType)
        
        
def draw_random_rects(img,N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        pt1x, pt2x = np.random.randint( 0, w, 2)
        pt1y, pt2y = np.random.randint( 0, h, 2)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = max(np.random.randint(-3, 10),-1)        
        cv2.rectangle(img, (pt1x,pt1y), (pt2x,pt2y), color, thickness, lineType)


def draw_random_ellipses(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]  
    axis_ext = w*0.1  
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        width, height = np.random.randint(0, axis_ext, 2)      
        angle = np.random.randint(0, 180)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = np.random.randint(-1, 9)   
        cv2.ellipse(img, (cx,cy), (width,height), angle, angle - 100, angle + 200, color, thickness, lineType)


def draw_random_polylines(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    num_pts = 3
    pts = np.zeros((num_pts,2),dtype=np.int32)
    for i in range(N):
        pts[:,0] = np.random.randint( 0, w, num_pts)
        pts[:,1] = np.random.randint( 0, h, num_pts)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = np.random.randint(1, 10)                   
        cv2.polylines(img, [pts], True, color, thickness, lineType)
        
        
def draw_random_polygons(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    num_pts = 3
    pts = np.zeros((num_pts,2),dtype=np.int32)
    for i in range(N):
        pts[:,0] = np.random.randint( 0, w, num_pts)
        pts[:,1] = np.random.randint( 0, h, num_pts)
        color = tuple(np.random.randint(0,255,3).tolist())          
        cv2.fillPoly(img, [pts], color, lineType)        


def draw_random_circles(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    radius_ext = w*0.1
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        color = tuple(np.random.randint(0,255,3).tolist())    
        radius = np.random.randint( 0, radius_ext)        
        thickness = np.random.randint(-1, 9)           
        cv2.circle(img, (cx,cy), radius, color, thickness, lineType )


def draw_random_text(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        random_char = random.choice(string.ascii_letters)
        font_face = np.random.randint( 0, 8 )
        scale = np.random.randint(0,5)+0.1
        color = tuple(np.random.randint(0,255,3).tolist())     
        thickness = np.random.randint(1, 10)                 
        cv2.putText(img, random_char, (cx,cy), font_face, scale, color, thickness, lineType);
    
    
def draw_random_img(shape): 
    #img_background = np.zeros(shape,dtype=np.uint8)           
    img_background = np.random.randint(255, size=shape,dtype=np.uint8)
    draw_random_rects(img_background)    
    draw_random_ellipses(img_background)                    
    draw_random_lines(img_background)           
    draw_random_polylines(img_background)    
    draw_random_polygons(img_background)   
    draw_random_circles(img_background)
    draw_random_text(img_background)
    img_background = cv2.GaussianBlur(img_background,ksize=(0,0),sigmaX=1)
    return img_background  

