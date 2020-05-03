import sys 
import numpy as np
import cv2 
from matplotlib import pyplot as plt

sys.path.append("../../")
from config import Config

from mplot_figure import MPlotFigure
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from utils_img import combine_images_horizontally, rotate_img, transform_img, add_background
from utils_geom import add_ones
from utils_features import descriptor_sigma_mad, compute_hom_reprojection_error
from utils_draw import draw_feature_matches

from feature_tracker_configs import FeatureTrackerConfigs

from timer import TimerFps


# ==================================================================================================
# N.B.: test the feature tracker and its feature matching capability 
# ==================================================================================================


timer = TimerFps(name='detection+description+matching')


#============================================
# Select Images   
#============================================  

img1, img2 = None, None       # var initialization
img1_box = None               # image 1 bounding box (initialization)
model_fitting_type = None     # 'homography' or 'fundamental' (automatically set below, this is an initialization)
draw_horizontal_layout=True   # draw matches with the two images in an horizontal or vertical layout (automatically set below, this is an initialization) 

test_type='graph'             # select the test type (there's a template below to add your test)
#  
if test_type == 'box': 
    img1 = cv2.imread('../data/box.png')          # queryImage  
    img2 = cv2.imread('../data/box_in_scene.png') # trainImage
    model_fitting_type='homography' 
    draw_horizontal_layout = True 
#
if test_type == 'graph': 
    img1 = cv2.imread('../data/graf/img1.ppm') # queryImage
    img2 = cv2.imread('../data/graf/img3.ppm') # trainImage   img2, img3, img4
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    model_fitting_type='homography' 
    draw_horizontal_layout = True 
#
if test_type == 'kitti': 
    img1 = cv2.imread('../data/kitti06-12-color.png')
    img2 = cv2.imread('../data/kitti06-12-R-color.png')     
    #img2 = cv2.imread('../data/kitti06-13-color.png')     
    model_fitting_type='fundamental' 
    draw_horizontal_layout = False     
# 
if test_type == 'churchill': 
    img1 = cv2.imread('../data/churchill/1.ppm') 
    img2 = cv2.imread('../data/churchill/6.ppm')
    model_fitting_type='homography' 
    draw_horizontal_layout = True     
#
if test_type == 'mars': 
    # Very hard. This works with ROOT_SIFT, SUPERPOINT, CONTEXTDESC, LFNET, KEYNET ...     
    img1 = cv2.imread('../data/mars1.png') # queryImage
    img2 = cv2.imread('../data/mars2.png') # trainImage
    model_fitting_type='homography' 
    draw_horizontal_layout = True         
# 
# if test_type == 'your test':   # add your test here 
#     img1 = cv2.imread('...') 
#     img2 = cv2.imread('...')
#     model_fitting_type='...' 
#     draw_horizontal_layout = True     
    
if img1 is None:
    raise IOError('Cannot find img1')    
if img2 is None: 
    raise IOError('Cannot find img2')  
    
#============================================
# Transform Images (Optional)
#============================================  
    
M  = None     # rotation matrix on first image, if used 
H  = None     # homography matrix on first image, if used 
M2 = None     # rotation matrix on second image, if used 
H2 = None     # homography matrix on second image, if used 
    
# optionally apply a transformation to the first image 
if False: 
    img1, img1_box, M = rotate_img(img1, angle=20, scale=1.0)  # rotation and scale       
    #img1, img1_box, H = transform_img(img1, rotx=0, roty=-40, rotz=0, tx=0, ty=0, scale=1, adjust_frame=True) # homography 
    
    
# optionally regenerate the second image (override) by transforming the first image with a rotation or homography (here you have a ground-truth)
# N.B.: this procedure does not generate additional 'outlier-background' features: matching is much easier without a 'disturbing' 'background'. 
#       In order to add/generate a disturbing background, you can use the function add_background() (reported below)
if False: 
    #img2, img2_box, M2 = rotate_img(img1, angle=0, scale=1.0)  # rotation and scale       
    img2, img2_box, H2 = transform_img(img1, rotx=20, roty=30, rotz=40, tx=0, ty=0, scale=1.05, adjust_frame=True)   # homography 
    # optionally add a random background in order to generate 'outlier' features
    img2 = add_background(img2, img2_box, img_background=None) 


#============================================
# Init Feature Tracker   
#============================================  

num_features=2000 

tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
#tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

# select your tracker configuration (see the file feature_tracker_configs.py) 
tracker_config = FeatureTrackerConfigs.TEST
tracker_config['num_features'] = num_features
tracker_config['match_ratio_test'] = 0.8        # 0.7 is the default in feature_tracker_configs.py
tracker_config['tracker_type'] = tracker_type
print('feature_manager_config: ',tracker_config)

feature_tracker = feature_tracker_factory(**tracker_config)

#============================================
# Compute keypoints and descriptors  
#============================================  
    
# loop for measuring time performance 
N=1
for i in range(N):
    
    # Find the keypoints and descriptors in img1
    kps1, des1 = feature_tracker.detectAndCompute(img1)
    
    timer.start()
    # Find the keypoints and descriptors in img2    
    kps2, des2 = feature_tracker.detectAndCompute(img2)
    # Find matches    
    idx1, idx2 = feature_tracker.matcher.match(des1, des2)
    timer.refresh()


print('#kps1: ', len(kps1))
if des1 is not None: 
    print('des1 shape: ', des1.shape)
print('#kps2: ', len(kps2))
if des2 is not None: 
    print('des2 shape: ', des2.shape)    

print('number of matches: ', len(idx1))

# Convert from list of keypoints to an array of points 
kpts1 = np.array([x.pt for x in kps1], dtype=np.float32) 
kpts2 = np.array([x.pt for x in kps2], dtype=np.float32)

# Get keypoint size 
kps1_size = np.array([x.size for x in kps1], dtype=np.float32)  
kps2_size = np.array([x.size for x in kps2], dtype=np.float32) 

# Build arrays of matched keypoints, descriptors, sizes 
kps1_matched = kpts1[idx1]
des1_matched = des1[idx1][:]
kps1_size = kps1_size[idx1]

kps2_matched = kpts2[idx2]
des2_matched = des2[idx2][:]
kps2_size = kps2_size[idx2]

# compute sigma mad of descriptor distances
sigma_mad, dists = descriptor_sigma_mad(des1_matched,des2_matched,descriptor_distances=feature_tracker.descriptor_distances)
print('3 x sigma-MAD of descriptor distances (all): ', 3 * sigma_mad)


#============================================
# Model fitting for extrapolating inliers 
#============================================  

hom_reproj_threshold = 3.0  # threshold for homography reprojection error: maximum allowed reprojection error in pixels (to treat a point pair as an inlier)
fmat_err_thld = 3.0         # threshold for fundamental matrix estimation: maximum allowed distance from a point to an epipolar line in pixels (to treat a point pair as an inlier)  

# Init inliers mask
mask = None 
   
h1,w1 = img1.shape[:2]  
if kps1_matched.shape[0] > 10:
    print('model fitting for',model_fitting_type)
    if model_fitting_type == 'homography': 
        # If enough matches are found, they are passed to find the perpective transformation. Once we get the 3x3 transformation matrix, 
        # we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it on img2.  
        # N.B.: this can be properly applied only when the view change corresponds to a proper homography transformation between the two sets of keypoints 
        #       e.g.: keypoints lie on a plane, view change corresponds to a pure camera rotation  
        H, mask = cv2.findHomography(kps1_matched, kps2_matched, cv2.RANSAC, ransacReprojThreshold=hom_reproj_threshold)   
        if img1_box is None: 
            img1_box = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
        else:
            img1_box = img1_box.reshape(-1,1,2)     
        pts_dst = cv2.perspectiveTransform(img1_box,H)
        # draw the transformed box on img2  
        img2 = cv2.polylines(img2,[np.int32(pts_dst)],True,(0, 0, 255),3,cv2.LINE_AA)    
        
        reprojection_error = compute_hom_reprojection_error(H, kps1_matched, kps2_matched, mask)
        print('reprojection error: ', reprojection_error)
    else:  
        F, mask = cv2.findFundamentalMat(kps1_matched, kps2_matched, cv2.RANSAC, fmat_err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
else:
    mask = None 
    print('Not enough matches are found for', model_fitting_type)
    
    
#============================================
# Drawing  
#============================================  

img_matched_inliers = None 
if mask is not None:    
    # Build arrays of matched inliers 
    mask_idxs = (mask.ravel() == 1)    
    
    kps1_matched_inliers = kps1_matched[mask_idxs]
    kps1_size_inliers = kps1_size[mask_idxs]
    des1_matched_inliers  = des1_matched[mask_idxs][:]    
    kps2_matched_inliers = kps2_matched[mask_idxs]   
    kps2_size_inliers = kps2_size[mask_idxs]    
    des2_matched_inliers  = des2_matched[mask_idxs][:]        
    print('num inliers: ', len(kps1_matched_inliers))
    print('inliers percentage: ', len(kps1_matched_inliers)/max(len(kps1_matched),1.)*100,'%')
        
    sigma_mad_inliers, dists = descriptor_sigma_mad(des1_matched_inliers,des2_matched_inliers,descriptor_distances=feature_tracker.descriptor_distances)
    print('3 x sigma-MAD of descriptor distances (inliers): ', 3 * sigma_mad)  
    #print('distances: ', dists)  
    img_matched_inliers = draw_feature_matches(img1, img2, kps1_matched_inliers, kps2_matched_inliers, kps1_size_inliers, kps2_size_inliers,draw_horizontal_layout)    
                          
                          
img_matched = draw_feature_matches(img1, img2, kps1_matched, kps2_matched, kps1_size, kps2_size,draw_horizontal_layout)
                          
                                                
fig1 = MPlotFigure(img_matched, title='All matches')
if img_matched_inliers is not None: 
    fig2 = MPlotFigure(img_matched_inliers, title='Inlier matches')
MPlotFigure.show()