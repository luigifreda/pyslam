import sys 
import numpy as np
import cv2 
from matplotlib import pyplot as plt

sys.path.append("../../")
from config import Config

from feature_tracker import feature_tracker_factory, TrackerTypes 
from feature_manager import feature_manager_factory, FeatureDetectorTypes, FeatureDescriptorTypes
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from geom_helpers import combine_images_horizontally

img1 = cv2.imread('../data/box.png',0)          # queryImage
img2 = cv2.imread('../data/box_in_scene.png',0) # trainImage

# this works with ROOT_SIFT 
#img1 = cv2.imread('../data/mars1.png',0) # queryImage
#img2 = cv2.imread('../data/mars2.png',0) # trainImage

num_features=2000 
"""
select your feature tracker 
N.B.: ORB detector (not descriptor) does not work as expected!
"""
#tracker_type = TrackerTypes.DES_BF
tracker_type = TrackerTypes.DES_FLANN

#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.BRISK, tracker_type = tracker_type)     
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.AKAZE, descriptor_type = FeatureDescriptorTypes.AKAZE, tracker_type = tracker_type)    
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 8, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.SIFT, descriptor_type = FeatureDescriptorTypes.SIFT, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.SURF, descriptor_type = FeatureDescriptorTypes.SURF, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ROOT_SIFT, descriptor_type = FeatureDescriptorTypes.ROOT_SIFT, tracker_type = tracker_type)
feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.SUPERPOINT, descriptor_type = FeatureDescriptorTypes.SUPERPOINT, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.TFEAT, tracker_type = tracker_type)

# Find the keypoints and descriptors 
kp1, des1 = feature_tracker.detectAndCompute(img1)
kp2, des2 = feature_tracker.detectAndCompute(img2)
idx1, idx2 = feature_tracker.matcher.match(des1, des2)

print('number of matches: ', len(idx1))

# Convert from list of keypoints to an array of points 
kp1 = np.array([x.pt for x in kp1], dtype=np.float32) 
kp2 = np.array([x.pt for x in kp2], dtype=np.float32) 

# Build arrays of matched points 
kp1_matched = kp1[idx1]
kp2_matched = kp2[idx2]

# If enough matches are found, they are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
# we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it on img2.      
h1,w1 = img1.shape[:2]
if kp1_matched.shape[0] > 10:
    M, mask = cv2.findHomography(kp1_matched, kp2_matched, cv2.RANSAC,5.0)
    pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
    pts_dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(pts_dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found for homography")
    
# Draw features associations
img3 = combine_images_horizontally(img1,img2)
for i,pts in enumerate(zip(kp1_matched, kp2_matched)):
    p1, p2 = np.rint(pts).astype(int)
    a,b = p1.ravel()
    c,d = p2.ravel()
    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.line(img3, (a,b),(c+w1,d), color, 1)
    cv2.circle(img3,(a,b),2, color,-1)   
    cv2.circle(img3,(c+w1,d),2, color,-1) 
                        
# Show image                         
plt.imshow(img3)
plt.show()
