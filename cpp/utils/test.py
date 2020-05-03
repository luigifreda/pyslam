import sys 
import numpy as np
import time 

import cv2
from matplotlib import pyplot as plt


sys.path.append("./lib/")
from orbslam2_features import ORBextractor

# read image 
img = cv2.imread('./kitti06-436.png',cv2.IMREAD_COLOR)

# main settings 
num_features=2000
num_levels=8
scale_factor=2

# declare ORB extractor
orb_extractor = ORBextractor(num_features, scale_factor, num_levels)

des = None

N=20
time_array=[]
for i in range(N):
    start = time.time()
    
    # just detect (extract keypoints) 
    #kps = orb_extractor.detect(img)    
        
    # detect and compute 
    kps, des = orb_extractor.detectAndCompute(img)
         
    # convert keypoint tuples in cv2.KeyPoints 
    kps2 = [cv2.KeyPoint(*kp) for kp in kps]
    time_array.append(time.time()-start)

#print('kps: ', kps)

mean_time = np.mean(time_array)
print('mean time: ', mean_time)

print('#kps: ', len(kps))
if des is not None: 
    print('des shape: ', des.shape)
    
# check = isinstance(kps[0],cv2.KeyPoint)
# print('check: ', check)

# keypoint = cv2.KeyPoint()
# print('keypoint type:',type(keypoint))
# print('keypoint kps[0]:',type(kps[0]))



imgDraw = cv2.drawKeypoints(img, kps2, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show image                         
plt.imshow(imgDraw)
plt.show()