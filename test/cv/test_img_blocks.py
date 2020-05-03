import sys
sys.path.append("../../")

import cv2
import numpy as np

from utils_img import img_blocks 

cols = 600
rows = 600
img = np.zeros((rows,cols,3), np.uint8)
#img[:,0:cols//2] = (255,0,0)      # (B, G, R)
#img[:,cols//2:cols] = (0,255,0)

block_generator = img_blocks(img, 4, 4)
for b, i, j in block_generator:
    color = list(np.random.choice(range(256), size=3))
    b[:,:] = color 
    print('block: (',i,',',j,') with color:', color)
    
cv2.imshow('image gray',img)
#sys.exit()

k= cv2.waitKey(0)

cv2.destroyAllWindows()