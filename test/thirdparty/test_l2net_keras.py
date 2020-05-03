import sys 
sys.path.append("../../")
import config
config.cfg.set_lib('l2net_keras') 

import cv2 
import numpy as np 

from L2_Net import L2Net 


#  One of "L2Net-HP", "L2Net-HP+", "L2Net-LIB", "L2Net-LIB+", "L2Net-ND", "L2Net-ND+", "L2Net-YOS", "L2Net-YOS+",
net_name = 'L2Net-HP'
l2net = L2Net(net_name,do_tf_logging=False)

if False: 
    patches = np.random.rand(100, 32, 32, 1)
else: 
    patches = np.random.rand(100, 32, 32)    
    patches = np.expand_dims(patches, -1)
descrs = l2net.calc_descriptors(patches)

print('done!')