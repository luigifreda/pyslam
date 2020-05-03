import sys 
import numpy as np
import math 

sys.path.append("../../")
from config import Config

import g2o


position = np.zeros(3)    # delta translation 
orientation = g2o.Quaternion()

pose = g2o.Isometry3d(orientation, position)

print('pose: ', pose.matrix())

position[0] = 1

print('pose: ', pose.matrix())

orientation2 = g2o.Quaternion(g2o.AngleAxis(math.pi,[0,0,1]))

print('position', position)
print('orientation2*position', orientation2*position)


