import sys 
import numpy as np
import math 


from pyslam.config import Config

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


rotation2= np.eye(3).reshape(3,3)
position2= np.zeros((3,1)).ravel()
sim3 = g2o.Sim3(np.eye(3), position2, 1)
print(f'sim3: R', sim3.rotation().matrix(), ' t', sim3.translation(), ' s', sim3.scale())