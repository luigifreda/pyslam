import sys 
import numpy as np
import math 

sys.path.append("../../")
from config import Config

import g2o


opt = g2o.SparseOptimizer() 
block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())       
solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
opt.set_algorithm(solver)

flag = g2o.Flag()
print('flag: ', flag.value)

opt.set_force_stop_flag(flag)
flag.value = False 
print('opt flag: ', opt.force_stop_flag())
flag.value = True 
print('opt flag: ', opt.force_stop_flag())

