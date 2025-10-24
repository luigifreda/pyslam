#!/usr/bin/env python3
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
import numpy as np


import pyslam.config as config
from pyslam.config_parameters import Parameters

USE_CPP = True
Parameters.USE_CPP_CORE = USE_CPP

import pyslam.slam.cpp as cpp_module


CKDTree2d = cpp_module.CKDTree2d
CKDTree3d = cpp_module.CKDTree3d
CKDTreeDyn = cpp_module.CKDTreeDyn

if __name__ == "__main__":

    P2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=np.float64)
    k2 = CKDTree2d(P2)

    dists, idx = k2.query(np.array([0.9, 0.9]), k=2)  # -> (array([0.141..., ...]), array([3, 1]))
    idx_r = k2.query_ball_point(np.array([0.9, 0.9]), r=0.25)  # -> indices as np.int64

    P3 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    k3 = CKDTree3d(P3)
    print(k3.n, k3.d)  # 3, 3

    PM = np.random.randn(1000, 5).astype(np.float64)
    km = CKDTreeDyn(PM)
    d, i = km.query(np.random.randn(5), k=8)
