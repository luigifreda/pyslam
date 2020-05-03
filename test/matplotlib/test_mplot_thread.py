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

import sys
import numpy as np
import math
import time 
sys.path.append("../../")

from mplot_thread import Mplot2d, Mplot3d
from utils import getchar

if __name__ == "__main__":

    is_draw_3d = True
    plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

    time.sleep(1)
    
    is_draw_matched_points = True 
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    traj3d_gt = []
    traj3d_est = []

    #getchar()

    img_id = 0
    
    do_loop = True
    while do_loop:

        if is_draw_3d:           # draw 3d trajectory 
            traj3d_gt.append((img_id,img_id,img_id))
            traj3d_est.append((2*img_id,2*img_id,2*img_id))
            plt3d.drawTraj(traj3d_gt,'ground truth',color='r',marker='.')
            plt3d.drawTraj(traj3d_est,'estimated',color='g',marker='.')

        if is_draw_err:         # draw error signals 
            errx = [img_id, img_id]
            erry = [img_id, 2*img_id]
            errz = [img_id, 3*img_id] 
            err_plt.draw(errx,'err_x',color='g')
            err_plt.draw(erry,'err_y',color='b')
            err_plt.draw(errz,'err_z',color='r')

        if is_draw_matched_points:
            matched_kps_signal = [img_id, -img_id]
            inliers_signal = [img_id, -2*img_id]                    
            matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
            matched_points_plt.draw(inliers_signal,'# inliers',color='g')                          
    
        key = err_plt.get_key()
        if key != chr(0):
            print('key: ', err_plt.get_key())
            if key == 'q':
                plt3d.quit()
                err_plt.quit()
                matched_points_plt.quit()
                do_loop = False 

        img_id+=1 
        time.sleep(0.04)               
