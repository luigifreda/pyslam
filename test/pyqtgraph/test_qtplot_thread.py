import pyqtgraph as pg
import numpy as np
import time
import math

import sys
sys.path.append("../../")

from qplot_thread import Qplot2d, Qplot3d


if __name__ == "__main__":
    plotter1 = Qplot2d(xlabel='Time', ylabel='Amplitude1', title='Real-Time Plot 1')
    plotter2 = Qplot2d(xlabel='Time', ylabel='Amplitude2', title='Real-Time Plot 2')

    plotter3d = Qplot3d(xlabel='Time', ylabel='Amplitude1', zlabel='Amplitude2', title='Real-Time Plot 3D')

    x_data, y1_data, y2_data = [], [], []

    traj3d_gt = []
    traj3d_est = []
    
    i = 0
    do_loop = True
    while do_loop:
        # x_data.append(i)
        # y1_data.append(math.cos(i * 2 * math.pi / 10))
        # y2_data.append(math.sin(i * 2 * math.pi / 10))
        x_data = i
        y1_data = 0.01* i + 10 * math.sin(i * 2 * math.pi / 1000) * math.cos(i * 2 * math.pi / 10)
        y2_data = 0.01* i + math.sin(i * 2 * math.pi / 10)
        i += 1
        
        plotter1.draw([x_data, y1_data], 'Curve 1', color='r')
        plotter1.draw([x_data, y2_data], 'Curve 2', color='b')
        
        plotter2.draw([x_data, y1_data], 'Curve 1', color='r')
        plotter2.draw([x_data, y2_data], 'Curve 2', color='b')
        
        traj3d_gt.append((i,y1_data,i))
        traj3d_est.append((i+0.5,y1_data,i+0.5))        
        plotter3d.draw(traj3d_gt,'ground truth',color='r')
        plotter3d.draw(traj3d_est,'estimated',color='g')            
        
        time.sleep(0.04)
        
        key = plotter1.get_key()
        if not key: key = plotter2.get_key()
        if not key: key = plotter3d.get_key()

        if key and key != chr(0):
            print('key: ', key)
            if key == 'q':
                plotter1.quit()
                plotter2.quit()
                plotter3d.quit()
                do_loop = False