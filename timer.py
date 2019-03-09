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

import cv2
from moving_average import MovingAverage

class Timer: 
    def __init__(self, name = '', is_verbose = False):
        self._name = name 
        self._is_verbose = is_verbose
        self._start = None 
        self._now = None
        self._elapsed = None         
        self.start()

    def start(self):
        self._start = cv2.getTickCount()

    def elapsed(self, do_print = False):
        self._now = cv2.getTickCount()
        self._elapsed = (self._now - self._start)/cv2.getTickFrequency()        
        if do_print is True:            
            print('Timer ', self._name, ' - elapsed: ', self._elapsed)        


class TimerFps(Timer):
    def __init__(self, name, average_width = 10, is_verbose = True): 
        super().__init__(name, is_verbose)   
        self.moving_average = MovingAverage(average_width)

    def refresh(self): 
        self.elapsed()
        self.moving_average.getAverage(self._elapsed)
        self._start = self._now
        if self._is_verbose is True:
            dT = self.moving_average.getAverage()
            print('Timer ', self._name, ' - fps: ', 1./dT, ', dT: ', dT)
        